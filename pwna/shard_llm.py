"""
Shard-level over-the-air attention merge inside Hugging Face causal LMs (direction D1-G).

Physical picture. The FC (edge server) runs the decoder. It holds an anchor block (the first
`anchor` tokens: BOS / system prompt / attention sink) and every token from `air_start` on (the
question and all generated tokens). The long context [anchor, air_start) is split into N contiguous
shards, one per device (its KV cache). For every query at position >= air_start, in every layer and
query head, the FC sets the shift c from its OWN logits (anchor + tail; no extra max round),
devices return block-partial sums d_j, n_j, and the merge is formed over the air (pwna.ota).
Queries before air_start (the context itself) are exact: the KV cache is assumed exact.

Typical use:
    eng = ShardEngine(seed=0); eng.install(model)
    eng.set_layout(anchor=64, air_start=T - E, n_shards=8)
    eng.calibrate(); model(cal_ids); eng.build_tables(M_list=[1, 4], q_list=[0.5, 0.8, 0.95])
    eng.evaluate("fixed", snr_db=20, M=4, mc=0, q=0.8); model(eval_ids)
    eng.off()
"""
import importlib
import math
import os
import zlib

import torch
import torch.nn.functional as F

from ota import (combiner_gains, csi_tstar, draw_channels, group_zf_noise_enhancement, imperfect_csi,
                 kappa_gains, ladder_detect, ota_merge, secure_fixed_merge, split_policy, tone_threshold,  # noqa: F401
                 zf_noise_enhancement)

DIGITAL = ("secdig", "secdigo", "digbf16")   # digital merges (no channel): see ShardEngine._digital

FAMILIES = ("llama", "qwen2", "qwen3", "mistral", "smollm3", "gemma", "gemma2", "gemma3", "phi3")
T_GRID_LOG10 = (-6.0, 12.0, 145)
LADDER_MARGIN_LO_DB = float(os.environ.get("PWNA_LADDER_LO_DB", 3.0))    # ladder range: calibrated 0.1%
LADDER_MARGIN_HI_DB = float(os.environ.get("PWNA_LADDER_HI_DB", 10.0))   # quantile of B minus LO to the
                                                                          # 99.9% quantile plus HI; above: overflow


def _repeat_kv(x, n):
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


class ShardEngine:
    def __init__(self, seed=0, combiner="best", device="cpu", floor="proj", vmax_margin=1.1):
        self.seed = seed
        self.combiner = combiner
        self.device = device
        self.floor = floor
        self.vmax_margin = vmax_margin
        self.mode = "off"
        self.anchor = self.air_start = None
        self.bounds = []
        self.cal = {}
        self.tables = None
        self.cfg = {}
        self.stats = {"sat": [], "calls": 0}
        self.salt = None            # extra RNG key (prompt / segment index); None keeps the old streams
        self._gcache = {}           # combiner gains per channel draw (identical across SNRs / policies)
        self.sigma_w = 0.0          # log-normal large-scale spread across devices [dB], fixed per device
        self.static_channel = False  # True: channel constant over the positions of a session
        # channel realism (P8): independent channels per query head (OFDM: every head's payload in its
        # own coherence block), C coherence blocks inside one instance (the device backs off uniformly
        # to its weakest block, so g_j = min_c g_jc), per-device residual phase (deg, common to the
        # instance) and phase ramp across the instance's REs (deg, std of the end-to-end difference,
        # e.g. a residual timing offset), and the I/Q-packed superposition (implied by any rotation)
        self.per_head = False
        self.coh_blocks = 1
        self.phase_deg = 0.0
        self.ramp_deg = 0.0
        self.iq = False

    # ------------------------------------------------------------------ patching
    def install(self, model):
        """Registers the merge as a custom attention implementation with SDPA-style masks (no dense
        T x T mask for full or single-token forwards); falls back to patching eager attention."""
        try:
            from transformers import AttentionInterface
            from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
            AttentionInterface.register("pwna", self._make_patch(_repeat_kv))
            AttentionMaskInterface.register("pwna", sdpa_mask)
            model.config._attn_implementation = "pwna"
            return self
        except Exception:
            pass
        for fam in FAMILIES:
            try:
                mod = importlib.import_module(f"transformers.models.{fam}.modeling_{fam}")
            except Exception:
                continue
            if hasattr(mod, "eager_attention_forward") and hasattr(mod, "repeat_kv"):
                mod.eager_attention_forward = self._make_patch(mod.repeat_kv)
        model.config._attn_implementation = "eager"
        return self

    def _make_patch(self, repeat_kv):
        eng = self

        def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            ks = repeat_kv(key, module.num_key_value_groups)
            vs = repeat_kv(value, module.num_key_value_groups)
            Tk, Tq = ks.shape[-2], query.shape[2]
            base = Tk - Tq
            mask = None if attention_mask is None else attention_mask[:, :, :, :Tk]
            if mask is not None and mask.dtype == torch.bool:  # SDPA-style mask: True = attend
                mask = torch.zeros(mask.shape, dtype=query.dtype, device=query.device).masked_fill(
                    ~mask, float("-inf"))
            if mask is None and 1 < Tq < Tk:                  # cached prefix + multi-token tail
                rows = torch.arange(base, Tk, device=query.device)[:, None]
                cols = torch.arange(Tk, device=query.device)[None, :]
                mask = torch.zeros(Tq, Tk, dtype=query.dtype, device=query.device).masked_fill(
                    cols > rows, float("-inf"))[None, None]
            if Tq == Tk and Tq > 1:
                out = F.scaled_dot_product_attention(query, ks, vs, is_causal=True, scale=scaling)
            else:
                out = F.scaled_dot_product_attention(query, ks, vs, attn_mask=mask, scale=scaling)
            if eng.mode in ("calibrate", "eval", "exact") and base + Tq > eng.air_start:
                q0 = max(0, eng.air_start - base)
                logits = torch.matmul(query[:, :, q0:].float(), ks.float().transpose(2, 3)) * scaling
                if mask is not None:
                    logits = logits + mask[:, :, q0:].float()
                elif Tq == Tk:
                    rows = torch.arange(base + q0, base + Tq, device=logits.device)[:, None]
                    cols = torch.arange(Tk, device=logits.device)[None, :]
                    logits = logits.masked_fill(cols > rows, float("-inf"))
                o_air = eng._process(module.layer_idx, logits, vs.float(), base + q0)
                if o_air is not None:
                    out = out.clone()
                    out[:, :, q0:] = o_air.to(out.dtype)
            return out.transpose(1, 2).contiguous(), None

        return patched

    # ------------------------------------------------------------------ layout / modes
    def set_layout(self, anchor, air_start, n_shards):
        self.anchor, self.air_start = int(anchor), int(air_start)
        edges = torch.linspace(self.anchor, self.air_start, n_shards + 1).round().long().tolist()
        self.bounds = [(edges[j], edges[j + 1]) for j in range(n_shards)]

    def off(self):
        self.mode = "off"

    def exact(self):
        self.mode = "exact"

    def calibrate(self):
        self.mode = "calibrate"

    def evaluate(self, policy, snr_db, M=1, mc=0, q=0.8, ladder_db=1.5, zf_group=0, tone_reps=2):
        """policy may carry suffixes: '@csi<x>' (device-side channel-estimation NMSE x), '@wf<k>'
        (weak-link fallback: the k devices with the weakest effective gain securely aggregate their
        partials digitally instead of transmitting over the air), '@q<x>' (quantile of the blind
        designs, overrides q), '@off<x>' (scale x dB above the design: controlled saturation),
        '@kap<k>' (noise-matched level report of every device, ota.kappa_gains), '@fz<p>' (digital
        thermometer handshake over Z_p: every non-empty level reads as empty with probability 1/p, so the
        FC drops k levels with probability p^-k; 'ladder' only) and
        '@b<k>' (ring bits of the secure digital merges), e.g. 'ladderdet_eff@off3' or 'secdig@b24'."""
        self.mode = "eval"
        csi, wf, off, bits, kap, mbits, fz = 0.0, 0, 0.0, 32, 0.0, 8, 0
        policy, *suffixes = policy.split("@")
        for s in suffixes:
            if s.startswith("csi"):
                csi = float(s[3:])
            elif s.startswith("wf"):
                wf = int(s[2:])
            elif s.startswith("q"):
                q = float(s[1:])
            elif s.startswith("fz"):
                fz = int(s[2:])
            elif s.startswith("m") and s[1:].isdigit():
                mbits = int(s[1:])
            elif s.startswith("kap"):
                kap = float(s[3:])
            elif s.startswith("off"):
                off = float(s[3:])
            elif s.startswith("b"):
                bits = int(s[1:])
            else:
                raise ValueError(f"unknown policy suffix @{s}")
        self.cfg = dict(policy=policy, snr_db=float(snr_db), M=int(M), csi_nmse=csi, wf=wf, off_db=off, bits=bits, kappa=kap, margin_bits=mbits, fz=fz,
                        mc=list(mc) if isinstance(mc, (list, tuple)) else [int(mc)], q=float(q),
                        ladder_db=float(ladder_db), tone_reps=int(tone_reps),
                        zf_group=int(zf_group) if zf_group else max(1, int(M) // 2))
        self.stats = {"sat": [], "calls": 0}

    # ------------------------------------------------------------------ core
    def _partials(self, logits, vs):
        """logits (B,H,T,Tk) fp32 with -inf on masked keys; vs (B,H,Tk,dv). Returns
        dA (B,H,T), nA (B,H,T,dv), d (B,H,T,N), n (B,H,T,N,dv)."""
        A, S = self.anchor, self.air_start
        c = torch.maximum(logits[..., :A].amax(-1), logits[..., S:].amax(-1))
        e = torch.exp((logits - c[..., None]).clamp(max=80.0))
        dA = e[..., :A].sum(-1) + e[..., S:].sum(-1)
        nA = torch.matmul(e[..., :A], vs[:, :, :A]) + torch.matmul(e[..., S:], vs[:, :, S:])
        d = torch.stack([e[..., lo:hi].sum(-1) for lo, hi in self.bounds], -1)
        n = torch.stack([torch.matmul(e[..., lo:hi], vs[:, :, lo:hi]) for lo, hi in self.bounds], -2)
        return dA, nA, d, n

    def _beta(self, N, device, mc):
        """Per-device log-normal large-scale gains with unit mean (equal average SNR for every
        sigma_w), fixed within a decode session (all layers and positions) and redrawn per session
        (segment / prompt and channel draw)."""
        key = ("beta", N, self.sigma_w, self.salt, mc)
        if key not in self._gcache:
            s = self.sigma_w * math.log(10.0) / 10.0
            z = torch.randn(N, generator=torch.Generator().manual_seed(
                zlib.crc32(f"{self.seed}|beta|{self.salt}|{mc}|{N}".encode())))
            self._gcache[key] = torch.exp(s * z - s * s / 2.0).to(device)
        return self._gcache[key]

    def _gen(self, mc, layer, pos, tag=""):
        """tag selects an independent stream (e.g. the handshake tones), so that the data noise of a
        policy stays identical to that of the other policies (common random numbers)."""
        key = (f"{self.seed}|{mc}|{layer}|{pos}" if self.salt is None
               else f"{self.seed}|{self.salt}|{mc}|{layer}|{pos}") + tag
        return torch.Generator(device=self.device).manual_seed(zlib.crc32(key.encode()))

    def _process(self, layer, logits, vs, pos0):
        dA, nA, d, n = self._partials(logits, vs)
        if self.mode == "exact":
            return (nA + n.sum(-2)) / (dA + d.sum(-1))[..., None]
        if self.mode == "calibrate":
            rec = self.cal.setdefault(layer, {"d": [], "dA": [], "ob2": 0.0, "cnt": 0.0, "vmax": 0.0})
            B, H, T, N = d.shape
            rec["d"].append(d.permute(1, 0, 2, 3).reshape(H, B * T, N).cpu())
            rec["dA"].append(dA.permute(1, 0, 2).reshape(H, B * T).cpu())
            rec["vmax"] = torch.maximum(torch.as_tensor(rec["vmax"]),
                                        vs.norm(dim=-1).amax(dim=(0, 2)).cpu())
            obar = n / d.clamp_min(1e-30)[..., None]
            rec["ob2"] = rec["ob2"] + (obar ** 2).sum(dim=(0, 2, 3, 4)).cpu()
            rec["cnt"] = rec["cnt"] + float(B * T * N * obar.shape[-1])
            return None
        cfg, tab = self.cfg, self.tables
        B, H, T, N = d.shape
        M = cfg["M"]
        sig_o = tab["sig_o"][layer].to(d.device).view(1, H, 1)
        vmax = (tab["vmax"][layer].to(d.device).view(1, H, 1) * self.vmax_margin
                if self.floor == "proj" else None)
        if split_policy(cfg["policy"])[0] in DIGITAL:
            return self._digital(layer, dA, nA, d, n)
        Hc = H if self.per_head else 1
        C = max(1, int(self.coh_blocks))
        R = (n.shape[-1] + 2) // 2                     # complex REs of one instance (I/Q packing)
        blk = (C,) if C > 1 else ()
        outs, sats, losses = [], [], []
        for b in range(B):
            mc = cfg["mc"][b % len(cfg["mc"])]
            gen = self._gen(mc, layer, pos0)
            if self.static_channel:          # one draw per (session, layer), constant over positions
                h = draw_channels((1, Hc, 1, *blk), N, M, self._gen(mc, layer, "static"), d.device)
                h = h.expand(1, Hc, T, *blk, N, M).contiguous()
            else:
                h = draw_channels((1, Hc, T, *blk), N, M, gen, d.device)
            if self.sigma_w > 0:
                h = h * self._beta(N, d.device, mc).sqrt()[:, None]
            align = rot = None
            iq = self.iq or self.phase_deg > 0 or self.ramp_deg > 0
            if cfg.get("csi_nmse", 0.0) > 0:
                g, align = imperfect_csi(h, cfg["csi_nmse"], self.combiner, gen, complex_out=iq or C > 1)
                if C > 1:                    # (1, Hc, T, C, N): weakest block sets g; per-RE rotation
                    ridx = torch.arange(R, device=d.device) * C // R
                    rot = align.transpose(-1, -2).index_select(-1, ridx)          # (1, Hc, T, N, R)
                    align, g = None, g.amin(-2)
                elif iq:
                    rot, align = align, None
            elif M == 1:
                g = combiner_gains(h, self.combiner)                   # (1,Hc,T,[C,]N)
                g = g.amin(-2) if C > 1 else g
            else:
                ck = (self.salt, mc, layer, pos0, T, N, M, self.combiner, self.sigma_w, Hc, C)
                g = self._gcache.get(ck)
                if g is None:
                    if len(self._gcache) > 30000:
                        self._gcache.clear()
                    g = combiner_gains(h, self.combiner)
                    g = self._gcache[ck] = g.amin(-2) if C > 1 else g
            if self.phase_deg > 0 or self.ramp_deg > 0:
                # residual phase per (device, instance), shared by the heads of a layer, with its own
                # random stream so that the data noise is common to every phase setting
                pg = self._gen(mc, layer, pos0, "|phase")
                phi = math.radians(self.phase_deg) * torch.randn((1, 1, T, N, 1), generator=pg, device=d.device)
                ramp = math.radians(self.ramp_deg) * torch.randn((1, 1, T, N, 1), generator=pg, device=d.device)
                pos = torch.arange(R, device=d.device, dtype=phi.dtype) / max(1, R - 1) - 0.5
                prot = torch.polar(torch.ones_like(phi * pos), phi + ramp * pos)            # (1,1,T,N,R)
                rot = prot if rot is None else (rot[..., None] if rot.dim() == 4 else rot) * prot
            if rot is not None and not iq:
                iq = True
            if C > 1 and cfg["policy"].startswith("zf"):
                raise ValueError("zf_tdma with coherence blocks is not modelled")
            dAb, nAb, db, nb = dA[b:b + 1], nA[b:b + 1], d[b:b + 1], n[b:b + 1]
            if cfg.get("wf", 0) > 0:
                # weak-link fallback: the wf devices with the weakest (FC-known) effective gain securely
                # aggregate their exact partials, which the FC adds to its anchor; the rest go over the air
                weak = torch.zeros_like(g).scatter_(
                    -1, g.topk(min(cfg["wf"], N), dim=-1, largest=False).indices, 1.0)   # (1, 1, T, N)
                dAb = dAb + (db * weak).sum(-1)
                nAb = nAb + (nb * weak[..., None]).sum(-2)
                db, nb = db * (1.0 - weak), nb * (1.0 - weak)[..., None]
            pol = cfg["policy"]
            base, denom = split_policy(pol)
            g_scale = (kappa_gains(g, 10.0 ** (cfg["snr_db"] / 10.0), cfg["kappa"]) if cfg.get("kappa", 0) > 0
                       else None)
            qi = tab["q_list"].index(cfg["q"]) if base in ("fixed", "csi", "classical") else None
            eta_param = zeta = force_analog = None
            if base == "fixed":
                eta_param = tab["X"][M][layer][:, qi].to(d.device).view(1, H, 1)
            elif base == "classical":
                eta_param = tab["A"][layer][:, qi].to(d.device).view(1, H, 1)
            elif base == "csi":
                eta_param = csi_tstar(g, tab["sorted_d2"][layer].to(d.device), cfg["q"],
                                      tab["t_grid"].to(d.device))
            elif base == "ladder" or base.startswith("lstale") or base.startswith("ladderoff"):
                eta_param = cfg["ladder_db"]
            elif base == "zf_sep":
                if M < N:
                    raise ValueError("zf_sep needs M >= N")
                zeta = zf_noise_enhancement(h)
            elif base == "zf_tdma":
                zeta = group_zf_noise_enhancement(h, min(cfg["zf_group"], M))
                pol = "zf_sep"
            elif base == "ladderdet":
                eta_param, force_analog = self._ladder_detect(db, g if g_scale is None else g_scale, cfg, tab, layer,
                                                              self._gen(mc, layer, pos0, "|tones"))
                pol = "fixed" if denom == "analog" else f"fixed_{denom}"
            off = cfg.get("off_db", 0.0)
            if cfg.get("fz", 0) and base == "ladder":
                u = torch.rand(db.shape[:-1], generator=self._gen(mc, layer, pos0, "|fz"), device=d.device)
                drops = torch.floor(-torch.log(u.clamp_min(1e-30)) / math.log(cfg["fz"]))
                off = off + drops * cfg["ladder_db"]
            info = {}
            o, sat = ota_merge(dAb, nAb, db, nb, g, pol, cfg["snr_db"], sig_o, eta_param=eta_param,
                               zeta=zeta, gen=gen, vmax=vmax, align=align, force_analog=force_analog,
                               off_db=off, rot=rot, info=info, g_scale=g_scale)
            outs.append(o)
            sats.append(sat)
            if "mass_loss" in info:
                losses.append(info["mass_loss"])
        self.stats["sat"].append(float(torch.cat(sats).mean()))
        if losses:
            self.stats.setdefault("mass_loss", []).append(sum(losses) / len(losses))
        self.stats["calls"] += 1
        return torch.cat(outs, 0)

    def _digital(self, layer, dA, nA, d, n):
        """Digital merges, exact up to the encoding (the digital link is assumed error-free):
          secdig   secure aggregation in fixed point with an OFFLINE per-(layer, head) scale: the
                   calibrated maximum of d_j times 2^m (m headroom bits against clipping on unseen
                   text, '@m<m>', default 8), and that times the value-norm bound for n_j; one round
          secdigo  the same with an ORACLE per-instance scale (max_j d_j, max |n_jk|): a lower bound on
                   the ring size of any public scale, i.e. the most favourable secure digital baseline
          digbf16  plain (non-secure) digital merge with bfloat16 partials
        Ring bits from the '@b<k>' suffix."""
        cfg, tab = self.cfg, self.tables
        base = split_policy(cfg["policy"])[0]
        H = d.shape[1]
        if base == "digbf16":
            db, nb = d.bfloat16().float(), n.bfloat16().float()
            return (nA + nb.sum(-2)) / (dA + db.sum(-1))[..., None]
        bits = cfg["bits"]
        if base == "secdig":
            Vd = tab["dmax"][layer].to(d.device).view(1, H, 1) * 2.0 ** cfg.get("margin_bits", 8)
            Vn = Vd * tab["vmax"][layer].to(d.device).view(1, H, 1) * self.vmax_margin
            Vd, Vn = Vd.expand_as(dA), Vn.expand_as(dA)
        else:
            Vd, Vn = d.amax(-1), n.abs().amax(dim=(-2, -1))
        out = torch.empty_like(nA)
        for h0 in range(0, H, 8):           # float64 in chunks of heads to bound the memory
            sl = slice(h0, h0 + 8)
            out[:, sl] = secure_fixed_merge(dA[:, sl], nA[:, sl], d[:, sl], n[:, sl], bits, Vd[:, sl], Vn[:, sl])
        self.stats["sat"].append(0.0)
        self.stats["calls"] += 1
        return out

    def _ladder_detect(self, d, g, cfg, tab, layer, gen):
        """Per-head ladder over the head's calibrated range of B (0.1% quantile minus LO dB to the 99.9%
        quantile plus HI dB, median as the no-detection fallback), detected by ota.ladder_detect.
        Returns (B-hat (1, H, T), overflow (1, H, T))."""
        step = cfg["ladder_db"]
        _, H, T, N = d.shape
        q_db = tab["Bdb"][(cfg["M"], cfg.get("wf", 0))][layer].to(d.device)     # (H, 3): q0.001, q0.5, q0.999
        lmin = torch.floor((q_db[:, 0] - LADDER_MARGIN_LO_DB) / step).view(1, H, 1, 1)
        lmax = torch.ceil((q_db[:, 2] + LADDER_MARGIN_HI_DB) / step).view(1, H, 1, 1)
        lmed = torch.round(q_db[:, 1] / step).view(1, H, 1, 1).clamp(min=lmin, max=lmax)
        beta = self._beta(N, d.device, cfg["mc"][0]).view(1, 1, 1, N) if self.sigma_w > 0 else None
        bhat, ovf, stats = ladder_detect(d, g, 10.0 ** (cfg["snr_db"] / 10.0), cfg["M"] * cfg["tone_reps"],
                                         step, lmin, lmax, lmed, gen, beta=beta)
        for key, v in stats.items():
            self.stats.setdefault(key, []).append(v)
        return bhat, ovf

    # ------------------------------------------------------------------ offline tables
    def build_tables(self, M_list=(1, 4), q_list=(0.5, 0.8, 0.95), n_draw=64, wf_list=(0,)):
        """Offline statistics from the calibration pass: per (layer, head) value scale sig_o,
        fixed-design quantiles X_M of max_j d_j^2 / g_j (for each antenna count M), classical
        weight-only quantiles A of max_j d_j, sorted d^2 samples for the CSI design, and the quantiles
        Bdb[(M, wf)] of B that set the handshake ladder, with the wf weakest devices excluded (weak-link
        fallback). The calibration channels include the large-scale spread sigma_w (one draw per
        session), as in the evaluation."""
        q_list = [float(q) for q in q_list]
        qs = torch.tensor(q_list)
        tab = {"q_list": q_list, "sig_o": {}, "vmax": {}, "A": {}, "sorted_d2": {},
               "X": {M: {} for M in M_list}, "Bdb": {(M, wf): {} for M in M_list for wf in wf_list},
               "t_grid": torch.logspace(*T_GRID_LOG10)}
        gen = torch.Generator().manual_seed(self.seed + 12345)
        s = self.sigma_w * math.log(10.0) / 10.0
        for layer, rec in self.cal.items():
            D = torch.cat(rec["d"], 1).float()                      # (H, S, N)
            H, S, N = D.shape
            tab["sig_o"][layer] = (rec["ob2"] / rec["cnt"]).sqrt()
            tab.setdefault("dmax", {})[layer] = D.amax(dim=(1, 2))           # (H,): secure digital scale
            tab["vmax"][layer] = torch.as_tensor(rec["vmax"]).float()
            tab["A"][layer] = torch.quantile(D.amax(-1), qs, dim=1).T.contiguous()
            tab["sorted_d2"][layer] = torch.sort((D ** 2).reshape(H, -1), dim=1).values
            for M in M_list:
                h = draw_channels((n_draw, S), N, M, gen, "cpu")
                if self.sigma_w > 0:
                    beta = torch.exp(s * torch.randn((n_draw, 1, N), generator=gen) - s * s / 2.0)
                    h = h * beta.sqrt()[..., None]
                g = combiner_gains(h, self.combiner)                # (n_draw, S, N)
                ratio = (D ** 2)[:, None] / g[None]                 # (H, n_draw, S, N)
                tab["X"][M][layer] = torch.quantile(ratio.amax(-1).reshape(H, -1), qs, dim=1).T.contiguous()
                for wf in wf_list:
                    r = ratio
                    if wf > 0:
                        weak = torch.zeros_like(g).scatter_(-1, g.topk(min(wf, N), dim=-1, largest=False).indices, 1.0)
                        r = ratio * (1.0 - weak)[None]
                    bott = r.amax(-1).reshape(H, -1)
                    tab["Bdb"][(M, wf)][layer] = 10.0 * torch.log10(torch.quantile(
                        bott, torch.tensor([1e-3, 0.5, 1 - 1e-3]), dim=1).T.clamp_min(1e-30))
        self.tables = tab
        return tab
