"""
E2: over-the-air normalized aggregation for spatial-field (radio-map) estimation with K devices
(direction D1-G, second application; the distributed-GPR task of Sato-Ishibashi, TCCN 2026).

Field: log-normal shadowing residual S(x) [dB] on a square area with Gudmundson covariance
sd^2 exp(-|x - x'| / dcor) (the path-loss trend is known and removed); measurements y = S + noise.
One aggregation instance per test point; the L test points of a round share the channel draw.
Tasks
  poe  distributed GPR, product of experts: device k fits a GP on its own N_total / K measurements
       (uniform in its cell of a grid partition: local experts); weight = local posterior precision
       1 / sigma_k^2(x), bounded in [1/sd^2, 1/sigma_min^2]; value = local posterior mean mu_k(x).
       D >= K / sd^2 is known.
  nw   Nadaraya-Watson kernel regression: device k holds one measurement; weight
       exp(-|x - x_k|^2 / 2h^2) (exponential, unbounded dynamic range, like attention), value y_k;
       the FC adds a prior pseudo-observation (weight w0, value 0) as its anchor.
Schemes
  ideal                       noise-free ratio (= digital upload)
  genie                       eta from the true weights (the FC knows them)
  ladder                      one anonymous max handshake per instance (Delta dB ladder)
  fixed                       blind: offline q-quantile of max_k w_k^2 / g_k
  ladder_hybrid, genie_hybrid digital denominator
  ladder_trunc, ladder_hybrid_trunc  as above, but devices whose effective gain g_k is below a
                              threshold are not scheduled (truncated channel inversion; the FC knows
                              g, so the selection reveals no weight); the threshold quantile of g is
                              tuned per PSNR on the calibration set, like the clipping bounds of awa
  ladder_chk                  split ratio with the digital handshake (exact highest level), exact D
                              with the convex-hull projection and the consistency check
  ladderdet, ladderdet_chk    the deployed designs: ladder handshake detected from tones (ota.ladder_detect,
                              --tone_obs = M nu observations per level, ladder range from the calibration
                              set with an overflow level), analog ratio / split ratio with safeguards
  ladderdet_trunc, ladderdet_chk_trunc  the deployed designs with weak-link truncation
  ladder_eff, ladderdet_eff[_trunc]  saturation-aware split ratio: the secure denominator carries the
                              EFFECTIVE weights a_k = min(w_k, sqrt(P g_k) / eta), so saturation only
                              re-weights the ratio (ota.split_policy)
  airmoe, airmoe_hybrid, airmoe_eff  known weights with the MSE-optimal threshold scale of AirMoE
                              (bottleneck devices may saturate), analog / exact / effective denominator
  <scheme>@off<x>             scale x dB above the design (controlled saturation)
  <scheme>@kap<k>             noise-matched level report (ota.kappa_gains)
  awa                         Sato-Ishibashi: weights clipped to [dmin, dmax] (tuned by grid search on
                              the evaluation metric, i.e. an oracle stand-in for their Bayesian
                              optimization), per-slot digital norm feedback -> separate genie scales
                              for the numerator and denominator slots, common over the L test points
  pure                        awa without clipping
  simple                      equal weights (sum of values / K)
Power convention as in ota.py (per-symbol peak power P per device, real AWGN of unit variance);
large-scale gains beta_k (equal, or log-normal with --sigma_w dB) are folded into the effective gains;
PSNR = P at beta = 1. Metric: RMSE [dB] over the L test points of a round; median and 90th
percentile over rounds.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ota import combiner_gains, draw_channels, kappa_gains, ladder_detect, ota_merge, split_policy  # noqa: E402

OTA = ("genie", "ladder", "fixed", "ladder_hybrid", "genie_hybrid", "ladder_chk", "ladder_eff",
       "airmoe", "airmoe_hybrid", "airmoe_eff")
TRUNC = ("ladder_trunc", "ladder_hybrid_trunc")
DET = ("ladderdet", "ladderdet_chk", "ladderdet_eff")
DET_POL = {"ladderdet": "fixed", "ladderdet_chk": "fixed_chk", "ladderdet_eff": "fixed_eff"}
DET_TRUNC = ("ladderdet_trunc", "ladderdet_chk_trunc", "ladderdet_eff_trunc")
TRUNC_Q = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85)
LADDER_MARGIN_LO_DB, LADDER_MARGIN_HI_DB = 3.0, 10.0       # as in shard_llm


def dist(a, b):
    return np.sqrt(((a[..., :, None, :] - b[..., None, :, :]) ** 2).sum(-1))


def sample_field(rng, pts, sd, dcor):
    """pts (R, P, 2) -> zero-mean Gudmundson field (R, P), in chunks to bound memory."""
    R, P, _ = pts.shape
    out = np.empty((R, P))
    step = max(1, int(2.5e7 // (P * P)))
    for s in range(0, R, step):
        C = sd ** 2 * np.exp(-dist(pts[s:s + step], pts[s:s + step]) / dcor)
        C += 1e-8 * sd ** 2 * np.eye(P)
        Lc = np.linalg.cholesky(C)
        out[s:s + step] = np.einsum("rij,rj->ri", Lc, rng.standard_normal((len(Lc), P)))
    return out


def make_poe(rng, R, K, a):
    """Device k holds N_total / K measurements uniform in its own cell of a gx x gy partition of the
    area (local experts). Returns weights w (R, L, K), values v (R, L, K), truth (R, L), anchor
    weight dA, known floor on D."""
    n = max(1, a.N_total // K)
    gx = int(math.sqrt(K))
    while K % gx:
        gx -= 1
    gy = K // gx
    cell = np.stack(np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij"), -1).reshape(K, 1, 2)
    size = np.array([a.area / gx, a.area / gy])
    Xk = (cell + rng.uniform(0, 1, (R, K, n, 2))) * size
    Xt = rng.uniform(0, a.area, (R, a.L, 2))
    f = sample_field(rng, np.concatenate([Xk.reshape(R, -1, 2), Xt], 1), a.sd, a.dcor)
    yk = f[:, :K * n].reshape(R, K, n) + a.sigma_m * rng.standard_normal((R, K, n))
    truth = f[:, K * n:]
    Kxx = a.sd ** 2 * np.exp(-dist(Xk, Xk) / a.dcor) + a.sigma_m ** 2 * np.eye(n)
    Kxt = a.sd ** 2 * np.exp(-dist(Xk, Xt[:, None]) / a.dcor)          # (R, K, n, L)
    alpha = np.linalg.solve(Kxx, yk[..., None])[..., 0]
    V = np.linalg.solve(Kxx, Kxt)
    mu = np.einsum("rknl,rkn->rkl", Kxt, alpha)
    var = np.clip(a.sd ** 2 - np.einsum("rknl,rknl->rkl", Kxt, V), 1e-6, None)
    w = (1.0 / var).transpose(0, 2, 1)
    v = mu.transpose(0, 2, 1)
    return w, v, truth, 0.0, K / a.sd ** 2


def make_nw(rng, R, K, a):
    Xk = rng.uniform(0, a.area, (R, K, 2))
    Xt = rng.uniform(0, a.area, (R, a.L, 2))
    f = sample_field(rng, np.concatenate([Xk, Xt], 1), a.sd, a.dcor)
    yk = f[:, :K] + a.sigma_m * rng.standard_normal((R, K))
    truth = f[:, K:]
    w = np.exp(-dist(Xt, Xk) ** 2 / (2 * a.h ** 2))                       # (R, L, K)
    v = np.broadcast_to(yk[:, None, :], w.shape).copy()
    return w, v, truth, a.w0, None


def gains(rng_t, R, K, M, beta, combiner):
    h = draw_channels((R,), K, M, rng_t, "cpu") * beta.sqrt()[..., None]
    return combiner_gains(h, combiner).double()                             # (R, K)


def schedule(w, v, g, dfl, g_min):
    """Weights, partial numerators and the known floor on D after weak-link truncation: devices with
    g < g_min are not scheduled (the strongest device always is)."""
    d, n = w, (w * v)[..., None]
    if g_min is not None:
        keep = ((g >= g_min) | (g == g.amax(-1, keepdim=True))).to(w.dtype)       # (R, K)
        d = w * keep[:, None, :]
        n = n * keep[:, None, :, None]
        if dfl is not None:                         # the known floor on D scales with the scheduled devices
            dfl = dfl * keep.sum(-1, keepdim=True) / keep.shape[-1]
    return d, n, dfl


def run_ota(pol, w, v, dA, dfl, g, P_db, a, gen, X=None, g_min=None, off_db=0.0, kap=0.0):
    d, n, dfl = schedule(w, v, g, dfl, g_min)
    dAt = torch.full(d.shape[:-1], float(dA), dtype=d.dtype)
    nA = torch.zeros((*d.shape[:-1], 1), dtype=d.dtype)
    sig = torch.tensor(a.sd, dtype=d.dtype)
    base = split_policy(pol)[0]
    ep = a.ladder_db if base == "ladder" else (X if base == "fixed" else None)
    o, sat = ota_merge(dAt, nA, d, n, g[:, None, :], pol, P_db, sig, eta_param=ep, gen=gen,
                       vmax=torch.tensor(a.vmax_sd * a.sd, dtype=d.dtype), dfloor=dfl, off_db=off_db,
                       g_scale=kappa_gains(g[:, None, :], 10 ** (P_db / 10), kap) if kap > 0 else None)
    return o[..., 0], float(sat.mean())


def ladder_levels(w, g, step, g_min=None):
    """Ladder range from a calibration set: the 0.1% and 99.9% quantiles of B = max_k d_k^2 / g_k over
    the scheduled devices, widened by the margins of shard_llm, and the median as the fallback."""
    keep = torch.ones_like(g) if g_min is None else ((g >= g_min) | (g == g.amax(-1, keepdim=True))).to(g.dtype)
    B = ((w * keep[:, None, :]) ** 2 / g[:, None, :]).amax(-1).flatten()
    x = 10.0 * torch.log10(B.clamp_min(1e-300))
    x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(0))[:400000]]
    q = torch.quantile(x, torch.tensor([1e-3, 0.5, 1 - 1e-3], dtype=x.dtype))
    lmin = torch.floor((q[0] - LADDER_MARGIN_LO_DB) / step)
    lmax = torch.ceil((q[2] + LADDER_MARGIN_HI_DB) / step)
    return lmin, lmax, torch.round(q[1] / step).clamp(min=float(lmin), max=float(lmax))


def run_det(pol, w, v, dA, dfl, g, beta, P_db, a, gen, gen_t, levels, g_min=None, off_db=0.0, kap=0.0):
    """Deployed designs with the tone handshake: 'ladderdet' (analog ratio) or 'ladderdet_chk' (split
    ratio with the overflow fallback, the consistency check and the projection)."""
    d, n, dfl = schedule(w, v, g, dfl, g_min)
    lmin, lmax, lmed = (x.view(1, 1, 1).to(d.dtype) for x in levels)
    gl = kappa_gains(g[:, None, :], 10 ** (P_db / 10), kap)
    bhat, ovf, st = ladder_detect(d, gl, 10 ** (P_db / 10), a.tone_obs, a.ladder_db, lmin, lmax,
                                  lmed, gen_t, beta=beta[:, None, :].to(d.dtype))
    dAt = torch.full(d.shape[:-1], float(dA), dtype=d.dtype)
    nA = torch.zeros((*d.shape[:-1], 1), dtype=d.dtype)
    o, sat = ota_merge(dAt, nA, d, n, g[:, None, :], DET_POL[pol], P_db,
                       torch.tensor(a.sd, dtype=d.dtype), eta_param=bhat, gen=gen,
                       vmax=torch.tensor(a.vmax_sd * a.sd, dtype=d.dtype), dfloor=dfl, force_analog=ovf,
                       off_db=off_db)
    return o[..., 0], float(sat.mean()), st


def run_awa(w, v, g, P_db, a, gen, dmin=0.0, dmax=math.inf):
    P = 10 ** (P_db / 10)
    wc = w.clamp(dmin, dmax)
    m0, m1 = wc * v / a.sd, wc
    e0 = (P * (g / (m0 ** 2).amax(1).clamp_min(1e-30)).amin(-1)).sqrt()     # (R,)
    e1 = (P * (g / (m1 ** 2).amax(1).clamp_min(1e-30)).amin(-1)).sqrt()
    num = a.sd * (m0.sum(-1) + torch.randn(m0.shape[:2], generator=gen, dtype=w.dtype) / e0[:, None])
    den = m1.sum(-1) + torch.randn(m1.shape[:2], generator=gen, dtype=w.dtype) / e1[:, None]
    return num / den


def run_simple(v, g, P_db, a, gen):
    P = 10 ** (P_db / 10)
    s = v / a.sd
    e = (P * (g / (s ** 2).amax(1).clamp_min(1e-30)).amin(-1)).sqrt()
    K = v.shape[-1]
    return a.sd * (s.sum(-1) + torch.randn(s.shape[:2], generator=gen, dtype=v.dtype) / e[:, None]) / K


def rmse(est, truth):
    r = ((est - truth) ** 2).mean(-1).sqrt()
    r = torch.where(torch.isfinite(r), r, torch.full_like(r, 1e6))
    return float(r.median()), float(r.quantile(0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["poe", "nw"], default="poe")
    ap.add_argument("--K", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 40])
    ap.add_argument("--schemes", nargs="+",
                    default=["genie", "ladder", "fixed", "ladder_hybrid", "genie_hybrid", "ladder_chk",
                             "ladderdet", "ladderdet_chk", "ladder_trunc", "ladder_hybrid_trunc",
                             "ladderdet_trunc", "ladderdet_chk_trunc", "awa", "pure", "simple"])
    ap.add_argument("--tone_obs", type=int, default=16,
                    help="handshake observations per level, M nu (REs x FC antennas)")
    ap.add_argument("--area", type=float, default=500.0)
    ap.add_argument("--dcor", type=float, default=100.0)
    ap.add_argument("--sd", type=float, default=8.0)
    ap.add_argument("--sigma_m", type=float, default=1.0, help="measurement noise [dB]")
    ap.add_argument("--N_total", type=int, default=256, help="poe: measurements over all devices")
    ap.add_argument("--h", type=float, default=30.0, help="nw: kernel bandwidth [m]")
    ap.add_argument("--w0", type=float, default=math.exp(-2.0), help="nw: FC prior pseudo-weight")
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--cal_runs", type=int, default=500)
    ap.add_argument("--sigma_w", type=float, default=0.0, help="log-normal large-scale spread [dB]")
    ap.add_argument("--ladder_db", type=float, default=3.0)
    ap.add_argument("--q", type=float, default=0.8)
    ap.add_argument("--vmax_sd", type=float, default=5.0)
    ap.add_argument("--combiner", default="best")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "local", "field_sim.json"))
    a = ap.parse_args()
    make = make_poe if a.task == "poe" else make_nw
    rows = []
    t0 = time.time()
    for K in a.K:
        rng = np.random.default_rng([a.seed, K, 0])
        w, v, truth, dA, dfl = make(rng, a.runs, K, a)
        wc_, vc_, tc_, _, _ = make(np.random.default_rng([a.seed, K, 1]), a.cal_runs, K, a)
        w, v, truth = (torch.from_numpy(np.ascontiguousarray(x)) for x in (w, v, truth))
        wc_, vc_, tc_ = (torch.from_numpy(np.ascontiguousarray(x)) for x in (wc_, vc_, tc_))
        ideal = ((dA * 0 + (w * v).sum(-1)) / (dA + w.sum(-1)))
        r_ideal = rmse(ideal, truth)
        lw = torch.log10(w.amax(-1) / w.amin(-1).clamp_min(1e-300))
        print(f"[{a.task} K={K}] ideal RMSE med {r_ideal[0]:.3f} dB p90 {r_ideal[1]:.3f}; "
              f"log10 weight range per instance: median {float(lw.median()):.1f}, "
              f"N_eff median {float((w.sum(-1) ** 2 / (w ** 2).sum(-1)).median()):.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        rows.append(dict(task=a.task, K=K, M=0, snr_db=None, scheme="ideal", rmse_med=r_ideal[0],
                         rmse_p90=r_ideal[1]))
        s = a.sigma_w * math.log(10.0) / 10.0       # unit-mean log-normal: same average SNR for every sigma_w
        beta = torch.exp(s * torch.from_numpy(np.random.default_rng([a.seed, K, 2]).standard_normal(
            (a.runs, K))) - s * s / 2)
        beta_c = torch.exp(s * torch.from_numpy(np.random.default_rng([a.seed, K, 3]).standard_normal(
            (a.cal_runs, K))) - s * s / 2)
        for M in a.M:
            gt = torch.Generator().manual_seed(a.seed * 1000 + K * 10 + M)
            g = gains(gt, a.runs, K, M, beta.float(), a.combiner)
            gc = gains(gt, a.cal_runs, K, M, beta_c.float(), a.combiner)
            X = torch.quantile((wc_ ** 2 / gc[:, None, :]).amax(-1).flatten(), a.q)
            sub = torch.randperm(wc_.numel(), generator=torch.Generator().manual_seed(a.seed))[:200000]
            wq = torch.quantile(wc_.flatten()[sub],
                                torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99, 1.0], dtype=wc_.dtype))
            levels = ladder_levels(wc_, gc, a.ladder_db)
            for snr in a.snrs:
                gen = torch.Generator().manual_seed(a.seed * 7919 + K * 31 + M * 7 + int(snr))
                gen_t = torch.Generator().manual_seed(a.seed * 104729 + K * 31 + M * 7 + int(snr))
                for sch_full in a.schemes:
                    sch, *sfx = sch_full.split("@")      # '@off<x>': scale x dB above the design
                    off = sum(float(x[3:]) for x in sfx if x.startswith("off"))
                    kap = sum(float(x[3:]) for x in sfx if x.startswith("kap"))
                    extra = {}
                    if sch in OTA:
                        est, sat = run_ota(sch, w, v, dA, dfl, g, snr, a, gen, X=X, off_db=off, kap=kap)
                        extra["sat"] = sat
                    elif sch in DET:
                        est, sat, st = run_det(sch, w, v, dA, dfl, g, beta, snr, a, gen, gen_t, levels, off_db=off,
                                               kap=kap)
                        extra.update(sat=sat, **st)
                    elif sch in DET_TRUNC:
                        pol = sch[:-6]
                        best = None
                        for qs in TRUNC_Q:
                            gm = float(torch.quantile(gc.flatten(), qs)) if qs > 0 else None
                            lv = ladder_levels(wc_, gc, a.ladder_db, gm)
                            r = rmse(run_det(pol, wc_, vc_, dA, dfl, gc, beta_c, snr, a, gen, gen_t, lv, gm,
                                             off_db=off, kap=kap)[0], tc_)[0]
                            if best is None or r < best[0]:
                                best = (r, qs, gm, lv)
                        est, sat, st = run_det(pol, w, v, dA, dfl, g, beta, snr, a, gen, gen_t, best[3], best[2],
                                               off_db=off, kap=kap)
                        extra.update(sat=sat, trunc_q=best[1], **st)
                    elif sch in TRUNC:
                        pol = sch[:-6]
                        best = None
                        for qs in TRUNC_Q:
                            gm = float(torch.quantile(gc.flatten(), qs)) if qs > 0 else None
                            r = rmse(run_ota(pol, wc_, vc_, dA, dfl, gc, snr, a, gen, g_min=gm)[0], tc_)[0]
                            if best is None or r < best[0]:
                                best = (r, qs, gm)
                        est, sat = run_ota(pol, w, v, dA, dfl, g, snr, a, gen, g_min=best[2])
                        extra.update(sat=sat, trunc_q=best[1])
                    elif sch == "pure":
                        est = run_awa(w, v, g, snr, a, gen)
                    elif sch == "awa":
                        best = None
                        for i in range(len(wq)):
                            for j in range(i + 1, len(wq)):
                                lo = 0.0 if i == 0 else float(wq[i])
                                hi = math.inf if j == len(wq) - 1 else float(wq[j])
                                r = rmse(run_awa(wc_, vc_, gc, snr, a, gen, lo, hi), tc_)[0]
                                if best is None or r < best[0]:
                                    best = (r, lo, hi)
                        est = run_awa(w, v, g, snr, a, gen, best[1], best[2])
                        extra.update(dmin=best[1], dmax=best[2] if math.isfinite(best[2]) else None)
                    elif sch == "simple":
                        est = run_simple(v, g, snr, a, gen)
                    else:
                        raise ValueError(sch)
                    rm = rmse(est, truth)
                    rows.append(dict(task=a.task, K=K, M=M, snr_db=snr, scheme=sch_full, rmse_med=rm[0],
                                     rmse_p90=rm[1], **extra))
                    print(f"  {a.task} K={K:4d} M={M:2d} PSNR={snr:4.0f} {sch_full:13s} RMSE med {rm[0]:8.3f} "
                          f"p90 {rm[1]:10.3f}" + (f"  sat {extra['sat']:.3f}" if "sat" in extra else ""),
                          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(config=vars(a), rows=rows), open(a.out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
