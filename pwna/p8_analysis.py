"""
Analysis of the P8 campaign (pwna/remote/p8_runner.py) together with the paired P7 cells (same seeds,
segments and channel draws): one Markdown report with every number the rewritten paper needs.
  python pwna/p8_analysis.py [--p7 results/p7] [--p8 results/p8] [--field results/local] [--out results/p8_analysis]

Metric: required SNR = smallest SNR at which the mean NLL increase over the 16 (segment, draw) samples
falls to 0.05 nats (linear interpolation; '<=x' met at the first grid point; '--' not met on the grid).
Paired gaps: bootstrap over the sample index, resampled jointly for the two schemes (the samples share the
segment and the channel / noise draws), 2000 resamples; a scheme that is met at the first grid point is
censored there (the gap is then an upper bound).

Pre-registered choice of kappa (Section 5 of 实验设计_P8.md): on the CPython-docs development set
(p8_f1_q4_pydoc, M = 4, N in {8, 32, 64}) the proposed design with the digital handshake
'ladder_eff@kap<k>@fz256' is compared for k in {0.3, 1}: the smaller mean required SNR over N wins;
on a tie (e.g. both met at the first grid point everywhere) the smaller mean NLL averaged over the SNR grid
and N wins. The chosen kappa is then used unchanged for every model, context and corpus.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep_summary import need_snr  # noqa: E402

TH = 0.05
MODELS = (("q4", "Qwen3-4B"), ("l8", "Llama-3.1-8B"), ("q8", "Qwen3-8B"))
CTX = {"8k": 8192, "16k": 16384, "32k": 32768}
NS = {"8k": (8, 32, 64), "16k": (8, 32, 64), "32k": (32, 128)}
ARCH = {"q4": dict(layers=36, H=32, dv=128), "l8": dict(layers=32, H=32, dv=128), "q8": dict(layers=36, H=32, dv=128)}
KNOWN = ("genie_hybrid", "airmoe_hybrid", "airmoe_eff")          # known-weight baselines (genie from P7)
P7_DEP = ("ladderdet_chk", "ladder_chk")                            # the P7 private designs (deployed / digital hs)


class Res:
    """Fidelity / NIAH cells of several result directories, looked up by job name (p7_* or p8_*)."""

    def __init__(self, *dirs, supp=None):
        self.dirs, self.cache, self.supp = dirs, {}, supp

    def supp_name(self, name):
        """The negative-SNR supplement of an F1 job (p8_supp/p8s_f1_*), which shares its draws with the P8
        job and with the P7 job of the same configuration (genie): the two grids are read as one curve."""
        if self.supp and (name.startswith("p8_f1_") or name.startswith("p7_f1_") or name.startswith("p8_f5_")
                          or name == "p8_f2_kap"):
            return "p8s_" + name.split("_", 1)[1]
        return None

    def get_supp(self, name):
        sn = self.supp_name(name)
        if sn is None:
            return None
        key = ("supp", sn)
        if key not in self.cache:
            p = os.path.join(self.supp, sn + ".json")
            self.cache[key] = json.load(open(p)) if os.path.exists(p) else None
        return self.cache[key]

    def get(self, name):
        if name not in self.cache:
            self.cache[name] = None
            for d in self.dirs:
                p = os.path.join(d, name + ".json")
                if os.path.exists(p):
                    self.cache[name] = json.load(open(p))
                    break
        return self.cache[name]

    def cells(self, name, task="fidelity", supp=True, **kw):
        r = self.get(name)
        if r is None:
            return []
        out = [c for c in r.get("cells", []) if c["task"] == task and all(c.get(k) == v for k, v in kw.items())]
        rs = self.get_supp(name) if supp else None
        if rs is not None:
            have = {(c["T"], c["N"], c["M"], c["policy"], c["snr_db"]) for c in out}
            out += [c for c in rs.get("cells", []) if c["task"] == task and all(c.get(k) == v for k, v in kw.items())
                    and (c["T"], c["N"], c["M"], c["policy"], c["snr_db"]) not in have]
        return out

    def policies(self, name, task="fidelity"):
        return sorted({c["policy"] for c in self.cells(name, task)})

    def curve(self, name, T, N, M, pol, supp=True):
        cs = sorted(self.cells(name, supp=supp, T=T, N=N, M=M, policy=pol), key=lambda c: c["snr_db"])
        return [(c["snr_db"], c["nll"], c.get("nll_s")) for c in cs]


def req(cv):
    """(required SNR or None if not met, met at the first grid point) or None if not run."""
    if not cv:
        return None
    s = need_snr([(x, v) for x, v, _ in cv], TH)
    return s, s is not None and cv[0][1] <= TH


def fmt(r):
    if r is None:
        return "n/r"
    s, first = r
    if s is None:
        return "--"
    return f"<={s:g}" if first else f"{s:.1f}"


def val(r, inf=math.inf):
    return inf if r is None or r[0] is None else r[0]


def _req_idx(cv, idx):
    K = len(idx)
    return need_snr([(x, sum(s[i] for i in idx) / K) for x, _, s in cv], TH)


def paired_gap(cva, cvb, B=2000, seed=1):
    """Required SNR of a minus that of b: point value and 95% paired-bootstrap interval (inf if a is not met)."""
    ra, rb = req(cva), req(cvb)
    if ra is None or rb is None:
        return None
    point = val(ra) - val(rb)
    Sa, Sb = [s for *_, s in cva], [s for *_, s in cvb]
    if any(not s for s in Sa + Sb) or len({len(s) for s in Sa + Sb}) != 1:
        return point, None
    K = len(Sa[0])
    rng = random.Random(seed)
    g = []
    for _ in range(B):
        idx = [rng.randrange(K) for _ in range(K)]
        a, b = _req_idx(cva, idx), _req_idx(cvb, idx)
        g.append(math.inf if a is None else (a - (b if b is not None else math.inf)))
    g.sort()
    return point, (g[int(0.025 * B)], g[int(0.975 * B) - 1])


def fgap(x):
    if x is None:
        return "n/r"
    p, ci = x
    f = lambda v: "inf" if math.isinf(v) else ("-inf" if v == -math.inf else f"{v:+.1f}")
    return f(p) + ("" if ci is None else f" [{f(ci[0])}, {f(ci[1])}]")


def auc(cv):
    return None if not cv else sum(v for _, v, _ in cv) / len(cv)


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


# ------------------------------------------------------------------ kappa (pre-registered)
def choose_kappa(R):
    name, T = "p8_f1_q4_pydoc", CTX["8k"]
    score = {}
    for k in ("0.3", "1"):
        pol = f"ladder_eff@kap{k}@fz256"
        reqs = [val(req(R.curve(name, T, N, 4, pol, supp=False))) for N in NS["8k"]]   # the pre-registered rule:
        aucs = [auc(R.curve(name, T, N, 4, pol, supp=False)) for N in NS["8k"]]        # on the P8 0-25 dB grid
        if any(a is None for a in aucs):
            return None, {}
        score[k] = (sum(reqs) / len(reqs), sum(aucs) / len(aucs))
    best = min(score, key=lambda k: score[k])
    return best, score


# ------------------------------------------------------------------ sections
def sec_main(R, kap):
    prop_d, prop_t = f"ladder_eff@kap{kap}@fz256", f"ladderdet_eff@kap{kap}"
    other = "0.3" if kap == "1" else "1"
    out = [f"## 1. Main table (F1): required SNR (dB), M = 4 (M = 8 for Qwen3-4B 8k as '/ M8'); kappa = {kap}", "",
           "best known = the smallest of genie (P7), AirMoE exact D, AirMoE effective D at that configuration;"
           " gap = proposed (digital handshake) minus best known, paired 95% CI."
           + (" SNR grid -15 to 25 dB (P8 0-25 dB and the supplement -15 to -2.5 dB on the same draws) where a"
              " supplement job exists, else 0-25 dB." if R.supp else ""), ""]
    head = ["model", "ctx", "N", "genie", "AirMoE", "AirMoE-eff", "P7 split", "P7 split dig",
            "prop dig", "prop tone", f"dig k={other}", f"tone k={other}", "gap dig vs best known", "gap dig vs P7 split"]
    rows = []
    for m, label in MODELS:
        for T in ("8k", "16k", "32k"):
            n7, n8 = f"p7_f1_{m}_{T}", f"p8_f1_{m}_{T}"
            if R.get(n8) is None:
                continue
            for N in NS[T]:
                for M in ((4, 8) if (m, T) == ("q4", "8k") else (4,)):
                    t = CTX[T]
                    cv = {p: R.curve(n7, t, N, M, p) for p in ("genie_hybrid",) + P7_DEP}
                    cv.update({p: R.curve(n8, t, N, M, p) for p in ("airmoe_hybrid", "airmoe_eff", prop_d, prop_t,
                                                                     f"ladder_eff@kap{other}@fz256",
                                                                     f"ladderdet_eff@kap{other}")})
                    if not cv[prop_d]:
                        continue
                    known = [p for p in KNOWN if cv[p]]
                    bk = min(known, key=lambda p: (val(req(cv[p])), auc(cv[p]))) if known else None
                    rows.append([label, T, N if M == 4 else f"{N} (M8)"]
                                + [fmt(req(cv[p])) for p in ("genie_hybrid", "airmoe_hybrid", "airmoe_eff") + P7_DEP
                                   + (prop_d, prop_t, f"ladder_eff@kap{other}@fz256", f"ladderdet_eff@kap{other}")]
                                + [fgap(paired_gap(cv[prop_d], cv[bk])) + f" ({bk})" if bk else "n/r",
                                   fgap(paired_gap(cv[prop_d], cv["ladderdet_chk"]))])
    out += md_table(head, rows)
    extra = []
    for m in ("l8",):
        for T in ("16k", "32k"):
            n8 = f"p8_f1_{m}_{T}"
            for N in NS[T]:
                cv = R.curve(n8, CTX[T], N, 4, "ladderdet_hybrid")
                if cv:
                    extra.append(f"- Llama {T} N={N}: split ratio, exact D only (ladderdet_hybrid) {fmt(req(cv))}")
    if extra:
        out += ["", "Exact-D-only split ratio where the tone handshake overflows most:"] + extra
    # N sweep
    rows = []
    for N in (4, 8, 16, 32, 64, 128):
        src = "p8_f1_q4_8k_nsweep" if N in (4, 16, 128) else "p8_f1_q4_8k"
        src7 = "p7_f1_q4_8k_nsweep" if N in (4, 16, 128) else "p7_f1_q4_8k"
        cv = {p: R.curve(src, CTX["8k"], N, 4, p) for p in ("airmoe_hybrid", "airmoe_eff", prop_d, prop_t)}
        cv["genie_hybrid"] = R.curve(src7, CTX["8k"], N, 4, "genie_hybrid")
        cv["ladderdet_chk"] = R.curve(src7, CTX["8k"], N, 4, "ladderdet_chk")
        if any(cv.values()):
            rows.append([N] + [fmt(req(cv[p])) for p in ("genie_hybrid", "airmoe_hybrid", "airmoe_eff", "ladderdet_chk",
                                                         prop_d, prop_t)])
    if rows:
        out += ["", "### N sweep (Qwen3-4B, 8k, M = 4)", ""]
        out += md_table(["N", "genie", "AirMoE", "AirMoE-eff", "P7 split", "prop dig", "prop tone"], rows)
    return out


def sec_kappa(R, kap, score):
    out = ["## 0. Pre-registered kappa on the CPython-docs development set (p8_f1_q4_pydoc, M = 4)", ""]
    if kap is None:
        return out + ["not available yet", ""]
    rows = []
    for N in NS["8k"]:
        rows.append([N] + [fmt(req(R.curve("p8_f1_q4_pydoc", CTX["8k"], N, 4, p))) for p in
                           ("airmoe_hybrid", "airmoe_eff", "ladder_eff@kap0.3@fz256", "ladder_eff@kap1@fz256",
                            "ladderdet_eff@kap0.3", "ladderdet_eff@kap1")]
                    + [fmt(req(R.curve("p7_f3_pydoc", CTX["8k"], N, 4, p))) for p in ("genie_hybrid", "ladderdet_chk")])
    out += md_table(["N", "AirMoE", "AirMoE-eff", "dig k=0.3", "dig k=1", "tone k=0.3", "tone k=1", "genie (P7)",
                     "P7 split"], rows)
    out += ["", "score (mean required SNR, mean NLL over the grid): "
            + ", ".join(f"kappa={k}: {s[0]:.2f} dB, {s[1]:.4f}" for k, s in score.items()) + f" -> **kappa = {kap}**"]
    return out


def sec_grid(R, title, name, Ns, pols=None, M=4, T="8k"):
    pols = pols or R.policies(name)
    out = [f"### {title} ({name})", ""]
    if R.get(name) is None:
        return out + ["not run / not pulled", ""]
    rows = [[p] + [fmt(req(R.curve(name, CTX[T], N, M, p))) for N in Ns] for p in pols]
    return out + md_table(["policy"] + [f"N={N}" for N in Ns], rows) + [""]


def sec_design(R, kap):
    out = ["## 3. Design study (F2, Qwen3-4B 8k, M = 4)", "",
           "Components: '_eff' = A (saturation-aware split ratio), '@kap' = B (noise-matched level report);"
           " 'ladder_*@fz256' digital thermometer handshake over Z_256, 'ladderdet_*' tone handshake.", ""]
    ref = defaultdict(list)
    for p in ("airmoe_hybrid", "airmoe_eff", "ladder_eff@kap0.3@fz256", "ladder_eff@kap1@fz256", "ladderdet_eff@kap0.3",
              "ladderdet_eff@kap1"):
        ref[p] = [fmt(req(R.curve("p8_f1_q4_8k", CTX["8k"], N, 4, p))) for N in NS["8k"]]
    for p in ("ladderdet", "ladderdet_chk", "ladder_chk", "genie_hybrid"):
        ref[p + " (P7)"] = [fmt(req(R.curve("p7_f1_q4_8k", CTX["8k"], N, 4, p))) for N in NS["8k"]]
    for p in ("airmoe",):
        ref[p + " (P7 f2)"] = [fmt(req(R.curve("p7_f2_db3_r4", CTX["8k"], N, 4, p))) for N in NS["8k"]]
    rows = [[p] + v for p, v in ref.items()]
    name = "p8_f2_kap"
    for p in R.policies(name):
        rows.append([p] + [fmt(req(R.curve(name, CTX["8k"], N, 4, p))) for N in NS["8k"]])
    out += md_table(["policy", "N=8", "N=32", "N=64"], rows) + [""]
    # paired component gains at N = 32
    out += ["### Paired component contributions, N = 32 (required SNR difference, 95% CI)", ""]
    T, N = CTX["8k"], 32
    c = lambda nm, p: R.curve(nm, T, N, 4, p)
    pairs = [("A alone vs P7 split (tone)", c("p8_f2_kap", "ladderdet_eff"), c("p7_f1_q4_8k", "ladderdet_chk")),
             ("A+B vs A (tone)", c("p8_f1_q4_8k", f"ladderdet_eff@kap{kap}"), c("p8_f2_kap", "ladderdet_eff")),
             ("B with exact D vs P7 exact-D split (tone)", c("p8_f2_kap", "ladderdet@kap1"),
              c("p7_f2_db3_r4", "ladderdet_hybrid")),
             ("A+B vs A (digital)", c("p8_f1_q4_8k", f"ladder_eff@kap{kap}@fz256"), c("p8_f2_kap", "ladder_eff@fz256")),
             ("A+B vs B with exact D (digital)", c("p8_f1_q4_8k", "ladder_eff@kap1@fz256"),
              c("p8_f2_kap", "ladder_hybrid@kap1@fz256")),
             ("kappa vs fixed offset 3 dB (tone)", c("p8_f1_q4_8k", "ladderdet_eff@kap1"), c("p8_f2_kap", "ladderdet_eff@off3")),
             ("digital vs tone handshake", c("p8_f1_q4_8k", f"ladder_eff@kap{kap}@fz256"),
              c("p8_f1_q4_8k", f"ladderdet_eff@kap{kap}")),
             ("AirMoE-eff vs AirMoE (A given to the baseline)", c("p8_f1_q4_8k", "airmoe_eff"),
              c("p8_f1_q4_8k", "airmoe_hybrid"))]
    out += [f"- {lab}: {fgap(paired_gap(a, b))}" for lab, a, b in pairs] + [""]
    hs = []
    for nm, lab in (("p8_f2_db15", "step 1.5 dB, nu=4"), ("p8_f2_db6", "step 6 dB, nu=4"), ("p8_f2_r2", "step 3 dB, nu=2"),
                    ("p8_f2_r8", "step 3 dB, nu=8")):
        for p in R.policies(nm):
            hs.append([lab, p] + [fmt(req(R.curve(nm, T, n, 4, p))) for n in NS["8k"]])
    for p in (f"ladder_eff@kap1@fz256", "ladderdet_eff@kap1"):
        hs.append(["step 3 dB, nu=4 (F1)", p] + [fmt(req(R.curve("p8_f1_q4_8k", T, n, 4, p))) for n in NS["8k"]])
    out += ["### Handshake resources (step Delta, tones per level nu)", ""]
    out += md_table(["handshake", "policy", "N=8", "N=32", "N=64"], hs) + [""]
    # handshake statistics
    out += ["### Handshake statistics, N = 32, M = 4 (mean over the SNR grid)", ""]
    rows = []
    for nm, p in (("p8_f1_q4_8k", f"ladder_eff@kap{kap}@fz256"), ("p8_f1_q4_8k", f"ladderdet_eff@kap{kap}"),
                  ("p7_f1_q4_8k", "ladderdet_chk")):
        cs = R.cells(nm, supp=False, N=32, M=4, policy=p)
        if cs:
            g = lambda k: sum(c.get(k, float("nan")) for c in cs) / len(cs)
            rows.append([p, f"{g('levels'):.1f}", f"{g('sat'):.3f}", f"{g('nodetect'):.2e}", f"{g('overflow'):.2e}"])
    out += md_table(["policy", "levels", "saturated", "nodetect", "overflow"], rows)
    return out


def sec_secure(R):
    out = ["## 5. Secure digital merge at matched accuracy (F4, error-free digital link)", "",
           "NLL increase (nats) vs ring bits; pass = mean NLL <= 0.05. 'secdig' = offline global scale with"
           " m margin bits (default 8); 'secdigo' = oracle per-instance scale (lower bound for any public scale).", ""]
    minbits = {}
    for name in ("p8_f4_q4_8k", "p8_f4_q4_32k", "p8_f4_l8_8k", "p8_f4_q8_8k"):
        r = R.get(name)
        if r is None:
            out += [f"- {name}: not run / not pulled"]
            continue
        cs = R.cells(name)
        Ns = sorted({c["N"] for c in cs})
        pols = sorted({c["policy"] for c in cs}, key=lambda p: (p.split("@")[0], len(p), p))
        rows = []
        for p in pols:
            row = [p]
            for N in Ns:
                cc = [c for c in cs if c["policy"] == p and c["N"] == N]
                row.append("n/r" if not cc else f"{cc[0]['nll']:.4f}{' ok' if cc[0]['nll'] <= TH else ''}")
            rows.append(row)
        out += [f"### {name}", ""] + md_table(["policy"] + [f"N={N}" for N in Ns], rows) + [""]
        for base in ("secdig", "secdigo"):
            ok = []
            for p in pols:
                parts = p.split("@")
                if parts[0] != base or (base == "secdig" and any(x.startswith("m") for x in parts[1:])):
                    continue
                b = int(next(x[1:] for x in parts[1:] if x.startswith("b")))
                cc = [c for c in cs if c["policy"] == p]
                if cc and all(c["nll"] <= TH for c in cc):
                    ok.append(b)
            minbits[(name, base)] = min(ok) if ok else None
    out += ["Minimum passing ring bits (all N): " + ", ".join(f"{n} {b}: {v}" for (n, b), v in minbits.items())]
    return out, minbits


def sec_realism(R, kap):
    out = ["## 6. Channel realism (F5, Qwen3-4B 8k, N = 32, M = 4; combo also Llama)", ""]
    rows = []
    base_pols = ["ladderdet_chk", "genie_hybrid", "airmoe_eff", f"ladder_eff@kap1@fz256", "ladderdet_eff@kap1"]
    rows.append(["i.i.d. reference (P7/F1)"] + [fmt(req(R.curve("p8_f1_q4_8k" if p in ("airmoe_eff",) or "kap" in p else
                                                                 "p7_f1_q4_8k", CTX["8k"], 32, 4, p))) for p in base_pols])
    for sc in ("perhead", "coh2", "coh4", "ph5", "ph10", "ph20", "ramp20", "ramp45", "static", "sigw4", "sigw8", "combo",
               "combo_l8"):
        nm = f"p8_f5_{sc}"
        if R.get(nm) is None:
            rows.append([sc] + ["n/r"] * len(base_pols))
            continue
        suf = "@csi0.01" if sc.startswith("combo") else ""       # combo runs with --iq csi 0.01: suffixed names
        rows.append([sc] + [fmt(req(R.curve(nm, CTX["8k"], 32, 4, p + suf))) for p in base_pols])
    for v in ("0.01", "0.05", "0.1"):
        rows.append([f"csi {v} (I/Q)"] + [fmt(req(R.curve("p8_f5_csi", CTX["8k"], 32, 4, f"{p}@csi{v}"))) for p in base_pols])
    out += md_table(["scenario", "P7 split", "genie", "AirMoE-eff", "prop dig k=1", "prop tone k=1"], rows) + [""]
    extra = []
    for sc in ("sigw4", "sigw8"):
        for p in ("airmoe_hybrid", "ladder_eff@kap1@wf2@fz256", "ladderdet_eff@kap1@wf2"):
            cv = R.curve(f"p8_f5_{sc}", CTX["8k"], 32, 4, p)
            if cv:
                extra.append(f"- {sc} {p}: {fmt(req(cv))}")
    return out + extra


def sec_oneround(R):
    out = ["## 4. One-round designs without a handshake (F3, Qwen3-4B 8k, M = 4, grid 0-50 dB)", ""]
    out += sec_grid(R, "one round", "p8_f3_oneround", (8, 32))
    out += sec_grid(R, "P7 blind designs for comparison", "p7_f3_blind_q4", (8, 32))
    return out


def sec_niah(R, kap):
    out = ["## 7. NIAH (F6, teacher-forced, 120 trials per cell, M = 4): accuracy (%) at 5/10/15/20/25 dB", ""]
    rows = []
    for name in [f"p8_f6_{m}_{s}" for m, _ in MODELS for s in ("8k", "mk8k")] + ["p8_f6_q4_16k", "p8_f6_q4_32k"]:
        cs = R.cells(name, task="niah")
        if not cs:
            rows.append([name, "-", "-", "not run / not pulled"])
            continue
        clean = [c for c in cs if c["policy"] == "clean"]
        ceil = f"{100 * clean[0].get('tf_acc', float('nan')):.1f}" if clean else "-"
        for N in sorted({c["N"] for c in cs if c["policy"] != "clean"}):
            for p in sorted({c["policy"] for c in cs if c["policy"] != "clean"}):
                acc = {c["snr_db"]: 100 * c["acc"] for c in cs if c["N"] == N and c["policy"] == p}
                rows.append([name, ceil, f"N={N} {p}", " / ".join(f"{acc[s]:.1f}" if s in acc else "-"
                                                                   for s in (5.0, 10.0, 15.0, 20.0, 25.0))])
    return out + md_table(["job", "TF ceiling", "config", "acc 5/10/15/20/25 dB"], rows)


# ------------------------------------------------------------------ latency at matched accuracy
def latency(N, bw, reqs, L, b_den, b_glob, b_inst, M=4, rtt=0.5e-3, snr=15.0, layers=36, H=32, dv=128, nu=4,
            re_eff=0.72, gap=3.0, b_w=16, b_ring_hs=8):
    """Per-token latency (ms). Analog payloads with required SNR above the operating SNR are repeated
    ceil(10^((req - snr)/10)) times; I/Q packing: H (dv + 1) / 2 complex REs per analog merge.
      prop_dig  thermometer handshake over Z_256 (N H L bytes, secure aggregation), then the analog numerator
                and denominator in parallel with the secure sum of the effective weights (b_den bits): 2 RTT
      prop_tone the same with the tone handshake (H L nu on-off REs)
      known     best known-weight baseline: N H weights of b_w bits, then the analog numerator: 2 RTT
      sec_inst  secure digital merge with a per-instance scale (b_inst bits; +1 RTT for the scale)
      sec_glob  secure digital merge with an offline global scale (b_glob bits): 1 RTT
      digital   plain bf16 digital merge (no privacy): 1 RTT
      zf_tdma   ceil(N / G) slots of H dv / 2 complex REs with G = max(1, M / 2) devices per slot (zero forcing),
                plus the N H weights of b_w bits (no privacy): 1 RTT"""
    res = re_eff * bw
    rate = math.log2(1 + 10 ** ((snr - gap) / 10)) * min(M, N) * res

    def reps(p):
        s = reqs.get(p)
        return math.inf if s is None or math.isinf(s) else max(1, math.ceil(10 ** ((s - snr) / 10)))

    second = lambda p: max(reps(p) * H * (dv + 1) / 2 / res, N * H * b_den / rate)
    lat = {"prop_dig": 2 * rtt + N * H * L * b_ring_hs / rate + second("prop_dig"),
           "prop_tone": 2 * rtt + H * L * nu / res + second("prop_tone"),
           "known": 2 * rtt + N * H * b_w / rate + reps("known") * H * dv / 2 / res,
           "sec_inst": (2 * rtt + N * H * (dv + 1) * b_inst / rate) if b_inst else math.inf,
           "sec_glob": (rtt + N * H * (dv + 1) * b_glob / rate) if b_glob else math.inf,
           "digital": rtt + N * H * (dv + 1) * 16 / rate}
    if "zf_tdma" in reqs:
        lat["zf_tdma"] = rtt + max(reps("zf_tdma") * math.ceil(N / max(1, M // 2)) * H * dv / 2 / res, N * H * b_w / rate)
    return {k: v * layers * 1e3 for k, v in lat.items()}


def latency_inputs(R, kap, minbits):
    """(L, b_den, b_glob, b_inst, mean measured levels) of the latency ledger."""
    lev = [c["levels"] for c in R.cells("p8_f1_q4_8k", supp=False, M=4, policy=f"ladderdet_eff@kap{kap}") if "levels" in c]
    b_glob = max([v for (n, b), v in minbits.items() if b == "secdig" and v] or [0]) or None
    b_inst = max([v for (n, b), v in minbits.items() if b == "secdigo" and v] or [0]) or None
    fx = []
    for m, _ in MODELS:
        r = R.get(f"p7_f5_fx_{m}")
        if r:
            fx += [row["bits_handshake_eps1e-06"]["max"] for row in r["rows"]]
    return 32, (int(max(fx)) if fx else 32), b_glob, b_inst, (sum(lev) / len(lev) if lev else float("nan"))


def latency_reqs(R, kap, N):
    """Required SNRs (dB) of the analog payloads of the main model at 8k, M = 4 (inf: not met / not run)."""
    src = "p8_f1_q4_8k_nsweep" if N in (4, 16, 128) else "p8_f1_q4_8k"
    src7 = "p7_f1_q4_8k_nsweep" if N in (4, 16, 128) else "p7_f1_q4_8k"
    kn = [val(req(R.curve(src, CTX["8k"], N, 4, p))) for p in ("airmoe_hybrid", "airmoe_eff")]
    kn.append(val(req(R.curve(src7, CTX["8k"], N, 4, "genie_hybrid"))))
    zf = min(val(req(R.curve(nm, CTX["8k"], N, 4, "zf_tdma"))) for nm in ("p7_f1_q4_8k_zf", src7))
    return {"prop_dig": val(req(R.curve(src, CTX["8k"], N, 4, f"ladder_eff@kap{kap}@fz256"))),
            "prop_tone": val(req(R.curve(src, CTX["8k"], N, 4, f"ladderdet_eff@kap{kap}"))), "known": min(kn),
            "zf_tdma": zf}


def sec_latency(R, kap, minbits):
    out = ["## 8. Latency at matched accuracy (Qwen3-4B, 8k reqs, operating SNR 15 dB, M = 4)", ""]
    L, b_den, b_glob, b_inst, lev = latency_inputs(R, kap, minbits)
    out += [f"inputs: ladder levels L = {L} (used levels per instance, measured mean {lev:.1f}),"
            f" secure-sum bits of the denominator {b_den},"
            f" secure merge bits: global {b_glob}, per-instance {b_inst}", ""]
    rows = []
    keys = ("prop_dig", "prop_tone", "known", "sec_inst", "sec_glob", "digital", "zf_tdma")
    for bw in (20e6, 100e6):
        for N in (4, 8, 16, 32, 64, 128):
            reqs = latency_reqs(R, kap, N)
            if all(math.isinf(v) for v in reqs.values()):
                continue
            lat = latency(N, bw, reqs, L, b_den, b_glob, b_inst, **ARCH["q4"])
            rows.append([f"{bw / 1e6:.0f} MHz", N] + ["--" if math.isinf(lat[k]) else f"{lat[k]:.1f}" for k in keys])
    out += md_table(["band", "N", "prop dig", "prop tone", "known w.", "secure (inst.)", "secure (global)", "digital bf16",
                     "ZF-TDMA"], rows)
    return out


def sec_field(F):
    out = ["## 9. Spatial field (median RMSE dB; M = 1 / M = 4 at 0 / 10 / 20 dB)", ""]
    sch = ("genie_hybrid", "airmoe_hybrid", "airmoe_eff", "ladderdet_chk", "ladderdet_trunc", "ladder_eff@kap1",
           "ladderdet_eff@kap1", "ladderdet_eff@kap0.3", "awa")
    rows = []
    for task in ("nw", "poe"):
        for sw in (0, 10):
            p = os.path.join(F, f"field_{task}_sw{sw}_p8.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            g = {(r["K"], r["M"], r["scheme"], r["snr_db"]): r["rmse_med"] for r in d["rows"]}
            ideal = {r["K"]: r["rmse_med"] for r in d["rows"] if r["scheme"] == "ideal"}
            for K in sorted(ideal):
                for M in (1, 4):
                    rows.append([f"{task} sw{sw}", K, M, f"{ideal[K]:.2f}"]
                                + ["/".join("-" if g.get((K, M, s, x)) is None else f"{g[(K, M, s, x)]:.2f}"
                                            for x in (0.0, 10.0, 20.0)) for s in sch])
    return out + md_table(["scenario", "K", "M", "ideal"] + list(sch), rows)


def main():
    root = os.path.dirname(HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--p7", default=os.path.join(root, "results", "p7"))
    ap.add_argument("--p8", default=os.path.join(root, "results", "p8"))
    ap.add_argument("--supp", default=os.path.join(root, "results", "p8_supp"),
                    help="negative-SNR supplement of F1 (p8s_f1_*); merged into the F1 curves when present")
    ap.add_argument("--field", default=os.path.join(root, "results", "local"))
    ap.add_argument("--out", default=os.path.join(root, "results", "p8_analysis"))
    ap.add_argument("--kappa", default=None, help="override the pre-registered choice (dry runs only)")
    a = ap.parse_args()
    R = Res(a.p8, a.p7, supp=a.supp if os.path.isdir(a.supp) else None)
    kap, score = choose_kappa(R)
    kap = a.kappa or kap or "1"
    out = ["# P8 analysis (generated by pwna/p8_analysis.py)", ""]
    out += sec_kappa(R, kap if score else None, score) + [""]
    out += sec_main(R, kap) + [""]
    out += sec_design(R, kap) + [""]
    out += sec_oneround(R) + [""]
    s, minbits = sec_secure(R)
    out += s + [""]
    out += sec_realism(R, kap) + [""]
    out += sec_niah(R, kap) + [""]
    out += sec_latency(R, kap, minbits) + [""]
    out += sec_field(a.field) + [""]
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "summary.md")
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("written to", p)


if __name__ == "__main__":
    main()
