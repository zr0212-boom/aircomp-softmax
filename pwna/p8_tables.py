"""
LaTeX tables of the rewritten paper, generated from the P8 results with the paired P7 cells and the
negative-SNR supplement (see pwna/p8_analysis.py for the metric, the pairing and the choice of kappa).
  python pwna/p8_tables.py            -> paper/twc/tables/tab_*.tex and results/p8_analysis/paper_numbers.md
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from p8_analysis import (ARCH, CTX, KNOWN, MODELS, NS, Res, auc, choose_kappa, latency, latency_inputs,  # noqa: E402
                         latency_reqs, paired_gap, req, sec_secure, val)

OUT = os.path.join(ROOT, "paper", "twc", "tables")
NUM = []


def t(r):
    """Required SNR for a table cell."""
    if r is None:
        return "n/r"
    s, first = r
    if s is None:
        return "--"
    v = f"{abs(s):.1f}"
    v = ("$-$" + v) if s < -0.05 else v
    return ("$\\le$" + v) if first else v


def sgn(v):
    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.1f}"


def tg(x):
    if x is None:
        return "n/r"
    p, ci = x
    return sgn(p) + ("" if ci is None else f" [{sgn(ci[0])}, {sgn(ci[1])}]")


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(rows) + "\n")


def rng(vals):
    vals = [v for v in vals if v is not None and not math.isinf(v)]
    return f"{min(vals):.1f} to {max(vals):.1f}" if vals else "n/a"


def tab_llm(R, kap):
    pd, pt = f"ladder_eff@kap{kap}@fz256", f"ladderdet_eff@kap{kap}"
    rows, stats = [], {}
    for m, label in MODELS:
        first = True
        nrow = sum(len(NS[T]) for T in ("8k", "16k", "32k"))        # M = 8 rows: statistics and text only
        for T in ("8k", "16k", "32k"):
            n7, n8 = f"p7_f1_{m}_{T}", f"p8_f1_{m}_{T}"
            for M in ((4, 8) if (m, T) == ("q4", "8k") else (4,)):
                for N in NS[T]:
                    c = CTX[T]
                    cv = {p: R.curve(n7, c, N, M, p) for p in ("genie_hybrid", "ladderdet", "ladderdet_chk")}
                    cv.update({p: R.curve(n8, c, N, M, p) for p in ("airmoe_hybrid", "airmoe_eff", pd, pt)})
                    bk = min(KNOWN, key=lambda p: (val(req(cv[p])), auc(cv[p])))
                    g1, g2 = paired_gap(cv[pd], cv[bk]), paired_gap(cv[pd], cv["ladderdet_chk"])
                    g3 = paired_gap(cv[pt], cv[pd])
                    cells = [t(req(cv[p])) for p in ("genie_hybrid", "airmoe_hybrid", "airmoe_eff", "ladderdet",
                                                     "ladderdet_chk", pt, pd)]
                    best = min(("genie_hybrid", "airmoe_hybrid", "airmoe_eff"), key=lambda p: val(req(cv[p])))
                    mark = "" if bk == "airmoe_eff" else "$^\\dagger$"
                    if M == 4:
                        rows.append((f"\\multirow{{{nrow}}}{{*}}{{{label}}}" if first else "") + f" & {T} & {N} & "
                                    + " & ".join(cells) + f" & {tg(g1)}{mark} & {tg(g2)} \\\\")
                        first = False
                    else:
                        NUM.append(f"- M = 8, {label} {T} N={N}: " + " | ".join(cells) + f" | {tg(g1)} | {tg(g2)}")
                    st = stats.setdefault(label, {k: [] for k in ("pd", "pt", "gap", "gapci", "gsplit", "lad", "split",
                                                                    "eff", "air", "genie", "tone_dig", "lad_pd")})
                    st["pd"].append(val(req(cv[pd]))); st["pt"].append(val(req(cv[pt])))
                    st["gap"].append(g1[0]); st["gapci"].append((T, N, M, g1)); st["gsplit"].append(g2[0])
                    st["lad"].append(val(req(cv["ladderdet"]))); st["split"].append(val(req(cv["ladderdet_chk"])))
                    st["eff"].append(val(req(cv["airmoe_eff"]))); st["air"].append(val(req(cv["airmoe_hybrid"])))
                    st["genie"].append(val(req(cv["genie_hybrid"]))); st["tone_dig"].append(g3[0])
                    st["lad_pd"].append(val(req(cv["ladderdet"])) - val(req(cv[pd])))
                    _ = best
        rows.append("\\midrule")
    write("tab_llm.tex", rows[:-1])
    NUM.append("## Main table ranges (M = 4 and M = 8 rows)")
    allv = {k: [] for k in ("pd", "pt", "gap", "gsplit", "lad", "split", "eff", "air", "genie", "tone_dig", "lad_pd")}
    for label, st in stats.items():
        NUM.append(f"- {label}: prop dig {rng(st['pd'])}; prop tone {rng(st['pt'])}; analog ladder {rng(st['lad'])};"
                   f" non-sat split {rng(st['split'])}; AirMoE-eff {rng(st['eff'])}; AirMoE {rng(st['air'])};"
                   f" genie {rng(st['genie'])}; gap dig-best known {rng(st['gap'])}; gap dig-split {rng(st['gsplit'])};"
                   f" tone-dig {rng(st['tone_dig'])}; analog ladder - dig {rng(st['lad_pd'])}")
        ns = [(T, N, M) for T, N, M, g in st["gapci"] if g[1] and g[1][0] <= 0]
        NUM.append(f"  gap CI includes 0 at: {ns}")
        for k in allv:
            allv[k] += st[k]
    NUM.append("- all: " + "; ".join(f"{k} {rng(v)}" for k, v in allv.items()))
    NUM.append(f"- prop dig minus genie: {rng([a - b for a, b in zip(allv['pd'], allv['genie'])])};"
               f" prop dig minus AirMoE (exact D): {rng([a - b for a, b in zip(allv['pd'], allv['air'])])};"
               f" prop tone minus genie: {rng([a - b for a, b in zip(allv['pt'], allv['genie'])])};"
               f" prop tone minus non-sat split: {rng([a - b for a, b in zip(allv['pt'], allv['split'])])}")
    # N sweep
    line = []
    for N in (4, 8, 16, 32, 64, 128):
        rq = latency_reqs(R, kap, N)
        line.append(f"N={N}: dig {rq['prop_dig']:.1f}, tone {rq['prop_tone']:.1f}, known {rq['known']:.1f}, zf {rq['zf_tdma']:.1f}")
    NUM.append("- N sweep (Qwen3-4B 8k M=4): " + "; ".join(line))


def tab_design(R, kap):
    T = CTX["8k"]
    c = lambda nm, p: [t(req(R.curve(nm, T, N, 4, p))) for N in NS["8k"]]
    F1, F2, P7 = "p8_f1_q4_8k", "p8_f2_kap", "p7_f1_q4_8k"
    rows = []
    add = lambda lab, v: rows.append(lab + " & " + " & ".join(v) + " \\\\")
    sec = lambda s: rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{s}}}}} \\\\")
    sec("Known weights")
    add("Genie scale, exact $D$", c(P7, "genie_hybrid"))
    add("AirMoE scale, analog $\\hat D$", c(F2, "airmoe"))
    add("AirMoE scale, exact $D$", c(F1, "airmoe_hybrid"))
    add("AirMoE scale, arrived weights $A$", c(F1, "airmoe_eff"))
    rows.append("\\midrule")
    sec("Private weights, digital handshake")
    add("Exact $D$ with check (neither)", c(P7, "ladder_chk"))
    add("A only ($\\kappa=0$)", c(F2, "ladder_eff@fz256"))
    add("B only ($\\kappa=1$, exact $D$)", c(F2, "ladder_hybrid@kap1@fz256"))
    add("A+B, $\\kappa=0.1$", c(F2, "ladder_eff@kap0.1@fz256"))
    add("A+B, $\\kappa=0.3$", c(F1, "ladder_eff@kap0.3@fz256"))
    add("\\textbf{A+B, $\\kappa=1$ (proposed)}", c(F1, "ladder_eff@kap1@fz256"))
    add("A+B, $\\kappa=3$", c(F2, "ladder_eff@kap3@fz256"))
    add("A+B, $\\kappa=1$, $\\Delta=1.5$ / 6~dB", [a + " / " + b for a, b in zip(c("p8_f2_db15", "ladder_eff@kap1@fz256"),
                                                                                 c("p8_f2_db6", "ladder_eff@kap1@fz256"))])
    rows.append("\\midrule")
    sec("Private weights, tone handshake ($\\nu=4$)")
    add("Analog $\\hat D$ (neither)", c(P7, "ladderdet"))
    add("Exact $D$ with check (neither)", c(P7, "ladderdet_chk"))
    add("A only ($\\kappa=0$)", c(F2, "ladderdet_eff"))
    add("B only ($\\kappa=1$, analog $\\hat D$)", c(F2, "ladderdet@kap1"))
    add("A, fixed offset 3~dB", c(F2, "ladderdet_eff@off3"))
    add("A+B, $\\kappa=0.1$", c(F2, "ladderdet_eff@kap0.1"))
    add("A+B, $\\kappa=0.3$", c(F1, "ladderdet_eff@kap0.3"))
    add("\\textbf{A+B, $\\kappa=1$ (proposed)}", c(F1, "ladderdet_eff@kap1"))
    add("A+B, $\\kappa=3$", c(F2, "ladderdet_eff@kap3"))
    add("A+B, $\\kappa=1$, $\\nu=2$ / 8", [a + " / " + b for a, b in zip(c("p8_f2_r2", "ladderdet_eff@kap1"),
                                                                        c("p8_f2_r8", "ladderdet_eff@kap1"))])
    write("tab_design.tex", rows)
    # one-round designs
    NUM.append("## One-round designs (Qwen3-4B 8k, M = 4, N = 8 / 32)")
    for nm in ("p8_f3_oneround", "p7_f3_blind_q4"):
        for p in R.policies(nm):
            NUM.append(f"- {nm} {p}: " + " / ".join(t(req(R.curve(nm, T, N, 4, p))) for N in (8, 32)))
    for m in ("l8", "q8"):
        nm = f"p7_f3_blind_{m}"
        for p in R.policies(nm):
            NUM.append(f"- {nm} {p}: " + " / ".join(t(req(R.curve(nm, T, N, 4, p))) for N in (8, 32)))
    NUM.append("## Paired gaps for the design study (N = 8 / 32 / 64)")
    cc = lambda nm, p, N: R.curve(nm, T, N, 4, p)
    for lab, a, b in (("A+B dig vs B exact-D dig", (F1, "ladder_eff@kap1@fz256"), (F2, "ladder_hybrid@kap1@fz256")),
                      ("A+B tone vs B analog tone", (F1, "ladderdet_eff@kap1"), (F2, "ladderdet@kap1")),
                      ("A+B dig vs A dig", (F1, "ladder_eff@kap1@fz256"), (F2, "ladder_eff@fz256")),
                      ("A+B tone vs A tone", (F1, "ladderdet_eff@kap1"), (F2, "ladderdet_eff")),
                      ("A dig vs exact-D chk dig", (F2, "ladder_eff@fz256"), (P7, "ladder_chk")),
                      ("A tone vs exact-D chk tone", (F2, "ladderdet_eff"), (P7, "ladderdet_chk")),
                      ("kappa 3 vs 1 dig", (F2, "ladder_eff@kap3@fz256"), (F1, "ladder_eff@kap1@fz256")),
                      ("kappa 0.3 vs 1 dig", (F1, "ladder_eff@kap0.3@fz256"), (F1, "ladder_eff@kap1@fz256")),
                      ("tone vs dig (A+B k1)", (F1, "ladderdet_eff@kap1"), (F1, "ladder_eff@kap1@fz256")),
                      ("tone nu8 vs dig", ("p8_f2_r8", "ladderdet_eff@kap1"), (F1, "ladder_eff@kap1@fz256")),
                      ("AirMoE-eff vs AirMoE exact", (F1, "airmoe_eff"), (F1, "airmoe_hybrid")),
                      ("AirMoE exact vs genie", (F1, "airmoe_hybrid"), (P7, "genie_hybrid")),
                      ("prop dig vs genie", (F1, "ladder_eff@kap1@fz256"), (P7, "genie_hybrid")),
                      ("prop dig vs AirMoE exact", (F1, "ladder_eff@kap1@fz256"), (F1, "airmoe_hybrid")),
                      ("dig step 6 vs 3", ("p8_f2_db6", "ladder_eff@kap1@fz256"), (F1, "ladder_eff@kap1@fz256")),
                      ("dig step 1.5 vs 3", ("p8_f2_db15", "ladder_eff@kap1@fz256"), (F1, "ladder_eff@kap1@fz256")),
                      ("kappa1 vs off3 tone", (F1, "ladderdet_eff@kap1"), (F2, "ladderdet_eff@off3"))):
        NUM.append(f"- {lab}: " + " / ".join(tg(paired_gap(cc(*a, N), cc(*b, N))) for N in NS["8k"]))
    NUM.append("## Handshake / saturation statistics, Qwen3-4B 8k, M = 4 (per SNR)")
    for nm, p in ((F1, "ladder_eff@kap1@fz256"), (F1, "ladderdet_eff@kap1"), (P7, "ladderdet_chk"), (P7, "ladder_chk")):
        for N in (8, 32, 64):
            cs = sorted(R.cells(nm, supp=True, N=N, M=4, policy=p), key=lambda x: x["snr_db"])
            NUM.append(f"- {p} N={N}: " + "; ".join(
                f"{x['snr_db']:g} dB sat {x.get('sat', float('nan')):.3f} loss {x.get('mass_loss', float('nan')):.3f}"
                + (f" under {x['det_under']:.3f} ovf {x['overflow']:.1e}" if "det_under" in x else "")
                for x in cs if x["snr_db"] in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)))


def tab_real(R):
    T = CTX["8k"]
    pols = ["genie_hybrid", "airmoe_eff", "ladderdet_chk", "ladderdet_eff@kap1", "ladder_eff@kap1@fz256"]
    rows = []

    def row(lab, nm, suf=""):
        rows.append(lab + " & " + " & ".join(t(req(R.curve(nm if not callable(nm) else nm(p), T, 32, 4, p + suf)))
                                             for p in pols) + " \\\\")

    row("Reference (i.i.d.)", lambda p: "p8_f1_q4_8k" if ("kap" in p or p == "airmoe_eff") else "p7_f1_q4_8k")
    row("Independent per head", "p8_f5_perhead")
    row("Static over the session", "p8_f5_static")
    row("$C=2$ coherence blocks", "p8_f5_coh2")
    row("$C=4$ coherence blocks", "p8_f5_coh4")
    for d in ("10", "20"):
        row(f"Phase error ${d}^\\circ$", f"p8_f5_ph{d}")
    for d in ("45",):
        row(f"Phase ramp ${d}^\\circ$", f"p8_f5_ramp{d}")
    for v in ("0.05", "0.1"):
        row(f"CSI NMSE {v}", "p8_f5_csi", f"@csi{v}")
    row("Spread 4~dB", "p8_f5_sigw4")
    row("Spread 8~dB", "p8_f5_sigw8")
    fb = lambda p: " / ".join(t(req(R.curve(f"p8_f5_sigw{sw}", T, 32, 4, p))) for sw in ("4", "8"))
    rows.append("\\quad with 2-device fallback & -- & -- & -- & " + fb("ladderdet_eff@kap1@wf2") + " & "
                + fb("ladder_eff@kap1@wf2@fz256") + " \\\\")
    row("Combined", "p8_f5_combo", "@csi0.01")
    row("Combined, Llama-3.1-8B", "p8_f5_combo_l8", "@csi0.01")
    write("tab_real.tex", rows)
    NUM.append("## Realism: paired gaps prop dig vs AirMoE-eff and vs reference")
    for sc in ("perhead", "static", "coh2", "coh4", "ph5", "ph10", "ph20", "ramp20", "ramp45", "sigw4", "sigw8"):
        nm = f"p8_f5_{sc}"
        NUM.append(f"- {sc}: dig - AirMoE-eff " + tg(paired_gap(R.curve(nm, T, 32, 4, "ladder_eff@kap1@fz256"),
                                                              R.curve(nm, T, 32, 4, "airmoe_eff"))))


def tab_niah(R):
    rows = []
    lab = {"q4": "Qwen3-4B", "l8": "Llama-8B", "q8": "Qwen3-8B"}
    jobs = [(f"p8_f6_{m}_{s}", m, "8k", k) for m in ("q4", "l8", "q8") for s, k in (("8k", 1), ("mk8k", 4))]
    jobs += [("p8_f6_q4_16k", "q4", "16k", 1), ("p8_f6_q4_32k", "q4", "32k", 1)]
    worst = {}
    for name, m, ctx, keys in jobs:
        cs = R.cells(name, task="niah")
        clean = [c for c in cs if c["policy"] == "clean"]
        ceil = 100 * clean[0].get("tf_acc", float("nan")) if clean else float("nan")
        for N in sorted({c["N"] for c in cs if c["policy"] != "clean"}):
            acc = {(c["policy"], c["snr_db"]): 100 * c["acc"] for c in cs if c["N"] == N}
            for (p, sn), a in acc.items():
                if p != "clean":
                    worst[(p, sn, keys)] = min(worst.get((p, sn, keys), 1e9), a - ceil)
            if N < 32:                       # the table lists N >= 32; the text covers all cells
                continue
            f = lambda p, s: f"{acc[(p, s)]:.1f}".replace(".0", "") if (p, s) in acc else "--"
            rows.append(f"{lab[m]}, {ctx}, {N}" + (", 4 keys" if keys > 1 else "") + f" & {ceil:.1f}".replace(".0", "")
                        + " & " + " & ".join([f("airmoe_eff", 5.0), f("airmoe_eff", 10.0), f("ladder_eff@kap1@fz256", 5.0),
                                              f("ladder_eff@kap1@fz256", 10.0), f("ladderdet_eff@kap1", 5.0),
                                              f("ladderdet_eff@kap1", 10.0), f("ladderdet_eff@kap1", 15.0)]) + " \\\\")
            for (p, s), a in acc.items():
                if p != "clean":
                    k = (p, s, keys)
                    worst[k] = min(worst.get(k, 1e9), a - ceil)
    write("tab_niah.tex", rows)
    NUM.append("## NIAH: worst accuracy minus ceiling (percentage points) per policy, SNR, keys")
    for k in sorted(worst):
        NUM.append(f"- {k}: {worst[k]:+.1f}")


def tab_latency(R, kap, minbits):
    L, b_den, b_glob, b_inst, lev = latency_inputs(R, kap, minbits)
    rows = []
    keys = ("prop_dig", "prop_tone", "sec_inst", "sec_glob", "known", "zf_tdma", "digital")
    NUM.append(f"## Latency inputs: L = {L}, measured levels {lev:.1f}, b_den = {b_den}, b_glob = {b_glob}, b_inst = {b_inst}")
    for bw, Ns in ((100e6, (8, 32, 128)), (20e6, (8, 16, 32, 64, 128))):
        for N in Ns:
            lat = latency(N, bw, latency_reqs(R, kap, N), L, b_den, b_glob, b_inst, **ARCH["q4"])
            rows.append(f"{bw / 1e6:.0f}~MHz, {N} & " + " & ".join(f"{lat[k]:.0f}" if lat[k] >= 100 else f"{lat[k]:.1f}"
                                                                   for k in keys) + " \\\\")
    for bw in (20e6, 100e6):
        for N in (4, 8, 16, 32, 64, 128):
            lat = latency(N, bw, latency_reqs(R, kap, N), L, b_den, b_glob, b_inst, **ARCH["q4"])
            best_priv = min(lat["prop_dig"], lat["prop_tone"])
            NUM.append(f"- {bw / 1e6:.0f} MHz N={N}: " + ", ".join(f"{k} {lat[k]:.1f}" for k in keys)
                       + f"; sec_inst/best prop {lat['sec_inst'] / best_priv:.2f}, sec_glob/best prop {lat['sec_glob'] / best_priv:.2f},"
                         f" best prop - known {best_priv - lat['known']:.1f} ms, zf/best prop {lat['zf_tdma'] / best_priv:.2f}")
    # 6 dB step halves the ladder
    for N in (8, 32, 128):
        a = latency(N, 20e6, latency_reqs(R, kap, N), 16, b_den, b_glob, b_inst, **ARCH["q4"])
        NUM.append(f"- 20 MHz N={N} with L = 16 (6 dB step, same required SNR assumed): dig {a['prop_dig']:.1f}, tone {a['prop_tone']:.1f}")
    write("tab_latency.tex", rows)


def tab_field():
    F = os.path.join(ROOT, "results", "local")
    sch = ("airmoe_eff", "ladder_eff@kap1", "ladderdet_eff@kap1", "ladderdet_chk", "ladderdet_trunc", "awa")
    rows = []
    NUM.append("## Field, M = 1: RMSE at 0 / 10 / 20 dB")
    for sw, lab in ((0, "Equal"), (10, "Spread")):
        first = True
        for task, Ks in (("nw", (64, 256, 1024)), ("poe", (4, 32))):
            d = json.load(open(os.path.join(F, f"field_{task}_sw{sw}_p8.json")))
            g = {(r["K"], r["M"], r["scheme"], r["snr_db"]): r["rmse_med"] for r in d["rows"]}
            ideal = {r["K"]: r["rmse_med"] for r in d["rows"] if r["scheme"] == "ideal"}
            for K in Ks:
                rows.append((f"\\multirow{{5}}{{*}}{{{lab}}}" if first else "") + f" & {task.upper() if task == 'nw' else 'PoE'}, {K} & {ideal[K]:.2f} & "
                            + " & ".join(f"{g[(K, 1, s, 10.0)]:.1f} / {g[(K, 1, s, 20.0)]:.2f}" if g[(K, 1, s, 10.0)] >= 10
                                         else f"{g[(K, 1, s, 10.0)]:.2f} / {g[(K, 1, s, 20.0)]:.2f}" for s in sch) + " \\\\")
                first = False
                NUM.append(f"- {task} sw{sw} K={K} ideal {ideal[K]:.2f}: " + "; ".join(
                    f"{s} " + "/".join(f"{g[(K, 1, s, x)]:.2f}" for x in (0.0, 10.0, 20.0))
                    for s in sch + ("airmoe_hybrid", "genie_hybrid", "ladderdet_eff@kap0.3")))
        rows.append("\\midrule")
    write("tab_field.tex", rows[:-1])


def main():
    R = Res(os.path.join(ROOT, "results", "p8"), os.path.join(ROOT, "results", "p7"),
            supp=os.path.join(ROOT, "results", "p8_supp"))
    kap, score = choose_kappa(R)
    NUM.append(f"# Numbers behind the paper text (pwna/p8_tables.py); kappa = {kap}; dev-set score {score}")
    tab_llm(R, kap)
    tab_design(R, kap)
    tab_real(R)
    tab_niah(R)
    _, minbits = sec_secure(R)
    NUM.append(f"## Secure digital merge: minimum passing ring bits {minbits}")
    tab_latency(R, kap, minbits)
    tab_field()
    p = os.path.join(ROOT, "results", "p8_analysis", "paper_numbers.md")
    open(p, "w", encoding="utf-8").write("\n".join(NUM) + "\n")
    print("\n".join(NUM))


if __name__ == "__main__":
    main()
