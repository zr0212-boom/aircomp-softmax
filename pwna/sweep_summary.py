"""
Summarize g1_llm.py sweeps (any mix of N, M, policies, references, NIAH modes).
For fidelity: SNR at which the mean NLL increase first drops below 0.05 / 0.01 nats (log-linear
interpolation between grid points; ">max" if never), with a 95% CI half-width of the NLL at that
point over the (segment, channel-draw) samples, and top-1 agreement at 10 / 20 dB.
For NIAH: accuracy grid with Wilson 95% intervals.
Usage: python pwna/sweep_summary.py results/a.json [results/b.json ...]
"""
import json
import math
import sys
from collections import defaultdict


def need_snr(row, th):
    """row: sorted [(snr, nll)]; first crossing below th with linear interpolation in dB."""
    prev = None
    for s, v in row:
        if v <= th:
            if prev is None or prev[1] <= th:
                return s
            (s0, v0) = prev
            return s0 + (s - s0) * (v0 - th) / max(1e-12, v0 - v)
        prev = (s, v)
    return None


def ci95(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return 1.96 * sd / math.sqrt(len(xs))


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(paths):
    cells = []
    for p in paths:
        d = json.load(open(p))
        cells += [dict(c, _src=p) for c in d["cells"]]   # group per file: runs may share T / N / ref
        if d.get("floor"):
            fl = d["floor"]
            print(f"[{p}] numeric floor (exact merge vs clean SDPA), mean over segments:")
            by = defaultdict(list)
            for k, v in fl.items():
                T, N = k.split("|")[:2]
                by[(int(T), int(N))].append(v)
            for (T, N), vs in sorted(by.items()):
                print(f"   T={T} N={N}: KL={sum(v['kl'] for v in vs) / len(vs):.4f} "
                      f"NLL+={sum(v['nll'] for v in vs) / len(vs):+.4f} "
                      f"top1={sum(v['top1'] for v in vs) / len(vs):.3f}")
    fid = [c for c in cells if c["task"] == "fidelity"]
    groups = defaultdict(list)
    for c in fid:
        groups[(c["_src"], c["T"], c.get("N", 8), c.get("ref", "clean"))].append(c)
    for (src, T, N, ref), cs in sorted(groups.items()):
        print(f"\n=== [{src}] fidelity T={T} N={N} ref={ref}: SNR for NLL+<=0.05 / <=0.01 "
              f"(CI of NLL+ at that grid point), top-1 at 10/20 dB ===")
        rows = defaultdict(dict)
        for c in cs:
            rows[(c["policy"], c["M"])][c["snr_db"]] = c
        print("  policy        M  slots   NLL+<=0.05      NLL+<=0.01      KL<=0.05  KL<=0.01   top1@10 top1@20")
        for (pol, M), r in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
            out = []
            for metric, ths, wide in (("nll", (0.05, 0.01), True), ("kl", (0.05, 0.01), False)):
                seq = sorted((s, c[metric]) for s, c in r.items())
                for th in ths:
                    s = need_snr(seq, th)
                    if s is None:
                        txt = ">" + format(seq[-1][0], ".0f")
                    elif s == seq[0][0]:
                        txt = "<=" + format(s, ".0f")
                    elif wide:
                        g = min((x for x in r if x >= s), default=None)
                        ci = ci95(r[g].get(f"{metric}_s", [])) if g is not None else float("nan")
                        txt = f"{s:.1f}(±{ci:.3f})"
                    else:
                        txt = f"{s:.1f}"
                    out.append(f"{txt:>14}" if wide else f"{txt:>9}")
            t10 = r.get(10.0, {}).get("top1", float("nan"))
            t20 = r.get(20.0, {}).get("top1", float("nan"))
            slots = next(iter(r.values())).get("slots", 1)
            print(f"  {pol:13s} {M:2d}  {slots:3d}  " + " ".join(out) + f"   {t10:.3f}   {t20:.3f}")

    niah = [c for c in cells if c["task"] == "niah"]
    ng = defaultdict(list)
    for c in niah:
        ng[(c["_src"], c["T"], c.get("ref", "gen"))].append(c)
    for (src, T, mode), cs in sorted(ng.items()):
        clean = [c for c in cs if c["policy"] == "clean"]
        cl = (f"{clean[0]['acc']:.3f} (n={clean[0]['n']})"
              + (f", exact-merge {clean[0]['ex_acc']:.3f}" if "ex_acc" in clean[0] else "")
              + (f", tf_self={clean[0]['tf_acc']:.3f}" if "tf_acc" in clean[0] else "")) if clean else "n/a"
        print(f"\n=== [{src}] NIAH T={T} mode={mode}: clean {cl} ===")
        rows = defaultdict(dict)
        for c in cs:
            if c["policy"] != "clean":
                rows[(c.get("N", 8), c["policy"], c["M"])][c["snr_db"]] = c
        snrs = sorted({c["snr_db"] for c in cs if c["policy"] != "clean"})
        print("  N  policy         M " + "".join(f"{s:>16.0f}dB" for s in snrs))
        for (N, pol, M), r in sorted(rows.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
            line = f"  {N:2d} {pol:13s} {M:2d} "
            for s in snrs:
                if s in r:
                    lo, hi = wilson(r[s]["acc"], r[s]["n"])
                    line += f"   {r[s]['acc']:.3f}[{lo:.2f},{hi:.2f}]"
                else:
                    line += " " * 18
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
