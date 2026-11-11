"""
Completeness and consistency check of the P8 results against the job list of p8_runner.py: every job
has its output; the saved configuration matches the command line (and the environment, for the ladder
margins); every expected cell (context x N x M x SNR x policy, NIAH and calibration rows included) is
present with finite metrics.
  python pwna/remote/p8_verify.py [--dir results/p8]      # final results
  python pwna/remote/p8_verify.py --dry                           # results of p8_runner.py --dry
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from p8_runner import build_jobs  # noqa: E402

FLOAT_KEYS = ("ladder_db", "sigma_w", "phase_deg", "ramp_deg")
INT_KEYS = ("tone_reps", "niah_keys", "max_new", "segments", "mc", "eval", "needles", "coh_blocks")
STR_KEYS = ("combiner", "niah_mode", "task", "ref")


def parse(cmd):
    """Flags of a command line: {'--flag': [values]} (a flag without values maps to [])."""
    out, key = {}, None
    for tok in cmd[2:]:
        if tok.startswith("--"):
            key = tok
            out[key] = []
        elif key is not None:
            out[key].append(tok)
    return out


def expected_policies(pols, csi, M, N):
    pols = [p for p in pols if not (p == "zf_sep" and M < N)]
    out = []
    for v in csi:
        out += pols if v == 0 else [f"{p}@csi{v:g}" for p in pols if not p.startswith("zf")]
    return out


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def check_g1(job, f, res):
    prob = []
    cfg = res.get("config", {})
    for k in FLOAT_KEYS:
        if f"--{k}" in f and abs(float(cfg.get(k, math.nan)) - float(f[f"--{k}"][-1])) > 1e-9:
            prob.append(f"config {k}={cfg.get(k)} but command {f['--' + k][-1]}")
    for k in INT_KEYS:
        if f"--{k}" in f and int(cfg.get(k, -1)) != int(f[f"--{k}"][-1]):
            prob.append(f"config {k}={cfg.get(k)} but command {f['--' + k][-1]}")
    for k in STR_KEYS:
        if f"--{k}" in f and str(cfg.get(k)) != f[f"--{k}"][-1]:
            prob.append(f"config {k}={cfg.get(k)} but command {f['--' + k][-1]}")
    if bool(cfg.get("static_channel")) != ("--static_channel" in f):
        prob.append(f"config static_channel={cfg.get('static_channel')}")
    if (cfg.get("text_file") or None) != (f["--text_file"][-1] if "--text_file" in f else None):
        prob.append(f"config text_file={cfg.get('text_file')}")
    lo = float(job["env"].get("PWNA_LADDER_LO_DB", 3.0))
    rec = cfg.get("ladder_margins_db")
    if rec is None and job["env"]:
        prob.append("ladder margins not recorded")
    elif rec is not None and rec[0] != lo:
        prob.append(f"ladder margins {rec} but expected low margin {lo}")
    cells = res.get("cells", [])
    have = {(c["task"], c["T"], c.get("N"), c["M"], c["policy"], float(c["snr_db"])): c for c in cells}
    task = f["--task"][-1]
    Ts = [int(x) for x in f["--contexts"]]
    Ns = [int(x) for x in f["--Ns"]]
    Ms = [int(x) for x in f["--M"]]
    snrs = [float(x) for x in f["--snrs"]]
    csi = [float(x) for x in f.get("--csi_nmses", ["0"])]
    missing, bad = [], []
    for T in Ts:
        if task == "niah" and not any(c["task"] == "niah" and c["T"] == T and c["policy"] == "clean" for c in cells):
            missing.append(f"niah clean T={T}")
        for N in Ns:
            for M in Ms:
                for snr in snrs:
                    for pol in expected_policies(f["--policies"], csi, M, N):
                        c = have.get((task, T, N, M, pol, snr))
                        if c is None:
                            missing.append(f"{task} T={T} N={N} M={M} {pol} {snr:g}")
                        elif task == "fidelity" and not all(finite(c.get(k)) for k in ("kl", "nll", "top1")):
                            bad.append(f"T={T} N={N} M={M} {pol} {snr:g}: {c.get('kl')}, {c.get('nll')}")
                        elif task == "fidelity" and len(c.get("nll_s", [])) != int(f["--segments"][-1]) * int(f["--mc"][-1]):
                            bad.append(f"T={T} N={N} M={M} {pol} {snr:g}: {len(c.get('nll_s', []))} samples")
                        elif task == "niah" and not (0 <= c.get("acc", -1) <= 1):
                            bad.append(f"T={T} N={N} {pol} {snr:g}: acc {c.get('acc')}")
    if missing:
        prob.append(f"{len(missing)} missing cells, e.g. {missing[:3]}")
    if bad:
        prob.append(f"{len(bad)} invalid cells, e.g. {bad[:3]}")
    if res.get("fid_partial") or res.get("niah_partial"):
        prob.append("unfinished partial results left in the file")
    return prob, len(have)


def check_rows(job, f, res):
    n_exp = len(f.get("--contexts", [])) * len(f.get("--Ns", []))
    rows = res.get("rows", [])
    prob = [] if len(rows) == n_exp else [f"{len(rows)} rows, expected {n_exp}"]
    return prob, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()
    jobs, R = build_jobs(dry=a.dry)
    R = a.dir or (R if a.dry else os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "p8"))
    jobs = [j for j in jobs if j["mem"] > 0]              # corpus and model downloads have no output
    n_bad = 0
    for job in jobs:
        f = parse(job["cmd"])
        out = os.path.join(R, os.path.basename(job["out"]))
        if not os.path.exists(out):
            print(f"[MISSING] {job['name']}: no {os.path.basename(out)}")
            n_bad += 1
            continue
        if out.endswith(".npz"):
            import numpy as np
            z = np.load(out)
            keys = set(z.files)
            prob = [] if any(k.startswith("d_") for k in keys) and any(k.startswith("h_") for k in keys) else ["npz without d_/h_ arrays"]
            n = len(keys)
        else:
            res = json.load(open(out))
            prob, n = check_g1(job, f, res) if job["cmd"][1].endswith("g1_llm.py") else check_rows(job, f, res)
        print(f"[{'OK' if not prob else 'PROBLEM'}] {job['name']}: {n} entries" + ("" if not prob else " -- " + "; ".join(prob)))
        n_bad += bool(prob)
    print(f"\n{len(jobs) - n_bad} jobs OK, {n_bad} with problems")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
