"""
Optimized max-min receive combiner versus the heuristic of ota.py (reweighted principal eigenvector,
best of it and least squares per instance), on i.i.d. Rayleigh channels with attention weights from a
real model.
  --stage prep  (torch env)  draws channels, computes the heuristic combiner and samples weight vectors
                             d from one calibration pass of --model; writes --npz
  --stage opt   (scipy env)  maximizes min_j |m^H h_j|^2 over ||m|| <= 1 with SLSQP from the heuristic
                             and random starts; reports the gain in min_j g_j and in the bottleneck
                             B = max_j d_j^2 / g_j, and the gap to the device-side MRT bound
                             max_j d_j^2 / ||h_j||^2 (Proposition, combiner bounds); writes --out
Usage:
  python pwna/combiner_opt.py --stage prep --model HuggingFaceTB/SmolLM2-135M --npz results/comb.npz
  python pwna/combiner_opt.py --stage opt --npz results/comb.npz --out results/combiner_opt.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def prep(a):
    import torch
    from g1_llm import natural_text, split_text, take_tokens
    from ota import combiner_gains, combiner_vector, draw_channels
    from shard_llm import ShardEngine
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="eager",
                                                 dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model = model.to(dev).eval()
    eng = ShardEngine(device=dev).install(model)
    _, text = split_text(natural_text(a.text_file))
    ids = torch.tensor([take_tokens(tok, text, a.context)], device=dev)
    gen = torch.Generator().manual_seed(11)
    out = {}
    with torch.no_grad():
        for N in a.Ns:
            eng.set_layout(64, a.context - a.eval, N)
            eng.cal = {}
            eng.calibrate()
            model(ids, logits_to_keep=1)
            D = torch.cat([torch.cat(r["d"], 1).reshape(-1, N) for r in eng.cal.values()], 0).double()
            D = D[D.sum(-1) > 0]
            out[f"d_{N}"] = D[torch.randperm(D.shape[0], generator=gen)[:a.n_inst]].numpy()
            for M in a.Ms:
                h = draw_channels((a.n_inst,), N, M, gen, "cpu")
                out[f"h_{N}_{M}"] = h.numpy()
                out[f"mheur_{N}_{M}"] = combiner_vector(h, "best").numpy()
                out[f"gheur_{N}_{M}"] = combiner_gains(h, "best").double().numpy()
            print(f"prep N={N}: {D.shape[0]} weight vectors", flush=True)
    np.savez_compressed(a.npz, **out)


def invsum_slsqp(h, m0, rng, starts):
    """min_m sum_j 1 / |m^H h_j|^2 over ||m|| = 1 (proportional to E[S] for exchangeable weights)."""
    from scipy.optimize import minimize
    N, M = h.shape

    def gains(x):
        m = x[:M] + 1j * x[M:]
        return np.abs(h @ m.conj()) ** 2 / max(np.sum(x ** 2), 1e-12)

    best = None
    for k in range(starts + 1):
        m = m0 if k == 0 else rng.standard_normal(M) + 1j * rng.standard_normal(M)
        x0 = np.concatenate([m.real, m.imag]) / np.linalg.norm(m)
        r = minimize(lambda x: np.sum(1.0 / np.maximum(gains(x), 1e-12)), x0, method="BFGS",
                     options={"maxiter": 300})
        g = gains(r.x)
        if best is None or np.sum(1 / g) < np.sum(1 / best):
            best = g
    return best


def maxmin_slsqp(h, m0, rng, starts):
    from scipy.optimize import minimize
    N, M = h.shape

    def gains(x):
        m = x[:M] + 1j * x[M:2 * M]
        return np.abs(h @ m.conj()) ** 2

    def solve(m):
        m = m / np.linalg.norm(m)
        x0 = np.concatenate([m.real, m.imag, [gains(np.concatenate([m.real, m.imag])).min()]])
        cons = [{"type": "ineq", "fun": lambda x: gains(x[:2 * M]) - x[-1]},
                {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x[:2 * M] ** 2)}]
        r = minimize(lambda x: -x[-1], x0, constraints=cons, method="SLSQP",
                     options={"maxiter": 300, "ftol": 1e-10})
        x = r.x[:2 * M] / max(np.linalg.norm(r.x[:2 * M]), 1e-12)
        return gains(x)

    best = None
    for k in range(starts + 1):
        m = m0 if k == 0 else rng.standard_normal(M) + 1j * rng.standard_normal(M)
        g = solve(m)
        if best is None or g.min() > best.min():
            best = g
    return best


def opt(a):
    z = np.load(a.npz)
    rng = np.random.default_rng(5)
    rows = []
    for key in sorted(k for k in z.files if k.startswith("h_")):
        _, N, M = key.split("_")
        N, M = int(N), int(M)
        h = z[key].astype(np.complex128)
        m0, g_h = z[f"mheur_{N}_{M}"].astype(np.complex128), z[f"gheur_{N}_{M}"].astype(np.float64)
        d = z[f"d_{N}"]
        n = min(a.n_inst, h.shape[0], d.shape[0])
        if a.objective == "maxmin":
            g_o = np.stack([maxmin_slsqp(h[i], m0[i], rng, a.starts) for i in range(n)])
            worse = g_o.min(-1) < g_h[:n].min(-1)       # keep the heuristic if the solver did not improve
        else:
            g_o = np.stack([invsum_slsqp(h[i], m0[i], rng, a.starts) for i in range(n)])
            worse = (1 / g_o).sum(-1) > (1 / g_h[:n]).sum(-1)
        g_o[worse] = g_h[:n][worse]
        dmin = 10 * np.log10(g_o.min(-1) / g_h[:n].min(-1))
        B_h = (d[:n] ** 2 / g_h[:n]).max(-1)
        B_o = (d[:n] ** 2 / g_o).max(-1)
        B_mrt = (d[:n] ** 2 / (np.abs(h[:n]) ** 2).sum(-1)).max(-1)
        dB = 10 * np.log10(B_h / B_o)
        gap = 10 * np.log10(B_o / B_mrt)
        q = lambda x: [float(np.percentile(x, p)) for p in (10, 50, 90)]
        row = dict(objective=a.objective, N=N, M=M, n=n, min_gain_db_p10_50_90=q(dmin),
                   bottleneck_gain_db_p10_50_90=q(dB),
                   gap_to_mrt_bound_db_p10_50_90=q(gap), frac_bottleneck_worse=float((dB < -0.01).mean()))
        rows.append(row)
        print(row, flush=True)
    json.dump(dict(npz=a.npz, rows=rows), open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "opt"], required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--eval", type=int, default=256)
    ap.add_argument("--Ns", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--Ms", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n_inst", type=int, default=400)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--objective", choices=["maxmin", "invsum"], default="maxmin")
    ap.add_argument("--text_file", default=None)
    ap.add_argument("--npz", default=os.path.join(os.path.dirname(HERE), "results", "p7", "p7_f5_comb_q4.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "combiner", "combiner_opt.json"))
    a = ap.parse_args()
    prep(a) if a.stage == "prep" else opt(a)


if __name__ == "__main__":
    main()
