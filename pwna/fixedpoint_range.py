"""
Ring size of the secure aggregation of the denominators D = sum_j d_j in the split ratio, from one
calibration pass of a real model. Device j sends round(2^F d_j / V) modulo 2^b; the FC needs D to a
relative precision eps of the full denominator d_A + D, i.e. N 2^-(F+1) V <= eps (d_A + D), and the sum
needs ceil(log2(N + 1)) integer bits, so b = F + ceil(log2(N + 1)).
Two public scales V are compared:
  global      the largest d_j of the head on the calibration text (an offline constant);
  handshake   V = sqrt(B-hat max_j g_j) with B-hat the ladder level of B = max_j d_j^2 / g_j, which every
              device knows after the handshake and which bounds every d_j (d_j^2 <= B-hat g_j); --guard
              extra integer bits keep the sum exact when a detected level is up to 6*guard dB too low.
Channels: i.i.d. Rayleigh, M FC antennas, the default combiner, n_draw draws per position.
Usage: python pwna/fixedpoint_range.py --model <path> --contexts 8192 --Ns 8 32 --out results/fx.json
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from g1_llm import natural_text, split_text, take_tokens  # noqa: E402
from ota import combiner_gains, draw_channels  # noqa: E402
from shard_llm import ShardEngine  # noqa: E402


def bits(V, dA, D, N, eps, guard=0):
    return torch.log2(N * V / (2.0 * eps * (dA + D))).clamp_min(0).ceil() + math.ceil(math.log2(N + 1)) + guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192])
    ap.add_argument("--Ns", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--ladder_db", type=float, default=3.0)
    ap.add_argument("--guard", type=int, default=2)
    ap.add_argument("--n_draw", type=int, default=8)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-6])
    ap.add_argument("--anchor", type=int, default=64)
    ap.add_argument("--eval", type=int, default=256)
    ap.add_argument("--text_file", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="eager",
                                                 dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model = model.to(dev).eval()
    eng = ShardEngine(device=dev).install(model)
    cal_text, eval_text = split_text(natural_text(a.text_file))
    gen = torch.Generator().manual_seed(7)
    out = {"model": a.model, "config": vars(a), "rows": []}
    with torch.no_grad():
        for T in a.contexts:
            for N in a.Ns:
                vglob = {}
                for name, text in (("cal", cal_text), ("eval", eval_text)):
                    ids = torch.tensor([take_tokens(tok, text, T)], device=dev)
                    eng.set_layout(a.anchor, T - a.eval, N)
                    eng.cal = {}
                    eng.calibrate()
                    model(ids, logits_to_keep=1)
                    if name == "cal":
                        vglob = {l: torch.cat(r["d"], 1).double().amax(dim=(1, 2)) for l, r in eng.cal.items()}
                b_glob = {e: [] for e in a.eps}
                b_hs = {e: [] for e in a.eps}
                over = []
                for layer, rec in sorted(eng.cal.items()):
                    D = torch.cat(rec["d"], 1).double()                   # (H, S, N)
                    dA = torch.cat(rec["dA"], 1).double()                 # (H, S)
                    H, S, _ = D.shape
                    g = combiner_gains(draw_channels((a.n_draw, S), N, a.M, gen, "cpu"), "best").double()
                    B = (D[:, None] ** 2 / g[None]).amax(-1)              # (H, n_draw, S)
                    Bhat = 10.0 ** (torch.ceil(10.0 * torch.log10(B.clamp_min(1e-300)) / a.ladder_db)
                                    * a.ladder_db / 10.0)
                    V_hs = (Bhat * g.amax(-1)[None]).sqrt()               # (H, n_draw, S)
                    Dsum, dAx = D.sum(-1)[:, None].expand_as(B), dA[:, None].expand_as(B)
                    V_g = vglob[layer].view(H, 1, 1).expand_as(B)
                    over.append(float((D.amax(-1)[:, None] > V_g).double().mean()))
                    for e in a.eps:
                        b_glob[e].append(bits(V_g, dAx, Dsum, N, e).flatten())
                        b_hs[e].append(bits(V_hs, dAx, Dsum, N, e, a.guard).flatten())
                row = dict(T=T, N=N, M=a.M, frac_eval_above_global_scale=sum(over) / len(over))
                for e in a.eps:
                    for tag, bb in (("global", b_glob[e]), ("handshake", b_hs[e])):
                        x = torch.cat(bb)
                        sub = x[torch.randperm(x.numel(), generator=gen)[:4_000_000]]
                        row[f"bits_{tag}_eps{e:g}"] = dict(
                            median=float(sub.median()), p999=float(torch.quantile(sub, 0.999)),
                            max=float(x.max()))
                out["rows"].append(row)
                print(row, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
