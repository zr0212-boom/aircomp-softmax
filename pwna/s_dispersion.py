"""
Dispersion of the bottleneck scale S = sum_j d_j^2 (unit gains) across decode positions, per layer and
head, and of the participation ratio N_eff, from one calibration pass of a real model (Corollary 2:
the price of blindness is driven by E[S]/S).
Usage: python pwna/s_dispersion.py --model <path> --contexts 8192 --Ns 8 32 --out results/s_disp.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from g1_llm import natural_text, split_text, take_tokens  # noqa: E402
from shard_llm import ShardEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192])
    ap.add_argument("--Ns", type=int, nargs="+", default=[8, 32])
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
    _, eval_text = split_text(natural_text(a.text_file))
    out = {"model": a.model, "rows": []}
    with torch.no_grad():
        for T in a.contexts:
            ids = torch.tensor([take_tokens(tok, eval_text, T)], device=dev)
            for N in a.Ns:
                eng.set_layout(a.anchor, T - a.eval, N)
                eng.cal = {}
                eng.calibrate()
                model(ids, logits_to_keep=1)
                spans, spans_max, neff, dev_gt_anchor = [], [], [], []
                for layer, rec in sorted(eng.cal.items()):
                    D = torch.cat(rec["d"], 1).double()                  # (H, positions, N)
                    lS = torch.log10((D ** 2).sum(-1).clamp_min(1e-300))  # (H, positions)
                    lB = torch.log10((D ** 2).amax(-1).clamp_min(1e-300))
                    q = torch.tensor([0.1, 0.9], dtype=lS.dtype)
                    spans += (torch.quantile(lS, q, dim=1)[1] - torch.quantile(lS, q, dim=1)[0]).tolist()
                    spans_max += (torch.quantile(lB, q, dim=1)[1] - torch.quantile(lB, q, dim=1)[0]).tolist()
                    neff += (D.sum(-1) ** 2 / (D ** 2).sum(-1).clamp_min(1e-300)).flatten().tolist()
                spans_t, neff_t = torch.tensor(spans), torch.tensor(neff)
                row = dict(T=T, N=N, heads=len(spans),
                           span_log10S_median=float(spans_t.median()),
                           span_log10S_p90=float(torch.quantile(spans_t, 0.9)),
                           span_log10S_max=float(spans_t.max()),
                           span_log10maxd2_median=float(torch.tensor(spans_max).median()),
                           neff_median=float(neff_t.median()),
                           neff_p90=float(torch.quantile(neff_t[torch.randperm(neff_t.numel())[:2_000_000]], 0.9)),
                           frac_neff_lt2=float((neff_t < 2).float().mean()))
                out["rows"].append(row)
                print(row, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
