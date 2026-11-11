"""
G1 go/no-go (GPU): shard-level over-the-air attention merge inside an instruct LLM.

Layout. The FC holds the instruction prefix (anchor, first ANCHOR tokens) and everything from the
question on (question + generated tokens); the long context in between is split into N contiguous
device shards (their KV caches). Only queries from the question on go over the air.

Tasks
  niah      single-needle retrieval in natural text (needle at several depths); accuracy
            --niah_mode gen: greedy generation per trial
            --niah_mode tf : teacher-forced check of the exact shard merge's greedy answer (cut at the
                             end of the key): a trial is a hit iff every answer token is the arg-max
                             under the air merge, i.e. greedy decoding with the same channel draws
                             reproduces the reference answer up to the key. One forward per prompt;
                             slightly stricter than gen
  fidelity  teacher-forced decode window of EVAL tokens after a long natural-text context:
            KL(p_ref || p_air), next-token NLL increase, top-1 agreement with the reference
            --ref clean: the clean SDPA model; --ref exact: the exact shard merge (removes the bf16
            path-difference floor; the floor itself is stored under "floor")

Decision rule (from the research plan): at M = 8 (and 4) FC antennas, the ladder design is within
~1 pp NIAH accuracy and ~0.05 nats NLL of clean at <= 20-25 dB  ->  LLM stays the flagship application.

Example (AutoDL):
  python pwna/g1_llm.py --model Qwen/Qwen2.5-3B-Instruct --contexts 8192 --task fidelity \
      --Ns 8 16 32 --M 4 8 --ref exact --segments 4 --out pwna/results/sweep.json
Resumable: finished cells already in --out are skipped.
"""
import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ota import split_policy  # noqa: E402
from shard_llm import LADDER_MARGIN_HI_DB, LADDER_MARGIN_LO_DB, ShardEngine  # noqa: E402

ADJ = ("crimson azure golden silver violet amber emerald cobalt ivory velvet hollow gilded frosted "
       "molten arctic lunar solar nimble quiet scarlet").split()
NOUN = ("falcon lantern harbor cipher meadow glacier comet anvil willow beacon thicket orchard summit "
        "ember quartz raven turbine marsh canyon zephyr").split()
NAMES = ("Alice Bruno Chen Dmitri Elena Farid Greta Hiro Ines Jonas Kofi Lena Mateo Nadia Omar "
         "Priya").split()
BLIND = ("fixed", "csi", "classical")


def natural_text(path):
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    from pydoc_data.topics import topics
    return "\n\n".join(topics[k] for k in sorted(topics))


def split_text(text):
    h = len(text) // 2
    return text[:h], text[h:]


def take_tokens(tok, text, n, offset=0):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) < offset + n:
        reps = (offset + n) // max(1, len(ids)) + 1
        ids = (ids + tok("\n\n", add_special_tokens=False).input_ids) * reps
    return ids[offset:offset + n]


def build_niah(tok, hay_text, T, depth, rng, n_keys=1):
    """One needle at the given depth; with n_keys > 1, n_keys passphrases of distinct people (the
    queried one at the given depth, the distractors at random depths) and a question about one of
    them (multi-key retrieval in the style of RULER). Returns (ids, first query position, key)."""
    keys = []
    while len(keys) < n_keys:
        k = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}-{rng.randint(10, 99)}"
        if k not in keys:
            keys.append(k)
    if n_keys == 1:
        needles = [(depth, f" The secret passphrase is {keys[0]}. Remember it. ")]
        question = "What is the secret passphrase mentioned in the text? Answer with the passphrase only."
    else:
        names = rng.sample(NAMES, n_keys)
        needles = [(depth, f" The secret passphrase for {names[0]} is {keys[0]}. ")]
        needles += [(rng.uniform(0.05, 0.95), f" The secret passphrase for {nm} is {k}. ")
                    for nm, k in zip(names[1:], keys[1:])]
        question = f"What is the secret passphrase for {names[0]}? Answer with the passphrase only."
    instr = "Read the following text carefully.\n\n"
    budget = T - 160 - 16 * (n_keys - 1)
    hay = take_tokens(tok, hay_text, budget, offset=rng.randint(0, 5000))
    parts, last = [], 0
    for dp, text in sorted(needles):
        cut = int(dp * len(hay))
        parts += [tok.decode(hay[last:cut]), text]
        last = cut
    context = "".join(parts) + tok.decode(hay[last:])
    user = instr + context + "\n\n" + question
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    else:
        prompt = user + "\nAnswer:"
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    qpos = prompt.rfind(question)
    air = next(i for i, (s, _) in enumerate(enc["offset_mapping"]) if s >= qpos)
    return torch.tensor([enc["input_ids"]]), air, keys[0]


def gpu_peak():
    return (f", GPU peak {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GiB" if torch.cuda.is_available() else "")


def score(text, key):
    t = text.lower().replace(" ", "")
    return float(key.lower().replace(" ", "") in t)


def fidelity_metrics(logits, ref_lp, ids, s):
    """logits: last T - s positions only (forward with logits_to_keep = T - s)."""
    lp = F.log_softmax(logits[0, :-1].float(), -1)
    tgt = ids[0, s + 1:].to(lp.device)
    idx = torch.arange(tgt.numel(), device=lp.device)
    kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1)
    return dict(kl=float(kl.mean()), kl_p90=float(kl.quantile(0.9)),
                nll=float((ref_lp[idx, tgt] - lp[idx, tgt]).mean()),
                top1=float((lp.argmax(-1) == ref_lp.argmax(-1)).float().mean()))


def policy_list(args, M, N):
    pols = list(args.policies)
    if M < N:
        pols = [p for p in pols if p != "zf_sep"]
    out = []
    for v in args.csi_nmses:
        out += pols if v == 0 else [f"{p}@csi{v:g}" for p in pols if not p.startswith("zf")]
    return out


def cell_key(c):
    ref = c.get("ref") or ("clean" if c["task"] == "fidelity" else "gen")
    return (c["task"], c["T"], c.get("N", 8), c["policy"], c["snr_db"], c["M"], ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--task", choices=["niah", "fidelity", "both"], default="both")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--Ns", type=int, nargs="+", default=None, help="device counts to sweep (default [--N])")
    ap.add_argument("--anchor", type=int, default=64)
    ap.add_argument("--eval", type=int, default=256, help="decode window for the fidelity task")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--snrs", type=float, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--policies", nargs="+",
                    default=["genie", "ladder", "ladder_hybrid", "fixed", "classical", "zf_sep"])
    ap.add_argument("--ladder_db", type=float, default=1.5)
    ap.add_argument("--zf_group", type=int, default=0, help="zf_tdma devices per slot (0: max(1, M // 2))")
    ap.add_argument("--csi_nmses", type=float, nargs="+", default=[0.0],
                    help="device-side channel-estimation NMSE values (0 = perfect CSI)")
    ap.add_argument("--sigma_w", type=float, default=0.0, help="log-normal large-scale spread [dB]")
    ap.add_argument("--combiner", default="best", help="maxmin | lsq | best | eig | mrc_bound")
    ap.add_argument("--static_channel", action="store_true",
                    help="channel constant over the decode positions of a session (static devices)")
    ap.add_argument("--per_head_channel", action="store_true",
                    help="independent channels per query head (OFDM: each head's payload in its own coherence block)")
    ap.add_argument("--coh_blocks", type=int, default=1,
                    help="coherence blocks inside one instance (devices back off to their weakest block)")
    ap.add_argument("--phase_deg", type=float, default=0.0, help="per-device residual phase std [deg] (I/Q model)")
    ap.add_argument("--ramp_deg", type=float, default=0.0,
                    help="per-device phase ramp across the instance's REs, std of the end-to-end difference [deg]")
    ap.add_argument("--iq", action="store_true", help="I/Q-packed superposition with complex misalignment")
    ap.add_argument("--tone_reps", type=int, default=4,
                    help="ladderdet: REs per ladder level (spread over the band) combined by the energy detector")
    ap.add_argument("--q", type=float, default=0.8)
    ap.add_argument("--ref", choices=["clean", "exact"], default="clean")
    ap.add_argument("--segments", type=int, default=1, help="fidelity: eval text segments (offsets k*T)")
    ap.add_argument("--niah_mode", choices=["gen", "tf"], default="gen")
    ap.add_argument("--needles", type=int, default=10)
    ap.add_argument("--niah_keys", type=int, default=1, help="passphrases in the haystack (>1: multi-key)")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--mc", type=int, default=2, help="channel draws per prompt (batched)")
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--text_file", default=None, help="natural text (default: CPython pydoc topics)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "g1.json"))
    args = ap.parse_args()
    Ns = args.Ns or [args.N]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager",
                                                 dtype=dtype).to(dev).eval()
    eng = ShardEngine(seed=args.seed, device=dev, combiner=args.combiner).install(model)
    eng.sigma_w = args.sigma_w
    eng.static_channel = args.static_channel
    eng.per_head, eng.coh_blocks = args.per_head_channel, args.coh_blocks
    eng.phase_deg, eng.ramp_deg, eng.iq = args.phase_deg, args.ramp_deg, args.iq
    cal_text, eval_text = split_text(natural_text(args.text_file))
    eval_ids_all = tok(eval_text, add_special_tokens=False).input_ids
    bases = [split_policy(p.split("@")[0])[0] for p in args.policies]
    blind = any(b in BLIND or b == "ladderdet" for b in bases)
    q_list = sorted({args.q} | {float(s[1:]) for p in args.policies for s in p.split("@")[1:] if s.startswith("q")})
    wf_list = sorted({0} | {int(s[2:]) for p in args.policies for s in p.split("@")[1:] if s.startswith("wf")})

    results = json.load(open(args.out)) if os.path.exists(args.out) else {"cells": []}
    done = {cell_key(c) for c in results["cells"]}
    results["config"] = dict(vars(args), ladder_margins_db=[LADDER_MARGIN_LO_DB, LADDER_MARGIN_HI_DB])

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp"
        json.dump(results, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)

    def segment(T, k):
        off = (k * T) % max(1, len(eval_ids_all) - T)
        ids = eval_ids_all[off:off + T]
        if len(ids) < T:
            ids = take_tokens(tok, eval_text, T, offset=off)
        return torch.tensor([ids], device=dev)

    def add(cell, msg):
        results["cells"].append(cell)
        done.add(cell_key(cell))
        save()
        print(msg, flush=True)

    t_start = time.time()
    with torch.no_grad():
        for T in args.contexts:
            E = args.eval
            air = T - E
            cal_ids = torch.tensor([take_tokens(tok, cal_text, T)], device=dev)

            # ---- NIAH prompts and (tf mode) clean greedy targets, shared by every N
            prompts, targets = [], None
            if args.task in ("niah", "both"):
                rng = random.Random(args.seed + T)
                prompts = [build_niah(tok, eval_text, T, dp, rng, args.niah_keys)
                           for _ in range(args.needles) for dp in args.depths]
                if args.niah_mode == "tf":
                    # targets: greedy answer of the EXACT shard merge (full forwards, the same numeric path
                    # as the air evaluation), cut at the end of the key; the clean SDPA model's own
                    # generation accuracy is reported alongside
                    cache = results.setdefault("niah_targets", {}).get(str(T))
                    if cache and len(cache["targets"]) == len(prompts):
                        targets = cache["targets"]
                    else:
                        gc_eos = model.generation_config.eos_token_id
                        eos = {tok.eos_token_id} | set(gc_eos if isinstance(gc_eos, list) else [gc_eos])
                        targets, hits, ex_hits, tf_hits, texts = [], [], [], [], []
                        for ids, air_q, key in prompts:
                            ids = ids.to(dev)
                            eng.off()
                            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                                 max_new_tokens=max(args.max_new, 24), do_sample=False,
                                                 pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
                            text = tok.decode(out, skip_special_tokens=True)
                            eng.set_layout(args.anchor, air_q, Ns[0])
                            pkv = model(ids[:, :air_q], use_cache=True, logits_to_keep=1).past_key_values
                            eng.exact()
                            ans = []
                            for _ in range(max(args.max_new, 24)):
                                tail = torch.cat([ids[:, air_q:], torch.tensor([ans], device=dev,
                                                                               dtype=ids.dtype)], 1)
                                nxt = int(model(tail, past_key_values=pkv, use_cache=True,
                                                logits_to_keep=1).logits[0, -1].argmax())
                                pkv.crop(air_q)
                                ans.append(nxt)
                                if nxt in eos or score(tok.decode(ans, skip_special_tokens=True), key):
                                    break
                            ex_hit = score(tok.decode(ans, skip_special_tokens=True), key)
                            tgt = ans if ex_hit else tok(key, add_special_tokens=False).input_ids
                            tail = torch.cat([ids[:, air_q:], torch.tensor([tgt], device=dev)], 1)
                            lg = model(tail, past_key_values=pkv, use_cache=True,
                                       logits_to_keep=len(tgt) + 1).logits[0, :-1]
                            del pkv
                            tf_hits.append(float((lg.argmax(-1).cpu() == torch.tensor(tgt)).all()))
                            targets.append(tgt)
                            hits.append(score(text, key))
                            ex_hits.append(ex_hit)
                            texts.append(text)
                        results["niah_targets"][str(T)] = dict(targets=targets, hits=hits, ex_hits=ex_hits,
                                                               tf_hits=tf_hits, texts=texts)
                        acc = sum(hits) / len(hits)
                        add(dict(task="niah", T=T, N=0, policy="clean", snr_db=0.0, M=0, ref="tf",
                                 acc=acc, n=len(hits), ex_acc=sum(ex_hits) / len(ex_hits),
                                 tf_acc=sum(tf_hits) / len(tf_hits)),
                            f"  niah T={T} clean acc={acc:.3f} exact_acc={sum(ex_hits) / len(ex_hits):.3f} "
                            f"tf_self={sum(tf_hits) / len(tf_hits):.3f} (n={len(hits)})")

            for N in Ns:
                # ---- exact-merge check, numeric floor, calibration (sig_o, vmax; blind tables if needed)
                eng.set_layout(args.anchor, air, N)
                eng.salt = None
                eng.off()
                ref = model(cal_ids, logits_to_keep=E).logits.float()
                eng.exact()
                dmax = float((model(cal_ids, logits_to_keep=E).logits.float() - ref).abs().max())
                print(f"[T={T} N={N}] S1 exact shard merge: max|dlogit| = {dmax:.3e}", flush=True)
                results.setdefault("exact_check", {})[f"{T}|{N}"] = dmax
                del ref
                eng.cal = {}
                eng.calibrate()
                model(cal_ids, logits_to_keep=1)
                eng.build_tables(M_list=args.M if blind else [], q_list=q_list, wf_list=wf_list)

                # ---- fidelity task
                if args.task in ("fidelity", "both"):
                    # segment-outer: the context [0, air) is exact for every scheme, so its KV cache is
                    # computed once per segment and only the E-token tail is re-run per cell (cropped back
                    # after each run); one segment is cached at a time, and partial per-segment results
                    # are saved, so a run resumes at segment granularity. The channel and noise draws are
                    # keyed by (segment, draw, layer, position), so every cell sees the same draws
                    todo = [(M, snr, pol) for M in args.M for snr in args.snrs for pol in policy_list(args, M, N)
                            if ("fidelity", T, N, pol, snr, M, args.ref) not in done]
                    names = [f"{M}|{snr}|{pol}" for M, snr, pol in todo]
                    pk = f"{T}|{N}"
                    part = results.setdefault("fid_partial", {}).get(pk)
                    if part and part["names"] == names:
                        start, acc, sats, dets, secs = (part["done"], part["acc"], part["sats"], part["dets"],
                                                        part["secs"])
                    else:
                        start = 0
                        acc, sats, dets = ({nm: [] for nm in names}, {nm: [] for nm in names},
                                           {nm: {} for nm in names})
                        secs = {nm: 0.0 for nm in names}
                    for k in range(start, args.segments if todo else 0):
                        ids = segment(T, k)
                        eng.set_layout(args.anchor, air, N)
                        eng.salt = None
                        eng.off()
                        pkv = model(ids[:, :air], use_cache=True, logits_to_keep=1).past_key_values
                        lg = model(ids[:, air:], past_key_values=pkv, use_cache=True).logits
                        pkv.crop(air)
                        clean_lp = F.log_softmax(lg[0, :-1].float(), -1)
                        if args.ref == "exact":
                            eng.exact()
                            lg = model(ids[:, air:], past_key_values=pkv, use_cache=True).logits
                            pkv.crop(air)
                            fl = fidelity_metrics(lg, clean_lp, ids, air)
                            results.setdefault("floor", {})[f"{T}|{N}|{k}"] = fl
                            ref_lp = F.log_softmax(lg[0, :-1].float(), -1)
                            del clean_lp
                        else:
                            ref_lp = clean_lp
                        eng.salt = None if k == 0 else k
                        for (M, snr, pol), nm in zip(todo, names):
                            t0 = time.time()
                            for mc in range(args.mc):
                                eng.evaluate(pol, snr, M=M, mc=mc, q=args.q, ladder_db=args.ladder_db,
                                             zf_group=args.zf_group, tone_reps=args.tone_reps)
                                lg = model(ids[:, air:], past_key_values=pkv, use_cache=True).logits
                                pkv.crop(air)
                                acc[nm].append(fidelity_metrics(lg, ref_lp, ids, air))
                                sats[nm] += eng.stats["sat"]
                                for key in ("levels", "nodetect", "overflow", "det_under", "det_under12dB",
                                            "det_over10dB", "mass_loss"):
                                    dets[nm].setdefault(key, []).extend(eng.stats.get(key, []))
                            secs[nm] += time.time() - t0
                        eng.salt = None
                        del pkv, ref_lp
                        results["fid_partial"][pk] = dict(names=names, done=k + 1, acc=acc, sats=sats,
                                                          dets=dets, secs=secs)
                        save()
                        print(f"  fid T={T} N={N}: segment {k + 1}/{args.segments} x {len(todo)} cells "
                              f"({sum(secs.values()):.0f} s{gpu_peak()})", flush=True)
                    for (M, snr, pol), nm in zip(todo, names):
                        a = acc[nm]
                        cell = {m: sum(x[m] for x in a) / len(a) for m in a[0]}
                        cell.update({f"{m}_s": [x[m] for x in a] for m in ("kl", "nll", "top1")})
                        cell.update({key: sum(v) / len(v) for key, v in dets[nm].items() if v})
                        G = min(args.zf_group or max(1, M // 2), M)
                        cell.update(task="fidelity", T=T, N=N, policy=pol, snr_db=snr, M=M, ref=args.ref,
                                    sat=sum(sats[nm]) / max(1, len(sats[nm])),
                                    slots=math.ceil(N / G) if pol == "zf_tdma" else 1, sec=round(secs[nm], 1))
                        add(cell, f"  fid T={T} N={N} M={M:2d} SNR={snr:4.1f} {pol:13s} "
                                  f"KL={cell['kl']:.4f} NLL+={cell['nll']:+.4f} top1={cell['top1']:.3f}"
                                  + "".join(f" {key}={cell[key]:.4f}" for key in dets[nm] if key in cell))
                    results["fid_partial"].pop(pk, None)
                    save()

                # ---- NIAH task
                if args.task in ("niah", "both"):
                    def run_niah(pol, snr, M):
                        hits = []
                        for i, (ids, air_q, key) in enumerate(prompts):
                            eng.set_layout(args.anchor, air_q, N)
                            reps = 1 if pol == "clean" else args.mc
                            if args.niah_mode == "tf":
                                tgt = torch.tensor(targets[i], device=dev)
                                batch = torch.cat([ids.to(dev), tgt[None]], 1).repeat(reps, 1)
                            else:
                                batch = ids.repeat(reps, 1).to(dev)
                            if pol == "clean":
                                eng.off()
                            else:
                                eng.salt = i
                                eng.evaluate(pol, snr, M=M, mc=list(range(reps)), q=args.q,
                                             ladder_db=args.ladder_db, zf_group=args.zf_group, tone_reps=args.tone_reps)
                            if args.niah_mode == "tf":
                                lg = model(batch, logits_to_keep=tgt.numel() + 1).logits[:, :-1]
                                hits += (lg.argmax(-1) == tgt[None]).all(-1).float().tolist()
                            else:
                                out = model.generate(batch, attention_mask=torch.ones_like(batch),
                                                     max_new_tokens=args.max_new, do_sample=False,
                                                     pad_token_id=tok.eos_token_id)
                                for row in out[:, ids.shape[1]:]:
                                    hits.append(score(tok.decode(row, skip_special_tokens=True), key))
                        eng.salt = None
                        return sum(hits) / len(hits), len(hits), [int(h) for h in hits]

                    if args.niah_mode == "tf":
                        # prompt-outer: one cached prefix per prompt serves every (M, SNR, policy) cell;
                        # resumable at prompt granularity through results["niah_partial"]
                        todo = [(M, snr, pol) for M in args.M for snr in args.snrs
                                for pol in policy_list(args, M, N)
                                if ("niah", T, N, pol, snr, M, "tf") not in done]
                        names = [f"{M}|{snr}|{pol}" for M, snr, pol in todo]
                        pk = f"{T}|{N}"
                        part = results.setdefault("niah_partial", {}).get(pk)
                        start, hits = ((part["done"], part["hits"]) if part and part["names"] == names
                                       else (0, {nm: [] for nm in names}))
                        t0 = time.time()
                        for i in range(start, len(prompts) if todo else 0):
                            ids, air_q, key = prompts[i]
                            ids = ids.to(dev)
                            eng.set_layout(args.anchor, air_q, N)
                            eng.off()
                            pkv = model(ids[:, :air_q], use_cache=True, logits_to_keep=1).past_key_values
                            tgt = torch.tensor(targets[i], device=dev)
                            tail = torch.cat([ids[:, air_q:], tgt[None]], 1)
                            eng.salt = i
                            for (M, snr, pol), nm in zip(todo, names):
                                for mc in range(args.mc):
                                    eng.evaluate(pol, snr, M=M, mc=[mc], q=args.q, ladder_db=args.ladder_db,
                                                 zf_group=args.zf_group, tone_reps=args.tone_reps)
                                    lg = model(tail, past_key_values=pkv, use_cache=True,
                                               logits_to_keep=tgt.numel() + 1).logits[0, :-1]
                                    pkv.crop(air_q)
                                    hits[nm].append(int((lg.argmax(-1) == tgt).all()))
                            eng.salt = None
                            del pkv
                            if (i + 1) % 5 == 0 or i + 1 == len(prompts):
                                results["niah_partial"][pk] = dict(names=names, done=i + 1, hits=hits)
                                save()
                                print(f"  niah T={T} N={N}: {i + 1}/{len(prompts)} prompts x {len(todo)} cells "
                                      f"({time.time() - t0:.0f} s{gpu_peak()})", flush=True)
                        for (M, snr, pol), nm in zip(todo, names):
                            h = hits[nm]
                            add(dict(task="niah", T=T, N=N, policy=pol, snr_db=snr, M=M, ref="tf",
                                     acc=sum(h) / len(h), n=len(h), hits=h),
                                f"  niah T={T} N={N} M={M:2d} SNR={snr:4.1f} {pol:13s} acc={sum(h) / len(h):.3f} "
                                f"(n={len(h)})")
                        results["niah_partial"].pop(pk, None)
                        save()
                        todo_gen = []
                    else:
                        todo_gen = args.M
                    if args.niah_mode == "gen" and ("niah", T, N, "clean", 0.0, 0, "gen") not in done:
                        acc, n, hits = run_niah("clean", 0.0, 0)
                        add(dict(task="niah", T=T, N=N, policy="clean", snr_db=0.0, M=0, ref="gen",
                                 acc=acc, n=n, hits=hits), f"  niah T={T} clean acc={acc:.3f} (n={n})")
                    for M in todo_gen:
                        for snr in args.snrs:
                            for pol in policy_list(args, M, N):
                                if ("niah", T, N, pol, snr, M, args.niah_mode) in done:
                                    continue
                                t0 = time.time()
                                acc, n, hits = run_niah(pol, snr, M)
                                add(dict(task="niah", T=T, N=N, policy=pol, snr_db=snr, M=M,
                                         ref=args.niah_mode, acc=acc, n=n, hits=hits,
                                         sec=round(time.time() - t0, 1)),
                                    f"  niah T={T} N={N} M={M:2d} SNR={snr:4.1f} {pol:13s} acc={acc:.3f} "
                                    f"(n={n})")
                print(f"[T={T} N={N}] done, elapsed {time.time() - t_start:.0f} s", flush=True)
    eng.off()
    save()


if __name__ == "__main__":
    main()
