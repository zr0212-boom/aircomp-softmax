"""
Final experiment campaign (P7): every LLM result of the paper, with one protocol.

Models (dense, full attention in every layer): Qwen3-4B-Instruct-2507 (main model, downloaded first),
Llama-3.1-8B-Instruct, Qwen3-8B. Corpus: WikiText-103 (validation half for calibration, test half for
evaluation); the CPython documentation as a second corpus for the main model. Fidelity protocol:
exact shard merge as reference, 4 text segments x 4 channel draws, 256 teacher-forced positions,
SNR 0-25 dB in 2.5 dB steps. Deployed designs: ladder handshake with Delta = 3 dB and nu = 4 tones per
level, detected from tones (analog ratio: ladderdet; split ratio with safeguards: ladderdet_chk), the
split ratio with the digital handshake (ladder_chk) and known weights with the genie scale
(genie_hybrid). NIAH: teacher-forced, 60 prompts x 2 draws, single key and four keys; greedy
generation on the same draws for the main model at 8k.

Jobs run as subprocesses under a GPU-memory budget (estimates per job, GB), with dependencies and up
to MAX_JOBS at a time; a job that fails is retried once, alone on the GPU if it ran out of memory (the
experiment scripts resume from their partial results). Status in <out>/p7_status.json. When every job
has finished, the runner writes ALL_DONE, waits up to an hour for the PULLED marker of the local pull,
and shuts the instance down.

  python pwna/remote/p7_runner.py                 # on the server
  python pwna/remote/p7_runner.py --dry           # locally: every job with SmolLM2-135M on a tiny grid
  python pwna/remote/p7_runner.py --list          # print the jobs with their estimates
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PWNA = os.path.dirname(HERE)
SNRS = ["0", "2.5", "5", "7.5", "10", "12.5", "15", "17.5", "20", "22.5", "25"]
NIAH_SNRS = ["5", "10", "15", "20", "25"]
DEP = ["ladderdet", "ladderdet_chk", "ladder_chk", "genie_hybrid"]
MAX_JOBS = 3
BUDGET_GB = 31.0
# GPU memory (GB) of one job: (4B model, 8B model), by context
MEM = {"8k": (11.5, 19.5), "16k": (13.5, 22.0), "32k": (18.0, 26.5)}
# seconds per teacher-forced forward of one (segment, draw) sample, and per NIAH cell of 120 trials
SEC_FID = {"8k": (0.5, 0.65), "16k": (0.75, 1.0), "32k": (1.25, 1.6)}
SEC_NIAH = {"8k": (20, 30), "16k": (32, 48), "32k": (50, 75)}


def build_jobs(dry=False):
    if dry:
        py = sys.executable
        R = os.path.join(PWNA, "results", "local_tests", "p7_dry")
        models = {m: "HuggingFaceTB/SmolLM2-135M" for m in ("q4", "l8", "q8")}
        wt, ctx, snrs, niah_snrs = [], {"8k": "1024", "16k": "2048", "32k": "2048"}, ["10", "20"], ["10", "20"]
        proto = ["--segments", "1", "--mc", "1", "--eval", "64"]
        niah_proto = ["--needles", "1", "--depths", "0.5", "--mc", "1"]
        ns_32k = ["32", "64"]
    else:
        py = "/root/miniconda3/bin/python"
        R = "/root/autodl-tmp/pwna_results/p7"
        root = "/root/autodl-tmp/models"
        models = {"q4": f"{root}/Qwen3-4B-Instruct-2507", "l8": f"{root}/Llama-3.1-8B-Instruct",
                  "q8": f"{root}/Qwen3-8B"}
        wt = ["--text_file", "/root/autodl-tmp/corpora/wikitext103_valtest.txt"]
        ctx, snrs, niah_snrs = {"8k": "8192", "16k": "16384", "32k": "32768"}, SNRS, NIAH_SNRS
        proto = ["--segments", "4", "--mc", "4", "--eval", "256"]
        niah_proto = ["--needles", "20", "--mc", "2"]
        ns_32k = ["32", "128"]
    hs = ["--ladder_db", "3", "--tone_reps", "4"]                     # deployed handshake
    large = {"l8", "q8"}
    deps = {m: [] if dry else [f"dl_{m}"] for m in ("q4", "l8", "q8")}
    jobs = []

    def add(name, script, args, m, T, minutes, env=None, out_ext=".json"):
        out = os.path.join(R, name + out_ext)
        flag = "--npz" if out_ext == ".npz" else "--out"
        d = list(deps[m]) + (["prep_wt"] if not dry and "--text_file" in args else [])
        jobs.append(dict(name=name, cmd=[py, os.path.join(PWNA, script), "--model", models[m], *args, flag, out],
                         mem=MEM[T][m in large], deps=d, minutes=round(minutes), env=env or {},
                         out=out, log=os.path.join(R, name + ".out")))

    def fid(name, m, T, Ns, Ms, pols, extra=(), corpus=True, env=None, grid=None):
        grid = snrs if dry or grid is None else grid
        n_pol = len(pols) * (3 if "--csi_nmses" in extra else 1)
        minutes = len(Ns) * len(Ms) * len(grid) * n_pol * 16 * SEC_FID[T][m in large] / 60
        add(name, "g1_llm.py", ["--contexts", ctx[T], "--task", "fidelity", "--ref", "exact", *proto,
                                "--snrs", *grid, *hs, *extra, "--Ns", *Ns, "--M", *Ms, "--policies", *pols,
                                *(wt if corpus else [])], m, T, minutes, env)

    def niah(name, m, T, Ns, pols, extra=(), corpus=True, mode="tf", nsnr=len(NIAH_SNRS), s=None):
        cells = len(Ns) * nsnr * len(pols)
        minutes = 4 + cells * SEC_NIAH[T][m in large] / 60 * (3 if mode == "gen" else 1)
        add(name, "g1_llm.py", ["--contexts", ctx[T], "--task", "niah", "--niah_mode", mode, *niah_proto,
                                "--snrs", *(s or niah_snrs), *hs, *extra, "--Ns", *Ns, "--M", "4", "--policies", *pols,
                                *(wt if corpus else [])], m, T, minutes)

    if not dry:
        # a fresh instance: corpus and the three models (40.5 GB from ModelScope), downloaded one after the
        # other ('after': order only, a failed download does not cancel the next one), so that the jobs of
        # the main model start while the 8B models are still downloading
        jobs.append(dict(name="prep_wt", cmd=["bash", os.path.join(HERE, "get_wikitext.sh")], mem=0, deps=[],
                         minutes=2, env={}, out="", log=os.path.join(R, "prep_wt.out")))
        for name, repo, m, after, mins in (("dl_q4", "Qwen/Qwen3-4B-Instruct-2507", "q4", [], 15),
                                           ("dl_l8", "LLM-Research/Meta-Llama-3.1-8B-Instruct", "l8", ["dl_q4"], 30),
                                           ("dl_q8", "Qwen/Qwen3-8B", "q8", ["dl_l8"], 30)):
            jobs.append(dict(name=name, cmd=["bash", os.path.join(HERE, "dl_model.sh"), repo, models[m]], mem=0,
                             deps=[], after=after, minutes=mins, env={}, out="", log=os.path.join(R, name + ".out")))

    # F1: main table (Table III, Fig. 4): deployed designs, every model, 8k / 16k / 32k
    for m in ("q4", "l8", "q8"):
        fid(f"p7_f1_{m}_8k", m, "8k", ["8", "32", "64"], ["4", "8"] if m == "q4" else ["4"], DEP)
        fid(f"p7_f1_{m}_16k", m, "16k", ["8", "32", "64"], ["4"], DEP)
        fid(f"p7_f1_{m}_32k", m, "32k", ns_32k, ["4"], DEP)
    # N sweep of the main model at 8k with ZF-TDMA, for the latency table and the regime map (Fig. 3)
    fid("p7_f1_q4_8k_nsweep", "q4", "8k", ["4", "16", "128"], ["4"], DEP + ["zf_tdma"])
    fid("p7_f1_q4_8k_zf", "q4", "8k", ["8", "32", "64"], ["4"], ["zf_tdma"])
    # F4a: NIAH at 8k (Table V)
    for m in ("q4", "l8", "q8"):
        niah(f"p7_f4_{m}_8k", m, "8k", ["8", "32"], DEP)
    # F2: handshake (Table VI): ladder step, tones per level, ideal and digital handshakes, exact-D-only
    # split ratio, AirMoE scale; main model, 8k, M = 4 (the deployed cells come from p7_f1_q4_8k)
    N3 = ["8", "32", "64"]
    fid("p7_f2_db15_r2", "q4", "8k", N3, ["4"], ["ladder", "ladder_chk", "ladderdet", "ladderdet_chk"],
        ["--ladder_db", "1.5", "--tone_reps", "2"])
    fid("p7_f2_db15_r4", "q4", "8k", N3, ["4"], ["ladderdet", "ladderdet_chk"], ["--ladder_db", "1.5", "--tone_reps", "4"])
    fid("p7_f2_db3_r2", "q4", "8k", N3, ["4"], ["ladder", "ladderdet", "ladderdet_chk"], ["--ladder_db", "3", "--tone_reps", "2"])
    fid("p7_f2_db3_r4", "q4", "8k", N3, ["4"], ["ladderdet_hybrid", "airmoe", "airmoe_hybrid"])
    fid("p7_f2_lo20", "q4", "8k", N3, ["4"], ["ladderdet", "ladderdet_chk"], env={"PWNA_LADDER_LO_DB": "20"})
    # F3: ablations (Table VII): main model, 8k, N = 32, M = 4, deployed handshake
    one = (["32"], ["4"])
    fid("p7_f3_db6", "q4", "8k", *one, ["ladder", "ladderdet", "ladderdet_chk"], ["--ladder_db", "6", "--tone_reps", "4"])
    fid("p7_f3_comb_maxmin", "q4", "8k", *one, ["ladderdet", "ladderdet_chk"], ["--combiner", "maxmin"])
    fid("p7_f3_comb_lsq", "q4", "8k", *one, ["ladderdet", "ladderdet_chk"], ["--combiner", "lsq"])
    fid("p7_f3_csi", "q4", "8k", *one, ["ladderdet", "ladderdet_chk"], ["--csi_nmses", "0.01", "0.05", "0.1"])
    fid("p7_f3_sigw4", "q4", "8k", *one, ["ladderdet", "ladderdet_chk", "genie_hybrid"], ["--sigma_w", "4"])
    fid("p7_f3_sigw8", "q4", "8k", *one, ["ladderdet", "ladderdet_chk", "genie_hybrid", "ladderdet@wf2",
                                          "ladderdet_chk@wf2", "ladderdet@wf4", "ladderdet_chk@wf4"], ["--sigma_w", "8"])
    fid("p7_f3_static", "q4", "8k", *one, ["ladderdet", "ladderdet_chk", "ladder", "lstale-6", "lstale0", "lstale3",
                                           "lstale-6_hybrid", "lstale0_hybrid", "lstale3_hybrid"], ["--static_channel"])
    for m in ("q4", "l8", "q8"):                         # blind designs: up to 50 dB
        fid(f"p7_f3_blind_{m}", m, "8k", ["8", "32"], ["4"], ["fixed@q0.5", "fixed@q0.9", "fixed@q0.99", "fixed_chk@q0.99"]
            + (["fixed_chk@q0.9", "csi@q0.9", "classical@q0.9"] if m == "q4" else []), grid=["10", "20", "30", "40", "50"])
    fid("p7_f3_pydoc", "q4", "8k", N3, ["4"], DEP, corpus=False)
    # F4b: NIAH at 16k and with four keys (every model), 32k and the documentation haystack (main model),
    # greedy generation on the same draws as the teacher-forced test (main model, 8k)
    for m in ("q4", "l8", "q8"):
        niah(f"p7_f4_{m}_16k", m, "16k", ["8", "32"], DEP)
        niah(f"p7_f4_{m}_mk8k", m, "8k", ["8", "32"], DEP, ["--niah_keys", "4"])
    niah("p7_f4_q4_32k", "q4", "32k", ns_32k, DEP)
    niah("p7_f4_q4_pydoc8k", "q4", "8k", ["8", "32"], DEP, corpus=False)
    niah("p7_f4_q4_gen8k", "q4", "8k", ["8", "32"], ["ladderdet", "ladderdet_chk"], ["--max_new", "32"],
         mode="gen", nsnr=2, s=["10", "20"])
    # F5: S dispersion and N_eff, fixed-point ranges (every model), weights for the combiner study
    for m in ("q4", "l8", "q8"):
        add(f"p7_f5_sdisp_{m}", "s_dispersion.py", ["--contexts", ctx["8k"], ctx["16k"], "--Ns", "8", "32", "64", *wt],
            m, "16k", 6)
        add(f"p7_f5_fx_{m}", "fixedpoint_range.py", ["--contexts", ctx["8k"], "--Ns", "8", "32", "--M", "4",
                                                     "--ladder_db", "3", "--guard", "2", *wt], m, "8k", 6)
    add("p7_f5_comb_q4", "combiner_opt.py", ["--stage", "prep", "--context", ctx["8k"], *wt,
                                             *(["--n_inst", "40"] if dry else [])], "q4", "8k", 5, out_ext=".npz")
    return jobs, R


def machine_limits(dry):
    """GPU-memory budget (GiB) and job cap: the GPU's total memory (nvidia-smi) minus a margin, or
    PWNA_GPU_BUDGET; at most 4 jobs on cards above 60 GiB, and one job per 20 GiB of host memory (the
    weights pass through host memory while a job loads)."""
    if dry:
        return BUDGET_GB, MAX_JOBS
    budget = os.environ.get("PWNA_GPU_BUDGET")
    if budget is None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60).stdout
            budget = int(out.split()[0]) / 1024 - 0.8      # PyTorch sees ~0.5 GiB less; CUDA contexts
        except Exception:
            budget = BUDGET_GB
    budget = float(budget)
    cap = 4 if budget > 60 else MAX_JOBS
    try:
        ram = int(open("/proc/meminfo").read().split()[1]) / 2 ** 20
        cap = max(1, min(cap, int(ram // 20)))
    except Exception:
        pass
    return budget, cap


def run(jobs, R, dry):
    os.makedirs(R, exist_ok=True)
    budget_gb, max_jobs = machine_limits(dry)
    print(time.strftime("%F %T"), f"GPU budget {budget_gb:.1f} GiB, up to {max_jobs} GPU jobs", flush=True)
    status = {j["name"]: dict(state="pending", tries=0, exclusive=False) for j in jobs}
    byname = {j["name"]: j for j in jobs}
    prev = os.path.join(R, "p7_status.json")
    if os.path.exists(prev):                             # restart after an interruption: keep finished jobs
        for n, s in json.load(open(prev)).get("jobs", {}).items():
            if n in status and s.get("state") == "done":
                status[n] = dict(s, note="finished in an earlier session")
    running = {}
    t_start = time.time()
    # live tuning from measured usage: <R>/p7_tune.json, re-read whenever it changes, e.g.
    # {"budget": 31.5, "max_jobs": 3, "mem_scale": {"p7_f1_q4": 0.8}, "mem": {"p7_f1_l8_8k": 18}}
    tune_path, tune_mtime = os.path.join(R, "p7_tune.json"), None
    mem0 = {j["name"]: j["mem"] for j in jobs}

    def save():
        json.dump(dict(started=time.strftime("%F %T", time.localtime(t_start)),
                       elapsed_min=round((time.time() - t_start) / 60, 1),
                       jobs={n: {**{k: v for k, v in s.items() if k != "t0"}, "mem": byname[n]["mem"],
                                 "est_min": byname[n]["minutes"]} for n, s in status.items()}),
                  open(os.path.join(R, "p7_status.json"), "w"), indent=1)

    while True:
        if os.path.exists(tune_path) and os.path.getmtime(tune_path) != tune_mtime:
            tune_mtime = os.path.getmtime(tune_path)
            try:
                t = json.load(open(tune_path))
                budget_gb, max_jobs = float(t.get("budget", budget_gb)), int(t.get("max_jobs", max_jobs))
                scale = sorted(t.get("mem_scale", {}).items(), key=lambda kv: -len(kv[0]))
                for n, m in mem0.items():
                    f = next((v for k, v in scale if n.startswith(k)), 1.0)
                    byname[n]["mem"] = float(t.get("mem", {}).get(n, round(m * f, 2))) if m > 0 else 0
                print(time.strftime("%F %T"), f"tuning: budget {budget_gb} GiB, up to {max_jobs} GPU jobs, {t}", flush=True)
            except Exception as e:
                print(time.strftime("%F %T"), "ignoring bad tuning file:", repr(e), flush=True)
        for n, p in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[n]
            st = status[n]
            st["end"] = time.strftime("%F %T")
            st["minutes"] = round((time.time() - st["t0"]) / 60, 1)
            if rc == 0:
                st["state"] = "done"
            else:
                log = open(byname[n]["log"], errors="ignore").read()[-20000:] if os.path.exists(byname[n]["log"]) else ""
                oom = "out of memory" in log.lower()
                if st["tries"] < 2:
                    st.update(state="pending", exclusive=oom or st["exclusive"], last_rc=rc)
                else:
                    st.update(state="failed", last_rc=rc)
            print(time.strftime("%F %T"), n, st["state"], f"rc={rc}", flush=True)
        failed = {n for n, s in status.items() if s["state"] in ("failed", "skipped")}
        for n, s in status.items():                      # dependents of failed jobs cannot run
            if s["state"] == "pending" and any(d in failed for d in byname[n]["deps"]):
                s["state"] = "skipped"
        pending = [j for j in jobs if status[j["name"]]["state"] == "pending"]
        if not pending and not running:
            break
        used = sum(byname[n]["mem"] for n in running)
        n_gpu = sum(byname[n]["mem"] > 0 for n in running)
        exclusive_running = any(status[n]["exclusive"] for n in running)
        for j in pending:
            n = j["name"]
            if any(status[d]["state"] != "done" for d in j["deps"]):
                continue
            if any(status[d]["state"] in ("pending", "running") for d in j.get("after", [])):
                continue
            if dry and running:
                break
            if j["mem"] == 0:                            # downloads and corpus: no GPU, no job slot
                pass
            elif status[n]["exclusive"]:
                if any(byname[r]["mem"] > 0 for r in running):
                    continue
            elif exclusive_running or n_gpu >= max_jobs or used + j["mem"] > budget_gb:
                continue
            os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", **j["env"])
            logf = open(j["log"], "a")
            running[n] = subprocess.Popen(j["cmd"], stdout=logf, stderr=subprocess.STDOUT,
                                          cwd=os.path.dirname(PWNA), env=env)
            status[n].update(state="running", tries=status[n]["tries"] + 1, t0=time.time(),
                             start=time.strftime("%F %T"))
            used += j["mem"]
            n_gpu += j["mem"] > 0
            exclusive_running = exclusive_running or status[n]["exclusive"]
            print(time.strftime("%F %T"), "start", n, f"mem={j['mem']} GB, est {j['minutes']} min", flush=True)
            if status[n]["exclusive"]:
                break
        save()
        time.sleep(float(os.environ.get("PWNA_POLL", 2 if dry else 15)))
    save()
    bad = [n for n, s in status.items() if s["state"] != "done"]
    print(time.strftime("%F %T"), f"all jobs finished, {len(bad)} not done: {bad}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="run only these jobs")
    a = ap.parse_args()
    jobs, R = build_jobs(dry=a.dry)
    if a.only:
        jobs = [dict(j, deps=[d for d in j["deps"] if d in a.only], after=[d for d in j.get("after", []) if d in a.only])
                for j in jobs if j["name"] in a.only]
    if a.list:
        tot = 0
        for j in jobs:
            tot += j["minutes"] if j["mem"] else 0
            print(f"{j['name']:22s} mem {j['mem']:>5} GB  est {j['minutes']:>4} min  deps {','.join(j['deps']) or '-'}"
                  + (f"  after {','.join(j['after'])}" if j.get("after") else "") + (f"  env {j['env']}" if j["env"] else ""))
        print(f"{len(jobs)} jobs, {tot / 60:.1f} GPU-hours if run one at a time")
        return
    bad = run(jobs, R, a.dry)
    if a.dry:
        sys.exit(1 if bad else 0)
    open(os.path.join(R, "ALL_DONE"), "w").write(time.strftime("%F %T") + "\n" + json.dumps(bad) + "\n")
    for _ in range(120):                                 # up to 60 min for the local pull
        if os.path.exists(os.path.join(R, "PULLED")):
            break
        time.sleep(30)
    subprocess.run(["sync"])
    subprocess.run(["/usr/bin/shutdown"])


if __name__ == "__main__":
    main()
