"""
Supplementary experiment campaign (P8), on the P7 protocol and seeds, so that every P8 cell is paired
with the P7 cells of the same model, context, N and M.

New designs: the saturation-aware split ratio (secure denominator of the EFFECTIVE weights, '_eff') with
the noise-matched level report ('@kap<k>'), under the digital handshake ('ladder_*') and the tone
handshake ('ladderdet_*'). New baselines: the strongest known-weight designs (AirMoE threshold scale with
an exact or effective denominator), the secure digital merge in fixed point at matched accuracy (offline
and oracle scales) and the plain bfloat16 digital merge. New conditions: per-head channels, coherence
blocks inside an instance, residual phase and phase ramps with I/Q packing, complex CSI errors, the
large-scale spread with the new design, and a combined realistic scenario. Design notes: 实验设计_P8.md.

Jobs run as subprocesses under a GPU-memory budget, with dependencies, retries and live tuning as in
P7 (<out>/p8_status.json, <out>/p8_tune.json). When every job has finished, the runner writes ALL_DONE,
waits up to an hour for the PULLED marker of the local pull, and shuts the instance down.

  python pwna/remote/p8_runner.py                 # on the server
  python pwna/remote/p8_runner.py --dry           # locally: every job with SmolLM2-135M on a tiny grid
  python pwna/remote/p8_runner.py --list          # print the jobs with their estimates
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PWNA = os.path.dirname(HERE)
NEG_SNRS = ["-15", "-12.5", "-10", "-7.5", "-5", "-2.5"]   # supplement: below the F1 grid
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


def build_jobs(dry=False, supp=False):
    if dry:
        py = sys.executable
        R = os.path.join(os.path.dirname(PWNA), "results", "local_tests", "p8_dry")
        models = {m: "HuggingFaceTB/SmolLM2-135M" for m in ("q4", "l8", "q8")}
        wt, ctx, snrs, niah_snrs = [], {"8k": "1024", "16k": "2048", "32k": "2048"}, ["10", "20"], ["10", "20"]
        proto = ["--segments", "1", "--mc", "1", "--eval", "64"]
        niah_proto = ["--needles", "1", "--depths", "0.5", "--mc", "1"]
        ns_32k = ["32", "64"]
    else:
        py = "/root/miniconda3/bin/python"
        R = "/root/autodl-tmp/pwna_results/p8" + ("_supp" if supp else "")
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
    deps = {m: [] if dry or supp else [f"dl_{m}"] for m in ("q4", "l8", "q8")}
    jobs = []

    def add(name, script, args, m, T, minutes, env=None, out_ext=".json"):
        out = os.path.join(R, name + out_ext)
        flag = "--npz" if out_ext == ".npz" else "--out"
        d = list(deps[m]) + (["prep_wt"] if not (dry or supp) and "--text_file" in args else [])
        jobs.append(dict(name=name, cmd=[py, os.path.join(PWNA, script), "--model", models[m], *args, flag, out],
                         mem=MEM[T][m in large], deps=d, minutes=round(minutes), env=env or {},
                         out=out, log=os.path.join(R, name + ".out")))

    def fid(name, m, T, Ns, Ms, pols, extra=(), corpus=True, env=None, grid=None):
        grid = snrs if dry or grid is None else grid
        n_pol = len(pols) * (3 if "--csi_nmses" in extra else 1)
        slow = 3.0 if "--per_head_channel" in extra else (1.5 if any(x in extra for x in ("--iq", "--coh_blocks")) else 1.0)
        minutes = len(Ns) * len(Ms) * len(grid) * n_pol * 16 * SEC_FID[T][m in large] * slow / 60
        add(name, "g1_llm.py", ["--contexts", ctx[T], "--task", "fidelity", "--ref", "exact", *proto,
                                "--snrs", *grid, *hs, *extra, "--Ns", *Ns, "--M", *Ms, "--policies", *pols,
                                *(wt if corpus else [])], m, T, minutes, env)

    def niah(name, m, T, Ns, pols, extra=(), corpus=True, mode="tf", nsnr=len(NIAH_SNRS), s=None):
        cells = len(Ns) * nsnr * len(pols)
        minutes = 4 + cells * SEC_NIAH[T][m in large] / 60 * (3 if mode == "gen" else 1)
        add(name, "g1_llm.py", ["--contexts", ctx[T], "--task", "niah", "--niah_mode", mode, *niah_proto,
                                "--snrs", *(s or niah_snrs), *hs, *extra, "--Ns", *Ns, "--M", "4", "--policies", *pols,
                                *(wt if corpus else [])], m, T, minutes)

    if not (dry or supp):
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

    # P8 designs (pwna/ota.py): '_eff' saturation-aware split ratio (secure denominator of the effective
    # weights); '@kap<k>' noise-matched level report; 'ladder_*' digital (thermometer) handshake, i.e. the
    # exact highest level; 'ladderdet_*' tone handshake; 'airmoe_*' strongest known-weight baselines;
    # 'secdig*' secure digital merge in fixed point; 'digbf16' plain digital merge; '@fz256' thermometer
    # handshake over Z_256 (a non-empty level reads as empty with probability 1/256)
    KNOWN = ["airmoe_hybrid", "airmoe_eff"]
    PROP = ["ladder_eff@kap0.3@fz256", "ladder_eff@kap1@fz256", "ladderdet_eff@kap0.3", "ladderdet_eff@kap1"]
    MAIN = KNOWN + PROP
    N3 = ["8", "32", "64"]
    one = (["32"], ["4"])
    if supp:
        # supplement (after P8): the F1 cells whose required SNR is floor-limited at 0 dB (Llama, Qwen3-8B,
        # Qwen3-4B at large N), on the grid below it; the draws do not depend on the SNR, so the cells
        # pair with the P8 cells of the same job and the two grids are read as one curve
        grid = ["-10", "0"] if dry else NEG_SNRS
        for m, c in (("q4", "8k"), ("q4", "16k"), ("l8", "8k"), ("l8", "16k"), ("l8", "32k"), ("q8", "8k"),
                     ("q8", "16k")):
            Ns = ns_32k if c == "32k" else N3
            pols = MAIN + (["genie_hybrid"] if (m, c) == ("q8", "16k") else [])
            fid(f"p8s_f1_{m}_{c}", m, c, Ns, ["4", "8"] if (m, c) == ("q4", "8k") else ["4"], pols, grid=grid)
        fid("p8s_f1_q4_8k_nsweep", "q4", "8k", ["4", "16", "128"], ["4"], MAIN, grid=grid)
        fid("p8s_f1_q4_pydoc", "q4", "8k", N3, ["4"], MAIN, corpus=False, grid=grid)
        # F5 scenarios where AirMoE-eff or the proposed digital design meet the target at 0 dB, and the
        # kappa = 3 cell of F2 at N = 64 (same flags as the P8 jobs, so the draws are shared)
        sup = ["airmoe_eff", "ladder_eff@kap1@fz256"]
        combo = ["--per_head_channel", "--coh_blocks", "2", "--phase_deg", "10", "--ramp_deg", "20", "--iq",
                 "--csi_nmses", "0.01", "--sigma_w", "4"]
        for name, m, ex in (("perhead", "q4", ["--per_head_channel"]), ("ph5", "q4", ["--phase_deg", "5"]),
                            ("ph10", "q4", ["--phase_deg", "10"]), ("ramp20", "q4", ["--ramp_deg", "20"]),
                            ("ramp45", "q4", ["--ramp_deg", "45"]), ("static", "q4", ["--static_channel"]),
                            ("csi", "q4", ["--iq", "--csi_nmses", "0.01", "0.05", "0.1"]), ("combo_l8", "l8", combo)):
            fid(f"p8s_f5_{name}", m, "8k", *one, sup, ex, grid=grid)
        fid("p8s_f2_kap", "q4", "8k", ["64"], ["4"], ["ladder_eff@kap3@fz256"], grid=grid)
        return jobs, R
    # F1: main table, every model and context (Ns and Ms as in P7); the exact-D-only split ratio where the
    # tone handshake overflows most (Llama, 16k / 32k); the N sweep; the CPython-docs corpus, which is the
    # development set on which kappa is chosen (pre-registered: the better of 0.3 and 1 there)
    for m in ("q4", "l8", "q8"):
        for c in ("8k", "16k", "32k"):
            Ns = ns_32k if c == "32k" else N3
            extra_p = ["ladderdet_hybrid"] if (m == "l8" and c != "8k") else []
            fid(f"p8_f1_{m}_{c}", m, c, Ns, ["4", "8"] if (m, c) == ("q4", "8k") else ["4"], MAIN + extra_p)
    fid("p8_f1_q4_8k_nsweep", "q4", "8k", ["4", "16", "128"], ["4"], MAIN)
    fid("p8_f1_q4_pydoc", "q4", "8k", N3, ["4"], MAIN, corpus=False)
    # F2: design study, main model, 8k, M = 4: kappa; each component alone (effective denominator without
    # kappa, kappa with the exact denominator or the analog ratio); the fixed offset that kappa replaces;
    # AirMoE with the analog ratio; ladder step and tones per level
    fid("p8_f2_kap", "q4", "8k", N3, ["4"],
        ["ladder_eff@fz256", "ladder_eff@kap0.1@fz256", "ladder_eff@kap3@fz256", "ladderdet_eff", "ladderdet_eff@kap0.1",
         "ladderdet_eff@kap3", "ladder_hybrid@kap1@fz256", "ladderdet@kap1", "ladderdet_eff@off3", "airmoe"])
    fid("p8_f2_db6", "q4", "8k", N3, ["4"], ["ladder_eff@kap1@fz256", "ladderdet_eff@kap1"], ["--ladder_db", "6", "--tone_reps", "4"])
    fid("p8_f2_db15", "q4", "8k", N3, ["4"], ["ladder_eff@kap1@fz256", "ladderdet_eff@kap1"],
        ["--ladder_db", "1.5", "--tone_reps", "4"])
    fid("p8_f2_r2", "q4", "8k", N3, ["4"], ["ladderdet_eff@kap1"], ["--ladder_db", "3", "--tone_reps", "2"])
    fid("p8_f2_r8", "q4", "8k", N3, ["4"], ["ladderdet_eff@kap1"], ["--ladder_db", "3", "--tone_reps", "8"])
    # F3: one-round designs (no handshake): blind and stale scales with the effective denominator
    fid("p8_f3_oneround", "q4", "8k", ["8", "32"], ["4"],
        ["fixed_eff@q0.5", "fixed_eff@q0.9", "fixed_eff@q0.99", "lstale0_eff", "lstale3_eff", "lstale0_eff@kap1",
         "lstale3_eff@kap1"], grid=["0", "5", "10", "15", "20", "25", "30", "40", "50"])
    # F4: secure digital merge at matched accuracy: ring bits (the digital link is error-free: one SNR)
    SEC = ([f"secdig@b{b}" for b in (20, 24, 28, 32, 36, 40)] + ["secdig@b40@m16"]
           + [f"secdigo@b{b}" for b in (8, 10, 12, 14, 16, 20)] + ["digbf16"])
    for name, m, c, Ns in (("q4_8k", "q4", "8k", N3), ("q4_32k", "q4", "32k", ns_32k),
                           ("l8_8k", "l8", "8k", ["32"]), ("q8_8k", "q8", "8k", ["32"])):
        fid(f"p8_f4_{name}", m, c, Ns, ["4"], SEC, grid=["20"])
    # F5: channel realism, main model, 8k, N = 32, M = 4 (and the combined scenario for Llama)
    REAL = ["ladderdet_chk", "genie_hybrid", "airmoe_eff", "ladder_eff@kap1@fz256", "ladderdet_eff@kap1"]
    for name, ex in (("perhead", ["--per_head_channel"]), ("coh2", ["--coh_blocks", "2"]),
                     ("coh4", ["--coh_blocks", "4"]), ("ph5", ["--phase_deg", "5"]), ("ph10", ["--phase_deg", "10"]),
                     ("ph20", ["--phase_deg", "20"]), ("ramp20", ["--ramp_deg", "20"]), ("ramp45", ["--ramp_deg", "45"]),
                     ("csi", ["--iq", "--csi_nmses", "0.01", "0.05", "0.1"]), ("static", ["--static_channel"])):
        fid(f"p8_f5_{name}", "q4", "8k", *one, REAL, ex)
    for sw in ("4", "8"):
        fid(f"p8_f5_sigw{sw}", "q4", "8k", *one,
            REAL + ["airmoe_hybrid", "ladder_eff@kap1@wf2@fz256", "ladderdet_eff@kap1@wf2"], ["--sigma_w", sw])
    combo = ["--per_head_channel", "--coh_blocks", "2", "--phase_deg", "10", "--ramp_deg", "20", "--iq",
             "--csi_nmses", "0.01", "--sigma_w", "4"]
    fid("p8_f5_combo", "q4", "8k", *one, REAL, combo)
    fid("p8_f5_combo_l8", "l8", "8k", *one, REAL, combo)
    # F6: NIAH (teacher-forced): the proposed design and the strongest known-weight baseline
    NI = ["airmoe_eff", "ladder_eff@kap1@fz256", "ladderdet_eff@kap1"]
    for m in ("q4", "l8", "q8"):
        niah(f"p8_f6_{m}_8k", m, "8k", ["8", "32"], NI)
        niah(f"p8_f6_{m}_mk8k", m, "8k", ["8", "32"], NI, ["--niah_keys", "4"])
    niah("p8_f6_q4_16k", "q4", "16k", ["8", "32"], NI)
    niah("p8_f6_q4_32k", "q4", "32k", ns_32k, NI)
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
    prev = os.path.join(R, "p8_status.json")
    if os.path.exists(prev):                             # restart after an interruption: keep finished jobs
        for n, s in json.load(open(prev)).get("jobs", {}).items():
            if n in status and s.get("state") == "done":
                status[n] = dict(s, note="finished in an earlier session")
    running = {}
    t_start = time.time()
    # live tuning from measured usage: <R>/p8_tune.json, re-read whenever it changes, e.g.
    # {"budget": 31.5, "max_jobs": 3, "mem_scale": {"p8_f1_q4": 0.8}, "mem": {"p8_f1_l8_8k": 18}}
    tune_path, tune_mtime = os.path.join(R, "p8_tune.json"), None
    mem0 = {j["name"]: j["mem"] for j in jobs}

    def save():
        json.dump(dict(started=time.strftime("%F %T", time.localtime(t_start)),
                       elapsed_min=round((time.time() - t_start) / 60, 1),
                       jobs={n: {**{k: v for k, v in s.items() if k != "t0"}, "mem": byname[n]["mem"],
                                 "est_min": byname[n]["minutes"]} for n, s in status.items()}),
                  open(os.path.join(R, "p8_status.json"), "w"), indent=1)

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
            elif exclusive_running or n_gpu >= max_jobs or (used + j["mem"] > budget_gb and n_gpu > 0):
                continue                                 # a job above the whole budget still runs, alone
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
    ap.add_argument("--supp", action="store_true", help="the negative-SNR supplement of F1 (out: p8_supp)")
    a = ap.parse_args()
    jobs, R = build_jobs(dry=a.dry, supp=a.supp)
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
    if os.path.exists(os.path.join(R, "KEEP_ALIVE")):  # follow-up runs planned: the operator shuts down by hand
        print(time.strftime("%F %T"), "KEEP_ALIVE present: not shutting down", flush=True)
        return
    subprocess.run(["bash", "/usr/bin/shutdown"])     # AutoDL: the script has no shebang line


if __name__ == "__main__":
    main()
