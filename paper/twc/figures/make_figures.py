"""Figures of the paper, generated from results/ (run: python paper/twc/figures/make_figures.py)."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.environ.get("PWNA_FIG_OUT", HERE)
RES = os.environ.get("PWNA_FIELD", os.path.join(ROOT, "results", "local"))
P7 = os.environ.get("PWNA_P7", os.path.join(ROOT, "results", "p7"))
sys.path.insert(0, os.path.join(ROOT, "pwna"))
from p8_analysis import (ARCH, CTX, Res, choose_kappa, latency, latency_inputs, latency_reqs,  # noqa: E402
                         sec_secure)

P8 = os.environ.get("PWNA_P8", os.path.join(ROOT, "results", "p8"))
SUPP = os.environ.get("PWNA_P8_SUPP", os.path.join(ROOT, "results", "p8_supp"))

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "lines.linewidth": 1.0, "lines.markersize": 3,
                     "axes.grid": True, "grid.alpha": 0.3, "pdf.fonttype": 42})
W = 3.5
PD, PT = "ladder_eff@kap1@fz256", "ladderdet_eff@kap1"      # proposed design, digital / tone handshake
COL = {"ladderdet": "C7", "ladderdet_chk": "C1", "genie_hybrid": "C2", "airmoe_eff": "k", "awa": "C4",
       PD: "C3", PT: "C0", "ladder_eff@kap1": "C3", "ladderdet_trunc": "C1"}
LAB = {"ladderdet": "Analog ladder", "ladderdet_chk": "Non-saturating split", "genie_hybrid": "Known w., genie",
       "airmoe_eff": "Known w., AirMoE-eff", "awa": "AWA", PD: "Proposed, digital hs.", PT: "Proposed, tone hs.",
       "ladder_eff@kap1": "Proposed, digital hs.", "ladderdet_trunc": "Ladder, truncated"}
LS = {"genie_hybrid": "--", "airmoe_eff": "--", "ladderdet": ":", "ladderdet_chk": ":", "ladderdet_trunc": ":", "awa": "-."}


def load(path):
    with open(path) as f:
        return json.load(f)


def fig_theory():
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.55))
    n = 200_000
    beta = 10 ** (rng.normal(0, 6, 16) / 10)
    profiles = {"spread": np.ones(16), "concentrated": np.r_[1.0, 0.3 * np.ones(3), 0.02 * np.ones(12)]}
    for (name, d), c in zip(profiles.items(), ("C0", "C3")):
        g = rng.exponential(1.0, (n, d.size)) * beta
        B = (d ** 2 / g).max(1)
        S = float((d ** 2 / beta).sum())
        x = np.logspace(math.log10(S) - 1.5, math.log10(S) + 2.5, 60)
        ax[0].semilogx(x, [(B <= v).mean() for v in x], "o", color=c, mfc="none", ms=2.5, mew=0.6)
        ax[0].semilogx(x, np.exp(-S / x), "-", color=c, label=f"{name}, $S$={S:.2g}")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel(r"$\Pr\{B\leq x\}$")
    ax[0].legend(loc="upper left", frameon=False)
    ax[0].set_title("(a)", fontsize=7)
    snr_db = np.arange(0, 41, 2.5)
    eps = 1e-2
    m = 100_000
    for k, c in zip((1, 4, 16), ("C3", "C1", "C0")):
        d = np.r_[np.ones(k), np.zeros(16 - k)] + 1e-9
        D = d.sum()
        neff = D ** 2 / (d ** 2).sum()
        mc = []
        for s in snr_db:
            P = 10 ** (s / 10)
            g = rng.exponential(1.0, (m, 16))
            B = (d ** 2 / g).max(1)
            z = rng.standard_normal(m)
            mc.append(float(np.mean(z ** 2 * B / P / D ** 2 > eps)))
        th = 1 - (1 + 2 / (eps * 10 ** (snr_db / 10) * neff)) ** -0.5
        ax[1].semilogy(snr_db, mc, "o", color=c, mfc="none", ms=2.5, mew=0.6)
        ax[1].semilogy(snr_db, th, "-", color=c, label=rf"$N_{{\rm eff}}$={neff:.0f}")
    ax[1].set_xlabel("SNR (dB)")
    ax[1].set_ylabel(r"outage, $\epsilon=10^{-2}$")
    ax[1].legend(loc="lower left", frameon=False)
    ax[1].set_title("(b)", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_theory.pdf"))
    plt.close(fig)


def fig_llm_curves():
    R = Res(P8, P7, supp=SUPP if os.path.isdir(SUPP) else None)
    T = CTX["8k"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.65))
    for nm, pol in (("p7_f1_q4_8k", "ladderdet"), ("p7_f1_q4_8k", "ladderdet_chk"), ("p7_f1_q4_8k", "genie_hybrid"),
                    ("p8_f1_q4_8k", PT), ("p8_f1_q4_8k", PD), ("p8_f1_q4_8k", "airmoe_eff")):
        cs = sorted(R.cells(nm, T=T, N=32, M=4, policy=pol), key=lambda c: c["snr_db"])
        s = np.array([c["snr_db"] for c in cs])
        for a, key in zip(ax, ("nll", "kl")):
            mean = np.array([c[key] for c in cs])
            ci = np.array([1.96 * np.std(c[f"{key}_s"], ddof=1) / math.sqrt(len(c[f"{key}_s"])) for c in cs])
            a.plot(s, mean, "o", ls=LS.get(pol, "-"), color=COL[pol], label=LAB[pol], ms=2)
            a.fill_between(s, np.maximum(mean - ci, 1e-5), mean + ci, color=COL[pol], alpha=0.15, lw=0)
    ax[0].axhline(0.05, color="k", ls=":", lw=0.7)
    ax[0].set_yscale("symlog", linthresh=0.01)
    ax[0].set_ylabel("NLL increase (nats)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"KL$(p_{\rm ref}\|p_{\rm air})$")
    for a, t in zip(ax, ("(a)", "(b)")):
        a.set_xlabel("SNR (dB)")
        a.set_title(t, fontsize=7)
        a.set_xlim(-15.5, 25.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig_llm_curves.pdf"))
    plt.close(fig)


def fig_field():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.0))
    schemes = ("genie_hybrid", "airmoe_eff", "ladderdet_chk", "ladder_eff@kap1", PT, "awa")
    for a, (f, K, title) in zip(ax, (("field_nw_sw0_p8.json", 256, "(a) NW, $K$=256, equal path loss"),
                                     ("field_nw_sw10_p8.json", 256, "(b) NW, $K$=256, 10 dB spread"))):
        rows = load(os.path.join(RES, f))["rows"]
        ideal = next(r["rmse_med"] for r in rows if r["scheme"] == "ideal" and r["K"] == K)
        for sch in schemes:
            rs = sorted((r for r in rows if r["K"] == K and r["M"] == 1 and r["scheme"] == sch),
                        key=lambda r: r["snr_db"])
            if rs:
                a.plot([r["snr_db"] for r in rs], [r["rmse_med"] for r in rs], marker="o", ms=2,
                       color=COL[sch], label=LAB[sch], ls=LS.get(sch, "-"))
        a.axhline(ideal, color="k", ls=":", lw=0.7)
        a.set_ylim(ideal * 0.95, ideal * 2.2)
        a.set_xlabel("SNR (dB)")
        a.set_title(title, fontsize=6.5)
    ax[0].set_ylabel("median RMSE (dB)")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig_field.pdf"))
    plt.close(fig)


def fig_system():
    """Fig. 1: devices with KV shards, the FC with M antennas, and the per-layer protocol."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(W, 1.85))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=fc, ec="0.25", lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, color, ls="-", lw=0.8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=color,
                                     lw=lw, ls=ls, shrinkA=0, shrinkB=0))

    ys = (5.0, 3.55, 1.25)
    names = ("Device 1", "Device 2", "Device $N$")
    for y, nm, j in zip(ys, names, ("1", "2", "N")):
        box(0.15, y, 2.55, 1.05, f"{nm}\nKV shard $\\mathcal{{S}}_{j}$\n$(d_{j},\\bar{{\\mathbf{{o}}}}_{j})$",
            "#EAF1FB")
    ax.text(1.42, 2.75, "$\\vdots$", ha="center", va="center", fontsize=9)
    box(7.05, 1.25, 2.8, 4.8, "", "#FDF0E6")
    ax.text(8.45, 5.55, "Fusion center", ha="center", va="center", fontsize=6.3, weight="bold")
    ax.text(8.45, 4.75, "anchor block + tail\n$(d_A,\\mathbf{n}_A)$, shift $c$", ha="center", va="center",
            fontsize=5.8)
    ax.text(8.45, 3.55, "learn $\\hat\\ell$, set $\\eta$\n(ladder handshake)", ha="center", va="center",
            fontsize=5.8)
    ax.text(8.45, 2.15, "$\\hat{\\mathbf{o}}=\\hat{\\mathbf{n}}/\\max\\{d_A{+}A,$\n"
            "$\\|\\hat{\\mathbf{n}}\\|/V_{\\max}\\}$", ha="center", va="center", fontsize=5.6)
    for k in range(4):
        x = 7.2 + 0.18 * k
        ax.plot([x, x], [6.05, 6.3], color="0.25", lw=0.6)
        ax.plot([x - 0.06, x, x + 0.06], [6.3, 6.22, 6.3], color="0.25", lw=0.6)
    ax.text(8.0, 6.2, "$M$ antennas", fontsize=5.4, va="center")
    cols = {"q": "0.45", "t": "#C44E52", "a": "#4C72B0", "s": "#55A868"}
    arrow(7.0, 5.75, 2.75, 5.75, cols["q"], ls="--")
    ax.text(4.9, 5.92, "(0) broadcast query $\\mathbf{q}$", ha="center", fontsize=5.6, color=cols["q"])
    arrow(2.75, 5.15, 7.0, 4.25, cols["t"])
    ax.text(4.95, 4.95, "(1) level $\\ell_j$ of the report", ha="center", fontsize=5.6, color=cols["t"])
    for y in ys:
        arrow(2.75, y + 0.4, 7.0, 2.75, cols["a"], lw=0.7)
    ax.text(4.0, 2.72, "(2) analog numerator, weight $a_j$", ha="center",
            va="center", fontsize=5.6, color=cols["a"])
    arrow(2.75, 1.45, 7.0, 1.75, cols["s"], ls=":")
    ax.text(4.9, 0.82, "(2') masked arrived weight $a_j$ by secure aggregation", ha="center", fontsize=5.4,
            color=cols["s"])
    ax.text(4.9, 0.15, "per layer, head and decoded token; the FC learns only $\\hat\\ell$ and $A=\\sum_j a_j$",
            ha="center", fontsize=5.4, style="italic", color="0.3")
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, "fig_system.pdf"))
    plt.close(fig)


def fig_regime(M=4, snr=15.0):
    """Fastest scheme per token in the (N, bandwidth) plane: the latency model of p8_analysis.latency with the
    required SNRs of the main model's 8k N sweep (interpolated in log2 N), the ladder size and the measured
    ring sizes of the secure aggregation and of the secure digital merge at matched accuracy."""
    R = Res(P8, P7, supp=SUPP if os.path.isdir(SUPP) else None)
    kap, _ = choose_kappa(R)
    _, minbits = sec_secure(R)
    L, b_den, b_glob, b_inst, _ = latency_inputs(R, kap, minbits)
    grid = (4, 8, 16, 32, 64, 128)
    rq = {N: latency_reqs(R, kap, N) for N in grid}
    lg = [math.log2(N) for N in grid]
    Ns = 2 ** np.linspace(2, 7, 121)
    bws = 10 ** np.linspace(math.log10(5e6), math.log10(4e8), 121)
    order = ("prop_dig", "prop_tone", "sec_inst", "sec_glob", "known", "zf_tdma", "digital")
    priv = order[:4]
    best_priv = np.zeros((len(bws), len(Ns)), dtype=int)
    best_all = np.zeros_like(best_priv)
    for i, bw in enumerate(bws):
        for k, N in enumerate(Ns):
            reqs = {p: float(np.interp(math.log2(N), lg, [rq[n][p] for n in grid])) for p in rq[4]}
            lat = latency(N, bw, reqs, L, b_den, b_glob, b_inst, M=M, snr=snr, **ARCH["q4"])
            best_priv[i, k] = order.index(min(priv, key=lat.get))
            best_all[i, k] = order.index(min(order, key=lat.get))
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.8), sharey=True)
    cmap = matplotlib.colors.ListedColormap(["#C44E52", "#4C72B0", "#DD8452", "#CCB974", "#55A868", "#8172B2", "#937860"])
    for ax, best, title in ((axs[0], best_priv, "(a) private schemes"), (axs[1], best_all, "(b) all schemes")):
        ax.pcolormesh(Ns, bws / 1e6, best, cmap=cmap, vmin=-0.5, vmax=6.5, shading="auto", rasterized=True)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("devices $N$")
        ax.set_title(title, fontsize=7)
        ax.grid(False)
    axs[0].set_ylabel("bandwidth (MHz)")
    labels = ("proposed, digital hs.", "proposed, tone hs.", "secure dig., inst. scale", "secure dig., global scale",
              "known weights", "ZF-TDMA", "digital")
    present = sorted(set(np.unique(best_priv)) | set(np.unique(best_all)))
    handles = [matplotlib.patches.Patch(color=cmap(i), label=labels[i]) for i in present]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.8))
    fig.savefig(os.path.join(OUT, "fig_regime.pdf"))
    plt.close(fig)
    return ({labels[i]: round(float((best_priv == i).mean()), 3) for i in present},
            {labels[i]: round(float((best_all == i).mean()), 3) for i in present})


if __name__ == "__main__":
    fig_system()
    fig_theory()
    fig_llm_curves()
    fig_field()
    print("fig_regime area shares (private, all):", fig_regime())
    print("figures written to", OUT)

