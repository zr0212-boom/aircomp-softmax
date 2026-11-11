"""
G2: Monte-Carlo verification of the closed forms behind the private-weight normalized AirComp
theory. Every check has an explicit numerical criterion; the script exits with status 1 if any
check fails, so "PASS" means the closed form matched, not merely that the script ran.

Model. N devices with private weights d_j > 0 and average channel gains beta_j. The FC aligns all
devices to a common scale eta (AirMoE power convention: device j needs precoder power
eta^2 d_j^2 / g_j <= P). The alignment bottleneck is Y = min_j g_j / d_j^2.

Checks
  A  weighted bottleneck law (Rayleigh, heterogeneous beta_j):
        P{max_j d_j^2 / g_j <= x} = exp(-S / x),   S = sum_j d_j^2 / beta_j
  C  single-scalar decentralized power control, any-device saturation probability:
        fixed eta:          1 - exp(-eta^2 S / P)
        random S:           1 - E[exp(-eta^2 S / P)]          (Laplace transform of S)
        stale estimate S^:  E[1 - (1 - delta)^(S / S^)]  with  eta^2 = P ln(1/(1-delta)) / S^
  E  Nakagami-m fading or device-side MRT with m antennas (g_j ~ Gamma(m, beta_j / m)):
        P{Y > y} = prod_j Q(m, m y d_j^2 / beta_j),   small-y slope m (diversity order m)
  B  denominator relative-error outage under genie alignment (eta^2 = P Y), threshold eps on the
     squared relative error, a = S sigma^2 / (eps P D^2)  (= 1 / (eps SNR N_eff) when beta_j = 1):
        real noise    (one real dimension):  1 - (1 + 2a)^(-1/2)
        complex noise (CN on the I/Q pair):  a / (1 + a)
  T  tail of the conditional noise variance V = sigma^2 / (P Y): index m, i.e. infinite mean under
     single-antenna Rayleigh (m = 1) and finite mean for m >= 2. Checked with the Hill estimator for
     m = 1, 2; for m = 4 the Hill estimator is pre-asymptotic at feasible sample sizes, so the finite
     mean is checked directly (MC mean of V vs. numerical integral of the exact tail).
  K  combiner bounds (Proposition, combiner bounds): for a channel-dependent unit-norm combiner,
     P{B <= x} <= prod_j Q(M, d_j^2 / (beta_j x)) (device-side MRT law); for the max-min combiner
     (SLSQP from several starts), P{B <= x} >= exp(-d_max^2 sum_j 1/beta_j / x)
  H  handshake energy detection (non-coherent tones on nu REs x M antennas, channels CN(0, beta_j)
     i.i.d. per RE and antenna, complex noise of variance 1 per real dimension), simulated from the
     received samples: false-alarm rate Q(M nu, theta / 2) of an empty level, miss probability
     P(M nu, theta / (P S + 2)) of an occupied one (S = summed beta of its devices), and the bound
     (theta / (P S))^(M nu) / (M nu)!.

Run:  python pwna/theory_mc.py            (CPU, ~1-2 min; writes results/local/theory_mc.json)
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammainccinv

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "local", "theory_mc.json")
SEED = 20261106
CHUNK = 50_000


def weight_profiles(rng):
    """Representative private-weight vectors: uniform, geometric (peaked), softmax-like heavy tail,
    and a real-attention-like profile with one dominant shard (N_eff ~ 2)."""
    prof = {}
    prof["uniform_N16"] = np.ones(16)
    prof["geometric_N16"] = 0.5 ** np.arange(16)
    for n in (64, 256):
        z = rng.normal(0.0, 3.0, n)
        prof[f"softmax_N{n}"] = np.exp(z - z.max())
    d = np.array([1.0, 0.9, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])
    prof["attention_like_N8"] = d
    return prof


def path_loss(rng, n, hetero):
    if not hetero:
        return np.ones(n)
    r = rng.uniform(0.5, 2.0, n)
    b = r ** -3.0
    return b / b.mean()


def sample_Y(rng, d, beta, n, m=1):
    """Y = min_j g_j / d_j^2 with g_j ~ Gamma(m, beta_j / m) (m = 1: Rayleigh power)."""
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        k = min(CHUNK, n - s)
        if m == 1:
            g = rng.exponential(1.0, (k, d.size)) * beta
        else:
            g = rng.gamma(m, 1.0 / m, (k, d.size)) * beta
        out[s:s + k] = (g / d ** 2).min(axis=1)
    return out


def neff(d):
    return d.sum() ** 2 / (d ** 2).sum()


def record(results, name, ok, **kw):
    results.append(dict(check=name, ok=bool(ok), **kw))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  " +
          "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()),
          flush=True)


def check_A(rng, results, n=400_000, tol=0.004):
    for hetero in (False, True):
        for name, d in weight_profiles(rng).items():
            beta = path_loss(rng, d.size, hetero)
            S = float((d ** 2 / beta).sum())
            Y = sample_Y(rng, d, beta, n)
            B = 1.0 / Y
            med_pred = S / math.log(2.0)
            err = 0.0
            for mult in (0.3, 1.0, 3.0):
                x = mult * med_pred
                err = max(err, abs(float(np.mean(B <= x)) - math.exp(-S / x)))
            med_ratio = float(np.median(B)) / med_pred
            record(results, f"A/bottleneck_law/{name}/{'hetero' if hetero else 'equal'}_beta",
                   err <= tol and abs(med_ratio - 1) <= 0.01,
                   max_cdf_err=err, median_ratio=med_ratio, N_eff=neff(d))


def check_C(rng, results, n=400_000, tol=0.004):
    d = weight_profiles(rng)["softmax_N64"]
    beta = np.ones(d.size)
    S = float((d ** 2).sum())
    P = 10.0
    Y = sample_Y(rng, d, beta, n)
    for q in (0.5, 0.8, 0.95):
        eta2 = P * math.log(1.0 / q) / S
        sat_mc = float(np.mean(eta2 / P > Y))          # any device needs more than P
        record(results, f"C/fixed_eta/q={q}", abs(sat_mc - (1 - q)) <= tol,
               sat_mc=sat_mc, sat_theory=1 - q)
    # random S: each instance draws a fresh weight vector (softmax with random temperature)
    inst = 200_000
    N = 16
    temp = rng.uniform(0.5, 4.0, inst)
    z = rng.normal(0.0, 1.0, (inst, N)) * temp[:, None]
    dd = np.exp(z - z.max(axis=1, keepdims=True))
    Ss = (dd ** 2).sum(axis=1)
    g = rng.exponential(1.0, (inst, N))
    Yi = (g / dd ** 2).min(axis=1)
    for tau in (0.1, 0.5, 2.0):                         # eta^2 / P
        sat_mc = float(np.mean(tau > Yi))
        sat_th = float(1.0 - np.mean(np.exp(-tau * Ss)))
        record(results, f"C/random_S_laplace/tau={tau}", abs(sat_mc - sat_th) <= tol,
               sat_mc=sat_mc, sat_theory=sat_th)
    # stale estimate: S^ = S * lognormal error (one-step-delayed statistic)
    Shat = Ss * np.exp(rng.normal(0.0, 0.5, inst))
    for delta in (0.1, 0.3):
        tau_i = math.log(1.0 / (1.0 - delta)) / Shat
        sat_mc = float(np.mean(tau_i > Yi))
        sat_th = float(np.mean(1.0 - (1.0 - delta) ** (Ss / Shat)))
        record(results, f"C/stale_estimate/delta={delta}", abs(sat_mc - sat_th) <= tol,
               sat_mc=sat_mc, sat_theory=sat_th)


def check_E(rng, results, n=400_000, tol=0.004):
    d = weight_profiles(rng)["attention_like_N8"]
    for m in (2, 4):
        for hetero in (False, True):
            beta = path_loss(rng, d.size, hetero)
            Y = sample_Y(rng, d, beta, n, m=m)
            def ccdf(y):
                return float(np.prod(gammaincc(m, m * y * d ** 2 / beta)))
            def cdf_small(y):                  # P{Y <= y}, accurate for tiny probabilities
                return float(-np.expm1(np.log1p(-gammainc(m, m * y * d ** 2 / beta)).sum()))
            ymed = float(np.median(Y))
            err = max(abs(float(np.mean(Y > y)) - ccdf(y)) for y in (0.3 * ymed, ymed, 3 * ymed))
            # diversity: exact law vs the small-y asymptote C y^m, C = sum_j (m d_j^2/beta_j)^m / m!,
            # evaluated deep in the asymptotic regime (probability 1e-10, where m y d_j^2/beta_j << 1)
            C = float(((m * d ** 2 / beta) ** m).sum() / math.factorial(m))
            y0 = (1e-10 / C) ** (1.0 / m)
            asym_ratio = cdf_small(y0) / (C * y0 ** m)
            slope = math.log(cdf_small(y0) / cdf_small(y0 / 10)) / math.log(10.0)
            record(results, f"E/nakagami_m={m}/{'hetero' if hetero else 'equal'}_beta",
                   err <= tol and abs(slope - m) <= 0.02 and abs(asym_ratio - 1) <= 0.01,
                   max_ccdf_err=err, small_y_slope=slope, asymptote_ratio=asym_ratio)


def check_B(rng, results, n=400_000, tol=0.004):
    sigma2 = 1.0
    for name in ("attention_like_N8", "softmax_N64"):
        d = weight_profiles(rng)[name]
        D = float(d.sum())
        S = float((d ** 2).sum())
        Y = sample_Y(rng, d, np.ones(d.size), n)
        for snr_db in (10.0, 20.0):
            P = 10 ** (snr_db / 10.0)
            for eps in (1e-3, 1e-2):
                a = S * sigma2 / (eps * P * D ** 2)
                w_real = rng.normal(0.0, math.sqrt(sigma2), n)
                out_real = float(np.mean(w_real ** 2 / (P * Y * D ** 2) > eps))
                th_real = 1.0 - (1.0 + 2.0 * a) ** -0.5
                w2_cplx = rng.exponential(sigma2, n)                  # |CN(0, sigma2)|^2
                out_cplx = float(np.mean(w2_cplx / (P * Y * D ** 2) > eps))
                th_cplx = a / (1.0 + a)
                ok = abs(out_real - th_real) <= tol and abs(out_cplx - th_cplx) <= tol
                record(results, f"B/outage/{name}/snr={snr_db:g}dB/eps={eps:g}", ok,
                       real_mc=out_real, real_theory=th_real, cplx_mc=out_cplx,
                       cplx_theory=th_cplx, N_eff=neff(d))


def hill(x, k):
    xs = np.sort(x)[::-1]
    return 1.0 / float(np.mean(np.log(xs[:k] / xs[k])))


def check_T(rng, results, n=1_000_000, k=2_000, tol=0.15):
    d = weight_profiles(rng)["attention_like_N8"]
    for m in (1, 2):
        Y = sample_Y(rng, d, np.ones(d.size), n, m=m)
        alpha = hill(1.0 / Y, k)
        record(results, f"T/tail_index/m={m}", abs(alpha - m) <= tol,
               hill_index=alpha, expected=float(m), mean_finite=bool(m >= 2))
    m = 4
    Y = sample_Y(rng, d, np.ones(d.size), n, m=m)
    mean_mc = float(np.mean(1.0 / Y))
    # E[V] = int_0^inf P{1/Y > v} dv = int_0^inf P{Y < 1/v} dv, exact tail from check E
    def tail(v):
        with np.errstate(divide="ignore"):
            return float(-np.expm1(np.log1p(-gammainc(m, m * d ** 2 / v)).sum()))
    mean_th = quad(tail, 0.0, 10.0, limit=200)[0] + quad(tail, 10.0, np.inf, limit=200)[0]
    record(results, f"T/finite_mean/m={m}", abs(mean_mc / mean_th - 1) <= 0.02,
           mean_mc=mean_mc, mean_theory=mean_th)


def check_H(rng, results, n=400_000, p_lvl=1e-2):
    for M, nu, P_db, betas in ((4, 2, 10.0, (1.0,)), (8, 2, 5.0, (1.0,)), (4, 4, 5.0, (1.0, 0.5)),
                               (4, 1, 15.0, (1.0,))):
        k, P, S = M * nu, 10 ** (P_db / 10), float(sum(betas))
        theta = 2.0 * float(gammainccinv(k, p_lvl))
        fa = miss = 0
        for s in range(0, n, CHUNK):
            c = min(CHUNK, n - s)
            w = rng.standard_normal((c, k, 2))
            fa += int(((w ** 2).sum((1, 2)) > theta).sum())
            y = w.copy()
            for b in betas:
                y += math.sqrt(P * b / 2.0) * rng.standard_normal((c, k, 2))
            miss += int(((y ** 2).sum((1, 2)) <= theta).sum())
        fa_mc, miss_mc = fa / n, miss / n
        miss_th = float(gammainc(k, theta / (P * S + 2.0)))
        bound = (theta / (P * S)) ** k / math.gamma(k + 1)
        ok = (abs(fa_mc / p_lvl - 1) <= 0.1 and abs(miss_mc - miss_th) <= max(0.004, 0.1 * miss_th)
              and miss_th <= bound)
        record(results, f"H/detection/M={M},nu={nu},P={P_db:g}dB,S={S:g}", ok, fa_mc=fa_mc,
               fa_theory=p_lvl, miss_mc=miss_mc, miss_theory=miss_th, miss_bound=min(bound, 1.0))


def _maxmin_combiner(h, rng, starts=2):
    from scipy.optimize import minimize
    N, M = h.shape

    def gains(x):
        m = x[:M] + 1j * x[M:2 * M]
        return np.abs(h @ m.conj()) ** 2

    best = None
    eig = np.linalg.eigh(h.conj().T @ h)[1][:, -1]
    for k in range(starts + 1):
        m = eig if k == 0 else rng.standard_normal(M) + 1j * rng.standard_normal(M)
        m = m / np.linalg.norm(m)
        x0 = np.concatenate([m.real, m.imag, [gains(np.concatenate([m.real, m.imag])).min()]])
        cons = [{"type": "ineq", "fun": lambda x: gains(x[:2 * M]) - x[-1]},
                {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x[:2 * M] ** 2)}]
        r = minimize(lambda x: -x[-1], x0, constraints=cons, method="SLSQP", options={"maxiter": 200})
        x = r.x[:2 * M] / max(np.linalg.norm(r.x[:2 * M]), 1e-12)
        g = gains(x)
        if best is None or g.min() > best.min():
            best = g
    return best


def check_K(rng, results, n=4000, n_mm=800, tol=0.01):
    d = weight_profiles(rng)["attention_like_N8"][:4]
    for M in (2, 4):
        beta = path_loss(rng, d.size, True)
        h = (rng.standard_normal((n, d.size, M)) + 1j * rng.standard_normal((n, d.size, M))) / np.sqrt(2)
        h = h * np.sqrt(beta)[None, :, None]
        R = np.einsum("rja,rjb->rab", h.conj(), h)                     # sum_j h_j h_j^H (conj order)
        m = np.linalg.eigh(R)[1][..., -1]                              # channel-dependent combiner
        B = (d ** 2 / np.abs(np.einsum("rja,ra->rj", h, m.conj())) ** 2).max(-1)
        S_med = np.median(B)
        viol = 0.0
        for mult in (0.3, 1.0, 3.0):
            x = mult * S_med
            emp = float(np.mean(B <= x))
            bound = float(np.prod(gammaincc(M, d ** 2 / (beta * x))))
            viol = max(viol, emp - bound)
        record(results, f"K/combiner_upper_bound/M={M}", viol <= tol, max_violation=viol)
        Bmm = np.array([(d ** 2 / _maxmin_combiner(h[i], rng)).max() for i in range(n_mm)])
        viol = 0.0
        for mult in (0.3, 1.0, 3.0):
            x = mult * np.median(Bmm)
            emp = float(np.mean(Bmm <= x))
            bound = math.exp(-d.max() ** 2 * np.sum(1.0 / beta) / x)
            viol = max(viol, bound - emp)
        record(results, f"K/maxmin_lower_bound/M={M}", viol <= 3 * tol, max_violation=viol)


def check_F(rng, results, n=400_000, tol=0.004):
    """Coherence blocks: device j sees C independent Rayleigh gains g_jc (mean beta_j) inside one
    instance and backs off uniformly to its weakest block, so B_C = max_j max_c d_j^2 / g_jc and
    P{B_C <= x} = exp(-C S / x): the bottleneck law of Theorem 1 with S replaced by C S."""
    for C in (2, 4):
        for hetero in (False, True):
            for name in ("uniform_N16", "attention_like_N8"):
                d = weight_profiles(rng)[name]
                beta = path_loss(rng, d.size, hetero)
                S = float((d ** 2 / beta).sum())
                B = np.empty(n)
                for s in range(0, n, CHUNK):
                    k = min(CHUNK, n - s)
                    g = (rng.exponential(1.0, (k, C, d.size)) * beta).min(axis=1)
                    B[s:s + k] = (d ** 2 / g).max(axis=1)
                err = max(abs(float(np.mean(B <= x)) - math.exp(-C * S / x))
                          for x in (0.3 * C * S / math.log(2), C * S / math.log(2), 3 * C * S / math.log(2)))
                record(results, f"F/coherence_blocks/C{C}/{name}/{'hetero' if hetero else 'equal'}_beta",
                       err <= tol, max_cdf_err=err)


def check_V(rng, results, n=400_000, tol=0.004):
    """Noise-matched level report: device j announces d_j / (c_j + sqrt(kappa)), c_j = sqrt(P g_j).
    (i) With u = max_j d_j / (c_j + sqrt(kappa)) and a_j = min(d_j, c_j u), every device's saturation
    distortion obeys (d_j - a_j)^2 <= kappa u^2 (deterministic). (ii) Under Rayleigh fading,
    B~ = P u^2 = max_j d_j^2 / (sqrt(g_j) + sqrt(kappa / P))^2 has the law
    P{B~ <= x} = exp(-sum_j (d_j / sqrt(x) - sqrt(kappa / P))_+^2 / beta_j), which reduces to Theorem 1
    for kappa = 0. (iii) The saturation-aware ratio (exact sum of the effective weights) is a convex
    combination of the anchor and device outputs for any scale (checked on random instances)."""
    for kappa, P in ((0.3, 10.0), (1.0, 3.16), (3.0, 31.6)):
        for hetero in (False, True):
            for name in ("softmax_N64", "attention_like_N8"):
                d = weight_profiles(rng)[name]
                beta = path_loss(rng, d.size, hetero)
                Bt = np.empty(n)
                worst = 0.0
                for s in range(0, n, CHUNK):
                    k = min(CHUNK, n - s)
                    g = rng.exponential(1.0, (k, d.size)) * beta
                    c = np.sqrt(P * g)
                    u = (d / (c + math.sqrt(kappa))).max(axis=1)
                    a = np.minimum(d, c * u[:, None])
                    worst = max(worst, float(((d - a) ** 2 / (kappa * u[:, None] ** 2)).max()))
                    Bt[s:s + k] = P * u ** 2
                med = float(np.median(Bt))
                err = 0.0
                for x in (0.3 * med, med, 3.0 * med):
                    pred = math.exp(-float((np.maximum(d / math.sqrt(x) - math.sqrt(kappa / P), 0.0) ** 2 / beta).sum()))
                    err = max(err, abs(float(np.mean(Bt <= x)) - pred))
                record(results, f"V/kappa_law/k{kappa:g}_P{P:g}/{name}/{'hetero' if hetero else 'equal'}_beta",
                       err <= tol and worst <= 1.0 + 1e-9, max_cdf_err=err, max_distortion_over_kappa_u2=worst)
    # (iii) convexity of the saturation-aware ratio under arbitrary saturation
    k, N, dv = 20000, 16, 4
    d = np.exp(3 * rng.normal(size=(k, N)))
    ob = rng.normal(size=(k, N, dv))
    ob /= np.maximum(1.0, np.linalg.norm(ob, axis=-1, keepdims=True))
    dA, oA = np.exp(rng.normal(size=k)), np.zeros((k, dv))
    a = np.minimum(d, d * rng.uniform(0, 1.5, (k, N)))
    o = (dA[:, None] * oA + (a[..., None] * ob).sum(1)) / (dA + a.sum(1))[:, None]
    record(results, "V/eff_ratio_convex", float(np.linalg.norm(o, axis=-1).max()) <= 1.0 + 1e-9,
           max_norm=float(np.linalg.norm(o, axis=-1).max()))


def _opt_partition(x, p, K):
    """Optimal one-sided quantizer: minimum over partitions of the grid x (pmf p) into K intervals of
    sum_cells sum_{i in cell} p_i (x_top(cell) - x_i), by dynamic programming."""
    G = x.size
    P, PX = np.concatenate([[0.0], np.cumsum(p)]), np.concatenate([[0.0], np.cumsum(p * x)])
    cost = lambda a, b: x[b] * (P[b + 1] - P[a]) - (PX[b + 1] - PX[a])        # cell [a, b], inclusive
    best = np.array([cost(0, b) for b in range(G)])
    for _ in range(K - 1):
        new = best.copy()
        for b in range(1, G):
            a = np.arange(1, b + 1)
            new[b] = min(best[b], float(np.min(best[a - 1] + x[b] * (P[b + 1] - P[a]) - (PX[b + 1] - PX[a]))))
        best = new
    return float(best[-1])


def check_Q(rng, results, n_part=20000):
    """Converse for quantized non-saturating scales. A scale is non-saturating iff 1/eta^2 >= B / P; if
    the FC's scale is a function of a K-valued message about X = 10 log10 B, the dB penalty of a cell is
    its top minus X, so the cells may be taken as intervals. (i) X uniform on [0, W]: the minimum mean
    penalty is W / (2K), attained by the uniform ladder (step W / K); random partitions never beat it.
    (ii) X with density f (high resolution): the minimum is (int sqrt f)^2 / (2K) (1 + o(1)) <= W / (2K),
    attained by a companded ladder with local step proportional to f^(-1/2); the uniform ladder pays
    W / (2K) for any f on [0, W]."""
    W = 60.0
    for K in (4, 16, 64):
        lb = W / (2 * K)
        X = rng.uniform(0, W, 200_000)
        lad = float(np.mean(np.ceil(X / (W / K)) * (W / K) - X))            # uniform ladder, MC
        cuts = np.sort(rng.uniform(0, W, (n_part, K - 1)), axis=1)
        edges = np.concatenate([np.zeros((n_part, 1)), cuts, np.full((n_part, 1), W)], axis=1)
        pen = (np.diff(edges, axis=1) ** 2).sum(1) / (2 * W)                   # exact mean penalty
        record(results, f"Q/uniform_converse/K{K}", abs(lad - lb) / lb < 0.01 and float(pen.min()) >= lb * (1 - 1e-12),
               ladder_mc=lad, bound=lb, best_random_partition=float(pen.min()))
    # (ii) truncated Gaussian X (the bulk of log B within a head), grid DP vs the high-resolution formula
    G = 1500
    x = np.linspace(0, W, G)
    for sd in (6.0, 12.0):
        f = np.exp(-0.5 * ((x - W / 2) / sd) ** 2)
        p = f / f.sum()
        dens = p / (x[1] - x[0])
        hr_int = float(np.sum(np.sqrt(dens)) * (x[1] - x[0])) ** 2
        for K in (16, 32):
            opt = _opt_partition(x, p, K)
            hr = hr_int / (2 * K)
            uni = float(np.sum(p * (np.ceil(x / (W / K) - 1e-12) * (W / K) - x)))
            record(results, f"Q/companded_highres/sd{sd:g}/K{K}", abs(opt - hr) / hr < 0.05 and opt <= uni + 1e-9,
                   dp_optimum=opt, high_res=hr, uniform_ladder=uni)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    results = []
    check_A(rng, results)
    check_C(rng, results)
    check_E(rng, results)
    check_B(rng, results)
    check_T(rng, results)
    check_K(rng, results)
    check_H(rng, results)
    check_F(rng, results)
    check_V(rng, results)
    check_Q(rng, results)
    n_fail = sum(not r["ok"] for r in results)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "seconds": round(time.time() - t0, 1), "n_checks": len(results),
                   "n_fail": n_fail, "results": results}, f, indent=1)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed "
          f"({time.time() - t0:.1f} s) -> {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
