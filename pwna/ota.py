"""
Over-the-air normalized aggregation with private weights (AirMoE power convention), in torch.

Target per aggregation instance:  o = (n_A + sum_j n_j) / (d_A + sum_j d_j),  n_j = d_j * obar_j,
where (d_A, n_A) is exact at the FC (anchor) and device j holds the private weight d_j and its local
normalized output obar_j. Device j transmits the denominator stream (data 1) and d_v numerator
streams (data obar_jk / sig_o) with ONE precoder magnitude b_j = eta * d_j / sqrt(g_j), subject to
b_j^2 <= P; devices that would exceed P saturate at P. g_j is the effective power gain after the
FC's unit-norm receive combiner. Real AWGN of variance 1 per real stream; SNR = P.

Receive-scale (eta) designs, all one scalar per aggregation:
  genie      eta^2 = P min_j g_j / d_j^2               (knows the private weights; never saturates)
  fixed      eta^2 = P / X                             X: offline quantile of max_j d_j^2 / g_j
  csi        eta^2 = P / t*(g)                         t*: smallest t with prod_j F(g_j t) >= q, F the
                                                       offline CDF of d^2 (uses current CSI only)
  classical  eta^2 = P min_j g_j / A^2                 weight-agnostic: A = offline quantile of max_j d_j
  ladder     eta^2 = P / B^,  B^ = max_j d_j^2 / g_j rounded UP to a log ladder of step Delta dB:
             one anonymous over-the-air max handshake per aggregation (devices compute their own
             d_j^2 / g_j; the FC learns one scalar, not which device or any weight); never saturates,
             noise penalty <= Delta dB relative to genie
Modes: analog (D over the air) or hybrid (D sent digitally, exact). The analog estimate projects the
denominator onto its feasible set: the output is a convex combination of values, so ||o|| <= V_max
and D >= max(d_A, ||N||/V_max); o_hat = N_hat / max(D_hat, d_A, ||N_hat|| / V_max). Without V_max
the floor is d_A only, which fails when the devices hold most of the mass.
Baseline zf_sep (needs M >= N): ZF spatial separation, every device sends obar_j at full power,
d_j sent digitally. Baseline zf_tdma (any M): the same in ceil(N / G) orthogonal slots of G devices.
"""
import functools
import math

import torch


def complex_normal(shape, gen, device):
    re = torch.randn(shape, generator=gen, device=device)
    im = torch.randn(shape, generator=gen, device=device)
    return torch.complex(re, im) / math.sqrt(2.0)


def draw_channels(prefix, N, M, gen, device):
    """i.i.d. Rayleigh h_j ~ CN(0, I_M); returns (*prefix, N, M) complex64."""
    return complex_normal((*prefix, N, M), gen, device)


def lsq_gains(h, delta=1e-3):
    """Least-squares equal-gain combiner m ∝ (H H^H + delta I)^{-1} H 1 (H = [h_1..h_N]); equalizes
    all devices when M >= N, poor when M <= N."""
    H = h.transpose(-1, -2)
    eye = torch.eye(H.shape[-2], dtype=H.dtype, device=H.device)
    m = torch.linalg.solve(H @ H.conj().transpose(-1, -2) + delta * eye, H.sum(-1, keepdim=True))[..., 0]
    m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    return torch.einsum("...na,...a->...n", h, m.conj()).abs() ** 2


def combiner_gains(h, mode="best", iters=6):
    """Effective power gains |m^H h_j|^2 (..., N) for a unit-norm receive combiner m chosen from the
    channels only (the FC does not know the private weights).
      maxmin     reweighted principal-eigenvector heuristic for max_m min_j |m^H h_j|^2
      lsq        least-squares equal-gain combiner
      best       per instance, whichever of maxmin / lsq gives the larger min_j gain
      eig        principal eigenvector of sum_j h_j h_j^H / ||h_j||^2
      mrc_bound  ||h_j||^2 for every device at once (NOT achievable by one combiner; optimistic bound)"""
    M = h.shape[-1]
    if M == 1:
        return h[..., 0].abs() ** 2
    energy = (h.abs() ** 2).sum(-1)
    if mode == "mrc_bound":
        return energy
    if mode == "lsq":
        return lsq_gains(h)
    if mode == "best":
        a, b = combiner_gains(h, "maxmin", iters), lsq_gains(h)
        return torch.where((b.min(-1).values > a.min(-1).values)[..., None], b, a)
    hn = h / energy.clamp_min(1e-20).sqrt()[..., None]
    w = torch.ones(h.shape[:-1], device=h.device)
    best_g = best_min = None
    for _ in range(iters if mode == "maxmin" else 1):
        R = torch.einsum("...n,...na,...nb->...ab", w.to(hn.dtype), hn, hn.conj())
        m = torch.linalg.eigh(R)[1][..., -1]
        g = torch.einsum("...na,...a->...n", h, m.conj()).abs() ** 2
        gmin = g.min(-1).values
        if best_g is None:
            best_g, best_min = g, gmin
        else:
            upd = gmin > best_min
            best_g = torch.where(upd[..., None], g, best_g)
            best_min = torch.where(upd, gmin, best_min)
        w = (g / energy.clamp_min(1e-20)).clamp_min(1e-6) ** -2.0
        w = w / w.sum(-1, keepdim=True)
    return best_g


def combiner_vector(h, mode="best", iters=6):
    """Unit-norm receive combiner m (..., M) chosen from the channels h (..., N, M): the vector behind
    combiner_gains (maxmin / lsq / best / eig); M = 1 gives m = 1."""
    M = h.shape[-1]
    if M == 1:
        return torch.ones((*h.shape[:-2], 1), dtype=h.dtype, device=h.device)
    H = h.transpose(-1, -2)

    def lsq_m():
        eye = torch.eye(M, dtype=H.dtype, device=H.device)
        m = torch.linalg.solve(H @ H.conj().transpose(-1, -2) + 1e-3 * eye, H.sum(-1, keepdim=True))[..., 0]
        return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-20)

    def gmin(m):
        return (torch.einsum("...na,...a->...n", h, m.conj()).abs() ** 2).min(-1).values

    if mode == "lsq":
        return lsq_m()
    energy = (h.abs() ** 2).sum(-1)
    hn = h / energy.clamp_min(1e-20).sqrt()[..., None]
    w = torch.ones(h.shape[:-1], device=h.device)
    best_m = best_min = None
    for _ in range(iters if mode in ("maxmin", "best") else 1):
        R = torch.einsum("...n,...na,...nb->...ab", w.to(hn.dtype), hn, hn.conj())
        m = torch.linalg.eigh(R)[1][..., -1]
        g = torch.einsum("...na,...a->...n", h, m.conj()).abs() ** 2
        cur = g.min(-1).values
        if best_m is None:
            best_m, best_min = m, cur
        else:
            upd = cur > best_min
            best_m = torch.where(upd[..., None], m, best_m)
            best_min = torch.where(upd, cur, best_min)
        w = (g / energy.clamp_min(1e-20)).clamp_min(1e-6) ** -2.0
        w = w / w.sum(-1, keepdim=True)
    if mode == "best":
        ml = lsq_m()
        return torch.where((gmin(ml) > best_min)[..., None], ml, best_m)
    return best_m


def imperfect_csi(h, nmse, mode, gen, complex_out=False):
    """Device-side inversion with estimated channels h_hat = h + e, e ~ CN(0, nmse I): the FC picks its
    combiner from h_hat, device j inverts m^H h_hat_j. Returns (g_hat, align): the gains the designs
    see, and the real alignment factor Re(m^H h_j / m^H h_hat_j) that multiplies every stream of j
    (complex_out: the complex factor m^H h_j / m^H h_hat_j itself, for the I/Q model)."""
    h_hat = h + math.sqrt(nmse) * complex_normal(h.shape, gen, h.device)
    m = combiner_vector(h_hat, mode)
    c_true = torch.einsum("...na,...a->...n", h, m.conj())
    c_hat = torch.einsum("...na,...a->...n", h_hat, m.conj())
    r = c_true / c_hat
    return c_hat.abs() ** 2, (r if complex_out else r.real)


def zf_noise_enhancement(h):
    """zeta_j = [(H^H H)^{-1}]_jj for H = [h_1 .. h_N] (M x N, M >= N); returns (..., N)."""
    gram = torch.einsum("...na,...ma->...nm", h.conj(), h)
    return torch.linalg.inv(gram).diagonal(dim1=-2, dim2=-1).real.clamp_min(0.0)


def group_zf_noise_enhancement(h, G):
    """Orthogonal slots of at most G devices (contiguous groups), ZF inside each slot (G <= M);
    returns zeta (..., N). G = 1 is plain TDMA with MRC (zeta_j = 1 / ||h_j||^2)."""
    N = h.shape[-2]
    return torch.cat([zf_noise_enhancement(h[..., s:s + G, :]) for s in range(0, N, G)], -1)


def csi_tstar(g, sorted_d2, q, grid):
    """t* = smallest grid value t with prod_j F(g_j t) >= q, F = empirical CDF of d^2.
    g: (..., N) gains with leading dims broadcastable over heads as (B, 1, T, N);
    sorted_d2: (H, K) sorted calibration samples per head; grid: (G,) ascending.
    Returns (B, H, T)."""
    H, K = sorted_d2.shape
    x = g[..., None] * grid                                   # (B, 1, T, N, G)
    out = []
    for h in range(H):
        cdf = torch.searchsorted(sorted_d2[h].contiguous(), x[:, 0].contiguous(), right=True)
        logp = torch.log((cdf.float() / K).clamp_min(1e-12)).sum(-2)   # (B, T, G)
        ok = logp >= math.log(q)
        idx = torch.where(ok.any(-1), ok.float().argmax(-1), torch.full_like(ok[..., 0], len(grid) - 1,
                                                                              dtype=torch.long))
        out.append(grid[idx])
    return torch.stack(out, 1)


CHK_Z = 3.0     # consistency check of the split ratio, in noise standard deviations of the analog denominator
LADDER_PFA = 1e-3   # false-alarm budget of one head's whole ladder (split evenly over its levels)


def airmoe_eta2(d, g, P):
    """Known-weight scale of [AirMoE, Thm. 1], exact: minimize the per-stream MSE
    J(u) = sum_j (d_j - a_j)^2 + u^2 over the noise std u = 1/eta, where a_j = min(d_j, c_j u) and
    c_j = sqrt(P g_j). With t_j = d_j / c_j sorted decreasingly, the devices with t_j > u saturate; on
    each interval of u, J is a convex quadratic with minimizer sum_{i<=k} c_i d_i / (1 + sum_{i<=k} c_i^2)
    (k saturated devices), clipped to the interval; k = 0 gives the genie scale u = t_(1)."""
    c = (P * g).sqrt().expand_as(d).clamp_min(1e-30)
    t, order = (d / c).sort(dim=-1, descending=True)
    ds, cs = d.gather(-1, order), c.gather(-1, order)
    cd, cc, dd = (cs * ds).cumsum(-1), (cs * cs).cumsum(-1), (ds * ds).cumsum(-1)
    lower = torch.cat([t[..., 1:], torch.zeros_like(t[..., :1])], -1)
    u = torch.minimum(torch.maximum(cd / (1.0 + cc), lower), t)
    J = dd - 2.0 * u * cd + u * u * (cc + 1.0)
    u = torch.cat([t[..., :1], u], -1)
    J = torch.cat([t[..., :1] ** 2, J], -1)
    u_opt = u.gather(-1, J.argmin(-1, keepdim=True))[..., 0]
    return 1.0 / u_opt.clamp_min(1e-30) ** 2


@functools.lru_cache(maxsize=None)
def tone_threshold(r, pfa):
    """Energy-detector threshold over r observations of complex noise with variance 1 per real
    dimension: the energy is Gamma(r, 2), so solve Q(r, t / 2) = pfa by bisection."""
    a, lo, hi = torch.tensor(float(r), dtype=torch.float64), 0.0, 1000.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(torch.special.gammaincc(a, torch.tensor(mid / 2.0, dtype=torch.float64))) > pfa:
            lo = mid
        else:
            hi = mid
    return hi


def ladder_detect(d, g, P, k, step, lmin, lmax, lmed, gen, beta=None, pfa=LADDER_PFA):
    """Ladder handshake with non-coherent energy detection (Proposition 4). Device j quantizes
    d_j^2 / g_j up to a level of step `step` dB, clipped below at lmin; devices above lmax announce the
    overflow level. The tones of a level are unmodulated and spread over the band, so the energy of
    level l over the k = M nu observations (REs x FC antennas) is Gamma(k, P S_l + 2), S_l the summed
    large-scale gain of its devices. The FC takes the highest regular level above the threshold set
    by the false-alarm budget pfa of the whole ladder, or lmed if nothing is detected.
    d, g: (..., N); lmin, lmax, lmed: level indices broadcastable to (..., 1); beta: large-scale gains
    broadcastable to d, or None. Devices with d_j = 0 (not scheduled, or on a digital fallback) send no
    tone. Returns (B-hat (...), overflow detected (...) bool, stats dict)."""
    top_reg = lmax - lmin                                   # index of l_max
    L = int(top_reg.max().item()) + 2                       # + overflow level
    lev = torch.ceil(10.0 * torch.log10((d ** 2 / g).clamp_min(1e-30)) / step)
    idx = torch.where(lev > lmax, top_reg + 1, torch.maximum(lev, lmin) - lmin).long()
    b = (beta if beta is not None else torch.ones_like(d[..., :1])) * (d > 0)
    S = torch.zeros((*d.shape[:-1], L), device=d.device, dtype=d.dtype).scatter_add_(
        -1, idx, b.expand_as(d).contiguous().to(d.dtype))
    G = torch.zeros_like(S)
    for _ in range(k):
        G += torch.empty_like(S).exponential_(generator=gen)
    E = (P * S + 2.0) * G
    ar = torch.arange(L, device=d.device)
    thr = torch.tensor([tone_threshold(k, pfa / n) for n in range(1, L + 1)], device=d.device)
    det = (E > thr[(top_reg + 1).long()]) & (ar <= top_reg + 1)
    ovf = (det & (ar == top_reg + 1)).any(-1)
    det = det & (ar <= top_reg)
    top = ((L - 1) - det.flip(-1).float().argmax(-1)).float()
    top = torch.where(det.any(-1), top, (lmed - lmin)[..., 0].expand_as(top).float())
    top = torch.where(ovf, top_reg[..., 0].expand_as(top).float(), torch.minimum(top, top_reg[..., 0]))
    bhat = 10.0 ** ((lmin[..., 0] + top) * step / 10.0)
    ratio = bhat / (d ** 2 / g).amax(-1).clamp_min(1e-30)
    stats = dict(levels=float(top_reg.float().mean()) + 2.0,
                 nodetect=float((~(det.any(-1) | ovf)).float().mean()),
                 overflow=float(ovf.float().mean()),
                 det_under=float((ratio < 1).float().mean()),
                 det_under12dB=float((ratio < 10.0 ** -1.2).float().mean()),
                 det_over10dB=float((ratio > 10).float().mean()))
    return bhat, ovf, stats


def split_policy(policy):
    """'<scale>[_hybrid|_chk|_eff]' -> (scale, denominator): analog, hybrid (exact digital D) or chk
    (exact D with the convex-hull projection, unless the analog denominator, which is sent anyway,
    deviates from it by more than CHK_Z noise standard deviations, which reveals saturation or
    misalignment, or the caller forces it; then the analog ratio), or eff (saturation-aware split
    ratio: every device securely reports its EFFECTIVE weight a_j = min(d_j, sqrt(P g_j) / eta), which
    it knows once eta is broadcast, so the exact denominator matches the weights that actually arrive
    and the ratio stays a convex combination under any saturation)."""
    for suffix in ("_hybrid", "_chk", "_eff"):
        if policy.endswith(suffix):
            return policy[:-len(suffix)], suffix[1:]
    return policy, "analog"


def kappa_gains(g, P, kappa):
    """Noise-matched level report: device j announces d_j^2 / g~_j with g~_j = (sqrt(g_j) + sqrt(kappa / P))^2
    instead of d_j^2 / g_j, i.e. the smallest noise std u = 1/eta at which its own saturation distortion
    (d_j - sqrt(P g_j) u)^2 stays below kappa u^2. Computed by every device from its own d_j, g_j and the
    public P and kappa; the FC still learns only the (quantized) maximum. kappa = 0 is the plain ladder;
    for a deep fade the report is capped at d_j^2 P / kappa. A decentralized form of the MSE-optimal
    threshold scale of AirMoE (which needs every d_j at the FC)."""
    return (g.sqrt() + math.sqrt(kappa / P)) ** 2 if kappa > 0 else g


def iq_superpose(a, obar, sig_o, rot):
    """Superposition with I/Q packing: device j sends the real streams [1, obar_j / sig_o] (dv + 1
    values) as R = ceil((dv + 1) / 2) complex symbols (stream 2r on I, 2r + 1 on Q of RE r) with
    amplitude a_j; RE r of device j arrives rotated by the complex factor rot[j, r] (residual phase,
    misalignment from imperfect CSI). A rotation therefore attenuates a stream AND leaks it into its
    I/Q partner, unlike a real alignment factor. a (..., N); obar (..., N, dv); sig_o broadcastable
    to a[..., 0]; rot complex, broadcastable to (..., N) (per device) or (..., N, R) (per RE).
    Returns (denominator sum (...), numerator sum (..., dv)) before noise."""
    dv = obar.shape[-1]
    S = dv + 1
    R = (S + 1) // 2
    so = sig_o[..., None, None]
    X = torch.cat([torch.ones_like(obar[..., :1]), obar / so], -1)
    if 2 * R > S:
        X = torch.cat([X, torch.zeros_like(X[..., :1])], -1)
    Z = torch.complex(X[..., 0::2].float(), X[..., 1::2].float())             # (..., N, R)
    if rot.dim() == a.dim():
        rot = rot[..., None]
    Y = torch.einsum("...j,...jr->...r", a.to(Z.dtype), rot.to(Z.dtype) * Z)    # (..., R)
    Xh = torch.stack([Y.real, Y.imag], -1).flatten(-2)[..., :S].to(obar.dtype)
    return Xh[..., 0], Xh[..., 1:] * sig_o[..., None]


def secure_fixed_merge(dA, nA, d, n, bits, Vd, Vn):
    """Secure digital merge in fixed point: every device encodes d_j in [0, Vd] and every component of
    n_j in [-Vn, Vn] (clipped) on a ring of `bits` bits, with ceil(log2(N + 1)) bits of headroom for the
    sum (and a sign bit for n), so that the masked sum never wraps; the FC adds its exact anchor.
    Vd broadcastable to dA, Vn broadcastable to dA. Computed in float64 (exact for bits <= 48)."""
    N = d.shape[-1]
    hb = math.ceil(math.log2(N + 1))
    d64, n64 = d.double(), n.double()
    Vd = torch.as_tensor(Vd, dtype=torch.float64, device=d.device).clamp_min(1e-300)
    Vn = torch.as_tensor(Vn, dtype=torch.float64, device=d.device).clamp_min(1e-300)
    sd = (Vd / 2.0 ** (bits - hb))[..., None]
    sn = (Vn / 2.0 ** (bits - hb - 1))[..., None, None]
    qd = torch.round(torch.minimum(d64, Vd[..., None]) / sd) * sd
    qn = torch.round(torch.maximum(torch.minimum(n64, Vn[..., None, None]), -Vn[..., None, None]) / sn) * sn
    o = (nA.double() + qn.sum(-2)) / (dA.double() + qd.sum(-1))[..., None]
    return o.to(n.dtype)


def ota_merge(dA, nA, d, n, g, policy, snr_db, sig_o, eta_param=None, zeta=None, gen=None,
              vmax=None, dfloor=None, align=None, force_analog=None, off_db=0.0, rot=None, info=None,
              g_scale=None):
    """dA (...), nA (..., dv), d (..., N), n (..., N, dv), g broadcastable to d; sig_o, eta_param,
    vmax and dfloor (a known lower bound on D) broadcastable to dA. policy in
    {genie, fixed, csi, classical, ladder, airmoe, lstale<m>}[_hybrid|_chk|_eff] or zf_sep. align
    (broadcastable to d): real alignment factor of each device's analog streams under imperfect CSI
    (g is then the estimate). rot (complex, per device or per RE): rotation of each device's REs with
    I/Q packing (iq_superpose); replaces align when given. off_db: scale off_db dB above the design
    (controlled saturation of the bottleneck devices). force_analog (bool, broadcastable to dA):
    instances where the _chk estimator must use the analog ratio (e.g. a detected ladder overflow).
    info (dict or None): receives the mean fraction of the device mass lost to saturation. g_scale: gains
    that set the scale of genie / ladder / lstale (default g), e.g. the noise-matched gains of
    kappa_gains; saturation always uses the true g.
    Returns (o_hat (..., dv), saturation indicator (...))."""
    P = 10.0 ** (snr_db / 10.0)
    dev = d.device
    dsafe = d.clamp_min(1e-30)
    obar = n / dsafe[..., None]
    if policy == "zf_sep":
        noise = torch.randn(obar.shape, generator=gen, device=dev)
        obar_hat = obar + (sig_o[..., None, None] * (zeta / P).sqrt()[..., None]) * noise
        Nh = nA + torch.einsum("...j,...jk->...k", d, obar_hat)
        return Nh / (dA + d.sum(-1))[..., None], torch.zeros_like(dA)
    base, denom = split_policy(policy)
    gs = g if g_scale is None else g_scale
    if base == "genie":
        eta2 = P * (gs / dsafe ** 2).min(-1).values
    elif base == "fixed":
        eta2 = P / eta_param
    elif base == "csi":
        eta2 = P / eta_param
    elif base == "classical":
        eta2 = P * g.min(-1).values / eta_param ** 2
    elif base == "ladder" or base.startswith("ladderoff"):
        # 'ladderoff<x>': scale x dB above the ladder level, so the bottleneck devices saturate by up to x dB
        lo_db = float(base[9:]) if base.startswith("ladderoff") else 0.0
        bott_db = 10.0 * torch.log10((dsafe ** 2 / gs).amax(-1).clamp_min(1e-30))
        eta2 = P / 10.0 ** ((torch.ceil(bott_db / eta_param) * eta_param - lo_db) / 10.0)
    elif base == "airmoe":
        eta2 = airmoe_eta2(d, g, P)
    elif base.startswith("lstale"):
        # stale ladder (no handshake round): the level of the previous instance along the last axis
        # (previous decode position) plus a margin of m dB ('lstale<m>'); devices above it saturate
        m_db = float(base[6:]) if len(base) > 6 else 0.0
        bott_db = 10.0 * torch.log10((dsafe ** 2 / gs).amax(-1).clamp_min(1e-30))
        prev = torch.cat([bott_db[..., :1], bott_db[..., :-1]], dim=-1)
        eta2 = P / 10.0 ** ((torch.ceil(prev / eta_param) * eta_param + m_db) / 10.0)
    else:
        raise ValueError(policy)
    if torch.is_tensor(off_db) or off_db:
        eta2 = eta2 * 10.0 ** (off_db / 10.0)
    inv_eta = eta2.clamp_min(1e-30).rsqrt()
    a_dev = torch.minimum(d, (P * g).sqrt() * inv_eta[..., None])          # effective weight, known at j
    sat = (a_dev < d * (1 - 1e-6)).any(-1).float()
    if info is not None:
        info["mass_loss"] = float(((d - a_dev).sum(-1) / (dA + d.sum(-1)).clamp_min(1e-30)).mean())
    n0 = torch.randn(dA.shape, generator=gen, device=dev)
    nk = torch.randn(nA.shape, generator=gen, device=dev)
    if rot is not None:
        den_air, num_air = iq_superpose(a_dev, obar, sig_o, rot)
    else:
        a = a_dev * align if align is not None else a_dev
        den_air, num_air = a.sum(-1), torch.einsum("...j,...jk->...k", a, obar)
    Nh = nA + num_air + (sig_o * inv_eta)[..., None] * nk
    D = dA + d.sum(-1)
    if denom == "hybrid":                    # exact D, with the same convex-hull projection as below
        Dx = D if vmax is None else torch.maximum(D, Nh.norm(dim=-1) / vmax)
        return Nh / Dx[..., None], sat
    if denom == "eff":                       # exact sum of the effective weights, same projection
        De = dA + a_dev.sum(-1)
        Dx = De if vmax is None else torch.maximum(De, Nh.norm(dim=-1) / vmax)
        return Nh / Dx[..., None], sat
    Dh = dA + den_air + inv_eta * n0
    floor = dA if vmax is None else torch.maximum(dA, Nh.norm(dim=-1) / vmax)
    if dfloor is not None:
        floor = torch.maximum(floor, torch.as_tensor(dfloor, dtype=floor.dtype, device=dev))
    o_an = Nh / torch.maximum(Dh, floor)[..., None]
    if denom == "chk":
        use_an = (Dh - D).abs() > CHK_Z * inv_eta
        if force_analog is not None:
            use_an = use_an | force_analog
        Dx = D if vmax is None else torch.maximum(D, Nh.norm(dim=-1) / vmax)   # same projection as analog
        return torch.where(use_an[..., None], o_an, Nh / Dx[..., None]), sat
    return o_an, sat
