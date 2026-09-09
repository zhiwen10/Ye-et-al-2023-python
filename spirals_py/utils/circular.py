import warnings

import numpy as np
from scipy import stats


def circ_r(alpha, w=None, d=0, dim=0):
    """Translated from ephys/utils/circ_r.m (dim is 0-based; MATLAB default dim=1)."""
    alpha = np.asarray(alpha)
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != alpha.shape:
            raise ValueError("Input dimensions do not match")

    r = np.sum(w * np.exp(1j * alpha), axis=dim)
    r = np.abs(r) / np.sum(w, axis=dim)

    if d != 0:
        c = d / 2 / np.sin(d / 2)
        r = c * r
    return r


def circ_var(alpha, w=None, d=0, dim=0):
    """Translated from ephys/utils/circ_var.m"""
    r = circ_r(alpha, w, d, dim)
    S = 1 - r
    s = 2 * S
    return S, s


def circ_mean(alpha, w=None, dim=0):
    """Translated from axons/utils/circ_mean.m

    Returns the mean direction only; use circ_confmean for confidence limits.
    """
    alpha = np.asarray(alpha)
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != alpha.shape:
            raise ValueError("Input dimensions do not match")

    r = np.sum(w * np.exp(1j * alpha), axis=dim)
    return np.angle(r)


def circ_dist2(x, y=None):
    """Translated from axons/utils/circ_dist2.m"""
    x = np.asarray(x).ravel()
    if y is None:
        y = x
    else:
        y = np.asarray(y).ravel()
    r = np.angle(np.exp(1j * x)[:, None] / np.exp(1j * y)[None, :])
    return r


def circ_confmean(alpha, xi=0.05, w=None, d=0, dim=0):
    """Translated from axons/utils/circ_confmean.m"""
    alpha = np.asarray(alpha)
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != alpha.shape:
            raise ValueError("Input dimensions do not match")

    r = np.atleast_1d(circ_r(alpha, w, d, dim))
    n = np.atleast_1d(np.sum(w, axis=dim).astype(float))
    R = n * r
    c2 = stats.chi2.ppf(1 - xi, 1)

    t = np.zeros_like(r)
    for i in range(r.size):
        if r[i] < 0.9 and r[i] > np.sqrt(c2 / 2 / n[i]):
            t[i] = np.sqrt((2 * n[i] * (2 * R[i] ** 2 - n[i] * c2)) / (4 * n[i] - c2))  # equ. 26.24
        elif r[i] >= 0.9:
            t[i] = np.sqrt(n[i] ** 2 - (n[i] ** 2 - R[i] ** 2) * np.exp(c2 / n[i]))  # equ. 26.25
        else:
            t[i] = np.nan
            warnings.warn("Requirements for confidence levels not met.")

    t = np.arccos(np.clip(t / R, -1, 1))
    return t.reshape(np.shape(circ_r(alpha, w, d, dim)))


def circ_kappa(alpha, w=None):
    """Translated from axons/utils/circ_kappa.m"""
    alpha = np.asarray(alpha).ravel()
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w, dtype=float).ravel()

    N = alpha.size
    if N > 1:
        R = circ_r(alpha, w)
    else:
        R = alpha[0]

    if R < 0.53:
        kappa = 2 * R + R**3 + 5 * R**5 / 6
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        kappa = 1 / (R**3 - 4 * R**2 + 3 * R)

    if N < 15 and N > 1:
        if kappa < 2:
            kappa = max(kappa - 2 * (N * kappa) ** -1, 0)
        else:
            kappa = (N - 1) ** 3 * kappa / (N**3 + N)
    return kappa


def circ_mtest(alpha, dir, xi=0.05, w=None, d=0):
    """Translated from axons/utils/circ_mtest.m; returns (h, mu, ul, ll)."""
    alpha = np.asarray(alpha).ravel()
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w, dtype=float).ravel()
        if alpha.size != w.size:
            raise ValueError("Input dimensions do not match.")

    mu = circ_mean(alpha, w)
    t = circ_confmean(alpha, xi, w, d)
    ul = mu + t
    ll = mu - t

    h = bool(np.abs(circ_dist2([dir], [mu]))[0, 0] > t)
    return h, mu, ul, ll


def circ_wwtest(*args):
    """Translated from axons/utils/circ_wwtest.m; returns (pval, table)."""
    alpha, idx, w = _wwtest_process_input(args)

    u = np.unique(idx)
    s = u.size
    n = np.sum(w)

    pn = np.zeros(s)
    pr = np.zeros(s)
    for t in range(s):
        pidx = idx == u[t]
        pn[t] = np.sum(pidx * w)
        pr[t] = circ_r(alpha[pidx], w[pidx])

    r = circ_r(alpha, w)
    rw = np.sum(pn * pr) / n

    _wwtest_check_assumption(rw, np.mean(pn))

    kk = circ_kappa(np.array([rw]))
    beta = 1 + 3 / (8 * kk)  # correction factor
    A = np.sum(pr * pn) - r * n
    B = n - np.sum(pr * pn)

    F = beta * (n - s) * A / (s - 1) / B
    pval = 1 - stats.f.cdf(F, s - 1, n - s)

    table = [
        ["Source", "d.f.", "SS", "MS", "F", "P-Value"],
        ["Columns", s - 1, A, A / (s - 1), F, pval],
        ["Residual ", n - s, B, B / (n - s), None, None],
        ["Total", n - 1, A + B, None, None, None],
    ]
    return pval, table


def _wwtest_process_input(args):
    if len(args) == 4:
        alpha1 = np.asarray(args[0]).ravel()
        alpha2 = np.asarray(args[1]).ravel()
        w1 = np.asarray(args[2]).ravel()
        w2 = np.asarray(args[3]).ravel()
        alpha = np.concatenate([alpha1, alpha2])
        # replicates the original repo code (upstream circStat uses 2*ones for alpha2)
        idx = np.concatenate([np.ones(alpha1.size), np.ones(alpha2.size)])
        w = np.concatenate([w1, w2])
    elif len(args) == 2 and np.sum(np.abs(np.round(args[1]) - args[1])) > 1e-5:
        alpha1 = np.asarray(args[0]).ravel()
        alpha2 = np.asarray(args[1]).ravel()
        alpha = np.concatenate([alpha1, alpha2])
        idx = np.concatenate([np.ones(alpha1.size), 2 * np.ones(alpha2.size)])
        w = np.ones(alpha.size)
    elif len(args) == 2:
        alpha = np.asarray(args[0]).ravel()
        idx = np.asarray(args[1]).ravel()
        if idx.size != alpha.size:
            raise ValueError("Input dimensions do not match.")
        w = np.ones(alpha.size)
    elif len(args) == 3:
        alpha = np.asarray(args[0]).ravel()
        idx = np.asarray(args[1]).ravel()
        w = np.asarray(args[2]).ravel()
        if idx.size != alpha.size or w.size != alpha.size:
            raise ValueError("Input dimensions do not match.")
    else:
        raise ValueError("Invalid use of circ_wwtest.")
    return alpha, idx, w


def _wwtest_check_assumption(rw, n):
    if n >= 11 and rw < 0.45:
        warnings.warn("Test not applicable. Average resultant vector length < 0.45.")
    elif n < 11 and n >= 7 and rw < 0.5:
        warnings.warn(
            "Test not applicable. Average number of samples per population 6 < x < 11 "
            "and average resultant vector length < 0.5."
        )
    elif n >= 5 and n < 7 and rw < 0.55:
        warnings.warn(
            "Test not applicable. Average number of samples per population 4 < x < 7 "
            "and average resultant vector length < 0.55."
        )
    elif n < 5:
        warnings.warn("Test not applicable. Average number of samples per population < 5.")


def _uniques_ties_cumuls_relfreqs(A):
    a, t = np.unique(A, return_counts=True)
    m = np.cumsum(t).astype(float)
    n = m[-1]
    return a, t.astype(float), m, n, m / n


def watsons_U2(A1, A2):
    """Translated from axons/utils/watsons_U2.m"""
    A1 = np.asarray(A1).ravel()
    A2 = np.asarray(A2).ravel()

    a1, t1, m1, n1, m1_n1 = _uniques_ties_cumuls_relfreqs(A1)
    a2, t2, m2, n2, m2_n2 = _uniques_ties_cumuls_relfreqs(A2)

    n = n1 + n2

    vals = np.unique(np.concatenate([A1, A2]))
    table = np.zeros((vals.size, 3))  # cols: relfreq sample1, relfreq sample2, ties
    for i, v in enumerate(vals):
        loc1 = np.flatnonzero(a1 == v)
        loc2 = np.flatnonzero(a2 == v)
        if loc1.size:
            table[i, 0] += m1_n1[loc1[0]]
            table[i, 2] += t1[loc1[0]]
        elif i > 0:
            table[i, 0] = table[i - 1, 0]
        if loc2.size:
            table[i, 1] += m2_n2[loc2[0]]
            table[i, 2] += t2[loc2[0]]
        elif i > 0:
            table[i, 1] = table[i - 1, 1]

    d = table[:, 0] - table[:, 1]
    t = table[:, 2]
    td = np.sum(t * d)
    td2 = np.sum(t * d**2)

    U2 = (n1 * n2 / n**2) * (td2 - td**2 / n)
    return U2


def watsons_U2_perm_test(A1, A2, N, rng=None):
    """Translated from axons/utils/watsons_U2_perm_test.m"""
    if rng is None:
        rng = np.random.default_rng()
    U2_obs = watsons_U2(A1, A2)
    U2_H0 = np.zeros(N)
    A = np.concatenate([np.asarray(A1).ravel(), np.asarray(A2).ravel()])
    n1 = np.asarray(A1).size
    n12 = A.size

    for i in range(N):
        a = A[rng.permutation(n12)]
        U2_H0[i] = watsons_U2(a[:n1], a[n1:])

    p = np.mean(U2_H0 >= U2_obs)
    return p, U2_obs, U2_H0
