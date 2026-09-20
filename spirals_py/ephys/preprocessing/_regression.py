"""Kernel-regression stack of the ephys preprocessing pipeline.

Translations of MATLAB sources in YE-et-al-2023-spirals/ephys/utils:
- AP_regresskernel_mod.m -> AP_regresskernel_mod
- kernelPrediction.m     -> kernelPrediction
- mua_prediction_full.m  -> mua_prediction_full
- get_prediction.m       -> get_prediction
- get_variance_explained.m (+ spirals_mirror/utils/sseExplainedCal.m) ->
  get_variance_explained
- get_wf_mua2.m          -> get_wf_mua2
- divide_epoch.m         -> divide_epoch
- get_flowfield5.m       -> get_flowfield5
- get_flow_metric1.m     -> get_flow_metric1
- compare_flow1.m        -> compare_flow1

Dead MATLAB code skipped: the multi-modality branches of
AP_regresskernel_mod (predicted_signals_reduced / explained_var.partial,
only reachable with cell-array regressor lists, unused by pipeline4) and
the GPU/sparse branches of both regression functions (plain numpy lstsq
throughout).  mua_prediction_full's ``if nargin == 4`` perm check is a
MATLAB bug (perm would sit in varargin at nargin == 6); permutation is
applied by the callers, as in getdVPredictionPermute.m.

Note on kernel_t = [-0.5, 0.5] at 35 Hz: MATLAB round is half-away-from-
zero, so round(-17.5) : round(17.5) = -18 : 18, i.e. 37 kernel taps
(not 35; matlab_round from _task_helpers_s15 reproduces this exactly).
"""

import warnings

import numpy as np
from scipy.stats import zscore

from spirals_py.ephys.utils import (
    get_MUA_bin,
    get_wf2ephysT2,
    loadKSdir2,
)
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap4
from spirals_py.spirals.utils import loadUVt1
from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.utils.circular import circ_mean, circ_var
from spirals_py.utils.matio import nanmean
from spirals_py.utils.optical_flow import HS_flowfield


def _zscore(x):
    """MATLAB zscore(x, [], 2): N-1 normalization along dimension 2."""
    return zscore(x, axis=1, ddof=1)


def _nanmean_kd(a, axis=1):
    """MATLAB nanmean(..., keepdims): all-NaN slices give NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis, keepdims=True)


def _kernel_frames(kernel_t, sample_rate=35):
    """MATLAB round(kernel_t * sample_rate) inclusive range (int taps)."""
    return np.arange(
        matlab_round(kernel_t[0] * sample_rate),
        matlab_round(kernel_t[1] * sample_rate) + 1,
    ).astype(int)


def _build_regressor_design(regressors, t_shifts, discontinuities):
    """Time-shifted design matrix (shared by AP_regresskernel_mod.m and
    kernelPrediction.m, lines 96-133 / 59-96).

    regressors: (nReg, T); t_shifts: 1-D integer shifts; discontinuities:
    (T,) doubles, 1 at discontinuity samples.  Returns design (T, nReg *
    nShifts) with one column block per shift, as MATLAB
    reshape(cell2mat(...), [], nReg * nShifts) lays them out.
    """
    nReg, T = regressors.shape
    nShifts = t_shifts.size
    design = np.broadcast_to(
        regressors.T[:, :, None], (T, nReg, nShifts)
    ).copy()

    disc = np.zeros((T, nShifts), dtype=bool)
    disc[:] = discontinuities.reshape(T, 1).astype(bool)
    disc[:, t_shifts == 0] = False

    for j in range(nShifts):
        design[:, :, j] = np.roll(design[:, :, j], t_shifts[j], axis=0)
        disc[:, j] = np.roll(disc[:, j], t_shifts[j])

    disc_cum = disc.copy()
    neg = t_shifts < 0
    pos = t_shifts > 0
    if neg.any():
        block = disc[:, neg]
        disc_cum[:, neg] = np.cumsum(block[::-1], axis=0)[::-1] > 0
    if pos.any():
        disc_cum[:, pos] = np.cumsum(disc[:, pos], axis=0) > 0

    design[np.broadcast_to(disc_cum[:, None, :], design.shape)] = 0.0
    return design.transpose(0, 2, 1).reshape(T, -1)


def _setup_regression(regressors, signals, t_shifts, lambdas, zs, cvfold,
                      return_constant, use_constant, discontinuities):
    """Shared argument handling / design + ridge assembly of
    AP_regresskernel_mod.m and kernelPrediction.m."""
    if t_shifts is None or np.size(t_shifts) == 0:
        t_shifts = np.array([0])
    t_shifts = np.atleast_1d(np.asarray(t_shifts, dtype=int)).ravel()
    if zs is None or len(zs) == 0:
        zs = [False, True]
    if cvfold is None or cvfold == 0:
        cvfold = 1
    if return_constant is None:
        return_constant = False
    if use_constant is None:
        use_constant = True

    regressors = np.atleast_2d(np.asarray(regressors, dtype=float))
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    nSignals, T = signals.shape

    if zs[0]:
        regressors = _zscore(regressors)
    if zs[1]:
        signals = _zscore(signals)

    if discontinuities is None or len(discontinuities) == 0:
        discontinuities = np.zeros(T)
    discontinuities = np.asarray(discontinuities, dtype=float).ravel().copy()
    if discontinuities.size != T:
        raise ValueError("Discontinuities vector doesn't match signals length")
    discontinuities[[0, -1]] = 1  # binning and end are always discontinuities

    design = _build_regressor_design(regressors, t_shifts, discontinuities)
    ncol = design.shape[1]

    # ridge rows: lambda * eye (equal to sqrt(lambda) * eye at lambda = 1)
    if lambdas is not None and np.any(lambdas):
        if np.size(lambdas) != 1:
            raise ValueError("Number of lambdas doesn't match regressor groups")
        ridge = float(np.asarray(lambdas).ravel()[0]) * np.eye(ncol)
        if use_constant:
            ridge = np.pad(ridge, ((0, 1), (0, 1)))  # extra zero row/column
    else:
        ridge = np.zeros((0, 0))

    if use_constant:
        aug = np.hstack([design, np.ones((T, 1))])
    else:
        aug = design
    A = np.vstack([aug, ridge])
    B = np.vstack([signals.T, np.zeros((ridge.shape[0], nSignals))])

    info = {
        "t_shifts": t_shifts,
        "zs": zs,
        "cvfold": cvfold,
        "return_constant": return_constant,
        "use_constant": use_constant,
        "regressors": regressors,
        "signals": signals,
        "design": design,
        "aug": aug,
        "A": A,
        "B": B,
        "nSignals": nSignals,
        "T": T,
    }
    return info


def _check_predictable(signals, design):
    """NaN handling of AP_regresskernel_mod.m lines 176-188."""
    predictable_signals = ~np.isnan(signals).all(axis=1)
    sub = signals[predictable_signals]
    if not np.all(np.isnan(sub).all(axis=0) | ~np.isnan(sub).any(axis=0)):
        raise ValueError("NaN values vary across signals")
    predictable_samples = (
        np.any(design != 0, axis=1)
        & ~np.isnan(design).any(axis=1)
        & ~np.isnan(sub).any(axis=0)
    )
    return predictable_signals, predictable_samples


def _explained_var_total(signals, predicted_signals, predictable_signals,
                         predictable_samples):
    """Total explained variance (R^2), AP_regresskernel_mod.m lines 308-314."""
    nSignals = signals.shape[0]
    sig_p = signals[np.ix_(predictable_signals, predictable_samples)]
    pred_p = predicted_signals[np.ix_(predictable_signals, predictable_samples)]
    sse_residual = ((sig_p - pred_p) ** 2).sum(axis=1, keepdims=True)
    sse_total = ((sig_p - _nanmean_kd(sig_p)) ** 2).sum(
        axis=1, keepdims=True
    )
    ev = np.full((nSignals, 1), np.nan)
    ev[predictable_signals] = 1 - sse_residual / sse_total
    return ev


def AP_regresskernel_mod(regressors, signals, t_shifts=None, lambdas=None,
                         zs=None, cvfold=None, return_constant=None,
                         use_constant=None, discontinuities=None):
    """Translated from ephys/utils/AP_regresskernel_mod.m

    Linear regression of kernels from regressors (dim x time) to signals
    (dim x time) with consecutive-chunk cvfold cross-validation.  Returns
    (k, predicted_signals, explained_var): k is [k_main, k_constant]
    (k_main (nReg, nShifts, nSignals), k_constant (1, 1, nSignals)) when
    return_constant, else the (nReg, nShifts, nSignals) array;
    explained_var is {'total': (nSignals, 1)}.
    """
    info = _setup_regression(
        regressors, signals, t_shifts, lambdas, zs, cvfold, return_constant,
        use_constant, discontinuities,
    )
    signals = info["signals"]
    design = info["design"]
    A = info["A"]
    B = info["B"]
    T = info["T"]
    nSignals = info["nSignals"]
    cvfold = info["cvfold"]
    t_shifts = info["t_shifts"]

    predictable_signals, predictable_samples = _check_predictable(signals, design)
    n_rite = A.shape[0] - T  # ridge rows appended after the data rows
    cv_partition = np.full(T, np.nan)
    if predictable_samples.sum():
        cv_partition[predictable_samples] = np.minimum(
            np.floor(np.linspace(1, cvfold + 1, predictable_samples.sum())), cvfold
        )

    k_cv = np.full((A.shape[1], nSignals, cvfold), np.nan)
    predicted_signals = np.full((nSignals, T), np.nan)
    for curr_cv in range(1, cvfold + 1):
        if cvfold == 1:
            # (deviation: MATLAB extends test_idx with the ridge rows,
            # which would assign out of bounds; test on data rows only)
            train_idx = np.concatenate(
                [predictable_samples, np.ones(n_rite, dtype=bool)]
            )
            test_idx = predictable_samples.copy()
        else:
            train_idx = np.concatenate(
                [(cv_partition != curr_cv) & predictable_samples,
                 np.ones(n_rite, dtype=bool)]
            )
            test_idx = cv_partition == curr_cv
        if not np.all(np.any(A[train_idx], axis=0)):
            warnings.warn("Regressors in fold unfilled (not enough trials?)")
        sol = np.linalg.lstsq(A[train_idx], B[train_idx], rcond=None)[0]
        k_cv[:, predictable_signals, curr_cv - 1] = sol
        # MATLAB allows the T-long test logical on the T + n_rite rows
        pred = A[:T][test_idx] @ k_cv[:, predictable_signals, curr_cv - 1]
        predicted_signals[np.ix_(predictable_signals, test_idx)] = pred.T

    if (
        np.isnan(k_cv[:, predictable_signals, :]).any()
        or np.isnan(predicted_signals[np.ix_(predictable_signals,
                                              predictable_samples)]).any()
        or np.isinf(k_cv[:, predictable_signals, :]).any()
        or np.isinf(predicted_signals[predictable_signals, :]).any()
    ):
        raise ValueError("Inf/NaN in kernel or predicted signal")

    ev = {"total": _explained_var_total(
        signals, predicted_signals, predictable_signals, predictable_samples
    )}

    if info["return_constant"]:
        k_vector = k_cv.mean(axis=2)
    else:
        k_vector = k_cv[: A.shape[1] - int(info["use_constant"])].mean(axis=2)

    nReg = info["regressors"].shape[0]
    nShifts = t_shifts.size
    if info["return_constant"]:
        k = [
            k_vector[: nReg * nShifts].reshape(nReg, nShifts, nSignals, order="F"),
            k_vector[nReg * nShifts:].reshape(1, 1, nSignals),
        ]
    else:
        k = k_vector.reshape(nReg, nShifts, nSignals, order="F")
    return k, predicted_signals, ev


def kernelPrediction(k_cv, regressors, signals, t_shifts=None, lambdas=None,
                     zs=None, cvfold=None, return_constant=None,
                     use_constant=None, discontinuities=None):
    """Translated from ephys/utils/kernelPrediction.m

    Applies a fitted kernel (k_cv, (nCols [+1], nSignals) or a single
    cvfold page) to the design matrix of regressors.  Returns
    (predicted_signals, explained_var) with explained_var {'total'}.
    (The MATLAB cvfold loop repeats the identical assignment and is run
    once here; the commented-out NaN check of the original is kept
    skipped.)
    """
    info = _setup_regression(
        regressors, signals, t_shifts, lambdas, zs, cvfold, return_constant,
        use_constant, discontinuities,
    )
    signals = info["signals"]
    design = info["design"]
    aug = info["aug"]
    nSignals = info["nSignals"]
    T = info["T"]

    predictable_signals, predictable_samples = _check_predictable(signals, design)

    k_cv = np.asarray(k_cv, dtype=float)
    if k_cv.ndim == 3:
        k_cv = k_cv[:, :, 0]

    predicted_signals = np.full((nSignals, T), np.nan)
    predicted_signals[np.ix_(predictable_signals, predictable_samples)] = (
        aug[predictable_samples] @ k_cv[:, predictable_signals]
    ).T

    if (
        np.isnan(k_cv[:, predictable_signals]).any()
        or np.isnan(predicted_signals[np.ix_(predictable_signals,
                                              predictable_samples)]).any()
        or np.isinf(k_cv[:, predictable_signals]).any()
        or np.isinf(predicted_signals[predictable_signals, :]).any()
    ):
        raise ValueError("Inf/NaN in kernel or predicted signal")

    ev = {"total": _explained_var_total(
        signals, predicted_signals, predictable_signals, predictable_samples
    )}
    return predicted_signals, ev


def divide_epoch(dV1, epochN):
    """Translated from ephys/utils/divide_epoch.m

    Returns 1-based (MATLAB) indices: epoch_indx (epochN, 2) inclusive
    ranges, train_indx (concatenated odd epochs, always 5 as in the
    original), test_indx (even epochs).
    """
    sample_length = dV1.shape[1]
    epochSize = sample_length // epochN
    epoch_indx = np.column_stack([
        1 + epochSize * np.arange(epochN),
        epochSize * np.arange(1, epochN + 1),
    ])
    train_indx = np.concatenate([
        np.arange(epoch_indx[2 * i - 1, 0], epoch_indx[2 * i - 1, 1] + 1)
        for i in range(1, 6)
    ])
    test_indx = epoch_indx[1::2]
    return epoch_indx, train_indx, test_indx


def _stack_k_cv(k):
    """k{1}/k{2} of AP_regresskernel_mod -> the (nCols + 1, nSignals)
    kernel matrix used by kernelPrediction."""
    k_main = k[0]
    nReg, nShifts, nSignals = k_main.shape
    k_cv = k_main.reshape(nReg * nShifts, nSignals, order="F")
    return np.vstack([k_cv, k[1].reshape(1, nSignals)])


def mua_prediction_full(dV1_train, MUA_std_train, dV1_test, MUA_std_test,
                        kernel_t):
    """Translated from ephys/utils/mua_prediction_full.m

    5-fold cross-validated kernel fit on the training series, then
    per-batch prediction of the 5 test epochs.  Returns
    (dV_predict (nSV, epochSize, 5), k_cv).  The dead ``if nargin == 4``
    perm branch (see module docstring) and the unused explained_var2
    output are skipped.
    """
    kernel_frames = _kernel_frames(kernel_t)
    zs = [True, False]
    cvfold = 5
    lambda1 = 1
    return_constant = 1
    k, _predicted_spikes, _explained_var = AP_regresskernel_mod(
        MUA_std_train, dV1_train, kernel_frames, lambda1, zs, cvfold,
        return_constant,
    )
    k_cv = _stack_k_cv(k)
    cvfold = 1
    nB = MUA_std_test.shape[2]
    dV_predict = np.full(
        (dV1_test.shape[0], dV1_test.shape[1], nB), np.nan
    )
    for i in range(nB):
        dV_predict[:, :, i], _ev2 = kernelPrediction(
            k_cv, MUA_std_test[:, :, i], dV1_test[:, :, i], kernel_frames,
            lambda1, zs, cvfold,
        )
    return dV_predict, k_cv


def get_prediction(dV1, MUA_std, perm=0, rng=None):
    """Translated from ephys/utils/get_prediction.m

    Short-kernel ([-0.2, 0.2] s -> lags -7..7) per-unit variant: train on
    the concatenated odd epochs, predict each even epoch.  Returns
    (dV_raw, dV_predict, epoch_indx) with epoch_indx 1-based (MATLAB).
    perm permutes the regressor rows of the test set (rng replaces
    MATLAB's global randperm stream).
    """
    if rng is None:
        rng = np.random.default_rng()
    kernel_t = [-0.2, 0.2]
    kernel_frames = _kernel_frames(kernel_t)
    zs = [True, False]
    cvfold = 5
    lambda1 = 1
    return_constant = 1
    epoch_indx, train_indx, _test_indx = divide_epoch(dV1, 10)
    k, _ps, _ev = AP_regresskernel_mod(
        MUA_std[:, train_indx - 1], dV1[:, train_indx - 1], kernel_frames,
        lambda1, zs, cvfold, return_constant,
    )
    k_cv = _stack_k_cv(k)
    cvfold = 1
    dV_raw = []
    dV_predict = []
    for i in range(5):
        test_indx = np.arange(epoch_indx[2 * i, 0], epoch_indx[2 * i, 1] + 1)
        if perm:
            randIndx = rng.permutation(MUA_std.shape[0])
            MUA_std_test = MUA_std[np.ix_(randIndx, test_indx - 1)]
        else:
            MUA_std_test = MUA_std[:, test_indx - 1]
        pred, _ev2 = kernelPrediction(
            k_cv, MUA_std_test, dV1[:, test_indx - 1], kernel_frames,
            lambda1, zs, cvfold,
        )
        dV_predict.append(pred)
        dV_raw.append(dV1[:, test_indx - 1])
    return np.stack(dV_raw, axis=2), np.stack(dV_predict, axis=2), epoch_indx


def _sse_explained_cal(signals, predicted_signals):
    """Translated from spirals_mirror/utils/sseExplainedCal.m
    (used by get_variance_explained.m)."""
    sse_residual = ((signals - predicted_signals) ** 2).sum(axis=1, keepdims=True)
    sse_total = ((signals - _nanmean_kd(signals)) ** 2).sum(
        axis=1, keepdims=True
    )
    return 1 - sse_residual / sse_total


def get_variance_explained(U, dV_raw, dV_predict):
    """Translated from ephys/utils/get_variance_explained.m

    Pixelwise 1 - SSE/SST of Ur @ dV_raw vs Ur @ dV_predict, per epoch
    page.  Reshapes use MATLAB column-major order so pixels map as in
    the original.
    """
    x, y = U.shape[0], U.shape[1]
    Ur = np.asarray(U, dtype=float).reshape(x * y, U.shape[2], order="F")
    explained_var_all = []
    for i in range(dV_raw.shape[2]):
        traceE = Ur @ dV_raw[:, :, i]
        traceEp = Ur @ dV_predict[:, :, i]
        explained_var3 = _sse_explained_cal(traceE, traceEp)
        explained_var_all.append(explained_var3.reshape(x, y, order="F"))
    return np.stack(explained_var_all, axis=2)


def get_wf_mua2(ops):
    """Translated from ephys/utils/get_wf_mua2.m

    Composite loader: loadUVt1 (dV = [0, diff]) + loadKSdir2 +
    get_wf2ephysT2 (NaN-frame dropping) + get_MUA_bin.  Returns
    (Ut, mimg, V1, dV1, MUA_std).  The unused fname/registration block of
    the MATLAB file is skipped.
    """
    U, V, t, mimg = loadUVt1(ops.session_root)
    dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
    sp = loadKSdir2(ops.session_root)
    scale = 4
    Ut = U[::scale, ::scale, :50]
    syncTL, syncProbe, WF2ephysT1 = get_wf2ephysT2(ops, t)
    WF2ephysT = WF2ephysT1[~np.isnan(WF2ephysT1)]
    dV1 = np.asarray(dV[:, ~np.isnan(WF2ephysT1)], dtype=float)
    V1 = np.asarray(V[:50, ~np.isnan(WF2ephysT1)], dtype=float)
    MUA_std = get_MUA_bin(sp, WF2ephysT)
    dV1 = np.asarray(dV1[:50, :], dtype=float)
    return Ut, mimg, V1, dV1, MUA_std


def get_flowfield5(Ut, dV_raw, mimg1, flow, len_):
    """Translated from ephys/utils/get_flowfield5.m

    Phase map + Horn-Schunck flow of a raw dV segment.  Returns
    (tracePhase1_raw1, vxy_raw1, traceAmp_raw1): phase/amp (t, x, y) with
    the last frame dropped; vxy_raw1 = vxRaw + 1j * vyRaw ((t-1, x, y)).
    useGPU is ignored (no GPU backend); the empty vxPred/vyPred branch
    returns an empty complex array.
    """
    params = {"lowpass": 0, "gsmooth": 0}
    rate = 1
    t1 = np.array([0, 1 / 35, 2 / 35])
    frameN = dV_raw.shape[1]
    if frameN > len_:
        frameN = len_
    dV_raw1 = dV_raw[:, :frameN]
    # MATLAB U./mimg broadcasts the 2-D mimg over the 3rd dimension
    _trace2d, traceAmp, tracePhase = spiralPhaseMap4(
        Ut, dV_raw1, t1, params, rate, np.asarray(mimg1, dtype=float)[:, :, None]
    )
    traceAmp = traceAmp.transpose(2, 0, 1)
    tracePhase = tracePhase.transpose(2, 0, 1)
    if flow:
        vxRaw, vyRaw = HS_flowfield(tracePhase, False)
        vxy_raw1 = vxRaw + 1j * vyRaw
    else:
        vxy_raw1 = np.zeros((0,), dtype=complex)
    return tracePhase[:-1], vxy_raw1, traceAmp[:-1]


def _angdiff(alpha, beta):
    """MATLAB angdiff(alpha, beta): elementwise wrapped difference to
    [-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(alpha) - np.asarray(beta))))


def get_flow_metric1(traceAmp_raw, tracePhase1_raw, tracePhase1_pred,
                     vxy_raw, vxy_predict):
    """Translated from ephys/utils/get_flow_metric1.m

    Bins frames by mean 2-8 Hz amplitude (edges 0:0.00125:0.025) and
    computes circular mean/variance of the raw-vs-predicted phase
    difference and flow-direction difference per bin.  NaN entries are
    dropped per array before differencing, mirroring the MATLAB
    ``x(isnan(x(:))) = []`` shrinks.  Returns
    (N, edges, phase_mu, phase_var, flow_mu, flow_var).
    """
    amp_all = np.nansum(traceAmp_raw, axis=(1, 2)) / (
        ~np.isnan(traceAmp_raw[0])
    ).sum()
    edges = np.linspace(0, 0.025, 21)
    N, _ = np.histogram(amp_all, bins=edges)
    phase_mu = np.full(edges.size - 1, np.nan)
    phase_var = phase_mu.copy()
    flow_mu = phase_mu.copy()
    flow_var = phase_mu.copy()
    for i in range(edges.size - 1):
        a = np.flatnonzero((amp_all >= edges[i]) & (amp_all < edges[i + 1]))
        if a.size:
            pa = tracePhase1_raw[a]
            pb = tracePhase1_pred[a]
            phase_diff = _angdiff(pa[~np.isnan(pa)], pb[~np.isnan(pb)])
            phase_var[i] = circ_var(phase_diff)[0]
            phase_mu[i] = circ_mean(phase_diff)

            fa = vxy_raw[a].ravel()
            fa = np.angle(fa[~np.isnan(fa)])
            fb = vxy_predict[a].ravel()
            fb = np.angle(fb[~np.isnan(fb)])
            flow_diff = _angdiff(fa, fb)
            flow_var[i] = circ_var(flow_diff)[0]
            flow_mu[i] = circ_mean(flow_diff)
    return N.astype(float), edges, phase_mu, phase_var, flow_mu, flow_var


def compare_flow1(Ut, mimg1, dV1, MUA_std, BW, len_, rng=None):
    """Translated from ephys/utils/compare_flow1.m

    Trains the [-0.5, 0.5] s kernel on the concatenated odd epochs and
    predicts the 2nd even epoch 21 times: repetition 1 with the real MUA
    rows, repetitions 2-21 with the MUA rows re-permuted each repetition
    (MATLAB randperm stream -> rng).  Phase and flow fields are masked
    with NaN outside BW after HS_flowfield (which zeroes NaN-contaminated
    flow internally, as the MATLAB horn_schunck does).  Returns
    (N, edges, phase_mu, phase_var, flow_mu, flow_var, traceAmp_mean).
    """
    if rng is None:
        rng = np.random.default_rng()
    kernel_t = [-0.5, 0.5]
    kernel_frames = _kernel_frames(kernel_t)
    zs = [True, False]
    cvfold = 5
    lambda1 = 1
    return_constant = 1
    _epoch_indx, train_indx, test_indx = divide_epoch(dV1, 10)
    k, _predicted_spikes, _explained_var = AP_regresskernel_mod(
        MUA_std[:, train_indx - 1], dV1[:, train_indx - 1], kernel_frames,
        lambda1, zs, cvfold, return_constant,
    )
    k_cv = _stack_k_cv(k)
    cvfold = 1
    # use 2nd batch in test batch
    ibatch = 2
    test_i = np.arange(test_indx[ibatch - 1, 0], test_indx[ibatch - 1, 1] + 1) - 1

    dV_raw = dV1[:, test_i]
    tracePhase_raw, vxy_raw, traceAmp_raw = get_flowfield5(
        Ut, dV_raw, mimg1, True, len_
    )
    vxy_raw[:, ~BW] = np.nan
    traceAmp_raw[:, ~BW] = np.nan
    tracePhase_raw[:, ~BW] = np.nan
    traceAmp_mean = nanmean(traceAmp_raw, axis=(1, 2))

    MUA_std_test = MUA_std[:, test_i]
    dV1_test = dV1[:, test_i]
    out = [np.full((21, 20), np.nan) for _ in range(4)]  # mu/var x phase/flow
    phase_mu, phase_var, flow_mu, flow_var = out
    N = np.full((21, 20), np.nan)
    edges = np.full((21, 21), np.nan)
    for count in range(1, 22):
        # from count 2 start to scramble
        if count > 1:
            randIndx = rng.permutation(MUA_std.shape[0])
            MUA_std_test = MUA_std_test[randIndx, :]
        dV_predict, _ev2 = kernelPrediction(
            k_cv, MUA_std_test, dV1_test, kernel_frames, lambda1, zs, cvfold
        )
        tracePhase_predict, vxy_predict, _traceAmp_predict = get_flowfield5(
            Ut, dV_predict, mimg1, True, len_
        )
        vxy_predict[:, ~BW] = np.nan
        tracePhase_predict[:, ~BW] = np.nan
        (
            N[count - 1], edges[count - 1], phase_mu[count - 1],
            phase_var[count - 1], flow_mu[count - 1], flow_var[count - 1],
        ) = get_flow_metric1(
            traceAmp_raw, tracePhase_raw, tracePhase_predict, vxy_raw, vxy_predict
        )
    return N, edges, phase_mu, phase_var, flow_mu, flow_var, traceAmp_mean
