"""Private helpers shared by the FigS12/FigS13 ephys plot modules
(plotSpiralPredictionExample1/2, plotWavePredictionExample1/2, plotVarMap,
plotVarSummary, plotWaveMatchingSession, sort_sprials_by_arousal).

Translated from MATLAB sources in YE-et-al-2023-spirals:
- ephys/utils/get_flowfield_structure2.m
- ephys/utils/spiralPhaseMap4.m
- ephys/utils/compare_flowfield_angle.m
- ephys/utils/vectorAngle.m
- ephys/utils/flow_vector_scale.m
- spirals/utils/get_cortex_atlas_path.m
- utils/subplottight.m

Plus MATLAB-imwarp/transformPointsForward equivalents for affine2d tforms
saved as v7.3 .mat files (adaptation: MATLAB imwarp is not available in
Python; cv2.warpAffine with the inverse transform is the equivalent).
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert

from spirals_py.utils.io import loadStructureTree
from spirals_py.utils.optical_flow import HS_flowfield

# area paths used by all FigS12/S13 ephys plots
# (maskPath cells in ephys/plots/*.m and spirals/utils/get_cortex_atlas_path.m)
MASK_PATHS = [
    "/997/8/567/688/695/315/500/985/",  # MOp
    "/997/8/567/688/695/315/500/993/",  # MOs
    "/997/8/567/688/695/315/31/",  # ACA
    "/997/8/567/688/695/315/453/378/",  # SS2
    "/997/8/567/688/695/315/453/322/",  # SSp
    "/997/8/567/688/695/315/247/",  # AUD
    "/997/8/567/688/695/315/669/",  # VIS
    "/997/8/567/688/695/315/254/",  # RSP
    "/997/8/567/688/695/315/22",  # VISa
    "/997/8/567/688/695/315/541/",  # TEa
    "/997/8/567/688/695/315/677/",  # VISC
]


def _load_h5_var(path, varname):
    """Load one numeric variable from a v7.3 .mat file.

    MATLAB v7.3 stores arrays with dimensions reversed, so transpose back.
    """
    with h5py.File(path, "r") as f:
        a = np.asarray(f[varname][()])
    return a.transpose(*range(a.ndim)[::-1])


def _load_mat_var(path, varname):
    """Load one variable from a .mat file (v7 via scipy, v7.3 via h5py)."""
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        return m[varname]
    except NotImplementedError:
        return _load_h5_var(path, varname)


def _load_outline_mat(data_folder):
    """isocortex_horizontal_projection_outline.mat (v7.3, 10um resolution)."""
    p = Path(data_folder) / "tables" / "isocortex_horizontal_projection_outline.mat"
    projectedAtlas1 = _load_h5_var(p, "projectedAtlas1")
    projectedTemplate1 = _load_h5_var(p, "projectedTemplate1")
    return projectedAtlas1, projectedTemplate1


def _load_atlas50(data_folder):
    """horizontal_cortex_atlas_50um.mat / horizontal_cortex_template_50um.mat (v7.3)."""
    atlas1 = _load_h5_var(Path(data_folder) / "tables" / "horizontal_cortex_atlas_50um.mat", "atlas1")
    template1 = _load_h5_var(Path(data_folder) / "tables" / "horizontal_cortex_template_50um.mat", "template1")
    return atlas1, template1


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m"""
    st = loadStructureTree(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    return list(MASK_PATHS), st


def _load_tform(path):
    """Load the 3x3 matrix T of a MATLAB affine2d saved in a v7.3 .mat file.

    MATLAB convention: [u v 1] = [x y 1] @ T (translation in row 3).

    Some rf_tform_4x files are v7 with an MCOS-opaque affine2d that scipy
    cannot decode; for those a v7.3 sibling "<name>_v73.mat" (converted with
    MATLAB) is used instead.
    """
    try:
        f = h5py.File(path, "r")
    except OSError:
        f = h5py.File(Path(path).with_name(Path(path).stem + "_v73.mat"), "r")
    with f:
        found = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("TransformationMatrix"):
                found.append(name)

        f.visititems(visitor)
        if not found:
            raise ValueError(f"No TransformationMatrix found in {path}")
        T = np.asarray(f[found[0]][()], dtype=float).T  # v7.3 stores transposed
    return T


def _transform_points_forward(T, x, y):
    """MATLAB transformPointsForward for affine2d: [u v 1] = [x y 1] @ T."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = x * T[0, 0] + y * T[1, 0] + T[2, 0]
    v = x * T[0, 1] + y * T[1, 1] + T[2, 1]
    return u, v


def _imwarp2d(A, T, out_shape):
    """MATLAB imwarp(A, tform, 'OutputView', imref2d(out_shape)) for a 2-D image.

    out_shape is (rows, cols). Bilinear interpolation, fill value 0.
    cv2.warpAffine expects the forward map and inverts it internally:
    with MATLAB [x y 1] @ T = [u v 1], the cv2 2x3 matrix is T[:, :2].T.
    """
    rows, cols = out_shape
    M = T[:, :2].T  # 2x3 forward transform in cv2 convention
    return cv2.warpAffine(
        np.asarray(A, dtype=float), M, (cols, rows),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _imwarp(A, T, out_shape):
    """imwarp for 2-D or 3-D arrays (warps each slice along the last axis)."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        return _imwarp2d(A, T, out_shape)
    out = np.zeros((out_shape[0], out_shape[1], A.shape[2]))
    for k in range(A.shape[2]):
        out[:, :, k] = _imwarp2d(A[:, :, k], T, out_shape)
    return out


def _subplottight(fig, n, m, i):
    """Translated from utils/subplottight.m (i is 1-based)."""
    c = (i - 1) % m + 1
    r = (i - 1) // m + 1
    return fig.add_axes([(c - 1) / m, 1 - r / n, 1 / m, 1 / n])


def _spiralPhaseMap4(U, dV, rate=1, lowpass=0, gsmooth=0):
    """Translated from ephys/utils/spiralPhaseMap4.m (mimg input not used here).

    U: (x, y, nSV); dV: (nSV, t). Returns trace2d, traceAmp, tracePhase,
    each (x, y, t).
    """
    x, y = U.shape[0], U.shape[1]
    Fs = 35.0 / rate  # t = [0, 1/35, 2/35] in the MATLAB caller
    Ur = U.reshape(x * y, U.shape[2])
    meanTrace = Ur @ dV
    meanTrace = np.asarray(meanTrace, dtype=float)
    tsize = meanTrace.shape[1]
    # gsmooth == 0 and rate == 1 in all callers
    meanTrace = meanTrace - meanTrace.mean(axis=1, keepdims=True)
    meanTrace = meanTrace.T  # (t, x*y); filter works on each column
    if lowpass:
        f1, f2 = butter(2, 1 / (Fs / 2), btype="low")
    else:
        f1, f2 = butter(2, [2 / (Fs / 2), 8 / (Fs / 2)], btype="bandpass")
    # MATLAB filtfilt pads with 3*(max(len(a),len(b))-1) odd samples
    padlen = 3 * (max(len(f1), len(f2)) - 1)
    meanTrace = filtfilt(f1, f2, meanTrace, axis=0, padlen=padlen)
    traceHilbert = hilbert(meanTrace, axis=0)
    tracePhase = np.angle(traceHilbert)
    traceAmp = np.abs(traceHilbert)
    tracePhase = tracePhase.reshape(tsize, x, y).transpose(1, 2, 0)
    traceAmp = traceAmp.reshape(tsize, x, y).transpose(1, 2, 0)
    trace2d = meanTrace.reshape(tsize, x, y).transpose(1, 2, 0)
    return trace2d, traceAmp, tracePhase


def _get_flowfield_structure2(Ut1, dV1, epochs0, mimg, flow=True):
    """Translated from ephys/utils/get_flowfield_structure2.m

    epochs0: 0-based frame indices (MATLAB epochs minus 1).
    Returns a SimpleNamespace with tracePhase1/traceAmp1/trace2d1 (t, x, y)
    and vxRaw/vyRaw (t-1, x, y).
    """
    frameStart = epochs0[0]
    frameEnd = epochs0[-1]
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)  # extra 2*35 frames
    dV_raw_epoch = dV1[:50, frameTemp]
    trace2d1, traceAmp1, tracePhase1 = _spiralPhaseMap4(Ut1[:, :, :50], dV_raw_epoch)
    pad = 35  # 35/rate with rate == 1
    trace2d1 = trace2d1[:, :, pad:-pad] / mimg[:, :, None]
    tracePhase1 = tracePhase1[:, :, pad:-pad]
    traceAmp1 = traceAmp1[:, :, pad:-pad]

    trace2d1 = trace2d1.transpose(2, 0, 1)
    tracePhase1 = tracePhase1.transpose(2, 0, 1)
    traceAmp1 = traceAmp1.transpose(2, 0, 1)

    data_raw = SimpleNamespace()
    if flow:
        vxRaw, vyRaw = HS_flowfield(tracePhase1, False)
        data_raw.vxRaw = vxRaw
        data_raw.vyRaw = vyRaw

    tracePhase1 = tracePhase1[:-1]
    traceAmp1 = traceAmp1[:-1]
    trace2d1 = trace2d1[:-1]

    data_raw.tracePhase1 = tracePhase1
    data_raw.traceAmp1 = traceAmp1
    data_raw.trace2d1 = trace2d1
    return data_raw


def _vectorAngle2(v1, v2):
    """Two-vector case of ephys/utils/vectorAngle.m (midAngle = pi):
    angle from v1 to v2, normalized to [0, 2*pi)."""
    alpha1 = np.mod(np.arctan2(v1[:, 1], v1[:, 0]), 2 * np.pi)
    alpha2 = np.mod(np.arctan2(v2[:, 1], v2[:, 0]), 2 * np.pi)
    return np.mod(alpha2 - alpha1, 2 * np.pi)


def _compare_flowfield_angle(data_raw, data_predict):
    """Translated from ephys/utils/compare_flowfield_angle.m

    Returns complex array (t, x, y) whose angle is the flow-direction
    difference between raw and predicted frames.
    """
    x_size = data_raw.vxRaw.shape[1]
    y_size = data_raw.vxRaw.shape[2]
    frameN = data_raw.vxRaw.shape[0]
    Axy_all = np.zeros((frameN, x_size, y_size), dtype=complex)
    for i in range(frameN):
        vxRaw = data_raw.vxRaw[i]
        vyRaw = data_raw.vyRaw[i]
        vxPredict = data_predict.vxRaw[i]
        vyPredict = data_predict.vyRaw[i]
        vector1 = np.column_stack([vxRaw.ravel(), vyRaw.ravel()])
        vector2 = np.column_stack([vxPredict.ravel(), vyPredict.ravel()])
        A = _vectorAngle2(vector1, vector2)
        Axy_all[i] = np.exp(1j * A.reshape(x_size, y_size))
    return Axy_all


def _flow_vector_scale(vxRaw, vyRaw, skip, zoom_scale):
    """Translated from ephys/utils/flow_vector_scale.m"""
    vxRaw = np.asarray(vxRaw, dtype=float).squeeze()
    vyRaw = np.asarray(vyRaw, dtype=float).squeeze()
    vxRaw2 = np.full(vxRaw.shape, np.nan)
    vyRaw2 = np.full(vyRaw.shape, np.nan)
    vxRaw2[::skip, ::skip] = vxRaw[::skip, ::skip] * zoom_scale
    vyRaw2[::skip, ::skip] = vyRaw[::skip, ::skip] * zoom_scale
    return vxRaw2, vyRaw2


def _forder_flat_indices(cond):
    """MATLAB find(cond): linear indices into a column-major flattening."""
    return np.flatnonzero(cond.ravel(order="F"))


def _forder_columns(A):
    """Reshape (t, x, y) to (t, x*y) matching MATLAB A(:,:) linear indexing."""
    return A.reshape(A.shape[0], -1, order="F")


# cbrewer2('qual','Set2',8) used for the df/f traces
_SET2 = np.array(
    [
        [102, 194, 165],
        [252, 141, 98],
        [141, 160, 203],
        [231, 138, 195],
        [166, 216, 84],
        [255, 217, 47],
        [229, 196, 148],
        [179, 179, 179],
    ]
) / 255.0


def _plot_prediction_example(
    T, data_folder, save_folder, kk, epochs, frame, abcd, fig_tag, wave=False
):
    """Shared driver for ephys/plots/plotSpiralPredictionExample1/2.m and
    ephys/plots/plotWavePredictionExample1/2.m.

    kk: MATLAB 1-based row of T; epochs/frame: MATLAB 1-based index ranges;
    abcd: (a, b, c, d) zoom-rectangle rows a..b, cols c..d (MATLAB 1-based);
    fig_tag: e.g. 'FigS12a'/'FigS12b' first tag; wave=True overlays quivers.
    """
    from spirals_py.ephys.utils import (
        get_MUA_bin,
        get_session_info2,
        get_wf2ephysT2,
        loadKSdir2,
    )
    from spirals_py.spirals.utils import loadUVt1
    from spirals_py.utils.atlas import plotOutline
    from spirals_py.utils.circular import circ_var

    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # atlas brain horizontal projection and outline
    projectedAtlas1, projectedTemplate1 = _load_outline_mat(data_folder)
    atlas1, _ = _load_atlas50(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    scale3 = 5
    lineColor = "k"
    hemi = None

    # original and predicted dV by kernel regression
    ops = get_session_info2(T, kk - 1, data_folder)
    U, V, t, mimg = loadUVt1(ops.session_root)
    dV = np.concatenate([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)], axis=1)
    sp = loadKSdir2(ops.session_root)
    syncTL, syncProbe, WF2ephysT1 = get_wf2ephysT2(ops, t)
    WF2ephysT = WF2ephysT1[~np.isnan(WF2ephysT1)]
    dV1 = np.asarray(dV[:, ~np.isnan(WF2ephysT1)], dtype=float)
    MUA_std = get_MUA_bin(sp, WF2ephysT)
    dV1 = np.asarray(dV1[:50, :], dtype=float)

    fname = f"{ops.mn}_{ops.tdb}_{ops.en}"
    T_tform = _load_tform(data_folder / "ephys" / "rf_tform" / f"{fname}_tform.mat")
    BW2 = _load_h5_var(
        data_folder / "ephys" / "spirals_example" / f"{fname}_mask.mat", "BW2"
    ).astype(float)
    T4 = _load_tform(data_folder / "ephys" / "rf_tform_4x" / f"{fname}_tform_4x.mat")
    pred = data_folder / "ephys" / "dv_prediction" / f"{fname}_dv_predict.mat"
    dV_predict = _load_h5_var(pred, "dV_predict")
    explained_var_all = _load_h5_var(pred, "explained_var_all")

    epochs0 = np.arange(epochs[0] - 1, epochs[1])  # 0-based, inclusive range
    frame0 = np.asarray(frame) - 1
    point = np.array([[101, 91]], dtype=float) * 4
    pt_u, pt_v = _transform_points_forward(T_tform, point[:, 0], point[:, 1])
    point_t = np.column_stack([pt_u, pt_v])

    a, b, c, d = abcd  # height a:b, width c:d (MATLAB 1-based)

    sizeTemplate = [1320, 1140]
    mimgt = _imwarp2d(mimg, T_tform, sizeTemplate)
    Utransformed = _imwarp(U, T_tform, projectedAtlas1.shape)
    Utransformed1 = Utransformed[::8, ::8, :]
    BW = projectedAtlas1 > 0

    # load dV prediction
    scale = 4
    dV1 = dV1[:, : dV_predict.shape[1]]

    # angle difference for oscillation period
    data_raw = _get_flowfield_structure2(Utransformed1[:, :, :50], dV1, epochs0, mimgt[::8, ::8])
    data_predict = _get_flowfield_structure2(
        Utransformed1[:, :, :50], dV_predict, epochs0, mimgt[::8, ::8]
    )
    Axy_all = _compare_flowfield_angle(data_raw, data_predict)

    explained_var_all = np.where(explained_var_all < 0, 0, explained_var_all)
    mean_var = explained_var_all.mean(axis=2)
    mean_var_t = _imwarp2d(mean_var, T4, projectedTemplate1.shape)
    filename = f"{ops.mn}_{epochs[0]}"

    colorn = MUA_std.shape[0]
    colorGray = plt.cm.gray(np.linspace(0, 1, colorn))[:, :3]
    randi = np.random.permutation(colorn)
    colorAll = colorGray[randi]

    fig1 = plt.figure(figsize=(10, 6))
    gs = fig1.add_gridspec(3, 4)
    ax1 = fig1.add_subplot(gs[0, 0])
    mxRange = np.percentile(mimgt, 99.5)
    ax1.imshow(mimgt, cmap="gray", alpha=BW2, vmin=0, vmax=mxRange)
    ax1.set_aspect("equal")
    ax1.axis("off")
    plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax1)
    ax1.scatter(point_t[:, 1], point_t[:, 0], 6, "r")
    plt.colorbar(ax1.images[0], ax=ax1, fraction=0.046)

    ax2 = fig1.add_subplot(gs[1, 0])
    ax2.imshow(mean_var_t, cmap="inferno", alpha=BW2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax2)
    plt.colorbar(ax2.images[0], ax=ax2, fraction=0.046)

    point1 = np.round(point / scale).astype(int)  # [101, 91]
    ax3 = fig1.add_subplot(gs[0, 1:4])
    trace_predict = data_predict.trace2d1[:, point1[0, 0] - 1, point1[0, 1] - 1]
    trace_raw = data_raw.trace2d1[:, point1[0, 0] - 1, point1[0, 1] - 1]
    ax3.plot(
        WF2ephysT[epochs0[:-1]], trace_raw * 100, linewidth=1, color=_SET2[4]
    )  # GREEN
    ax3.plot(
        WF2ephysT[epochs0[:-1]], trace_predict * 100, linewidth=1, color=_SET2[7]
    )  # GRAY
    ax3.set_xlim(WF2ephysT[epochs0[0]], WF2ephysT[epochs0[-1]])
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("df/f (%)")
    ax3.axvline(WF2ephysT[epochs0[frame0[0]]], linestyle="--", color="k")
    ax3.axvline(WF2ephysT[epochs0[frame0[-1]]], linestyle="--", color="k")

    t_sample = epochs0[-1] - epochs0[0]
    t_length = t_sample / 35

    ax4 = fig1.add_subplot(gs[1, 1:4])
    t0 = WF2ephysT[epochs0[0]]
    t1 = WF2ephysT[epochs0[-1]]
    for i in range(len(sp.gcluster)):
        incl = (
            (sp.clu == sp.gcluster[i])
            & (sp.spikeTimes > t0)
            & (sp.spikeTimes < t1)
        )
        cluster_t = sp.spikeTimes[incl]
        cluster_depth = sp.spikeDepths[incl]
        mean_rate = cluster_t.size / t_length
        if mean_rate < 20 and cluster_t.size:
            ax4.vlines(
                cluster_t, cluster_depth, cluster_depth + 100, color=colorAll[i], linewidth=1
            )
    ax4.set_xlim(t0, t1)

    BW2down = BW2[::8, ::8]
    mean_var_t2 = mean_var_t[::8, ::8]
    indx = _forder_flat_indices((mean_var_t2 <= 0.4) & (BW2down == 0))
    ax10 = fig1.add_subplot(gs[2, 1:4])
    theta = np.angle(Axy_all)
    theta2d = _forder_columns(theta)
    theta2d = np.delete(theta2d, indx, axis=1)
    S, _ = circ_var(theta2d, dim=1)
    c1 = 1 - S
    ax10.scatter(np.arange(1, t_sample + 1), c1, c="k", alpha=0.4)
    ax10.set_xticks(np.arange(0, t_sample + 1e-6, 17.5))
    ax10.set_xticklabels([f"{v:g}" for v in np.arange(0, t_length + 1e-6, 0.5)])
    ax10.set_xlabel("Time (s)")
    ax10.set_xlim(1, t_sample)
    ax10.set_ylim(0, 1)
    fig1.savefig(save_folder / f"{fig_tag[0]}_{filename}.pdf", bbox_inches="tight")

    # frame montage
    phase_raw = data_raw.tracePhase1.transpose(1, 2, 0)
    phase_predict = data_predict.tracePhase1.transpose(1, 2, 0)
    vx_raw = data_raw.vxRaw.transpose(1, 2, 0)
    vy_raw = data_raw.vyRaw.transpose(1, 2, 0)
    vx_predict = data_predict.vxRaw.transpose(1, 2, 0)
    vy_predict = data_predict.vyRaw.transpose(1, 2, 0)

    BW_var = ((mean_var_t2 >= 0.1) & (BW2down > 0))
    indx = _forder_flat_indices(~BW_var)
    theta2d = _forder_columns(theta)
    theta2d = np.delete(theta2d, indx, axis=1)
    S, _ = circ_var(theta2d, dim=1)
    c1 = 1 - S

    if wave:
        skip = 5
        zoom_scale = 2
        vx_raw2, vy_raw2 = _flow_vector_scale(vx_raw, vy_raw, skip, zoom_scale)
        vx_predict2, vy_predict2 = _flow_vector_scale(vx_predict, vy_predict, skip, zoom_scale)
        mask = ~BW_var

        def _mask_flow(vx2, vy2):
            vx3 = vx2.copy()
            vy3 = vy2.copy()
            vx3[mask] = np.nan
            vy3[mask] = np.nan
            return vx3, vy3

        vx_raw3, vy_raw3 = _mask_flow(vx_raw2, vy_raw2)
        vx_predict3, vy_predict3 = _mask_flow(vx_predict2, vy_predict2)

    try:
        import colorcet as cc

        cmapC6 = cc.cm.CET_C6
    except ImportError:
        cmapC6 = "twilight"

    scale4 = 5 / 8
    frame_n = len(frame0)
    fig2 = plt.figure(figsize=(8, 4))
    for i, f0 in enumerate(frame0):
        frame_raw = phase_raw[:, :, f0]
        frame_predict = phase_predict[:, :, f0]

        ax1i = _subplottight(fig2, 4, frame_n, i + 1)
        ax1i.imshow(frame_raw, cmap=cmapC6, alpha=BW2down)
        ax1i.set_aspect("equal")
        ax1i.axis("off")
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale4, lineColor, ax=ax1i)
        _roi_patch(ax1i, a, b, c, d)

        ax2i = _subplottight(fig2, 4, frame_n, frame_n + i + 1)
        ax2i.imshow(frame_predict, cmap=cmapC6, alpha=BW2down)
        ax2i.set_aspect("equal")
        ax2i.axis("off")
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale4, lineColor, ax=ax2i)
        _roi_patch(ax2i, a, b, c, d)

        ax3i = _subplottight(fig2, 4, frame_n, 2 * frame_n + i + 1)
        ax3i.imshow(frame_raw, cmap=cmapC6, alpha=BW2down, vmin=-np.pi, vmax=np.pi)
        ax3i.set_aspect("equal")
        ax3i.axis("off")
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale4, lineColor, ax=ax3i)
        if wave:
            ax3i.quiver(
                vx_raw3[:, :, f0], vy_raw3[:, :, f0], color="k",
                angles="xy", scale_units="xy", scale=1, linewidth=0.5,
            )
        ax3i.set_ylim(b - 0.5, a - 0.5)
        ax3i.set_xlim(c - 0.5, d - 0.5)
        ax3i.set_title(f"{c1[f0]:g}", fontsize=6)

        ax4i = _subplottight(fig2, 4, frame_n, 3 * frame_n + i + 1)
        ax4i.imshow(frame_predict, cmap=cmapC6, alpha=BW2down, vmin=-np.pi, vmax=np.pi)
        ax4i.set_aspect("equal")
        ax4i.axis("off")
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale4, lineColor, ax=ax4i)
        if wave:
            ax4i.quiver(
                vx_predict3[:, :, f0], vy_predict3[:, :, f0], color="k",
                angles="xy", scale_units="xy", scale=1, linewidth=0.5,
            )
        ax4i.set_ylim(b - 0.5, a - 0.5)
        ax4i.set_xlim(c - 0.5, d - 0.5)

    fig2.savefig(save_folder / f"{fig_tag[1]}_{filename}_frames.pdf", bbox_inches="tight")
    return fig1, fig2


def _roi_patch(ax, a, b, c, d):
    """MATLAB patch('Faces',[1 2 3 4],'Vertices',[c a;d a;d b;c b]) (1-based)."""
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle((c - 1, a - 1), d - c + 1, b - a + 1, fill=False, edgecolor="k", linewidth=1)
    )
