"""Shared private helpers for the Extended Data Fig.1 / Fig.2 modules
(plotExampleSpiral2b, plotExampleSpiralSpectrum2/3, plotExampleSpiral3b,
plotSpiralDetectionPipeline).

Translations of MATLAB helpers that are not yet ported elsewhere:
- spirals/utils/spirals_detection/spiralPhaseMap5.m          -> _spiralPhaseMap5
- spirals/utils/select_area.m                                -> _select_area
- revision/power_spectrum/fft_spectrum.m                     -> _fft_spectrum
- spirals/utils/spirals_detection/setSpiralDetectionParams.m -> _setSpiralDetectionParams
- spirals/utils/spirals_detection/spiralAlgorithm.m          -> _spiralAlgorithm
- spirals/utils/spirals_detection/checkSpiral.m              -> _checkSpiral
- spirals/utils/spirals_detection/padZeros.m                 -> _padZeros
- spirals/utils/spirals_detection/checkClusterXY.m           -> _checkClusterXY
- spirals/utils/spirals_detection/clusterXYpoints.m          -> _clusterXYpoints
- spirals/utils/spirals_detection/doubleCheckSpiralsAlgorithm.m
                                                             -> _doubleCheckSpiralsAlgorithm
- spirals/utils/spirals_detection/spatialRefine.m            -> _spatialRefine
- spirals/utils/spirals_detection/spiralRadiusCheck2.m       -> _spiralRadiusCheck2
- MATLAB inROI() on a polyshape ROI                          -> _inROI/_load_roi_vertices
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert, medfilt2d

from spirals_py.spirals.plots.plotSpiralTimeSeries3d import (
    _get_cortex_atlas_path,
    _matlab_round,
)
from spirals_py.spirals.utils import loadUVt1


def _load_tform(path):
    """Load a v7.3 affine2d object; returns its 3x3 TransformationMatrix T in
    MATLAB orientation (last row [tx ty 1]), so that
    [u v 1] = [x y 1] * T (transformPointsForward / imwarp)."""
    with h5py.File(path, "r") as f:
        T = None
        for name in f["#refs#"]:
            g = f["#refs#"][name]
            if isinstance(g, h5py.Group) and "TransformationMatrix" in g:
                T = np.asarray(g["TransformationMatrix"]).T
                break
        if T is None:
            raise ValueError(f"No TransformationMatrix found in {path}")
        return T


def _transform_points_forward_row(T, x, y):
    """MATLAB transformPointsForward for affine2d: [u v 1] = [x y 1] * T.

    (Note: plotSpiralSequence3._transform_points_forward applies T as
    [x y 1] @ T.T, which drops the translation term of these tform files.)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xy1 = np.stack([x, y, np.ones_like(x)], axis=-1)
    uv = xy1 @ T
    return uv[..., 0], uv[..., 1]


def _imwarp_row(img, T, out_shape, stride=1):
    """MATLAB imwarp(img, tform, 'OutputView', imref2d(out_shape)), bilinear,
    FillValues=0, followed by (1:stride:end, 1:stride:end) sampling.

    The inverse map is applied in MATLAB's row-vector convention
    [x_in y_in 1] = [X Y 1] * inv(T). (plotSpiralTimeSeries3d._imwarp applies
    inv(T).T instead, which maps every output pixel of these tforms outside
    the input image and returns an all-zero warp.)"""
    from scipy import ndimage

    Tinv = np.linalg.inv(T)
    rows = np.arange(0, out_shape[0], stride) + 1  # 1-based output y centers
    cols = np.arange(0, out_shape[1], stride) + 1  # 1-based output x centers
    X, Y = np.meshgrid(cols, rows)
    ones = np.ones_like(X)
    uv = np.stack([X, Y, ones], axis=-1) @ Tinv
    xin = uv[..., 0] / uv[..., 2] - 1  # 0-based input column
    yin = uv[..., 1] / uv[..., 2] - 1  # 0-based input row
    coords = [yin, xin]

    def warp_plane(plane):
        return ndimage.map_coordinates(
            np.asarray(plane, dtype=float), coords, order=1, mode="constant", cval=0
        )

    if img.ndim == 2:
        return warp_plane(img)
    planes = [warp_plane(img[:, :, k]) for k in range(img.shape[2])]
    return np.stack(planes, axis=2)


def _load_session_row(data_folder, T, kk):
    """Load session `kk` (MATLAB 1-based row) of table T: returns
    (mn, tdb, en, fname, U, V, t, mimg, dV)."""
    data_folder = Path(data_folder)
    row = T.iloc[kk - 1]
    mn = str(row["MouseID"])
    tda = pd.Timestamp(row["date"])
    en = int(row["folder"])
    tdb = tda.strftime("%Y%m%d")
    fname = f"{mn}_{tdb}_{en}"
    session_root = data_folder / "spirals" / "svd" / fname
    U, V, t, mimg = loadUVt1(session_root)
    dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
    return mn, tdb, en, fname, U, V, t, mimg, dV


def _load_spirals_group_fftn(data_folder, fname, T_tform):
    """Load spirals/spirals_grouping/<fname>_spirals_group_fftn.mat
    (cell2mat(archiveCell) -> N x 5 [x, y, radius, direction, frame]) and map
    the centers through the affine tform into the 8x-downscaled atlas frame."""
    data_folder = Path(data_folder)
    group_file = data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat"
    rows = []
    with h5py.File(group_file, "r") as f:
        refs = f["archiveCell"][()]
        for ref in refs.flat:
            rows.append(np.asarray(f[ref]).T)  # (5, m) -> (m, 5)
    spirals = np.vstack(rows)  # cell2mat(archiveCell): N x 5
    sx, sy = _transform_points_forward_row(T_tform, spirals[:, 0], spirals[:, 1])
    spirals[:, 0:2] = _matlab_round(np.stack([sx, sy], axis=1) / 8)
    return spirals


def _load_session_masks(data_folder, fname):
    """Load spirals/spirals_example/<fname>_mask.mat (BW, BW1, BW2) as bools
    in MATLAB orientation."""
    data_folder = Path(data_folder)
    path = data_folder / "spirals" / "spirals_example" / f"{fname}_mask.mat"
    masks = {}
    with h5py.File(path, "r") as f:
        for key in ("BW", "BW1", "BW2"):
            masks[key] = np.asarray(f[key]).T.astype(bool)
    return masks


def _spiralPhaseMap5(U, dV, t, params, rate, mimg=None):
    """Translated from spirals/utils/spirals_detection/spiralPhaseMap5.m

    Returns (trace2d, traceAmp, tracePhase, rawTrace), each (x, y, tsize).
    """
    lowpass = params.get("lowpass", 0)
    gsmooth = params.get("gsmooth", 0)
    nSV = U.shape[2]
    Fs = (1 / np.median(np.diff(t))) * (1 / rate)
    x, y = U.shape[0], U.shape[1]

    if mimg is not None:
        U = U / mimg
    Ur = U.reshape(-1, nSV, order="F")  # MATLAB reshape(U, x*y, nSV)
    meanTrace = np.asarray(Ur @ dV, dtype=float)
    tsize = meanTrace.shape[1]

    if gsmooth:
        mt = meanTrace.reshape(x, y, tsize, order="F")
        for kk in range(tsize):
            mt[:, :, kk] = medfilt2d(mt[:, :, kk], 20)
        meanTrace = mt.reshape(-1, tsize, order="F")

    if rate != 1:
        n = int(np.floor((tsize - 1) / rate)) + 1
        tq = 1 + np.arange(n) * rate  # MATLAB 1:rate:tsize
        meanTrace = interp1d(np.arange(1, tsize + 1), meanTrace, axis=1, kind="linear")(tq)
        tsize = n

    meanTrace = meanTrace - meanTrace.mean(axis=1, keepdims=True)
    rawTrace = meanTrace

    if lowpass:
        f1, f2 = butter(2, 1 / (Fs / 2), btype="low")
    else:
        f1, f2 = butter(2, np.array([2.0, 8.0]) / (Fs / 2), btype="bandpass")
    meanTrace = filtfilt(f1, f2, meanTrace, axis=1)

    traceHilbert = hilbert(meanTrace, axis=1)
    tracePhase = np.angle(traceHilbert).reshape(x, y, tsize, order="F")
    traceAmp = np.abs(traceHilbert).reshape(x, y, tsize, order="F")
    trace2d = meanTrace.reshape(x, y, tsize, order="F")
    rawTrace = rawTrace.reshape(x, y, tsize, order="F")
    return trace2d, traceAmp, tracePhase, rawTrace


def _phase_colored_segments_pos(qt, trace_filt, trace_phase, yscale, yoffset, cmap):
    """LineCollection of segments colored by instantaneous phase (MATLAB
    colors segment j from a 100-sample colorcet C06 LUT at
    phase/(2*pi)+0.5; the continuous colormap is sampled directly, as in
    plotSpiralTimeSeries3d._phase_colored_segments, but with a positive
    y offset: y = trace_filt*yscale + yoffset)."""
    color1 = trace_phase / (2 * np.pi) + 0.5
    pts = np.stack([qt, trace_filt * yscale + yoffset], axis=-1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, linewidths=2)
    lc.set_color(cmap(np.clip(color1[:-1], 0, 1))[:, :3])
    return lc


def _select_area(sensoryArea, spath, st, coords, Utransformed, projectedAtlas1, projectedTemplate1, hemi, scale):
    """Translated from spirals/utils/select_area.m

    sensoryArea: iterable of structure_id_path prefixes (MATLAB
    startsWith any-of semantics). Returns (index, Uselected), where index
    holds 1-based, column-major (order="F") indices into the downsampled
    atlas, matching MATLAB find(projectedAtlas2)."""
    spath3 = spath.str.startswith(tuple(sensoryArea))
    idFilt1 = np.flatnonzero(spath3.to_numpy()) + 1  # MATLAB st.index (1-based rows)
    projectedAtlas1 = projectedAtlas1.copy()
    projectedTemplate1 = projectedTemplate1.copy()
    Lia1 = np.isin(projectedAtlas1, idFilt1)
    projectedAtlas1[~Lia1] = 0
    projectedTemplate1[~Lia1] = 0

    if hemi == "right":
        projectedAtlas1[:, : projectedAtlas1.shape[1] // 2] = 0
        projectedTemplate1[:, : projectedTemplate1.shape[1] // 2] = 0
    elif hemi == "left":
        projectedAtlas1[:, projectedAtlas1.shape[1] // 2 :] = 0
        projectedTemplate1[:, projectedTemplate1.shape[1] // 2 :] = 0

    projectedAtlas2 = projectedAtlas1[::scale, ::scale]
    index = np.flatnonzero(projectedAtlas2.ravel(order="F")) + 1
    UregDown2d = Utransformed[::scale, ::scale, ...]  # MATLAB allows trailing :
    n3 = UregDown2d.shape[2] if UregDown2d.ndim == 3 else 1  # MATLAB size(U,3)
    UregDown1d = UregDown2d.reshape(-1, n3, order="F")
    Uselected = UregDown1d[index - 1, :]
    return index, Uselected


def _fft_spectrum(trace):
    """Translated from revision/power_spectrum/fft_spectrum.m

    trace: ntrace x timestamps. Returns (freq1, psdx, psdx_mean) with psdx
    shaped (ntrace, nf) (MATLAB returns nf x ntrace)."""
    trace = np.asarray(trace, dtype=float)
    alpha_trace_mean = trace - trace.mean(axis=1, keepdims=True)
    N = alpha_trace_mean.shape[1]
    xdft = np.fft.fft(alpha_trace_mean, axis=1)
    Fs = 35.0
    xdft1 = xdft[:, : N // 2 + 1]  # MATLAB xdft(1:N/2+1,:)
    psdx = (1 / (Fs * N)) * np.abs(xdft1) ** 2
    psdx[:, 1:-1] = 2 * psdx[:, 1:-1]
    freq1 = np.arange(0, Fs / 2 + Fs / N / 2, Fs / N)  # MATLAB 0:Fs/N:Fs/2
    psdx_mean = psdx.mean(axis=0)
    return freq1, psdx, psdx_mean


def _exampleSpiralSpectrumCore(T, data_folder, save_folder, kk, first_frame, frame_anchor, out_name):
    """Shared core of plotExampleSpiralSpectrum2.m / plotExampleSpiralSpectrum3.m:
    power spectrum of the raw left-hemisphere cortex traces around the example
    spiral epoch, averaged over SSp/VIS/AUD/RSP pixels."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # left SSp (sensory areas within the isocortex)
    _, st = _get_cortex_atlas_path(data_folder)
    spath = st["structure_id_path"].astype(str)
    areaPath = [
        "/997/8/567/688/695/315/453/",  # SS
        "/997/8/567/688/695/315/247/",  # AUD
        "/997/8/567/688/695/315/669/",  # VIS
        "/997/8/567/688/695/315/254/",  # RSP
        "/997/8/567/688/695/315/22/312782546/",  # VISa
        "/997/8/567/688/695/315/22/417/",  # VISrl
        "/997/8/567/688/695/315/541/",  # TEa
        "/997/8/567/688/695/315/677/",  # VISC
        "/997/8/567/688/695/1089/",  # HPF
        "/997/8/567/688/695/315/677/",  # CTXsp
    ]
    sensoryArea = areaPath  # MATLAB strcat(areaPath(:)) keeps the string array

    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    with h5py.File(outline_file, "r") as f:
        projectedAtlas1 = np.asarray(f["projectedAtlas1"]).T
        projectedTemplate1 = np.asarray(f["projectedTemplate1"]).T

    scale = 8
    Utransformed = np.zeros(projectedAtlas1.shape)
    hemi = "left"
    indexSSp, _ = _select_area(
        sensoryArea, spath, st, None, Utransformed,
        projectedAtlas1, projectedTemplate1, hemi, scale,
    )
    BW_empty = np.zeros(projectedAtlas1[::scale, ::scale].shape)
    BW_SSp = BW_empty.copy()
    rr, cc = np.unravel_index(indexSSp - 1, BW_SSp.shape, order="F")
    BW_SSp[rr, cc] = 1

    mn, tdb, en, fname, U, V, t, mimg, dV = _load_session_row(data_folder, T, kk)
    T_tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")

    params = {"downscale": 8, "lowpass": 0, "gsmooth": 0}
    rate2 = 1
    frameStart = frame_anchor - first_frame
    frameEnd = frameStart + 70
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)  # extra 2*35 frames

    Utransformed_d = _imwarp_row(U, T_tform, projectedAtlas1.shape, stride=8)
    mimgtransformed = _imwarp_row(mimg, T_tform, projectedAtlas1.shape, stride=8)

    Ur = Utransformed_d.reshape(-1, Utransformed_d.shape[2], order="F")
    V1 = V[:, frameTemp - 1]
    rawTrace2 = Ur @ V1
    rawTrace2 = rawTrace2 - rawTrace2.mean(axis=1, keepdims=True)
    rawTrace2 = rawTrace2.reshape(Utransformed_d.shape[0], Utransformed_d.shape[1], -1, order="F")
    pad = int(35 / rate2)
    rawTrace2 = rawTrace2[:, :, pad:-pad]
    with np.errstate(divide="ignore", invalid="ignore"):
        rawTrace2 = rawTrace2 / mimgtransformed[:, :, None]
    trace_raw3 = rawTrace2.reshape(-1, rawTrace2.shape[2], order="F")

    trace_raw4 = trace_raw3[BW_SSp.ravel(order="F").astype(bool), :]
    index1 = ~np.isnan(trace_raw4[:, 0])
    trace_raw4 = trace_raw4[index1, :]

    freq1, psdx, _ = _fft_spectrum(trace_raw4)
    psdx_mean = psdx.mean(axis=0)

    freq_value = [0.2, 0.5, 1, 2, 4, 6, 8, 10]
    log_freq_value = np.log10(freq_value)
    freqN = psdx_mean.size

    hs = plt.figure(figsize=(3, 6))
    ax = hs.add_subplot(1, 1, 1)
    ax.plot(np.log10(freq1[1:freqN]), np.log10(psdx_mean[1:freqN]), color="k")
    ax.set_xlim(log_freq_value[0], log_freq_value[-1])
    ax.set_xticks(log_freq_value)
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "4", "6", "8", "10"])
    ax.set_xlabel("log10(Frequency)")
    ax.set_ylabel("log10(Power) (df/f^2)")
    ax.set_ylim(-7, -3)
    hs.savefig(save_folder / out_name, dpi=300)
    plt.show()
    return hs


def _load_roi_vertices(path):
    """Recover the polygon vertices of a polyshape `roi` saved in a MATLAB
    v7.3 .mat (e.g. spirals/full_roi/*_roi.mat). The vertices are stored as
    a 2 x N float dataset under #refs# (row 0 = x, row 1 = y)."""
    with h5py.File(path, "r") as f:
        candidates = []

        def _visit(name, obj):
            if (
                isinstance(obj, h5py.Dataset)
                and obj.ndim == 2
                and obj.shape[0] == 2
                and obj.dtype.kind == "f"
                and obj.shape[1] > 2
            ):
                candidates.append(name)

        f["#refs#"].visititems(_visit)
        if not candidates:
            raise ValueError(f"No polygon vertices found in {path}")
        return np.asarray(f["#refs#"][candidates[0]]).T  # (N, 2)


def _inROI(roi_vertices, x, y):
    """MATLAB inROI(roi, x, y) for a polyshape roi: logical index of the
    points (x, y) inside the ROI polygon."""
    path = MplPath(np.asarray(roi_vertices, dtype=float))
    pts = np.column_stack(
        [np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()]
    )
    return path.contains_points(pts)


def _setSpiralDetectionParams(U, t):
    """Translated from spirals/utils/spirals_detection/setSpiralDetectionParams.m"""
    params = {}
    params["downscale"] = 1
    params["lowpass"] = 0
    params["Fs"] = 35
    params["halfpadding"] = 120
    params["padding"] = 2 * params["halfpadding"]
    params["th"] = np.arange(1, 361, 36)
    params["rs"] = np.arange(10, 21, 5)
    params["gridsize"] = 10
    params["spiralRange"] = np.linspace(-np.pi, np.pi, 5)
    params["gsmooth"] = 0
    params["epochL"] = 100
    params["nt"] = np.size(t)
    params["frameN1"] = int(_matlab_round((params["nt"] - 70) / 100) * 100)
    params["frameRange"] = np.arange(36, 36 + params["frameN1"], params["epochL"])
    params["dThreshold"] = 15
    params["rsRCheck"] = np.arange(10, 101, 10)
    params["pgridx"], params["pgridy"] = np.meshgrid(
        np.arange(21 - 10, 21 + 10 + 1), np.arange(21 - 10, 21 + 10 + 1)
    )
    # generate coarse search grids with zeros padded at the edges
    U1 = U[:: params["downscale"], :: params["downscale"], :50]
    params["xsize"] = U1.shape[0]
    params["ysize"] = U1.shape[1]
    xsizePadded = params["xsize"] + params["padding"]
    ysizePadded = params["ysize"] + params["padding"]
    xx, yy = np.meshgrid(
        np.arange(np.min(params["rs"]) + 1, xsizePadded - np.min(params["rs"]) - 1 + 1, params["gridsize"]),
        np.arange(np.min(params["rs"]) + 1, ysizePadded - np.min(params["rs"]) - 1 + 1, params["gridsize"]),
    )
    params["xx"] = xx
    params["yy"] = yy
    return params


def _angdiff(ph):
    """MATLAB angdiff(ph): consecutive differences wrapped to [-pi, pi]."""
    return np.angle(np.exp(1j * np.diff(ph)))


def _histcounts_all(ph, edges):
    """MATLAB [N, edges] = histcounts(ph, edges); returns True if every bin
    is non-empty (MATLAB all(N))."""
    N, _ = np.histogram(ph, bins=edges)
    return bool(np.all(N > 0))


def _unwrap_phases(ph):
    """ph2 from MATLAB: ph2(1)=ph(1); ph2(i)=ph2(i-1)+angdiff(ph)(i-1)."""
    phdiff = _angdiff(ph)
    return np.concatenate([[ph[0]], ph[0] + np.cumsum(phdiff)])


def _checkSpiral(A, px, py, r, th, spiralRange):
    """Translated from spirals/utils/spirals_detection/checkSpiral.m"""
    cx = _matlab_round(r * np.cos(np.deg2rad(th)) + px).astype(int)
    cy = _matlab_round(r * np.sin(np.deg2rad(th)) + py).astype(int)
    ph = A[cy - 1, cx - 1]  # MATLAB sub2ind (1-based)
    ph2 = _unwrap_phases(ph)
    ph3 = np.abs(ph2 - ph2[0])
    allN = _histcounts_all(ph, spiralRange)
    AngleRange = np.abs(ph2[-1] - ph2[0])
    spiralTemp = np.array([px, py, r, 0], dtype=float)
    if AngleRange > 5 and AngleRange < 7 and allN:
        spiralTemp[3] = 1
    return spiralTemp


def _spiralAlgorithm(tracePhase, params):
    """Translated from spirals/utils/spirals_detection/spiralAlgorithm.m"""
    xxRoi = params["xxRoi"]
    yyRoi = params["yyRoi"]
    rs = params["rs"]
    th = params["th"]
    spiralRange = params["spiralRange"]
    pwAll1 = np.zeros((0, 2))
    for ixy in range(np.size(xxRoi)):
        px = xxRoi[ixy]
        py = yyRoi[ixy]
        pw = np.zeros((0, 4))
        for rn in range(rs.size):
            pw = np.vstack(
                [pw, _checkSpiral(tracePhase, px, py, rs[rn], th, spiralRange)[None, :]]
            )
        # if 2 out of 3 radius satisfy the criteria, then a candidate spiral center
        if pw[:, 3].sum() >= 2:
            pwAll1 = np.vstack([pwAll1, [[px, py]]])
    return pwAll1


def _padZeros(tracePhase1, halfpadding):
    """Translated from spirals/utils/spirals_detection/padZeros.m"""
    padding = 2 * halfpadding
    xsize = tracePhase1.shape[0]
    ysize = tracePhase1.shape[1]
    out_shape = (xsize + padding, ysize + padding) + tracePhase1.shape[2:]
    tracePhase = np.zeros(out_shape, dtype=tracePhase1.dtype)
    tracePhase[
        halfpadding : halfpadding + xsize, halfpadding : halfpadding + ysize, ...
    ] = tracePhase1
    return tracePhase


def _clusterXYpoints(XY, maxdist, minClusterSize=None, method="point", mergeflag=None):
    """Translated from spirals/utils/spirals_detection/clusterXYpoints.m
    (returns clustersXY only, as a list of (n, 2) arrays of point
    coordinates; points are handled 1-based internally to reproduce the
    MATLAB pairing/merge order)."""
    XY = np.asarray(XY, dtype=float)
    if minClusterSize is None:
        minClusterSize = 1
    kmax = XY.shape[0]

    # pairwise distances in MATLAB pairs order: (k..kmax, k-1) for k = 2..kmax
    pairs = []
    for k in range(2, kmax + 1):
        for a in range(k, kmax + 1):
            pairs.append((a, k - 1))
    D = [np.linalg.norm(XY[a - 1] - XY[b - 1]) for a, b in pairs]
    order = np.argsort(D, kind="stable")

    clusters = {}  # id -> list of 1-based point indices
    point_cluster = {}  # 1-based point -> cluster id
    inc = 0
    for oi in order:
        pt1, pt2 = pairs[oi]
        d = D[oi]
        if d > maxdist:
            break
        if pt1 not in point_cluster and pt2 not in point_cluster:
            inc += 1
            clusters[inc] = [pt1, pt2]
            point_cluster[pt1] = inc
            point_cluster[pt2] = inc
        elif pt1 not in point_cluster:
            id_ = point_cluster[pt2]
            # method 'point': cdist empty -> always add
            clusters[id_].append(pt1)
            point_cluster[pt1] = id_
        elif pt2 not in point_cluster:
            id_ = point_cluster[pt1]
            clusters[id_].append(pt2)
            point_cluster[pt2] = id_
        else:
            if mergeflag is not None and mergeflag.lower() == "merge":
                id1 = point_cluster[pt1]
                id2 = point_cluster[pt2]
                if id1 != id2:
                    clusters[id1].extend(clusters[id2])
                    for p in clusters[id2]:
                        point_cluster[p] = id1
                    clusters[id2] = []

    clusters = {k: v for k, v in clusters.items() if len(v) >= minClusterSize}
    points_in = set()
    for v in clusters.values():
        points_in.update(v)
    if minClusterSize < 2:  # convert single points into 'one-point' clusters
        nid = max(clusters.keys(), default=0)
        for p in range(1, kmax + 1):
            if p not in points_in:
                nid += 1
                clusters[nid] = [p]
    clustersXY = [XY[np.array(v) - 1, :] for v in clusters.values()]
    return clustersXY


def _checkClusterXY(pwAll1, dThreshold):
    """Translated from spirals/utils/spirals_detection/checkClusterXY.m"""
    clustersCentroids = np.zeros((0, 2))
    if pwAll1 is not None and pwAll1.shape[0]:
        clustersXY1 = _clusterXYpoints(pwAll1[:, 0:2], dThreshold, None, "point", "merge")
        # true spirals are detected multiple times in neighboring grids
        clustersXY1 = [c for c in clustersXY1 if c.shape[0] > 1]
        rows = [_matlab_round(c.mean(axis=0)) for c in clustersXY1]
        if rows:
            clustersCentroids = np.vstack(rows)
    return clustersCentroids


def _doubleCheckSpiralsAlgorithm(tracePhase, pwframe, params):
    """Translated from
    spirals/utils/spirals_detection/doubleCheckSpiralsAlgorithm.m"""
    rs = params["rs"]
    th = params["th"]
    spiralRange = params["spiralRange"]
    pwAll = np.zeros((0, 2))
    for ixy in range(pwframe.shape[0]):
        px = pwframe[ixy, 0]
        py = pwframe[ixy, 1]
        pw = np.zeros((0, 4))
        for rn in range(rs.size):
            pw = np.vstack(
                [pw, _checkSpiral(tracePhase, px, py, rs[rn], th, spiralRange)[None, :]]
            )
        # if 2 out of 3 radius satisfy the criteria, then a candidate spiral center
        if pw[:, 3].sum() >= 2:
            pwAll = np.vstack([pwAll, [[px, py]]])
    return pwAll


def _spatialRefine(tracePhase, pwframe, params):
    """Translated from spirals/utils/spirals_detection/spatialRefine.m"""
    pgridx = params["pgridx"]
    pgridy = params["pgridy"]
    spiralRange = params["spiralRange"]
    th = params["th"]
    pgridx_flat = pgridx.ravel(order="F")
    pgridy_flat = pgridy.ravel(order="F")
    spiralCT1 = np.zeros((0, 2))
    for ixy in range(pwframe.shape[0]):
        px = int(pwframe[ixy, 0])
        py = int(pwframe[ixy, 1])
        roi1 = tracePhase[py - 21 : py + 20, px - 21 : px + 20]  # (py-20):(py+20), 1-based
        # search for spirals within 20x20 pixels fine grid;
        # only need to conform by a single radius
        rs1 = [2]
        spiralD = np.zeros(pgridx.size)
        for kkk in range(pgridx.size):
            px1 = pgridx_flat[kkk]
            py1 = pgridy_flat[kkk]
            ii = 1
            while spiralD[kkk] == 0 and ii < 2:
                r = rs1[ii - 1]
                cx = _matlab_round(r * np.cos(np.deg2rad(th)) + px1).astype(int)
                cy = _matlab_round(r * np.sin(np.deg2rad(th)) + py1).astype(int)
                ph = roi1[cy - 1, cx - 1]
                ph2 = _unwrap_phases(ph)
                ph3 = np.abs(ph2 - ph2[0])
                allN = _histcounts_all(ph, spiralRange)
                AngleRange = ph3[-1]
                if AngleRange > 5 and AngleRange < 7 and allN:
                    spiralD[kkk] = 1
                ii += 1
        pgridx1 = pgridx_flat[spiralD.astype(bool)]
        pgridy1 = pgridy_flat[spiralD.astype(bool)]
        if spiralD.sum() > 0:
            spiralCT = _matlab_round(
                np.mean(np.column_stack([pgridx1, pgridy1]), axis=0)
            )
            spiralCT1 = np.vstack(
                [spiralCT1, [[px - 21 + spiralCT[0], py - 21 + spiralCT[1]]]]
            )
    return spiralCT1


def _spiralRadiusCheck2(tracePhase, pwframe, params):
    """Translated from spirals/utils/spirals_detection/spiralRadiusCheck2.m"""
    th = params["th"]
    spiralRange = params["spiralRange"]
    rsRCheck = params["rsRCheck"]
    pwAll5 = np.zeros((0, 4))
    for ixy in range(pwframe.shape[0]):
        px = pwframe[ixy, 0]
        py = pwframe[ixy, 1]
        spiralR = np.zeros((rsRCheck.size, 2))
        phAll = np.zeros((rsRCheck.size, th.size))
        phdiff = None  # MATLAB keeps the previous iteration's phdiff on NaN
        for ii in range(rsRCheck.size):
            r = rsRCheck[ii]
            cx = _matlab_round(r * np.cos(np.deg2rad(th)) + px).astype(int)
            cy = _matlab_round(r * np.sin(np.deg2rad(th)) + py).astype(int)
            ph = tracePhase[cy - 1, cx - 1].astype(float)
            if not np.isnan(ph).any():
                phdiff = _angdiff(ph)
            else:
                indx1 = np.isnan(ph)
                if ii == 0:
                    ph[indx1] = 0.0  # MATLAB phAll(ii-1,...) is undefined here
                else:
                    ph[indx1] = phAll[ii - 1, indx1]
            phAll[ii, :] = ph
            ph2 = _unwrap_phases(ph)
            ph2 = ph2 - ph2[0]
            ph3 = np.abs(ph2)
            allN = _histcounts_all(ph, spiralRange)
            AngleRange = ph3[-1]
            if AngleRange > 4 and AngleRange < 8 and allN:
                spiralR[ii, 0] = 1
                if ph2[-1] > 0:
                    spiralR[ii, 1] = 1  # counterclockwise
                else:
                    spiralR[ii, 1] = -1
        if spiralR[:, 0].sum() > 0:
            indxZ = int(np.flatnonzero(spiralR[:, 0])[0])
            spiralR1 = spiralR.copy()
            if indxZ > 0:
                spiralR1[:indxZ, 1] = spiralR1[indxZ, 1]
            # find radius where it changed direction as end of spiral,
            # otherwise is max radius of 100
            diff_idx = np.flatnonzero(spiralR1[:, 1] != spiralR1[indxZ, 1])
            if diff_idx.size:
                maxR = rsRCheck[diff_idx[0] - 1]
            else:
                maxR = rsRCheck[-1]
            # direction is set as the direction at the smallest radius
            direction = spiralR[indxZ, 1]
            pwAll5 = np.vstack([pwAll5, [[px, py, maxR, direction]]])
    return pwAll5
