"""Translated from spirals/plots/plotSpiralTimeSeries3d.m

Plots an example spiral epoch for session ZYE_0012/2020-10-16/5: mean image
with 7 equidistant pixels around a spiral center (panel a), and raw +
band-passed (2-8 Hz) phase-colored time series at those pixels (panels c).

Shared private helpers used by the other Figure-1 modules live here:
_get_cortex_atlas_path, _load_atlas_data, _load_tform, _imwarp,
_load_example_session, _matlab_round, _cbrewer, _phase_colored_segments.
"""

from pathlib import Path

import colorcet as cc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import ndimage
from scipy.signal import find_peaks

from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.atlas import overlayOutlines, plotOutline
from spirals_py.utils.io import load_outline_coords_h5


def _matlab_round(a):
    """MATLAB round(): half away from zero (np.round is half-to-even)."""
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _cbrewer(_type, name, n):
    """Approximates dependencies/cbrewer2: samples the same ColorBrewer map."""
    return plt.get_cmap(name)(np.linspace(0, 1, n))[:, :3]


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m"""
    data_folder = Path(data_folder)
    st = pd.read_csv(data_folder / "tables" / "structure_tree_safe_2017.csv")
    maskPath = [
        "/997/8/567/688/695/315/500/985/",  # MOp
        "/997/8/567/688/695/315/500/993/",  # MOs
        "/997/8/567/688/695/315/31/",       # ACA
        "/997/8/567/688/695/315/453/378/",  # SS2
        "/997/8/567/688/695/315/453/322/",  # SSp
        "/997/8/567/688/695/315/247/",      # AUD
        "/997/8/567/688/695/315/669/",      # VIS
        "/997/8/567/688/695/315/254/",      # RSP
        "/997/8/567/688/695/315/22",        # VISa
        "/997/8/567/688/695/315/541/",      # TEa
        "/997/8/567/688/695/315/677/",      # VISC
    ]
    return maskPath, st


def _load_atlas_data(data_folder):
    """Load atlas1 (50 um), outline coords and projectedAtlas1 (10 um).

    Sources: tables/horizontal_cortex_atlas_50um.mat and
    tables/isocortex_horizontal_projection_outline.mat (v7.3 -> h5py,
    transposed back to MATLAB orientation).
    """
    data_folder = Path(data_folder)
    with h5py.File(data_folder / "tables" / "horizontal_cortex_atlas_50um.mat", "r") as f:
        atlas1 = np.asarray(f["atlas1"]).T
    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    coords = load_outline_coords_h5(outline_file)
    with h5py.File(outline_file, "r") as f:
        projectedAtlas1 = np.asarray(f["projectedAtlas1"]).T
    return atlas1, coords, projectedAtlas1


def _load_tform(path):
    """Load a v7.3 affine2d object; returns its 3x3 TransformationMatrix T in
    MATLAB orientation, with [u v 1] = [x y 1] * T (transformPointsForward)."""
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


def _imwarp(img, T, out_shape, stride=1):
    """MATLAB imwarp(img, tform, 'OutputView', imref2d(out_shape)), bilinear,
    FillValues=0. stride>1 samples the output grid at 1:stride:end, matching
    imwarp followed by (1:stride:end, 1:stride:end) slicing. img may be 2-D or
    3-D (each 2-D plane is warped, as imwarp does for N-D input)."""
    Tinv = np.linalg.inv(T)
    rows = np.arange(0, out_shape[0], stride) + 1  # 1-based output y centers
    cols = np.arange(0, out_shape[1], stride) + 1  # 1-based output x centers
    X, Y = np.meshgrid(cols, rows)
    ones = np.ones_like(X)
    uv = np.stack([X, Y, ones], axis=-1) @ Tinv  # (R, C, 3); [X Y 1] @ inv(T)
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


def _load_example_session(data_folder):
    """Load the ZYE_0012/2020-10-16/5 example session and its registration.

    Returns U, V, t, mimg, dV, masks(dict with BW/BW1/BW2), tform T,
    and the session folder name."""
    data_folder = Path(data_folder)
    mn, tdb, en = "ZYE_0012", "20201016", 5
    subfolder = f"{mn}_{tdb}_{en}"
    session_root = data_folder / "spirals" / "svd" / subfolder
    U, V, t, mimg = loadUVt1(session_root)
    dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
    masks = {}
    with h5py.File(data_folder / "tables" / "mask_ZYE12.mat", "r") as f:
        for key in ("BW", "BW1", "BW2"):
            masks[key] = np.asarray(f[key]).T.astype(bool)
    T = _load_tform(data_folder / "tables" / f"{subfolder}_tform.mat")
    return U, V, t, mimg, dV, masks, T, subfolder


def _phase_colored_segments(qt, trace_filt, trace_phase, yoffset, yscale, cmap):
    """Build LineCollection segments colored by instantaneous phase.

    MATLAB colors segment j by interp1 over a 100-sample colorcet C06 LUT at
    color1 = phase/(2*pi)+0.5; here the continuous colormap is sampled
    directly at color1 (equivalent up to LUT resolution)."""
    color1 = trace_phase / (2 * np.pi) + 0.5  # from [-0.5,0.5] to [0,1]
    pts = np.stack([qt, trace_filt * yscale - yoffset], axis=-1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, linewidths=2)
    lc.set_color(cmap(np.clip(color1[:-1], 0, 1))[:, :3])
    return lc


def plotSpiralTimeSeries3d(data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralTimeSeries3d.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1, coords, projectedAtlas1 = _load_atlas_data(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    U, V, t, mimg, dV, masks, T, subfolder = _load_example_session(data_folder)
    BW = masks["BW"]

    downscale = 8
    params = {"downscale": downscale, "lowpass": 0, "gsmooth": 0}
    rate = 0.1

    out_shape = projectedAtlas1.shape  # (1320, 1140)
    Utransformed = _imwarp(U, T, out_shape, stride=downscale)
    mimgtransformed = _imwarp(mimg, T, out_shape, stride=downscale)
    with np.errstate(divide="ignore", invalid="ignore"):
        mimgtransformedRBG = mimgtransformed / mimgtransformed.max()
    mimgtransformedRGB = np.stack([mimgtransformedRBG] * 3, axis=-1)

    freq = [2, 8]
    tStart, tEnd = 1681, 1683  # find spirals between tStart:tEnd
    frameStart = int(np.argmax(t > tStart)) + 1  # MATLAB 1-based
    frameEnd = int(np.argmax(t > tEnd)) + 1
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)  # 1-based, inclusive
    dV1 = dV[:, frameTemp - 1]
    trace2d1, traceAmp1, tracePhase1 = spiralPhaseMap_freq(
        Utransformed, dV1, t, params, freq, rate
    )
    pad = int(35 / rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        trace2d1 = trace2d1[:, :, pad:-pad] / mimgtransformed[:, :, None]
    tracePhase1 = tracePhase1[:, :, pad:-pad]

    Ur = Utransformed.reshape(-1, Utransformed.shape[2], order="F")
    rawTrace1 = Ur @ dV1
    rawTrace1 = rawTrace1 - rawTrace1.mean(axis=1, keepdims=True)
    tsize = dV1.shape[1]
    n = int(np.floor((tsize - 1) / rate)) + 1
    tq = 1 + np.arange(n) * rate
    from scipy.interpolate import interp1d

    rawTrace1 = interp1d(np.arange(1, tsize + 1), rawTrace1, axis=1, kind="linear")(tq)
    rawTrace1 = rawTrace1.reshape(Utransformed.shape[0], Utransformed.shape[1], n, order="F")
    with np.errstate(divide="ignore", invalid="ignore"):
        rawTrace = rawTrace1[:, :, pad:-pad] / mimgtransformed[:, :, None]

    # 7 equidistant pixels on a circle around the spiral center (1-based)
    center = np.array([75, 111])  # (row, col) in downscaled image
    r = 15
    th2 = np.linspace(0, 360, 8) + 294
    th2 = th2[:-1]
    px1, py1 = center[1], center[0]
    cx2 = _matlab_round(r * np.cos(np.deg2rad(th2)) + px1)
    cy2 = _matlab_round(r * np.sin(np.deg2rad(th2)) + py1)
    pixel = np.stack([cy2, cx2], axis=1).astype(int)  # (row, col), 1-based

    a, b = 60, 97  # height
    c, d = 92, 129  # width
    v1 = np.array([[c, a], [d, a], [d, b], [c, b]]) - 1  # 0-based for matplotlib

    nFrames = trace2d1.shape[2]
    trace_filt = np.zeros((pixel.shape[0], nFrames))
    trace_raw = np.zeros((pixel.shape[0], nFrames))
    trace_phase = np.zeros((pixel.shape[0], nFrames))
    for i in range(pixel.shape[0]):
        ri, ci = pixel[i, 0] - 1, pixel[i, 1] - 1
        trace_raw[i] = rawTrace[ri, ci, :]
        trace_filt[i] = trace2d1[ri, ci, :]
        trace_phase[i] = tracePhase1[ri, ci, :]

    t1 = t[frameStart - 1 : frameEnd]
    nq = int(np.floor((t1.size - 1) / rate)) + 1
    tq1 = 1 + np.arange(nq) * rate
    qt = np.interp(tq1, np.arange(1, t1.size + 1), t1)

    scale3 = 5 / 8
    color2 = _cbrewer("seq", "YlOrRd", pixel.shape[0] + 2)[2:][::-1]

    frames_to_plot = 9
    first_frame = 22
    last_frame = first_frame + frames_to_plot
    t1a, t1b = t1[first_frame - 1], t1[last_frame - 1]
    first_frame1 = int(np.argmax(qt - t1a > 0)) + 1  # MATLAB 1-based
    last_frame1 = int(np.argmax(qt - t1b > 0)) + 1
    frame0 = first_frame1 - 30

    cmap_c06 = cc.cm["CET_C6"]

    h1ac = plt.figure(figsize=(8, 4))
    # MATLAB subplot(1,4,1) / subplot(1,4,[2,3]) / subplot(1,4,4): the middle
    # traces panel spans half the figure width (gridspec replaces manual
    # set_position, which tight_layout would override).
    gs = h1ac.add_gridspec(1, 3, width_ratios=[1, 2, 1], wspace=0.3)

    ax1 = h1ac.add_subplot(gs[0, 0])
    ax1.imshow(mimgtransformedRGB, alpha=BW.astype(float))
    overlayOutlines(coords, downscale, ax=ax1)
    plotOutline(maskPath[:11], st, atlas1, [], scale3, "k", ax=ax1)
    ax1.scatter(pixel[:, 1] - 1, pixel[:, 0] - 1, s=16, c=color2)
    ax1.scatter(center[1] - 1, center[0] - 1, s=16, c="k")
    ax1.add_patch(
        plt.Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=2)
    )

    for col, (xvals, xlim) in enumerate(
        [(qt, (t1[0], t1[-1])), (qt - qt[first_frame1 - 1], None)], start=1
    ):
        if col == 1:
            ax = h1ac.add_subplot(gs[0, 1])
        else:
            ax = h1ac.add_subplot(gs[0, 2])
        yscale = 150
        for i in range(pixel.shape[0]):
            yoff = 3 * i
            ax.plot(xvals, trace_raw[i] * yscale - yoff, "k", linewidth=1)
            ax.add_collection(
                _phase_colored_segments(
                    xvals, trace_filt[i], trace_phase[i], yoff, yscale, cmap_c06
                )
            )
            win = trace_filt[i, frame0 - 1 : last_frame1 + 10]
            locs, _ = find_peaks(win, prominence=0.001)
            ax.scatter(
                xvals[frame0 + locs],
                win[locs] * yscale - yoff + 0.2,
                s=12,
                c="k",
            )
        ax.tick_params(labelsize=8)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            qta = xvals
            ax.set_xlim(qta[frame0 - 1], qta[last_frame1 + 10 - 1])
        ax.set_yticklabels([])
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.axvline(xvals[first_frame1 - 1], ls="--")
        ax.axvline(xvals[last_frame1 - 1], ls="--")
        if col == 1:
            ax.plot([t1[0] + 0.5, t1[0] + 0.5], [0, 1.5], "r")
            for i in range(pixel.shape[0]):
                ax.scatter(xvals[0], -3 * i, s=16, c=[color2[i]])

    h1ac.savefig(save_folder / "Fig1ac_overlay8.pdf", bbox_inches="tight")
    plt.show()
    return h1ac
