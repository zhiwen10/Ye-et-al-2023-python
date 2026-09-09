"""Translated from spirals/plots/plotExampleSpiral2b.m (Extended Data Fig.1g-h).

Plots the LK_0003 example spirals: a 4 x 10 sequence of dF/F and phase
frames (two full-cortex rows and two zoomed rows with detected spirals
overlaid, FigS1h), and the mean image with 7 region pixels plus raw +
phase-colored band-passed time series at those pixels (FigS1g).
"""

from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _imwarp_row,
    _load_session_masks,
    _load_session_row,
    _load_spirals_group_fftn,
    _load_tform,
    _phase_colored_segments_pos,
    _spiralPhaseMap5,
)
from spirals_py.spirals.plots.plotExampleOscillation import _cbrewer2
from spirals_py.spirals.plots.plotSpiralTimeSeries3d import (
    _get_cortex_atlas_path,
    _load_atlas_data,
    _matlab_round,
)
from spirals_py.utils.atlas import overlayOutlines, plotOutline


def plotExampleSpiral2b(T, data_folder, save_folder):
    """Translated from spirals/plots/plotExampleSpiral2b.m; returns (hs1g, hs1h)."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # load atlas brain horizontal projection and outline
    atlas1, coords, projectedAtlas1 = _load_atlas_data(data_folder)
    BW = projectedAtlas1 != 0
    maskPath, st = _get_cortex_atlas_path(data_folder)

    kk = 11  # LK_0003
    mn, tdb, en, fname, U, V, t, mimg, dV = _load_session_row(data_folder, T, kk)
    T_tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
    spirals = _load_spirals_group_fftn(data_folder, fname, T_tform)
    masks = _load_session_masks(data_folder, fname)
    BW2 = masks["BW2"]

    pixSize = 3.45 / 1000 / 0.6 * 3  # mm / pix for spiral radius

    params = {"downscale": 8, "lowpass": 0, "gsmooth": 0}
    rate = 1
    first_frame = 34
    frameStart = 70885 - first_frame
    frameEnd = frameStart + 70
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)  # extra 2*35 frames
    dV1 = dV[:, frameTemp - 1]

    out_shape = projectedAtlas1.shape
    Utransformed1 = _imwarp_row(U, T_tform, out_shape, stride=params["downscale"])
    mimgtransformed = _imwarp_row(mimg, T_tform, out_shape, stride=params["downscale"])
    with np.errstate(divide="ignore", invalid="ignore"):
        mimgtransformedRBG = mimgtransformed / mimgtransformed.max()
    mimgtransformedRGB = np.stack([mimgtransformedRBG] * 3, axis=-1)

    trace2d1, traceAmp1, tracePhase1, rawTrace1 = _spiralPhaseMap5(
        Utransformed1, dV1, t, params, rate
    )
    pad = int(35 / rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        trace2d1 = trace2d1[:, :, pad:-pad] / mimgtransformed[:, :, None]
        rawTrace = rawTrace1[:, :, pad:-pad] / mimgtransformed[:, :, None]
    dff = trace2d1 * 100
    tracePhase1 = tracePhase1[:, :, pad:-pad]

    # 7 equidistant pixels on a circle around the spiral center (1-based)
    center = np.array([69, 33])
    r = 12
    th2 = np.linspace(0, 360, 8) + 294
    th2 = th2[:-1]
    px1, py1 = center[1], center[0]
    cx2 = _matlab_round(r * np.cos(np.deg2rad(th2)) + px1)
    cy2 = _matlab_round(r * np.sin(np.deg2rad(th2)) + py1)
    pixel = np.stack([cy2, cx2], axis=1).astype(int)  # (row, col), 1-based

    a, b = 35, 85  # height
    c, d = 20, 70  # width
    v1 = np.array([[c, a], [d, a], [d, b], [c, b]]) - 1  # 0-based for matplotlib

    lineColor = "w"
    hemi = []
    scale3 = 5 / 8

    phase_min, phase_max = -3.14, 3.14
    cmap_c06 = cc.cm["CET_C6"]
    th2c = np.arange(1, 361, 5)

    hs1h, axs = plt.subplots(
        4, 10, figsize=(10, 5), gridspec_kw={"wspace": 0.02, "hspace": 0.08}
    )
    frame_count = 1
    for k in range(1, 10 + 1):
        frame = first_frame + (k - 1)  # MATLAB 1-based
        frame_real = frameStart + first_frame + k - 1
        spiral_temp = spirals[spirals[:, 4] == frame_real, :]

        ax3 = axs[0, k - 1]
        ax3.imshow(
            dff[:, :, frame - 1], cmap="viridis", alpha=BW2.astype(float),
            vmin=-1, vmax=1,
        )
        ax3.set_aspect("equal")
        ax3.axis("off")
        plotOutline(maskPath[:11], st, atlas1, hemi, scale3, lineColor, ax=ax3)
        ax3.set_title(f"frame{frame_count}", fontsize=8)
        ax3.add_patch(plt.Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=1))

        ax4 = axs[1, k - 1]
        ax4.imshow(
            tracePhase1[:, :, frame - 1], cmap=cmap_c06, alpha=BW2.astype(float),
            vmin=phase_min, vmax=phase_max,
        )
        ax4.set_aspect("equal")
        ax4.axis("off")
        plotOutline(maskPath[:11], st, atlas1, hemi, scale3, lineColor, ax=ax4)
        ax4.add_patch(plt.Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=1))

        for row, (data, vmin, vmax, cmap) in enumerate(
            [
                (dff[:, :, frame - 1], -1, 1, "viridis"),
                (tracePhase1[:, :, frame - 1], phase_min, phase_max, cmap_c06),
            ],
            start=2,
        ):
            ax = axs[row, k - 1]
            ax.imshow(data, cmap=cmap, alpha=BW2.astype(float), vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.axis("off")
            plotOutline(maskPath[:11], st, atlas1, hemi, scale3, lineColor, ax=ax)
            for kk2 in range(spiral_temp.shape[0]):
                px1 = spiral_temp[kk2, 0]
                py1 = spiral_temp[kk2, 1]
                r = spiral_temp[kk2, 2] * pixSize / (0.01 * 8)
                cx2 = _matlab_round(r * np.cos(np.deg2rad(th2c)) + px1)
                cy2 = _matlab_round(r * np.sin(np.deg2rad(th2c)) + py1)
                color1 = "w" if spiral_temp[kk2, 3] == 1 else "k"  # CCW white / CW black
                cx2c = np.concatenate([cx2, cx2[:1]]) - 1  # 0-based for matplotlib
                cy2c = np.concatenate([cy2, cy2[:1]]) - 1
                ax.plot(cx2c, cy2c, color=color1, linewidth=1)
                ax.scatter(spiral_temp[kk2, 0] - 1, spiral_temp[kk2, 1] - 1, s=8, c=color1)
            ax.set_ylim(b - 1, a - 1)  # MATLAB ylim([a,b]) with reversed y axis
            ax.set_xlim(c - 1, d - 1)

        frame_count += 1
    frameS = frameStart + first_frame
    hs1h.savefig(save_folder / f"FigS1h_{fname}_{frameS}.png", dpi=300)

    # upsampled pass for the time series figure
    rate1 = 0.1
    trace2d1, traceAmp1, tracePhase1, rawTrace1 = _spiralPhaseMap5(
        Utransformed1, dV1, t, params, rate1
    )
    pad = int(35 / rate1)
    with np.errstate(divide="ignore", invalid="ignore"):
        trace2d1 = trace2d1[:, :, pad:-pad] / mimgtransformed[:, :, None]
        rawTrace = rawTrace1[:, :, pad:-pad] / mimgtransformed[:, :, None]
    dff = trace2d1 * 100
    tracePhase1 = tracePhase1[:, :, pad:-pad]

    nFrames = trace2d1.shape[2]
    trace_filt = np.zeros((7, nFrames))
    trace_raw = np.zeros((7, nFrames))
    trace_phase = np.zeros((7, nFrames))
    for i in range(7):
        trace_raw[i] = rawTrace[pixel[i, 0] - 1, pixel[i, 1] - 1, :]
        trace_filt[i] = trace2d1[pixel[i, 0] - 1, pixel[i, 1] - 1, :]
        trace_phase[i] = tracePhase1[pixel[i, 0] - 1, pixel[i, 1] - 1, :]

    t1 = t[frameStart - 1 : frameEnd]
    nq = int(np.floor((t1.size - 1) / rate1)) + 1
    tq1 = 1 + np.arange(nq) * rate1
    qt = np.interp(tq1, np.arange(1, t1.size + 1), t1)

    color2 = _cbrewer2("seq", "YlOrRd", 9)
    frames_to_plot = 10
    last_frame = first_frame + frames_to_plot
    t1a, t1b = t1[first_frame - 1], t1[last_frame - 1]
    first_frame_to_plot1 = int(np.argmax(qt - t1a > 0)) + 1  # MATLAB 1-based
    last_frame_to_plot1 = int(np.argmax(qt - t1b > 0)) + 1

    hs1g = plt.figure(figsize=(7, 5))
    ax1 = hs1g.add_subplot(1, 2, 1)
    # MATLAB uses the full-resolution BW as AlphaData of the 8x-downscaled
    # mean image; the downsampled BW is used here so the shapes match.
    ax1.imshow(mimgtransformedRGB, alpha=(BW[:: params["downscale"], :: params["downscale"]]).astype(float))
    overlayOutlines(coords, params["downscale"], ax=ax1)
    plotOutline(maskPath[:11], st, atlas1, hemi, scale3, "k", ax=ax1)
    ax1.scatter(pixel[:, 1] - 1, pixel[:, 0] - 1, s=16, c=color2[2:])
    ax1.scatter(center[1] - 1, center[0] - 1, s=16, c="k")
    ax1.add_patch(plt.Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=1))
    ax1.axis("off")
    ax1.set_aspect("equal")

    ax2 = hs1g.add_subplot(1, 2, 2)
    for i in range(7):
        ax2.plot(qt, trace_raw[i] * 200 + 3 * i, "k", linewidth=1)
        ax2.add_collection(
            _phase_colored_segments_pos(qt, trace_filt[i], trace_phase[i], 200, 3 * i, cmap_c06)
        )
    ax2.tick_params(labelsize=8)
    ax2.set_xlim(t1[0], t1[-1])
    ax2.set_yticklabels([])
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.axvline(qt[first_frame_to_plot1 - 1], ls="--")
    ax2.axvline(qt[last_frame_to_plot1 - 1], ls="--")
    ax2.plot([t1[0] + 0.5, t1[0] + 0.5], [0, 2], "r")
    hs1g.savefig(save_folder / f"FigS1g_{fname}_{frameS}_cetc6_3.png", dpi=300)
    plt.show()
    return hs1g, hs1h
