"""Translated from spirals/plots/plotSpiralTimeSeries.m

Plots an example spiral frame for session ZYE_0012/2020-10-16/5: mean image
with 7 region pixels, one phase frame with the Horn-Schunck optical-flow
field, the raw + phase-colored time series at the 7 pixels, and a zoomed
phase/flow inset.
"""

from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from spirals_py.spirals.plots.plotSpiralTimeSeries3d import (
    _cbrewer,
    _get_cortex_atlas_path,
    _imwarp,
    _load_atlas_data,
    _load_example_session,
    _matlab_round,
    _phase_colored_segments,
)
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.utils.atlas import overlayOutlines, plotOutline
from spirals_py.utils.optical_flow import HS_flowfield


def plotSpiralTimeSeries(data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralTimeSeries.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1, coords, projectedAtlas1 = _load_atlas_data(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    U, V, t, mimg, dV, masks, T, subfolder = _load_example_session(data_folder)
    BW2 = masks["BW2"]

    downscale = 8
    params = {"downscale": downscale, "lowpass": 0, "gsmooth": 0}
    rate = 0.1

    # 7 region pixels on the 10 um template, then downscaled (1-based)
    pixel = np.array(
        [
            [900, 800],  # VISp
            [775, 650],  # RSP
            [590, 750],  # SSp-ul
            [520, 850],  # SSp-ll
            [480, 960],  # SSp-m
            [550, 960],  # SSp-n
            [682, 950],  # SSp-bfd
        ]
    )
    pixel = _matlab_round(pixel / downscale).astype(int)

    out_shape = projectedAtlas1.shape
    Utransformed = _imwarp(U, T, out_shape, stride=downscale)
    mimgtransformed = _imwarp(mimg, T, out_shape, stride=downscale)
    with np.errstate(divide="ignore", invalid="ignore"):
        mimgtransformedRBG = mimgtransformed / mimgtransformed.max()
    mimgtransformedRGB = np.stack([mimgtransformedRBG] * 3, axis=-1)

    freq = [2, 8]
    tStart, tEnd = 1681, 1684  # find spirals between tStart:tEnd
    frameStart = int(np.argmax(t > tStart)) + 1  # MATLAB 1-based
    frameEnd = int(np.argmax(t > tEnd)) + 1
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)
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
    rawTrace1 = interp1d(np.arange(1, tsize + 1), rawTrace1, axis=1, kind="linear")(tq)
    rawTrace1 = rawTrace1.reshape(Utransformed.shape[0], Utransformed.shape[1], n, order="F")
    with np.errstate(divide="ignore", invalid="ignore"):
        rawTrace = rawTrace1[:, :, pad:-pad] / mimgtransformed[:, :, None]

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
    color2 = _cbrewer("seq", "YlOrRd", 9)
    nameList = ["VISp", "RSP", "SSp_ul", "SSp_ll", "SSp_m", "SSp_n", "SSp_bfd"]
    frames_to_plot = 18
    first_frame = 22
    last_frame = first_frame + frames_to_plot
    t1a, t1b = t1[first_frame - 1], t1[last_frame - 1]
    first_frame1 = int(np.argmax(qt - t1a > 0)) + 1
    last_frame1 = int(np.argmax(qt - t1b > 0)) + 1

    example_frame_to_plot = first_frame + 8
    t1c = t1[example_frame_to_plot - 1]
    frame = int(np.argmax(qt - t1c > 0)) + 1  # MATLAB 1-based
    cmap_c06 = cc.cm["CET_C6"]

    h1ac = plt.figure(figsize=(7, 9))

    ax1 = h1ac.add_subplot(2, 2, 1)
    ax1.imshow(mimgtransformedRGB, alpha=BW2.astype(float))
    overlayOutlines(coords, downscale, ax=ax1)
    plotOutline(maskPath[:11], st, atlas1, [], scale3, "k", ax=ax1)
    cb1 = h1ac.colorbar(
        ax1.images[0], ax=ax1, fraction=0.046 * 0.5, pad=0.14
    )
    ax1.scatter(pixel[:, 1] - 1, pixel[:, 0] - 1, s=16, c=color2[2:])

    ax4 = h1ac.add_subplot(2, 2, 2)
    framea = tracePhase1[:, :, frame - 1]
    frameb = tracePhase1[:, :, frame - 1 + 10]
    frame_ab = np.stack([framea, frameb], axis=0)
    vxRaw, vyRaw = HS_flowfield(frame_ab, 0)
    vxRaw = vxRaw.squeeze()
    vyRaw = vyRaw.squeeze()
    vxRaw2 = np.full(vxRaw.shape, np.nan)
    vyRaw2 = np.full(vyRaw.shape, np.nan)
    skip = 3
    zoom_scale = 2
    vxRaw2[::skip, ::skip] = vxRaw[::skip, ::skip] * zoom_scale
    vyRaw2[::skip, ::skip] = vyRaw[::skip, ::skip] * zoom_scale
    vxRaw2[~BW2] = np.nan
    vyRaw2[~BW2] = np.nan

    im_phase = ax4.imshow(framea, cmap=cmap_c06, alpha=BW2.astype(float))
    ax4.set_aspect("equal")
    plotOutline(maskPath[:11], st, atlas1, [], scale3, "k", ax=ax4)
    Yq, Xq = np.mgrid[0 : vxRaw2.shape[0], 0 : vxRaw2.shape[1]]
    ax4.quiver(
        Xq, Yq, vxRaw2, vyRaw2, angles="xy", scale_units="xy", scale=1,
        color="k", linewidths=1,
    )
    a, b = 60, 97  # height
    c, d = 92, 129  # width
    v = np.array([[c, a], [d, a], [d, b], [c, b]]) - 1  # 0-based
    ax4.add_patch(plt.Polygon(v, closed=True, fill=False, edgecolor="k", linewidth=2))
    cb2 = h1ac.colorbar(im_phase, ax=ax4, fraction=0.046 * 0.5, pad=0.14)

    ax2 = h1ac.add_subplot(2, 2, 3)
    for i in range(pixel.shape[0]):
        yoff = 3 * i
        ax2.plot(qt, trace_raw[i] * 100 + yoff, "k", linewidth=1)
        ax2.add_collection(
            _phase_colored_segments(qt, trace_filt[i], trace_phase[i], -yoff, 100, cmap_c06)
        )
    ax2.tick_params(labelsize=8)
    ax2.set_xlim(t1[0], t1[-1])
    ax2.set_yticklabels([])
    for i in range(pixel.shape[0]):
        ax2.text(t1[0] - 0.8, i * 3, nameList[i], color=color2[i + 2])
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.axvline(qt[first_frame1 - 1], ls="--")
    ax2.axvline(qt[last_frame1 - 1], ls="--")
    ax2.plot([t1[0] + 0.5, t1[0] + 0.5], [0, 1], "r")

    framea_zoom = framea[a - 1 : b, c - 1 : d]
    ax5 = h1ac.add_subplot(2, 2, 4)
    im_phase5 = ax5.imshow(framea_zoom, cmap=cmap_c06)
    ax5.set_aspect("equal")
    ax5.axis("off")
    vz = vxRaw2[a - 1 : b, c - 1 : d]
    wz = vyRaw2[a - 1 : b, c - 1 : d]
    Yz, Xz = np.mgrid[0 : vz.shape[0], 0 : vz.shape[1]]
    ax5.quiver(Xz, Yz, vz, wz, angles="xy", scale_units="xy", scale=1, color="k")
    cb5 = h1ac.colorbar(im_phase5, ax=ax5, fraction=0.046 * 0.5, pad=0.14)

    h1ac.tight_layout()
    h1ac.savefig(save_folder / "Fig1ac_example_time_series&flow_V.pdf")
    plt.show()
    return h1ac
