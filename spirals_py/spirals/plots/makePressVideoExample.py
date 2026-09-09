"""Translated from press/makePressVideoExample.m

Renders the 2-8 Hz phase map of the ZYE_0012/2020-10-16/5 example epoch
(1681.44-1681.88 s) with cortical outlines, a 2 mm scale bar and a timestamp,
and writes it as an mp4 video (cv2.VideoWriter instead of MATLAB VideoWriter;
the optional ffmpeg gif pass is left out, as in the commented MATLAB code).
"""

import re
from pathlib import Path

import colorcet as cc
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots.plotSpiralTimeSeries3d import (
    _get_cortex_atlas_path,
    _imwarp,
    _load_atlas_data,
    _load_example_session,
    _matlab_round,
)
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap4
from spirals_py.utils.atlas import plotOutline2


def _outline_area(areaPath, st, section, hemi):
    """Area mask computed by utils/plotOutline2.m (returned as area1 there)."""
    indx = []
    spath = st["structure_id_path"].astype(str)
    for p in areaPath:
        indx.extend(np.flatnonzero(spath.str.contains(re.escape(p), na=False)))
    idAll = st["id"].iloc[indx].to_numpy(dtype=float)
    area = np.isin(section.astype(float), idAll).astype(float)
    return area


def plotSpiralTimeSeriesFrame(ax, phase, BW3, cmap):
    im = ax.imshow(phase, cmap=cmap, vmin=-np.pi, vmax=np.pi,
                   alpha=BW3.astype(float))
    ax.set_aspect("equal")
    ax.axis("off")
    return im


def makePressVideoExample(data_folder, save_folder):
    """Translated from press/makePressVideoExample.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1, coords, projectedAtlas1 = _load_atlas_data(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    U, V, t, mimg, dV, masks, T, subfolder = _load_example_session(data_folder)
    BW2 = masks["BW2"]

    downscale = 4
    params = {"downscale": downscale, "lowpass": 0, "gsmooth": 0}
    rate = 0.1
    tStart, tEnd = 1681.44, 1681.88
    outScale = 1.5
    frameRate = 30
    lineW = 3.5
    textColor = "k"
    scale3 = 5 / downscale
    lineColor = "k"

    out_shape = projectedAtlas1.shape
    Utransformed = _imwarp(U, T, out_shape, stride=downscale)

    frameStart = int(np.argmax(t > tStart)) + 1  # MATLAB 1-based
    frameEnd = int(np.argmax(t > tEnd)) + 1
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)
    dV1 = dV[:, frameTemp - 1]
    _, _, tracePhase1 = spiralPhaseMap4(Utransformed, dV1, t, params, rate)
    pad = int(35 / rate)
    tracePhase1 = tracePhase1[:, :, pad:-pad]

    t1 = t[frameStart - 1 : frameEnd]
    nq = int(np.floor((t1.size - 1) / rate)) + 1
    tq = 1 + np.arange(nq) * rate
    qt = np.interp(tq, np.arange(1, t1.size + 1), t1)
    qt1 = qt - qt[0]
    nFrames = qt1.size

    imgH, imgW = tracePhase1.shape[0], tracePhase1.shape[1]
    um_per_pix = 10 * downscale
    barLen_px = 2000 / um_per_pix

    # masks at the downscaled resolution (MATLAB imresize, bilinear)
    BW3 = cv2.resize(BW2.astype(np.uint8), (imgW, imgH),
                     interpolation=cv2.INTER_LINEAR) > 0
    area1 = _outline_area(maskPath[:11], st, atlas1, [])
    BW4 = cv2.resize(area1.astype(np.float32), (imgW, imgH),
                     interpolation=cv2.INTER_LINEAR)
    BW3 = BW3 & (BW4 != 0)

    figW = int(2 * _matlab_round(imgW * outScale / 2))
    figH = int(2 * _matlab_round(imgH * outScale / 2))
    dpi = 100
    ha = plt.figure(figsize=(figW / dpi, figH / dpi), dpi=dpi, facecolor="w")
    ax3 = ha.add_axes([0, 0, 1, 1])

    cmap_c06 = cc.cm["CET_C6"]
    im_phase = plotSpiralTimeSeriesFrame(ax3, tracePhase1[:, :, 0], BW3, cmap_c06)

    plotOutline2(maskPath[:11], st, atlas1, [], scale3, lineColor, ax=ax3)
    for ln in ax3.lines:
        ln.set_linewidth(lineW)

    mgn = round(0.05 * imgW)
    xR = imgW - mgn
    xL = xR - barLen_px
    yB = imgH - 45
    ax3.plot([xL, xR], [yB, yB], color=textColor, linewidth=3)
    ax3.text((xL + xR) / 2, yB + 5, "2 mm", color=textColor, fontsize=26,
             ha="center", va="top")
    text_ts = ax3.text(0.03, 0.95, f"{qt1[0]:.1f} s", transform=ax3.transAxes,
                       fontsize=26, fontweight="bold", color=textColor)
    ax3.set_xlim(-0.5, imgW - 0.5)
    ax3.set_ylim(imgH - 0.5, -0.5)

    video_name = save_folder / f"{subfolder}_{tStart}-{tEnd}.mp4"
    vw = cv2.VideoWriter(
        str(video_name), cv2.VideoWriter_fourcc(*"mp4v"), frameRate, (figW, figH)
    )
    if not vw.isOpened():
        print(f"Could not open video writer for {video_name}; skipping video.")
        plt.close(ha)
        return None
    for i in range(nFrames):
        im_phase.set_data(tracePhase1[:, :, i])
        text_ts.set_text(f"{qt1[i]:.1f} s")
        ha.canvas.draw()
        fr = np.asarray(ha.canvas.buffer_rgba())[:, :, :3]
        fr = cv2.resize(fr, (figW, figH))
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    plt.close(ha)
    print(f"Saved video: {video_name} ({nFrames} frames)")
    return video_name
