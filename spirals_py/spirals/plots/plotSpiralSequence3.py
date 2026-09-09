"""Translated from spirals/plots/plotSpiralSequence3.m

Plots an example 9-frame spiral sequence for session ZYE_0012/2020-10-16/5:
dF/F frames, phase frames, and zoomed dF/F + phase crops with the detected
spiral positions/radii overlaid (white = counterclockwise, black = clockwise).
"""

from pathlib import Path

import colorcet as cc
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
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq2
from spirals_py.utils.atlas import plotOutline


def _transform_points_forward(T, x, y):
    """MATLAB transformPointsForward for affine2d: [u v 1] = [x y 1] * T."""
    xy1 = np.stack([np.asarray(x, float), np.asarray(y, float), np.ones_like(np.asarray(x, float))], axis=-1)
    uv = xy1 @ T
    return uv[..., 0], uv[..., 1]


def plotSpiralSequence3(data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralSequence3.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1, coords, projectedAtlas1 = _load_atlas_data(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    U, V, t, mimg, dV, masks, T, fname = _load_example_session(data_folder)
    BW2 = masks["BW2"]

    group_file = data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat"
    rows = []
    with h5py.File(group_file, "r") as f:
        refs = f["archiveCell"][()]
        for ref in refs.flat:
            rows.append(np.asarray(f[ref]).T)  # (5, m) -> (m, 5)
    spirals = np.vstack(rows)  # cell2mat(archiveCell): N x 5
    sx, sy = _transform_points_forward(T, spirals[:, 0], spirals[:, 1])
    spirals[:, 0:2] = _matlab_round(np.stack([sx, sy], axis=1) / 8)

    pixSize = 3.45 / 1000 / 0.6 * 3  # mm / pix for spiral radius

    downscale = 8
    params = {"downscale": downscale, "lowpass": 0, "gsmooth": 0}
    rate = 1

    out_shape = projectedAtlas1.shape
    Utransformed = _imwarp(U, T, out_shape, stride=downscale)
    mimgtransformed = _imwarp(mimg, T, out_shape, stride=downscale)

    freq = [2, 8]
    tStart, tEnd = 1681, 1686  # find spirals between tStart:tEnd
    frameStart = int(np.argmax(t > tStart)) + 1  # MATLAB 1-based
    frameEnd = int(np.argmax(t > tEnd)) + 1
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)
    dV1 = dV[:, frameTemp - 1]
    trace2d1, traceAmp1, tracePhase1 = spiralPhaseMap_freq2(
        Utransformed, dV1, t, params, freq, rate
    )
    pad = int(35 / rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        trace2d1 = trace2d1[:, :, pad:-pad] / mimgtransformed[:, :, None]
    tracePhase1 = tracePhase1[:, :, pad:-pad]

    scale3 = 5 / 8
    a, b = 60, 97  # height
    c, d = 92, 129  # width
    v1 = np.array([[c, a], [d, a], [d, b], [c, b]]) - 1  # 0-based

    lineColor = "k"
    frames = 9
    subplotn = 10
    first_frame = 23

    dff = trace2d1 * 100
    phase_min, phase_max = -np.pi, np.pi
    th2 = np.arange(1, 361, 5)
    cmap_c06 = cc.cm["CET_C6"]

    h1b, axs = plt.subplots(
        4, subplotn, figsize=(10, 8), gridspec_kw={"wspace": 0.02, "hspace": 0.08}
    )
    frame_count = 1
    for k in range(1, frames + 1):
        frame = first_frame + (k - 1)  # MATLAB 1-based index into dff/phase
        frame_real = frameStart + first_frame + k - 1
        spiral_temp = spirals[spirals[:, 4] == frame_real, :]

        ax3 = axs[0, k - 1]
        ax3.imshow(dff[:, :, frame - 1], cmap="viridis", alpha=BW2.astype(float),
                   vmin=-1.5, vmax=1.5)
        ax3.set_aspect("equal")
        ax3.axis("off")
        plotOutline(maskPath[:11], st, atlas1, [], scale3, lineColor, ax=ax3)
        ax3.set_title(f"frame{frame_count}", fontsize=8)
        ax3.add_patch(plt.Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=2))

        ax4 = axs[1, k - 1]
        ax4.imshow(tracePhase1[:, :, frame - 1], cmap=cmap_c06,
                   alpha=BW2.astype(float), vmin=phase_min, vmax=phase_max)
        ax4.set_aspect("equal")
        ax4.axis("off")
        plotOutline(maskPath[:11], st, atlas1, [], scale3, lineColor, ax=ax4)
        ax4.set_title(f"frame{frame_count}", fontsize=8)
        ax4.add_patch(plt.Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=2))

        for row, (data, vmin, vmax, cmap) in enumerate(
            [
                (dff[:, :, frame - 1], -1.5, 1.5, "viridis"),
                (tracePhase1[:, :, frame - 1], phase_min, phase_max, cmap_c06),
            ],
            start=2,
        ):
            ax = axs[row, k - 1]
            ax.imshow(data, cmap=cmap, alpha=BW2.astype(float), vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.axis("off")
            plotOutline(maskPath[:11], st, atlas1, [], scale3, lineColor, ax=ax)
            for kk in range(spiral_temp.shape[0]):
                px1 = spiral_temp[kk, 0]
                py1 = spiral_temp[kk, 1]
                r = spiral_temp[kk, 2] * pixSize / (0.01 * 8)
                cx2 = _matlab_round(r * np.cos(np.deg2rad(th2)) + px1)
                cy2 = _matlab_round(r * np.sin(np.deg2rad(th2)) + py1)
                color1 = "w" if spiral_temp[kk, 3] == 1 else "k"  # CCW white / CW black
                cx2c = np.concatenate([cx2, cx2[:1]]) - 1  # 0-based for matplotlib
                cy2c = np.concatenate([cy2, cy2[:1]]) - 1
                ax.plot(cx2c, cy2c, color=color1, linewidth=2)
                ax.scatter(spiral_temp[kk, 0] - 1, spiral_temp[kk, 1] - 1, s=8, c=color1)
            ax.set_ylim(b - 1, a - 1)  # MATLAB ylim([a,b]) with reversed y axis
            ax.set_xlim(c - 1, d - 1)

        frame_count += 1

    # 10th column of the first row: last dff frame with a colorbar
    axc = axs[0, subplotn - 1]
    im_raw = axc.imshow(dff[:, :, frame - 1], cmap="viridis",
                        alpha=BW2.astype(float), vmin=-1.5, vmax=1.5)
    axc.set_aspect("equal")
    axc.axis("off")
    plotOutline(maskPath[:11], st, atlas1, [], scale3, lineColor, ax=axc)
    axc.set_title(f"frame{frame_count}", fontsize=8)
    axc.add_patch(plt.Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=1))
    h1b.colorbar(im_raw, ax=axc, fraction=0.046, pad=0.04)
    for j in range(1, 4):
        axs[j, subplotn - 1].axis("off")

    h1b.savefig(save_folder / "Fig1b_example_spiral_sequence4.pdf")
    plt.show()
    return h1b
