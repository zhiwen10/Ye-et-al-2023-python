"""Translated from whisker/preprocessing/plotMeanTracePhase7.m"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from spirals_py.utils.atlas import plotOutline

from ._whisker_utils import SSP_SUB_PATHS, _bandpass_phase, _colorcet_c06, _subplottight


def _draw_spirals(ax, spiral_temp, th2):
    for row in spiral_temp:
        px1, py1, r, d = row[0], row[1], row[2], row[3]
        cx2 = np.round(r * np.cos(np.deg2rad(th2)) + px1)
        cy2 = np.round(r * np.sin(np.deg2rad(th2)) + py1)
        # MATLAB: counterclockwise (1) white, clockwise (-1) black
        color1 = "w" if d == 1 else "k"
        ax.plot(
            np.append(cx2, cx2[0]),
            np.append(cy2, cy2[0]),
            color=color1,
            linewidth=2,
        )
        ax.scatter([px1], [py1], s=8, c=color1)


def plotMeanTracePhase7(wf_mean2, BW2, maskPath, st, atlas1, spirals):
    """Translated from whisker/preprocessing/plotMeanTracePhase7.m

    wf_mean2: [rows, cols, frames] mean evoked map (660x570x141).
    spirals: Nx5 [x, y, r, direction, frame] with MATLAB 1-based frame numbers.
    matplotlib's viridis stands in for MATLAB's parula colormap.
    Returns (fig, meanTrace2, tracePhase).
    """
    maskPath = list(maskPath[:11]) + SSP_SUB_PATHS
    lineColor = "k"
    hemi = None
    scale3 = 5 / 2
    v = np.array([[80, 50], [130, 50], [130, 100], [80, 100]]) * 4

    meanTrace2, tracePhase = _bandpass_phase(wf_mean2)
    mina, maxa = -0.01, 0.01
    th2 = np.arange(1, 361, 5)
    alpha = BW2.astype(float)
    cmap_c06 = _colorcet_c06()

    subN = 14
    fig = plt.figure(figsize=(9.5, 4.0))
    for i in range(1, subN + 1):
        idx = 68 + i - 1  # MATLAB frame 68+i (1-based) -> 0-based
        spiral_temp = spirals[spirals[:, 4] == 68 + i, :]

        ax1 = _subplottight(fig, 4, 15, i)
        ax1.imshow(wf_mean2[:, :, idx], cmap="viridis", vmin=mina, vmax=maxa, alpha=alpha)
        ax1.axis("off")
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax1)
        if i == 3:
            plotOutline(maskPath[11:18], st, atlas1, hemi, scale3, lineColor, ax=ax1)
        ax1.add_patch(Polygon(v, closed=True, fill=False, edgecolor="k", linewidth=0.8))
        pos = ax1.get_position()
        ax1.set_position([pos.x0, pos.y0, pos.width * 0.95, pos.height * 0.95])

        ax2 = _subplottight(fig, 4, 15, 15 + i)
        ax2.imshow(tracePhase[:, :, idx], cmap=cmap_c06, vmin=-np.pi, vmax=np.pi, alpha=alpha)
        ax2.axis("off")
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax2)
        if i == 3:
            plotOutline(maskPath[11:18], st, atlas1, hemi, scale3, lineColor, ax=ax2)
        ax2.add_patch(Polygon(v, closed=True, fill=False, edgecolor="k", linewidth=0.8))

        ax3 = _subplottight(fig, 4, 15, 15 * 2 + i)
        pos = ax3.get_position()
        ax3.set_position([pos.x0 + 0.01, pos.y0, pos.width - 0.02, pos.height - 0.02])
        ax3.imshow(wf_mean2[:, :, idx], cmap="viridis", vmin=mina, vmax=maxa, alpha=alpha)
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax3)
        if i == 3:
            plotOutline(maskPath[11:18], st, atlas1, hemi, scale3, lineColor, ax=ax3)
        ax3.axis("off")
        if spiral_temp.size:
            _draw_spirals(ax3, spiral_temp, th2)
        ax3.set_xlim(80 * 4, 130 * 4)
        ax3.set_ylim(100 * 4, 50 * 4)

        ax4 = _subplottight(fig, 4, 15, 15 * 3 + i)
        pos = ax4.get_position()
        ax4.set_position([pos.x0 + 0.01, pos.y0, pos.width - 0.02, pos.height - 0.02])
        ax4.imshow(tracePhase[:, :, idx], cmap=cmap_c06, vmin=-np.pi, vmax=np.pi, alpha=alpha)
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax4)
        if i == 3:
            plotOutline(maskPath[11:18], st, atlas1, hemi, scale3, lineColor, ax=ax4)
        ax4.axis("off")
        if spiral_temp.size:
            _draw_spirals(ax4, spiral_temp, th2)
        ax4.set_xlim(80 * 4, 130 * 4)
        ax4.set_ylim(100 * 4, 50 * 4)

    axc = _subplottight(fig, 3, 15, 15)
    im = axc.imshow(wf_mean2[:, :, 68 + 12 - 1], cmap="viridis", vmin=mina, vmax=maxa, alpha=alpha)
    axc.axis("off")
    plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=axc)
    cb = fig.colorbar(im, ax=axc, shrink=0.5)
    pos = axc.get_position()
    axc.set_position([pos.x0, pos.y0, pos.width * 0.95, pos.height * 0.95])

    return fig, meanTrace2, tracePhase
