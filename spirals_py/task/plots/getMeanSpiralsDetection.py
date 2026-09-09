from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

from spirals_py.task.preprocessing.trace_filt_nan import trace_filt_nan
from spirals_py.task.plots.plotCorrectMapsFlow import (
    _colorcet_c06,
    _get_cortex_atlas_path,
    _load_atlas1,
    _load_task_mask,
    _parula,
    _subplottight,
)
from spirals_py.utils.atlas import plotOutline

_REPO_SPIRALS_FILE = Path(__file__).resolve().parent / "task_spirals_group_fftn.mat"


def _imresize(vol, out_hw):
    """Adaptation: MATLAB imresize (bicubic) approximated by OpenCV INTER_CUBIC,
    applied frame by frame."""
    out = np.empty((out_hw[0], out_hw[1], vol.shape[2]))
    for k in range(vol.shape[2]):
        out[:, :, k] = cv2.resize(
            vol[:, :, k], (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC
        )
    return out


def _load_grouped_spirals(path):
    """Load archiveCell (v7.3 cell array of Nx5 double matrices) and
    concatenate like MATLAB cell2mat. Columns: x, y, radius, direction, frame."""
    spirals = []
    with h5py.File(path, "r") as f:
        for ref in f["archiveCell"][()].flat:
            spirals.append(np.asarray(f[ref]).T)
    return np.vstack(spirals)


def getMeanSpiralsDetection(data_folder, save_folder, spirals_file=None):
    """Translated from task/plots/getMeanSpiralsDetection.m

    Plot mean widefield maps (upsampled to 660x570) with detected spirals
    overlaid as circles on raw and phase frames.

    The spiral detection itself is commented out in the MATLAB source; the
    precomputed task_spirals_group_fftn.mat is loaded instead. It ships with
    the paper repository (task/plots/) and was copied next to this module
    (spirals_file overrides).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    if spirals_file is None:
        spirals_file = _REPO_SPIRALS_FILE
    atlas1 = _load_atlas1(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    BW2 = _load_task_mask(data_folder, 2)
    with h5py.File(
        data_folder / "task" / "task_mean_maps" / "task_mean_maps_all_mice.mat", "r"
    ) as f:
        trace_mean_all = np.asarray(f["trace_mean_all"]).transpose(3, 2, 1, 0)

    hemi = None
    scale3 = 5 / 2
    lineColor = "k"
    v = np.array([[36, 33], [62, 33], [62, 59], [36, 59]]) * 8 - 1
    parula = _parula()
    c06 = _colorcet_c06()

    wf_mean2 = trace_mean_all[:, :, :, 0]
    wf_mean3 = _imresize(wf_mean2, (660, 570))
    traceFilt2, tracePhase2 = trace_filt_nan(wf_mean3)

    cmin = -0.01
    cmax = 0.01
    spirals = _load_grouped_spirals(spirals_file)

    h5b = plt.figure(figsize=(9.5, 4))
    th2 = np.deg2rad(np.arange(1, 361, 5))
    for i in range(14):
        iframe = i + 68  # 0-based; MATLAB iframe = i+68 (1-based)
        spiral_temp = spirals[spirals[:, 4] == 68 + i + 1, :]
        frame_temp1 = wf_mean3[:, :, iframe]
        frame_temp2 = tracePhase2[:, :, iframe]

        ax1 = _subplottight(h5b, 4, 15, i + 1)
        ax1.imshow(frame_temp1, cmap=parula, vmin=cmin, vmax=cmax, alpha=BW2)
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax1)
        ax1.set_aspect("equal")
        ax1.axis("off")
        ax1.add_patch(
            Polygon(v, closed=True, edgecolor="k", facecolor="none", linewidth=0.8)
        )

        ax2 = _subplottight(h5b, 4, 15, i + 16)
        ax2.imshow(frame_temp2, cmap=c06, vmin=-np.pi, vmax=np.pi, alpha=BW2)
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax2)
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.add_patch(
            Polygon(v, closed=True, edgecolor="k", facecolor="none", linewidth=0.8)
        )

        for row, frame in ((31, frame_temp1), (46, frame_temp2)):
            ax = _subplottight(h5b, 4, 15, i + row)
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width - 0.01, pos.height - 0.01])
            ax.imshow(
                frame,
                cmap=parula if row == 31 else c06,
                vmin=cmin if row == 31 else -np.pi,
                vmax=cmax if row == 31 else np.pi,
                alpha=BW2,
            )
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(287, 495)  # MATLAB xlim([36,62]*8)
            ax.set_ylim(471, 263)  # MATLAB ylim([33,59]*8)
            if spiral_temp.size:
                for kk in range(spiral_temp.shape[0]):
                    px1, py1, r, direction = spiral_temp[kk, 0:4]
                    cx2 = np.round(r * np.cos(th2) + px1)
                    cy2 = np.round(r * np.sin(th2) + py1)
                    # direction 1 = counterclockwise -> white, else black
                    color1 = "w" if direction == 1 else "k"
                    ax.plot(
                        np.append(cx2, cx2[0]) - 1,
                        np.append(cy2, cy2[0]) - 1,
                        color=color1,
                        linewidth=2,
                    )
                    ax.plot(px1 - 1, py1 - 1, "o", color=color1, markersize=3)
    cb4 = _subplottight(h5b, 4, 15, 15)
    im_cb = cb4.imshow(wf_mean3[:, :, 81], cmap=parula, vmin=cmin, vmax=cmax)
    cb4.axis("off")
    cb = h5b.colorbar(im_cb, ax=cb4)
    a = cb.ax.get_position()
    cb.ax.set_position([a.x0 - 0.02, a.y0, a.width, a.height / 4])
    h5b.savefig(save_folder / "Fig5b_correct_mean_maps4.pdf", bbox_inches="tight")
    return h5b
