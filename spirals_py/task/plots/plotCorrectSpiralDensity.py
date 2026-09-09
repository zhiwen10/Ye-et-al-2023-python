from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.task.plots.plotCorrectMapsFlow import (
    _get_cortex_atlas_path,
    _load_atlas1,
)
from spirals_py.utils.atlas import overlayOutlines, plotOutline
from spirals_py.utils.io import load_outline_coords_h5


def plotCorrectSpiralDensity(data_folder, save_folder):
    """Translated from task/plots/plotCorrectSpiralDensity.m

    Spiral density maps (spirals/(mm^2*s)) before and after stimulus onset
    for correct / incorrect / miss trials.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    atlas1 = _load_atlas1(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    coords = load_outline_coords_h5(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )
    pixSize = 0.01  # mm/pix after registration
    pixArea = pixSize**2

    spiral_folder = data_folder / "task" / "spirals"
    framesN = 7  # numel(62:68)
    hist_bin = 40
    with h5py.File(spiral_folder / "spiral_density_map_trial_types.mat", "r") as f:
        unique_spirals_pre = [np.asarray(f[r]).T for r in f["unique_spirals_pre"][()].flat]
        unique_spirals_post = [
            np.asarray(f[r]).T for r in f["unique_spirals_post"][()].flat
        ]
        trialN = np.asarray(f["trialN"]).ravel()

    labels_all = [
        "Correct pre-stim",
        "Correct post-stim",
        "Incorrect pre-stim",
        "Incorrect post-stim",
        "Miss pre-stim",
        "Miss post-stim",
    ]

    h5d = plt.figure(figsize=(8, 8))
    lineColor = "k"
    scale3 = 5 / 1
    cmax = np.zeros(6)
    count = 0
    for j in range(3):
        for k in range(2):
            unique_spirals = unique_spirals_pre[j] if k == 0 else unique_spirals_post[j]
            ax1 = h5d.add_subplot(3, 2, j * 2 + k + 1)
            unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)
            unique_spirals_unit = unique_spirals_unit / trialN[j] * 35 / framesN
            cmax[count] = unique_spirals_unit.max()
            ax1.scatter(
                unique_spirals[:, 0],
                unique_spirals[:, 1],
                s=3,
                c=unique_spirals_unit,
                cmap="hot",
            )
            overlayOutlines(coords, 1, ax=ax1)
            plotOutline(maskPath[0:11], st, atlas1, None, scale3, lineColor, ax=ax1)
            ax1.set_xlim(0, 1140)
            ax1.set_ylim(0, 1320)
            ax1.invert_yaxis()  # MATLAB set(gca,'Ydir','reverse')
            ax1.set_aspect("equal")
            ax1.set_xticks(np.arange(0, 1001, 200))
            ax1.set_yticks(np.arange(0, 1201, 200))
            ax1.set_title(labels_all[j * 2 + k])
            count += 1

    cmax2 = cmax.max()
    for j in range(6):
        ax1 = h5d.axes[j]
        for coll in ax1.collections:
            coll.set_clim(0, cmax2)
        sm = plt.cm.ScalarMappable(
            cmap="hot", norm=plt.Normalize(0, cmax2)
        )
        cb1 = h5d.colorbar(sm, ax=ax1)
        cb1.set_ticks([0, cmax2])
        cb1.set_ticklabels(["0", str(round(cmax2 * 10) / 10)])
    h5d.savefig(save_folder / "Fig5d_task_spiral_density.pdf", bbox_inches="tight")
    return h5d
