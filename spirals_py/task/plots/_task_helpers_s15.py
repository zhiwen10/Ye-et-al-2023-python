"""Private helpers shared by the FigS15 task plot translations
(plotMeanMapsAll, plotMeanTraceAll, plotSpiralTrialExample,
plotPassiveSpiralPrePost, plotCorrectSpiralPrePost,
plotSpiralCountOverTime).

Sources (MATLAB):
- tables/isocortex_horizontal_projection_outline.mat (projectedAtlas1, v7.3)
- task_svd/*_block.mat (MATLAB v5 struct via scipy.io.loadmat)
- shared body of task/plots/plotPassiveSpiralPrePost.m and
  task/plots/plotCorrectSpiralPrePost.m (identical except for inputs)
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from spirals_py.task.plots._task_helpers import (
    _schmitt,
    load_h5_var,
    tsToT,
)
from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.io import load_outline_coords_h5


def schmittTimes(t, sig, thresh):
    """Corrected port of dependencies/spikes/analysis/helpers/schmittTimes.m.

    MATLAB allows a logical index shorter than the indexed array (the
    trailing elements are ignored); _task_helpers.schmittTimes raises
    IndexError instead, so this variant is used here.
    """
    t = np.asarray(t, dtype=float).ravel()
    sig = np.asarray(sig, dtype=float).ravel()
    schmittSig = _schmitt(sig, thresh[0], thresh[1])
    down = (schmittSig[:-1] == 1) & (schmittSig[1:] == -1)
    up = (schmittSig[:-1] == -1) & (schmittSig[1:] == 1)
    flipsDown = t[: down.size][down]
    flipsUp = t[: up.size][up]
    flipTimes = np.sort(np.concatenate([flipsUp, flipsDown]))
    return flipTimes, flipsUp, flipsDown


def getPhotodiodeTime(session_root, win):
    """Translated from task/utils/getPhotodiodeTime.m (v7.3 files via h5py),
    using the corrected schmittTimes above."""
    session_root = Path(session_root)
    pd_sig = load_h5_var(session_root / "photodiode_raw.mat", "pd").ravel()
    tlTimes = load_h5_var(session_root / "photodiode_timestamps_Timeline.mat", "tlTimes")
    tt = tsToT(tlTimes, pd_sig.size)

    allPD, flipsUp, flipsDown = schmittTimes(tt, pd_sig, [0.5, 0.8])
    flipsUp = flipsUp[(flipsUp >= win[0]) & (flipsUp <= win[1])]
    flipsUp = flipsUp[:-1]

    dff_allPD = np.diff(allPD)
    indx = dff_allPD < 0.6
    indx = np.concatenate([[True], indx])  # MATLAB: indx = [1; indx]
    return allPD[~indx]


def matlab_round(x):
    """MATLAB round() semantics (half away from zero), unlike numpy's
    round-half-to-even."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def load_projectedAtlas1(data_folder):
    """projectedAtlas1 from tables/isocortex_horizontal_projection_outline.mat
    in MATLAB orientation (1320 x 1140)."""
    with h5py.File(
        Path(data_folder) / "tables" / "isocortex_horizontal_projection_outline.mat",
        "r",
    ) as f:
        return np.asarray(f["projectedAtlas1"]).T


def load_block(session_root):
    """Load the *_block.mat file of a task session (MATLAB v5)."""
    session_root = Path(session_root)
    blocks = [
        p for p in session_root.iterdir() if p.name.lower().endswith("_block.mat")
    ]
    if len(blocks) != 1:
        raise FileNotFoundError(f"expected one *_block.mat in {session_root}")
    m = loadmat(blocks[0], squeeze_me=True, struct_as_record=False)
    return m["block"]


def plot_spiral_pre_post(
    data_folder, save_folder, mat_file, frames_pre, labels, out_name, fig_size=(8, 8)
):
    """Shared body of plotPassiveSpiralPrePost.m / plotCorrectSpiralPrePost.m.

    Scatter plots of spiral density (spirals/(mm^2*s)) before and after
    stimulus onset, one panel per element of the unique_spirals_all cell.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    coords = load_outline_coords_h5(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )
    pixSize = 0.01  # mm/pix after registration
    pixArea = pixSize**2
    framesN = len(frames_pre)
    hist_bin = 40

    with h5py.File(data_folder / "task" / "spirals" / mat_file, "r") as f:
        unique_spirals_all = [
            np.asarray(f[r]).T for r in f["unique_spirals_all"][()].flat
        ]
        trialN = np.asarray(f["trialN"]).ravel()

    fig = plt.figure(figsize=fig_size)
    cmax = np.zeros(2)
    axes = []
    for j in range(2):
        unique_spirals = unique_spirals_all[j]
        ax1 = fig.add_subplot(1, 2, j + 1)
        axes.append(ax1)
        unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)
        unique_spirals_unit = unique_spirals_unit / trialN[j] * 35 / framesN
        cmax[j] = unique_spirals_unit.max()
        ax1.scatter(
            unique_spirals[:, 0], unique_spirals[:, 1], s=3, c=unique_spirals_unit, cmap="hot"
        )
        overlayOutlines(coords, 1, ax=ax1)
        ax1.invert_yaxis()  # MATLAB set(gca,'Ydir','reverse')
        ax1.set_aspect("equal")
        ax1.set_xlim(0, 1140)
        ax1.set_ylim(0, 1320)
        ax1.set_xticks(np.arange(0, 1001, 200))
        ax1.set_yticks(np.arange(0, 1201, 200))
        ax1.set_title(labels[j])

    cmax2 = cmax.max()
    for j in range(2):
        ax1 = axes[j]
        for coll in ax1.collections:
            coll.set_clim(0, cmax2)
        sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, cmax2))
        cb1 = fig.colorbar(sm, ax=ax1)
        cb1.set_ticks([0, cmax2])
        cb1.set_ticklabels(["0", "%g" % (matlab_round(cmax2 * 10) / 10)])
    fig.savefig(save_folder / out_name, bbox_inches="tight")
    return fig
