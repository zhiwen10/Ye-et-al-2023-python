"""Translated from spirals/plots/plotSpiralsSymmetryRatio.m (Extended Data
Fig.7ef): spiral matching ratio between hand-drawn ROIs (M1/S1/M2, left
and right) across the 15 sessions.

The ROIs are drawpolygon objects stored in roiSelection.mat; since h5py
cannot deserialize MCOS graphics objects, their Position vertices are
recovered in _load_rois (mapping verified against the ROI geometry) and
inROI is replaced by a point-in-polygon test. As in MATLAB, a pair of
same-frame spirals in two ROIs "matches" when their rotation directions
are opposite (DirDiff == 0 in UniqueMatchPoints)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _getSymmetryRatio,
    _load_rois,
    _load_spirals_grouping,
    _session_info,
    _transformPointsForward,
)
from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_tform


def plotSpiralsSymmetryRatio(T, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralsSymmetryRatio.m

    Returns the figure handle hs7ef.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    rois = _load_rois(data_folder / "spirals" / "spirals_symmetry" / "roiSelection.mat")

    n_sessions = T.shape[0]
    ratio_all = np.full((7, n_sessions), np.nan)
    for kk in range(n_sessions):
        mn, tdb, en, fname = _session_info(T, kk)
        # load spiral centers (>40 pixels radius) and the atlas tform
        cells, _ = _load_spirals_grouping(
            data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            min_duration=2,
        )
        filteredSpirals2 = np.vstack(cells)
        filteredSpirals2[filteredSpirals2[:, 3] == -1, 3] = 0
        spirals2 = filteredSpirals2
        tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
        sx, sy = _transformPointsForward(tform, spirals2[:, 0], spirals2[:, 1])
        spirals2[:, 0] = sx
        spirals2[:, 1] = sy
        ratio = _getSymmetryRatio(
            rois["roiM1L"], rois["roiSL"], rois["roiM1R"], rois["roiSR"],
            rois["roiM2L"], rois["roiM2R"], spirals2,
        )
        ratio_all[:, kk] = ratio

    mn1 = T["MouseID"].tolist()
    area_name = [
        "M1-S1-left", "M1-S1-right", "S1-left-right", "M1-left-right",
        "M2-left-right", "M1-M2-left", "M1-M2-right",
    ]
    ratio_mean = ratio_all.mean(axis=1)
    ratio_std = ratio_all.std(axis=1, ddof=1)
    rng = np.random.default_rng()
    r = rng.normal(0, 0.05, n_sessions)  # MATLAB normrnd(0,0.05,[1,15])

    # MATLAB scatter without explicit color cycles through the default
    # ColorOrder, one color per condition
    ml_color_order = [
        "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#77BBBB", "#E1007A",
    ]

    hs7ef = plt.figure()
    ax = hs7ef.add_subplot(1, 1, 1)
    ax.bar(np.arange(1, 8), ratio_mean, color=[0.5, 0.5, 0.5])
    for i in range(1, 8):
        ax.scatter(np.full(n_sessions, i) + r, ratio_all[i - 1, :], s=36,
                   color=ml_color_order[i - 1], zorder=3)
    ax.errorbar(np.arange(1, 8), ratio_mean, yerr=ratio_std, linewidth=2, capsize=18,
                color=[0, 0, 0], linestyle="none", zorder=4)
    ax.set_xticks(np.arange(1, 8))
    ax.set_xticklabels(area_name, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("matching ratio")

    hs7ef.savefig(save_folder / "Figs7ef_spirals_symmetry_ratio.png", bbox_inches="tight")
    plt.show()
    return hs7ef
