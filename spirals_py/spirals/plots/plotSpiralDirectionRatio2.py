"""Translated from spirals/plots/plotSpiralDirectionRatio2.m (Extended Data
Fig.7b): counter-clockwise spiral ratio, left vs right hemisphere, across
the 15 sessions (paired dots + mean +/- STD).

Faithful to the MATLAB control flow: 'left' uses the sensory-area mask
from select_area(..., 'left'), while 'right' is re-derived inside the
session loop as the whole right half of the cortex-filtered atlas (the
top-level right-hemisphere select_area result is overwritten there in
MATLAB and therefore unused).
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _get_cortex_atlas_path,
    _index_xy,
    _ismember_rows,
    _load_spirals_grouping,
    _select_area,
    _session_info,
    _transformPointsForward,
)
from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_tform


def plotSpiralDirectionRatio2(T, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralDirectionRatio2.m

    Returns the figure handle hs7b.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # load atlas brain horizontal projection and outline
    with h5py.File(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat", "r"
    ) as f:
        projectedAtlas1 = f["projectedAtlas1"][()].T
        projectedTemplate1 = f["projectedTemplate1"][()].T
    _, st = _get_cortex_atlas_path(data_folder)

    # keep only cortex in the atlas (paths under /997/8/567/)
    spath = st["structure_id_path"].astype(str)
    spath2 = spath.str.startswith("/997/8/567/")
    idFilt = np.flatnonzero(spath2.to_numpy()) + 1  # MATLAB st.index
    Lia = np.isin(projectedAtlas1, idFilt)
    projectedAtlas1[~Lia] = 0
    projectedTemplate1[~Lia] = 0

    # sensory area paths (areaPath(8) is unassigned in the MATLAB cell)
    areaPath = [
        "/997/8/567/688/695/315/453/322/",  # SSp (SS)
        "/997/8/567/688/695/315/247/",  # AUD
        "/997/8/567/688/695/315/669/",  # VIS
        "/997/8/567/688/695/315/254/",  # RSP
        "/997/8/567/688/695/315/22/312782546/",  # VISa
        "/997/8/567/688/695/315/22/417/",  # VISrl
        "/997/8/567/688/695/315/541/",  # TEa
        "/997/8/567/688/695/315/677/",  # VISC
        "/997/8/567/688/695/1089/",  # HPF
    ]
    sensoryArea = areaPath  # strcat(areaPath(:))

    # left-hemisphere sensory-area mask (used for spirals_left below)
    BW_SSp_left = _select_area(sensoryArea, st, projectedAtlas1, "left")

    # mask for whole brain (cortex)
    BW = projectedAtlas1.astype(bool)
    brain_index = _index_xy(BW)
    brain_index_left = _index_xy(BW_SSp_left)

    direction_ratio = np.full((15, 2), np.nan)
    for kk in range(15):
        mn, tdb, en, fname = _session_info(T, kk)
        tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
        cells, _ = _load_spirals_grouping(
            data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            min_duration=2,
        )
        filteredSpirals2 = np.vstack(cells)
        filteredSpirals2[filteredSpirals2[:, 3] == -1, 3] = 0

        sx, sy = _transformPointsForward(tform, filteredSpirals2[:, 0], filteredSpirals2[:, 1])
        filteredSpirals2[:, 0] = np.round(sx)
        filteredSpirals2[:, 1] = np.round(sy)

        # keep spirals within the brain boundary
        lia = _ismember_rows(filteredSpirals2[:, :2], brain_index)
        filteredSpirals2 = filteredSpirals2[lia, :]

        # right half of the cortex-filtered atlas (recomputed every session,
        # as in MATLAB)
        projectedAtlas_right = projectedAtlas1.copy()
        projectedAtlas_right[:, : projectedAtlas_right.shape[1] // 2] = 0
        brain_index_right = _index_xy(projectedAtlas_right.astype(bool))

        lia_right = _ismember_rows(filteredSpirals2[:, :2], brain_index_right)
        spirals_right = filteredSpirals2[lia_right, :]
        lia_left = _ismember_rows(filteredSpirals2[:, :2], brain_index_left)
        spirals_left = filteredSpirals2[lia_left, :]

        # 1 is counterclockwise
        direction_ratio[kk, 0] = np.divide(
            spirals_left[:, 3].sum(), spirals_left.shape[0], dtype=float
        )
        direction_ratio[kk, 1] = np.divide(
            spirals_right[:, 3].sum(), spirals_right.shape[0], dtype=float
        )

    # t-test
    _, p = ttest_rel(direction_ratio[:, 0], direction_ratio[:, 1])
    print(f"paired t-test ccw ratio left vs right: p = {p:.4g}")

    mean_ratio = direction_ratio.mean(axis=0)
    std_ratio = direction_ratio.std(axis=0, ddof=1)

    hs7b = plt.figure(figsize=(2, 2.5))
    ax = hs7b.add_subplot(1, 1, 1)
    ax.bar([0, 1], mean_ratio, color=[0.8, 0.8, 0.8], edgecolor=[0, 0, 0])
    ax.scatter(np.zeros(15), direction_ratio[:, 0], s=36, facecolor=[0.5, 0.5, 0.5],
               edgecolor="k", zorder=3)
    ax.scatter(np.ones(15), direction_ratio[:, 1], s=36, facecolor=[0.5, 0.5, 0.5],
               edgecolor="k", zorder=3)
    for k in range(direction_ratio.shape[0]):
        ax.plot([0, 1], [direction_ratio[k, 0], direction_ratio[k, 1]], "k", zorder=2)
    ax.errorbar(0, mean_ratio[0], yerr=std_ratio[0], linewidth=1, capsize=15, color="k",
                fmt="none", zorder=4)
    ax.errorbar(1, mean_ratio[1], yerr=std_ratio[1], linewidth=1, capsize=15, color="k",
                fmt="none", zorder=4)
    ax.set_ylim(0, 0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["left", "right"])
    ax.set_ylabel("ccw spiral ratio")

    hs7b.savefig(save_folder / "FigS7b_spirals_direction_all.png", bbox_inches="tight")
    plt.show()
    return hs7b
