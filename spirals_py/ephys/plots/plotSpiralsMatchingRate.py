"""Translated from ephys/plots/plotSpiralsMatchingRate.m
(Figure 4g: spiral matching ratio, real vs shuffle condition)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._ephys_helpers import _load_cell_array_h5, ttest_paired_h

# cbrewer2('qual','Set2',8) green/red/gray used for the df/f traces


def plotSpiralsMatchingRate(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotSpiralsMatchingRate.m

    Returns h4g.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    left = _load_cell_array_h5(
        data_folder / "ephys" / "spirals_compare" / "spiral_compare_sessions_neighbor.mat",
        "spiral_left_match_all",
    ).ravel()
    right = _load_cell_array_h5(
        data_folder / "ephys" / "spirals_compare" / "spiral_compare_sessions_neighbor.mat",
        "spiral_right_match_all",
    ).ravel()
    left_perm = _load_cell_array_h5(
        data_folder
        / "ephys"
        / "spirals_compare"
        / "spiral_compare_sessions_neighbor_permute.mat",
        "spiral_left_match_all_perm",
    ).ravel()
    right_perm = _load_cell_array_h5(
        data_folder
        / "ephys"
        / "spirals_compare"
        / "spiral_compare_sessions_neighbor_permute.mat",
        "spiral_right_match_all_perm",
    ).ravel()

    area = ["THAL", "STR", "CORTEX", "MB"]

    match_all = np.full((len(T), 2), np.nan)
    match_perm = np.full((len(T), 2), np.nan)
    for kk in range(len(T)):
        match1 = left[kk][:, 10]
        match2 = right[kk][:, 10]
        match_temp = np.concatenate([match1, match2])
        match_all[kk, 0] = match_temp.size
        match_all[kk, 1] = match_temp.sum() / match_temp.size

        match1p = left_perm[kk][:, 10]
        match2p = right_perm[kk][:, 10]
        match_tempp = np.concatenate([match1p, match2p])
        match_perm[kk, 0] = match_tempp.size
        match_perm[kk, 1] = match_tempp.sum() / match_tempp.size

    color2 = ["g", "r", "c", "m"]
    h4g = plt.figure(figsize=(6, 3))
    count1 = 0
    for kk in [1, 2, 4]:  # 'THAL', 'STR', 'MB'
        indx = T["Area"].str.contains(area[kk - 1], na=False).to_numpy()
        match_all_temp = match_all[indx, :]
        match_perm_temp = match_perm[indx, :]
        count1 += 1
        ax = h4g.add_subplot(1, 3, count1)
        ax.scatter(
            np.ones(match_all_temp[:, 1].shape), match_all_temp[:, 1],
            s=18, c=color2[kk - 1],
        )
        ax.scatter(
            2 * np.ones(match_perm_temp[:, 1].shape), match_perm_temp[:, 1],
            s=18, c="k",
        )
        for r_real, r_perm in zip(match_all_temp[:, 1], match_perm_temp[:, 1]):
            ax.plot([1, 2], [r_real, r_perm], "k", linewidth=0.5)
        ax.axhline(0.5, linestyle="--", color="k")
        ax.set_ylim(0.3, 0.9)
        _ha, pa2 = ttest_paired_h(match_all_temp[:, 1], match_perm_temp[:, 1])
        ax.set_xlabel("Spiral counts")
        ax.set_ylabel("Spiral matching ratio")
        ax.text(1.5, 0.7, f"p = {pa2:g}")
        ax.set_yticks(np.arange(0.3, 0.9 + 1e-9, 0.1))
        ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.3, 0.9 + 1e-9, 0.1)])

    h4g.savefig(save_folder / "Fig4g_spirals_matching_ratio.pdf", bbox_inches="tight")
    return h4g
