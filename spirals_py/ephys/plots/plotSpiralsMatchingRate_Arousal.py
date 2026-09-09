"""Translated from revision2/prediction_motion_energy/
plotSpiralsMatchingRate_Arousal.m
(Extended Data Fig.13g: spiral matching ratio for arousal-sorted
spirals, real vs shuffle condition)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._ephys_helpers import _load_cell_array_h5, filter_facevideo_table, ttest_paired_h


def plotSpiralsMatchingRate_Arousal(T, data_folder, save_folder):
    """Original MATLAB file:
    revision2/prediction_motion_energy/plotSpiralsMatchingRate_Arousal.m

    Returns hs13g. The facevideo table filter is skipped when the session
    table lacks that column.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    left_all = _load_cell_array_h5(
        data_folder / "ephys" / "spirals_compare" / "spiral_compare_sessions_arousal.mat",
        "spiral_left_match_all",
    )
    right_all = _load_cell_array_h5(
        data_folder / "ephys" / "spirals_compare" / "spiral_compare_sessions_arousal.mat",
        "spiral_right_match_all",
    )

    T = filter_facevideo_table(T)

    area = ["THAL", "STR", "CORTEX", "MB"]

    n_sess = left_all.shape[0]
    match_all = np.full((n_sess, 2), np.nan)
    match_perm = np.full((n_sess, 2), np.nan)
    for kk in range(n_sess):
        match1 = left_all[kk, 0][:, 10]
        match2 = right_all[kk, 0][:, 10]
        match_temp = np.concatenate([match1, match2])
        match_all[kk, 0] = match_temp.size
        match_all[kk, 1] = match_temp.sum() / match_temp.size

        match1p = left_all[kk, 1][:, 10]
        match2p = right_all[kk, 1][:, 10]
        match_tempp = np.concatenate([match1p, match2p])
        match_perm[kk, 0] = match_tempp.size
        match_perm[kk, 1] = match_tempp.sum() / match_tempp.size

    color2 = ["g", "r", "c", "m"]
    hs13g = plt.figure(figsize=(6, 3))
    count1 = 0
    for kk in [1, 2, 4]:  # 'THAL', 'STR'; 'MB'
        count1 += 1
        indx = T["Area"].str.contains(area[kk - 1], na=False).to_numpy()
        match_all_temp = match_all[indx, :]
        match_perm_temp = match_perm[indx, :]
        ax = hs13g.add_subplot(1, 3, count1)
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
        ax.plot([1, 2], [0.5, 0.5], "k--")
        ax.set_ylim(0.3, 0.9)
        _ha2, pa2 = ttest_paired_h(match_all_temp[:, 1], match_perm_temp[:, 1])
        ax.set_xlabel("Spiral counts")
        ax.set_ylabel("Spiral matching ratio")
        ax.text(1.5, 0.7, f"p = {pa2:g}")
        ax.set_yticks(np.arange(0.3, 0.9 + 1e-9, 0.1))
        ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.3, 0.9 + 1e-9, 0.1)])
        ax.set_xlim(0.8, 2.2)

    hs13g.savefig(
        save_folder / "FigS13g_spirals_matching_ratio_arousal2.pdf", bbox_inches="tight"
    )
    return hs13g
