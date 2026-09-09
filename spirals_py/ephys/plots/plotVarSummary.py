"""Translated from ephys/plots/plotVarSummary.m
(Extended Data Fig.13b-e: variance explained summary and relationship
with neuron number)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._ephys_helpers import _load_mat_var, ttest2_h
from spirals_py.ephys.utils import get_session_info2

# cbrewer2('qual','Set2',8)-style colors used per area


def plotVarSummary(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotVarSummary.m

    Returns hs13be. (Deviation: the MATLAB Ta1/Ta stats tables are only
    computed, never displayed; here the p-values are kept as arrays.)
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    area = ["THAL", "STR", "CORTEX", "MB"]

    # mean and max variances
    max_var = []
    mean_var = []
    for current_area in [1, 2, 4]:
        indx = T["Area"].str.contains(area[current_area - 1], na=False)
        current_T = T[indx.astype(bool)]
        max_var.append(np.asarray(current_T["maxVar"], dtype=float))
        mean_var.append(np.asarray(current_T["meanVar"], dtype=float))

    # significance test
    mean_var_all = np.array([v.mean() for v in mean_var])
    max_var_all = np.array([v.mean() for v in max_var])
    std_mean_all = np.array([v.std(ddof=1) for v in mean_var])
    std_max_all = np.array([v.std(ddof=1) for v in max_var])
    count = np.array([v.size for v in max_var], dtype=float)
    sem_mean_all = std_mean_all / np.sqrt(count)
    sem_max_all = std_max_all / np.sqrt(count)
    pairs = [(0, 1), (0, 2), (1, 2)]
    p1 = np.zeros((3, 2))
    for i, (a, b) in enumerate(pairs):
        p1[i, 0] = ttest2_h(mean_var[a], mean_var[b])[1]
        p1[i, 1] = ttest2_h(max_var[a], max_var[b])[1]

    # neuron counts
    data_folder1 = data_folder / "ephys" / "var_ordered"
    numVar_all = []
    for current_area in [1, 2, 4]:
        indx = T["Area"].str.contains(area[current_area - 1], na=False)
        current_T = T[indx.astype(bool)]
        numVar = np.full((len(current_T), 3), np.nan)
        for kk in range(len(current_T)):
            ops = get_session_info2(current_T, kk, data_folder)
            var_name = f"{ops.mn}_{ops.tdb}_{ops.en}_explained_var.mat"
            mean_var2 = _load_mat_var(data_folder1 / var_name, "mean_var2").ravel()
            max_var1 = mean_var2.max()
            indx1 = np.flatnonzero(mean_var2 >= 0.8 * max_var1)[0]
            numVar[kk, :] = [mean_var2.size, indx1 + 1, mean_var2[indx1]]
        numVar_all.append(numVar)

    mean_area = np.vstack([v.mean(axis=0) for v in numVar_all])
    std_area = np.vstack([v.std(axis=0, ddof=1) for v in numVar_all])
    count_area = np.tile(
        np.array([[v.shape[0] for v in numVar_all]]).T, (1, 3)
    ).astype(float)
    sem_area = std_area / np.sqrt(count_area)

    # stats
    p = np.zeros((3, 3))
    for i, (a, b) in enumerate(pairs):
        p[i, 0] = ttest2_h(numVar_all[a][:, 0], numVar_all[b][:, 0])[1]
        p[i, 1] = ttest2_h(numVar_all[a][:, 1], numVar_all[b][:, 1])[1]
        p[i, 2] = ttest2_h(numVar_all[a][:, 2], numVar_all[b][:, 2])[1]

    color1 = ["g", "r", "m"]
    hs13be = plt.figure(figsize=(7, 7))
    ax1 = hs13be.add_subplot(2, 2, 1)
    for i in range(3):
        ax1.scatter(max_var[i], mean_var[i], s=8, c=color1[i])
        ax1.errorbar(
            max_var_all[i], mean_var_all[i],
            yerr=sem_mean_all[i], xerr=sem_max_all[i],
            marker="o", color=color1[i],
        )
    ax1.set_xlim(0.2, 1.0)
    ax1.set_ylim(0, 0.8)
    ax1.set_yticks(np.arange(0, 0.8 + 1e-9, 0.2))
    ax1.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 0.8 + 1e-9, 0.2)])
    ax1.set_xlabel("Max var explained")
    ax1.set_ylabel("Mean var explained")

    ax2 = hs13be.add_subplot(2, 2, 2)
    for i in range(3):
        current_numVar = numVar_all[i]
        ax2.scatter(current_numVar[:, 0], current_numVar[:, 2], s=8, c=color1[i])
        ax2.errorbar(
            mean_area[i, 0], mean_area[i, 2],
            yerr=sem_area[i, 2], xerr=sem_area[i, 0],
            marker="o", color=color1[i],
        )
    ax2.set_ylim(0, 0.8)
    ax2.set_yticks(np.arange(0, 0.8 + 1e-9, 0.2))
    ax2.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 0.8 + 1e-9, 0.2)])
    ax2.set_xlabel("Neuron number")
    ax2.set_ylabel("Mean var explained")

    for count1, current_area in enumerate([1, 2, 4]):
        ax = hs13be.add_subplot(2, 6, count1 + 7)
        indx = T["Area"].str.contains(area[current_area - 1], na=False)
        current_T = T[indx.astype(bool)]
        for kk in range(len(current_T)):
            ops = get_session_info2(current_T, kk, data_folder)
            var_name = f"{ops.mn}_{ops.tdb}_{ops.en}_explained_var.mat"
            mean_var2 = _load_mat_var(data_folder1 / var_name, "mean_var2").ravel()
            ax.plot(np.arange(1, mean_var2.size + 1), mean_var2, color=color1[count1])
            ax.set_title(area[current_area - 1])
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 0.8)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_xticklabels(["0", "50", "100", "150", "200"])
        ax.set_yticks(np.arange(0, 0.8 + 1e-9, 0.2))
        ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 0.8 + 1e-9, 0.2)])
        ax.set_xlabel("Neuron number")
        ax.set_ylabel("Mean var explained")

    ax4 = hs13be.add_subplot(2, 2, 4)
    for i in range(3):
        current_numVar = numVar_all[i]
        ax4.scatter(current_numVar[:, 1], current_numVar[:, 2], s=8, c=color1[i])
        ax4.errorbar(
            mean_area[i, 1], mean_area[i, 2],
            yerr=sem_area[i, 2], xerr=sem_area[i, 1],
            marker="o", color=color1[i],
        )
    ax4.set_xlim(0, 50)
    ax4.set_ylim(0, 0.8)
    ax4.set_xticks(np.arange(0, 50 + 1e-9, 10))
    ax4.set_xticklabels([f"{int(v)}" for v in np.arange(0, 50 + 1e-9, 10)])
    ax4.set_yticks(np.arange(0, 0.8 + 1e-9, 0.2))
    ax4.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 0.8 + 1e-9, 0.2)])
    ax4.set_xlabel("Neuron number")
    ax4.set_ylabel("80% of mean var explained")

    hs13be.savefig(save_folder / "FigS13be_variance_summary.pdf", bbox_inches="tight")
    return hs13be
