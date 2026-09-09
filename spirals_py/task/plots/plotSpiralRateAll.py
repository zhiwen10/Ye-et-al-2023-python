from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

from spirals_py.task.plots._task_helpers2 import load_high_perf_task_sessions, load_task_outcome
from spirals_py.utils.plotting import shadedErrorBar


def plotSpiralRateAll(data_folder, save_folder):
    """Translated from task/plots/plotSpiralRateAll.m

    Spiral-rate change around stimulus onset: half-cortex spiral (CW)
    rate for correct/incorrect/miss trials (top) and full-spiral rate
    (bottom) across 4 mice.

    Adaptations:
    - The session table and the *_task_trial_ID.mat load of the MATLAB
      code are not used downstream (T1/trial_all unused) and are kept
      only as the session-table read.
    - MATLAB ttest(...) p-values are computed with scipy ttest_rel and
      printed; they are not part of the figure.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    fnames = ["ZYE_0085", "ZYE_0088", "ZYE_0090", "ZYE_0091"]
    labels = ["correct", "incorrect", "miss"]

    sampleN = 71
    half_spiral_rate_all = np.zeros((4, 3, sampleN))
    for i in range(4):
        mn = fnames[i]
        load_high_perf_task_sessions(data_folder, mn)  # MATLAB reads T1 (unused)
        T_all = load_task_outcome(data_folder, mn)

        with h5py.File(
            data_folder / "task" / "spirals_half" / f"{mn}_half_spirals4.mat", "r"
        ) as f:
            # stored (2, 71, nTrials) -> MATLAB (nTrials, 71, 2)
            halfSpiral_all = np.asarray(f["halfSpiral_all"]).transpose(2, 1, 0)

        left = T_all["left_contrast"].to_numpy()
        right = T_all["right_contrast"].to_numpy()
        lab = T_all["label"].to_numpy()
        for j in range(3):
            if j == 0:
                indx = (lab == labels[j]) & ((left - right) >= 0.25)
            else:
                indx = lab == labels[j]
            halfSpiral_temp = halfSpiral_all[indx, :, :]
            indx2 = halfSpiral_temp[:, :, 1] == -1  # MATLAB halfSpiral_temp(:,:,2)
            cw_sum = indx2.sum(axis=0)
            cw_rate = cw_sum / indx.sum()
            half_spiral_rate_all[i, j, :] = cw_rate

    half_rate_mean = half_spiral_rate_all.mean(axis=0)
    half_rate_sem = half_spiral_rate_all.std(axis=0) / np.sqrt(4)
    # half spiral stats
    pre_rate = half_spiral_rate_all[:, :, 35]  # MATLAB (:,:,36)
    post_rate = half_spiral_rate_all[:, :, 44]  # MATLAB (:,:,45)
    _, p2 = ttest_rel(pre_rate, post_rate, axis=0)

    # full spiral raster plot
    with h5py.File(
        data_folder / "task" / "spirals_large" / "task_spiral_count_radius_100.mat", "r"
    ) as f:
        # stored (141, 4, 3) -> MATLAB (3 labels, 4 mice, 141 time)
        spiral_count_sum_all = np.asarray(f["spiral_count_sum_all"]).transpose(2, 1, 0)

    # full spiral stats
    spiral_count_mean2 = spiral_count_sum_all[:, :, 74:82].mean(axis=2)  # 75:82
    spiral_count_mean3 = spiral_count_sum_all[:, :, 62:70].mean(axis=2)  # 63:70
    _, p3 = ttest_rel(spiral_count_mean2.T, spiral_count_mean3.T, axis=0)
    print(
        f"plotSpiralRateAll: half-spiral pre vs post p={np.round(p2, 4)}, "
        f"full-spiral pre vs post p={np.round(p3, 4)}"
    )

    # MATLAB also computes spira_mean_all_pre1/post1 and spiral_sem_pre1/
    # post1 from squeeze(mean(spiral_count_sum_all(1,:,63:70),3)) -- dead
    # code on a 4x1 squeeze result, never used downstream; skipped.

    # half spiral raster plot
    color1 = ["k", "g", "r"]
    h5e = plt.figure(figsize=(8, 5))
    for i in range(3):
        ax = h5e.add_subplot(2, 3, i + 1)
        shadedErrorBar(
            np.arange(1, 72),
            half_rate_mean[i, :],
            half_rate_sem[i, :],
            lineProps=color1[i],
            ax=ax,
        )
        ax.set_ylim(0, 0.15)
        ax.axvline(36, linestyle="--", color="k")
        ax.set_xticks([1, 18, 36, 54, 71])
        ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
        ax.set_xlim(1, 71)
        ax.set_yticks([0, 0.05, 0.1, 0.15])
        ax.set_yticklabels(["0", "5%", "10%", "15%"])

    for i in range(3):
        spiral_count_sum_all1 = spiral_count_sum_all[i, :, :]  # (4 mice, 141)
        spiral_count_mean = spiral_count_sum_all1.mean(axis=0)
        spiral_count_sem = spiral_count_sum_all1.std(axis=0) / np.sqrt(4)
        ax = h5e.add_subplot(2, 3, i + 4)
        shadedErrorBar(
            np.arange(1, 72),
            spiral_count_mean[34:105],  # MATLAB 35:105
            spiral_count_sem[34:105],
            lineProps=color1[i],
            ax=ax,
        )
        ax.axvline(36, linestyle="--", color="k")
        ax.set_xticks([1, 18, 36, 54, 71])
        ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
        ax.set_ylim(0, 0.06)
        ax.set_xlim(1, 71)
        ax.set_yticks([0, 0.02, 0.04, 0.06])
        ax.set_yticklabels(["0", "2%", "4%", "6%"])

    h5e.savefig(save_folder / "Fig5e_half_and_full_spirals.png", bbox_inches="tight")
    return h5e
