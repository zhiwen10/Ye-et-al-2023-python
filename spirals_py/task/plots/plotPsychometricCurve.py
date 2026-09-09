from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

from spirals_py.task.plots._task_helpers import read_table_cell_array


def plotPsychometricCurve(data_folder, save_folder):
    """Translated from task/plots/plotPsychometricCurve.m

    Psychometric curves across all sessions of the 4 mice: left/right
    choice probability, miss/no-go probability and median reaction time
    per signed contrast.

    Adaptations:
    - T_trial_counts_all is a v7.3 cell array of MATLAB tables with
      empty cells; decoded with _task_helpers.read_table_cell_array
      (tables are returned in MATLAB column-major cell order).
    - MATLAB ttest p-values are computed with scipy ttest_rel and
      printed; they are not part of the figure.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    path = data_folder / "task" / "psychometric_curve" / "Task_performance_across_sessions.mat"
    empty_mask, tables = read_table_cell_array(path, "T_trial_counts_all")
    n_sess, n_mice = empty_mask.shape  # MATLAB (11 sessions, 4 mice)
    it = iter(tables)
    cell = [[None] * n_mice for _ in range(n_sess)]
    for j in range(n_mice):  # MATLAB column-major cell order
        for i in range(n_sess):
            if not empty_mask[i, j]:
                cell[i][j] = next(it)

    total_count_all = np.full((6, 11, 4), np.nan)
    total_all = np.full((11, 4), np.nan)
    for i in range(11):
        for j in range(4):
            trial_count_temp = cell[i][j]
            if trial_count_temp is not None:
                total_count_temp = trial_count_temp["total"].to_numpy()  # column 8
                total_count_temp2 = np.empty(6)
                for k in range(5):  # sum left or right same contrasts
                    total_count_temp2[k] = (
                        total_count_temp[k] + total_count_temp[10 - k]
                    )
                total_count_temp2[5] = total_count_temp[5]
                total_all_temp = total_count_temp2.sum()

                total_count_all[:, i, j] = total_count_temp2
                total_all[i, j] = total_all_temp
    total_all2 = np.nansum(total_all, axis=0)  # MATLAB sum(...,'omitnan')
    total_all_mean = round(total_all2.mean())
    total_all_sem = round(total_all2.std() / np.sqrt(4))
    sessionsa = ~np.isnan(total_all)
    sessions = sessionsa.sum(axis=0)

    total_count_mean = np.nansum(total_count_all, axis=1)  # squeeze(sum(...,2,'omitnan'))
    total_count_mean2 = total_count_mean.mean(axis=1)
    total_count_sem2 = total_count_mean.std(axis=1) / np.sqrt(4)
    total_count_mean3 = total_count_mean2 / total_count_mean2[0]
    total_count_mean3 = np.round(total_count_mean3 * 10) / 10
    contrast1 = np.array([100, 50, 25, 12.5, 6.25, 0])

    fname = ["ZYE_0085", "ZYE_0088", "ZYE_0090", "ZYE_0091"]
    contrast = np.array([-1, -0.5, -0.25, -0.125, -0.06, 0, 0.06, 0.125, 0.25, 0.5, 1])

    import h5py

    with h5py.File(path, "r") as f:
        # stored (4, 11) -> MATLAB (11, 4)
        right_choice_all_mean = np.asarray(f["right_choice_all_mean"]).T
        right_choice_all_sem = np.asarray(f["right_choice_all_sem"]).T
        no_go_all_mean = np.asarray(f["no_go_all_mean"]).T
        no_go_all_sem = np.asarray(f["no_go_all_sem"]).T
        left_choice_all_mean = np.asarray(f["left_choice_all_mean"]).T
        left_choice_all_sem = np.asarray(f["left_choice_all_sem"]).T
        rt_median_all_mean = np.asarray(f["rt_median_all_mean"]).T
        rt_median_all_sem = np.asarray(f["rt_median_all_sem"]).T

    hs14a = plt.figure(figsize=(8, 9))
    for m in range(4):
        ax = hs14a.add_subplot(4, 4, 1 + m * 4)
        ax.errorbar(
            contrast,
            right_choice_all_mean[:, m],
            yerr=right_choice_all_sem[:, m],
            fmt="or-",
            markeredgecolor="k",
            markersize=4,
        )
        ax.set_ylim(0, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-100", "-50", "0", "50", "100"])
        ax.set_xlabel("Contrast (%)")
        ax.set_ylabel("Left choice  probability")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = hs14a.add_subplot(4, 4, 2 + m * 4)
        ax.errorbar(
            contrast,
            no_go_all_mean[:, m],
            yerr=no_go_all_sem[:, m],
            fmt="or-",
            markeredgecolor="k",
            markersize=4,
        )
        ax.set_ylim(0, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-100", "-50", "0", "50", "100"])
        ax.set_xlabel("Contrast (%)")
        ax.set_ylabel("Miss/NoGo  probability")
        ax.set_title(
            f"{fname[m]} (Trials: {total_all2[m]:.0f}; sessions: {sessions[m]:.0f})"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = hs14a.add_subplot(4, 4, 3 + m * 4)
        ax.errorbar(
            contrast,
            left_choice_all_mean[:, m],
            yerr=left_choice_all_sem[:, m],
            fmt="or-",
            markeredgecolor="k",
            markersize=4,
        )
        ax.set_ylim(0, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-100", "-50", "0", "50", "100"])
        ax.set_xlabel("Contrast (%)")
        ax.set_ylabel("Right choice probability")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = hs14a.add_subplot(4, 4, 4 + m * 4)
        ax.errorbar(
            contrast,
            rt_median_all_mean[:, m],
            yerr=rt_median_all_sem[:, m],
            fmt="or-",
            markeredgecolor="k",
            markersize=4,
        )
        ax.set_ylim(0, 2)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-100", "-50", "0", "50", "100"])
        ax.set_xlabel("Contrast (%)")
        ax.set_ylabel("Reaction time (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    left_choice_mean_100 = left_choice_all_mean[10, :].mean()
    left_choice_sem_100 = left_choice_all_mean[10, :].std() / np.sqrt(4)
    left_choice_mean_6 = left_choice_all_mean[6, :].mean()
    left_choice_sem_6 = left_choice_all_mean[6, :].std() / np.sqrt(4)
    _, p1 = ttest_rel(left_choice_all_mean[10, :], left_choice_all_mean[6, :])

    rt_mean_100 = rt_median_all_mean[10, :].mean()
    rt_sem_100 = rt_median_all_mean[10, :].std() / np.sqrt(4)
    rt_mean_6 = rt_median_all_mean[6, :].mean()
    rt_sem_6 = rt_median_all_mean[6, :].std() / np.sqrt(4)
    _, p2 = ttest_rel(rt_median_all_mean[10, :], rt_median_all_mean[6, :])
    print(
        f"plotPsychometricCurve: 100% vs 6% left-choice p={p1:.4g}, "
        f"reaction-time p={p2:.4g}"
    )

    hs14a.savefig(
        save_folder / "FigS14a_Psychometric_curve_all_mice.png", bbox_inches="tight"
    )
    return hs14a
