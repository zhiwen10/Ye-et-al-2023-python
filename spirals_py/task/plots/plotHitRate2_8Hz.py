from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.task.plots._task_helpers2 import (
    hit_rate_panels,
    load_task_freq_arrays,
    load_task_outcome,
)
from spirals_py.task.preprocessing.sort_ratio_by_contrast2 import sort_ratio_by_contrast2


def plotHitRate2_8Hz(data_folder, save_folder):
    """Translated from task/plots/plotHitRate2_8Hz.m

    Psychometric curve split by pre-stimulus 2-8 Hz amplitude (high-amp
    quartile) combined with 0.05-2 Hz low-amp quartile and 2-8 Hz onset
    phase (near ±pi/2 vs near 0), and the hit-rate change, across 4 mice.

    Adaptations:
    - The anovan call is commented out in the MATLAB source and is
      skipped here as well.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    fname = ["ZYE_0085", "ZYE_0088", "ZYE_0090", "ZYE_0091"]
    freq_high = [2, 8]
    freq_low = [0.05, 2]
    hit_rate_high = np.full((5, 4), np.nan)
    miss_rate_high = np.full((5, 4), np.nan)
    hit_rate_low = np.full((5, 4), np.nan)
    miss_rate_low = np.full((5, 4), np.nan)
    trial_count_high_all = np.full((5, 4), np.nan)
    trial_count_low_all = np.full((5, 4), np.nan)

    for m in range(4):
        mn = fname[m]
        T_all = load_task_outcome(data_folder, mn)
        arrs_high = load_task_freq_arrays(
            data_folder
            / "task"
            / "task_outcome"
            / f"{mn}_task_freq_to{freq_high[1]}Hz.mat",
            ["amp_all", "phase_all"],
        )
        amp_high = arrs_high["amp_all"]
        phase_all = arrs_high["phase_all"]
        arrs_low = load_task_freq_arrays(
            data_folder
            / "task"
            / "task_outcome"
            / f"{mn}_task_freq_to{freq_low[1]}Hz.mat",
            ["amp_all"],
        )
        amp_low = arrs_low["amp_all"]

        pixel_indx = 141  # MATLAB 1-based
        amp_high1 = amp_high[122:141, 0, :].mean(axis=0)  # MATLAB 123:141
        amp_low1 = amp_low[0:141, 0, :].mean(axis=0)  # MATLAB 1:141
        p_high1 = np.percentile(amp_high1, 25)
        p_low1 = np.percentile(amp_low1, 25)
        phase1 = phase_all[pixel_indx - 1, 0, :]
        T_high = T_all[
            (amp_high1 > p_high1)
            & (amp_low1 < p_low1)
            & (phase1 > -np.pi / 2)
            & (phase1 < np.pi / 2)
        ]
        T_low = T_all[
            (amp_high1 > p_high1)
            & (amp_low1 < p_low1)
            & ((phase1 < -np.pi * 1 / 2) | (phase1 > np.pi * 1 / 2))
        ]
        # right hemisphere
        (
            T_high_ratio,
            rt_high_median,
            trial_count_high,
            reaction_time_sort_high,
        ) = sort_ratio_by_contrast2(T_high)
        (
            T_low_ratio,
            rt_low_median,
            trial_count_low,
            reaction_time_sort_low,
        ) = sort_ratio_by_contrast2(T_low)

        hit_rate_high[:, m] = T_high_ratio["correct"]
        hit_rate_low[:, m] = T_low_ratio["correct"]
        trial_count_high_all[:, m] = trial_count_high
        trial_count_low_all[:, m] = trial_count_low

    subj = np.array([1, 2, 3, 4])
    subjs = np.tile(subj, (5, 1))
    subjs2 = np.concatenate([subjs, subjs])
    hit_rates = np.concatenate([hit_rate_high, hit_rate_low])
    contrast_matrix1 = np.array([[6.25], [12.5], [25], [50], [100]])
    contrast_matrix2 = np.tile(contrast_matrix1, (1, 4))
    contrast_matrix = np.concatenate([contrast_matrix2, contrast_matrix2])
    voltage_label1 = np.ones((5, 4))
    voltage_label2 = np.ones((5, 4)) + 1
    voltage_label = np.concatenate([voltage_label1, voltage_label2])
    subjs2 = subjs2.ravel()
    hit_rates = hit_rates.ravel()
    contrast_matrix = contrast_matrix.ravel()
    voltage_label = voltage_label.ravel()

    hit_rate_change = hit_rate_high - hit_rate_low
    miss_rate_change = miss_rate_high - miss_rate_low
    hit_rate_change_mean = hit_rate_change.mean(axis=1)
    hit_rate_change_sem = hit_rate_change.std(axis=1) / np.sqrt(6)  # MATLAB sqrt(6)

    h5jk = plt.figure(figsize=(6, 3))
    hit_rate_panels(
        h5jk,
        hit_rate_high,
        hit_rate_low,
        hit_rate_change,
        hit_rate_change_mean,
        hit_rate_change_sem,
        ["6%", "12%", "25%", "50%", "100%"],
    )
    h5jk.savefig(save_folder / "Fig5jk_hit_ratio_2_8Hz.png", bbox_inches="tight")
    return h5jk
