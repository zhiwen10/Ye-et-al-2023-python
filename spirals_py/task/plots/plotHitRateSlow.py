from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.task.plots._task_helpers2 import (
    anovan2_random,
    hit_rate_panels,
    load_task_freq_arrays,
    load_task_outcome,
)
from spirals_py.task.preprocessing.sort_ratio_by_contrast2 import sort_ratio_by_contrast2


def plotHitRateSlow(data_folder, save_folder):
    """Translated from task/plots/plotHitRateSlow.m

    Psychometric curve split by pre-stimulus 0.05-2 Hz power (low-power
    quartile 'high' vs high-power quartile 'low', MATLAB naming kept) and
    the hit-rate change between quartiles, across 4 mice.

    Adaptations:
    - MATLAB anovan(...,'model',2,'random',3) is approximated by a
      statsmodels Type-II ANOVA (see _task_helpers2.anovan2_random); the
      result (pp_hit) is not used for the figure.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    fname = ["ZYE_0085", "ZYE_0088", "ZYE_0090", "ZYE_0091"]
    hit_rate_high = np.full((5, 4), np.nan)
    miss_rate_high = np.full((5, 4), np.nan)
    hit_rate_low = np.full((5, 4), np.nan)
    miss_rate_low = np.full((5, 4), np.nan)
    rt_high = np.full((5, 4), np.nan)
    rt_low = np.full((5, 4), np.nan)
    freq = [0.05, 2]

    for m in range(4):
        mn = fname[m]
        T_all = load_task_outcome(data_folder, mn)
        arrs = load_task_freq_arrays(
            data_folder / "task" / "task_outcome" / f"{mn}_task_freq_to{freq[1]}Hz.mat",
            ["amp_all"],
        )
        amp_all = arrs["amp_all"]  # (281, comp, trial)

        amp = amp_all[0:141, 0, :].mean(axis=0)  # MATLAB 1:141, pixel 1
        p1 = np.percentile(amp, 25)
        p2 = np.percentile(amp, 75)
        T_high = T_all[amp < p1]
        T_low = T_all[amp > p2]
        # right hemisphere
        T_high_ratio, rt_high_median = sort_ratio_by_contrast2(T_high)[:2]
        T_low_ratio, rt_low_median = sort_ratio_by_contrast2(T_low)[:2]

        hit_rate_high[:, m] = T_high_ratio["correct"]
        miss_rate_high[:, m] = T_high_ratio["miss"]
        rt_high[:, m] = rt_high_median
        hit_rate_low[:, m] = T_low_ratio["correct"]
        miss_rate_low[:, m] = T_low_ratio["miss"]
        rt_low[:, m] = rt_low_median

    subj = np.array([1, 2, 3, 4])
    subjs = np.tile(subj, (5, 1))
    subject = np.concatenate([subjs, subjs])
    hit_rates = np.concatenate([hit_rate_high, hit_rate_low])
    miss_rates = np.concatenate([miss_rate_high, miss_rate_low])
    rt = np.concatenate([rt_high, rt_low])
    contrast_matrix1 = np.array([[6.25], [12.5], [25], [50], [100]])
    contrast_matrix2 = np.tile(contrast_matrix1, (1, 4))
    contrasts_l = np.concatenate([contrast_matrix2, contrast_matrix2])
    voltage_label1 = np.ones((5, 4))
    voltage_label2 = np.ones((5, 4)) + 1
    voltage_label = np.concatenate([voltage_label1, voltage_label2])

    subject = subject.ravel()
    rt = rt.ravel()
    hit_rates = hit_rates.ravel()
    miss_rates = miss_rates.ravel()
    contrasts_l = contrasts_l.ravel()
    voltage_label2c = voltage_label.ravel()
    voltage_label3 = voltage_label.ravel()
    label = np.full(40, "", dtype=object)
    label[voltage_label3 == 1] = "high"
    label[voltage_label3 == 2] = "low"

    pp_hit = anovan2_random(hit_rates, contrasts_l, voltage_label2c, subject)

    hit_rate_change = hit_rate_high - hit_rate_low
    miss_rate_change = miss_rate_high - miss_rate_low
    hit_rate_change_mean = hit_rate_change.mean(axis=1)
    hit_rate_change_sem = hit_rate_change.std(axis=1) / np.sqrt(6)  # MATLAB sqrt(6)

    h5gh = plt.figure(figsize=(6, 3))
    hit_rate_panels(
        h5gh,
        hit_rate_high,
        hit_rate_low,
        hit_rate_change,
        hit_rate_change_mean,
        hit_rate_change_sem,
        ["6%", "12.5%", "25%", "50%", "100%"],
    )
    h5gh.savefig(save_folder / "Fig5gh_hit_ratio_005_2Hz.png", bbox_inches="tight")
    return h5gh
