import numpy as np
import pandas as pd

_RATIO_COLS = ["miss", "correct", "incorrect", "reject", "falarmL", "falarmR"]


def sort_ratio_by_contrast2(T_all):
    """Translated from task/preprocessing/sort_ratio_by_contrast2.m

    T_all: DataFrame with columns left_contrast, right_contrast, label,
    wheel_onset and the indicator columns miss/correct/incorrect/reject/
    falarmL/falarmR.
    Returns (T_ratio, rt_median, trial_count, reaction_time_sort).
    """
    contrast = np.array([0.06, 0.125, 0.25, 0.5, 1.0])
    ratio = np.full((5, 6), np.nan)
    trial_count = np.zeros(5)
    rt_median = np.full(5, np.nan)
    reaction_time_sort = []
    cdiff = np.abs(T_all["left_contrast"].to_numpy() - T_all["right_contrast"].to_numpy())
    for i in range(5):
        T_temp = T_all[cdiff == contrast[i]]
        ratio[i, :] = T_temp[_RATIO_COLS].to_numpy(dtype=float).sum(axis=0) / len(T_temp)
        sel = T_temp["label"].isin(["correct", "falarmL", "falarmR"])
        rt = T_temp.loc[sel, "wheel_onset"].to_numpy(dtype=float)
        reaction_time_sort.append(rt)
        rt_median[i] = rt.mean()
        trial_count[i] = len(T_temp)
    T_ratio = pd.DataFrame(
        np.column_stack([contrast, ratio]),
        columns=["contrast"] + _RATIO_COLS,
    )
    return T_ratio, rt_median, trial_count, reaction_time_sort
