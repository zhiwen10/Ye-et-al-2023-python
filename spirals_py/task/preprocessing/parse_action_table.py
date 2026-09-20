import numpy as np
import pandas as pd

from spirals_py.task.preprocessing._task_utils import (
    CONTRAST_SIGNED,
    CONTRAST_STIM,
    RATIO_COLS,
    build_outcome_table,
)


def parse_action_table(block):
    """Translated from task/preprocessing/parse_action_table.m

    Per-trial behavior table (non-repeat trials only) with outcome
    indicator columns, label and reaction time, plus the outcome-ratio
    table T_ratio per signed contrast.

    Returns (T, T_ratio).  The reaction_time_sort / rt_mean / rt_sem of
    the MATLAB source are computed but never returned and are skipped.

    MATLAB indexes contrastLeftValues / feedbackValues directly with the
    (ntrial-long) norepeat mask; here the arrays are first truncated to
    ntrial (identical when the lengths match, robust when the saved
    arrays are longer, as in plotTaskSessionEpochExample.py).
    """
    ntrial = np.size(block.events.endTrialValues)
    norepeatValues = np.asarray(block.events.repeatNumValues).ravel()[:ntrial]
    norepeat_indx = norepeatValues == 1

    response = np.asarray(block.events.responseValues).ravel()[:ntrial][norepeat_indx]
    feedback = np.asarray(block.events.feedbackValues).ravel()[:ntrial][norepeat_indx]
    left_contrast = (
        np.asarray(block.events.contrastLeftValues).ravel()[:ntrial][norepeat_indx]
    )
    right_contrast = (
        np.asarray(block.events.contrastRightValues).ravel()[:ntrial][norepeat_indx]
    )
    # reaction time
    response_times = np.asarray(block.events.responseTimes).ravel()[:ntrial]
    stimulus_on_times = np.asarray(block.events.stimulusOnTimes).ravel()[:ntrial]
    reaction_time = (response_times - stimulus_on_times)[norepeat_indx]

    T = build_outcome_table(
        left_contrast, right_contrast, response, feedback, reaction_time
    )

    ratio = np.full((11, 6), np.nan)
    for i in range(11):
        T_temp = T[
            (T.left_contrast == CONTRAST_STIM[i, 0])
            & (T.right_contrast == CONTRAST_STIM[i, 1])
        ]
        if len(T_temp):
            ratio[i, :] = (
                T_temp[RATIO_COLS].to_numpy(dtype=float).sum(axis=0) / len(T_temp)
            )

    T_ratio = pd.DataFrame(
        np.column_stack([CONTRAST_SIGNED, ratio]), columns=["contrast"] + RATIO_COLS
    )
    return T, T_ratio
