from pathlib import Path

import numpy as np
import pandas as pd

from spirals_py.task.plots._task_helpers_s15 import load_block
from spirals_py.task.preprocessing._task_io import nanmean, nanstd, save_mat
from spirals_py.task.preprocessing._task_utils import (
    CONTRAST_STIM,
    RATIO_COLS,
    TASK_MICE,
    build_outcome_table,
    load_task_sessions,
    session_dirs,
)
from spirals_py.task.preprocessing.getWheelOnsetTime import getWheelOnsetTime


def getPsychometricCurve(data_folder, save_folder):
    """Translated from task/preprocessing/getPsychometricCurve.m

    Psychometric curves for all subjects: per signed contrast, the
    median wheel-onset time and the left/right choice and no-go
    probabilities, averaged across the high-performance task sessions of
    each mouse (mean and SEM across sessions).

    Saved as Task_performance_across_sessions.mat with
    T_trial_counts_all (a cell array of per-session trial-count tables,
    stored as structs of column arrays), and the
    *_mean / *_sem matrices.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    rt_median_all_mean = np.full((11, 4), np.nan)
    rt_median_all_sem = np.full((11, 4), np.nan)
    left_choice_all_mean = np.full((11, 4), np.nan)
    left_choice_all_sem = np.full((11, 4), np.nan)
    right_choice_all_mean = np.full((11, 4), np.nan)
    right_choice_all_sem = np.full((11, 4), np.nan)
    no_go_all_mean = np.full((11, 4), np.nan)
    no_go_all_sem = np.full((11, 4), np.nan)
    T_trial_counts = {}  # (kk, m) -> DataFrame, MATLAB T_trial_counts_all{kk, m}
    max_sessions = 0

    for m in range(4):
        mn = TASK_MICE[m]
        T1 = load_task_sessions(data_folder, mn)
        sessions = len(T1)
        max_sessions = max(max_sessions, sessions)
        rt_median_all = np.full((11, sessions), np.nan)
        left_choice_all = np.full((11, sessions), np.nan)
        right_choice_all = np.full((11, sessions), np.nan)
        no_go_all = np.full((11, sessions), np.nan)

        for kk in range(sessions):
            _, session_root = session_dirs(data_folder, T1, kk)
            # load block (MATLAB loads [td _ block_en _ mn _Block.mat];
            # the single *_Block.mat of the session is used here)
            block = load_block(session_root)

            ntrial = np.size(block.events.endTrialValues)
            norepeatValues = np.asarray(block.events.repeatNumValues).ravel()[:ntrial]
            response = np.asarray(block.events.responseValues).ravel()[:ntrial]
            norepeat_indx = norepeatValues == 1
            response = response[norepeat_indx]
            feedback = (
                np.asarray(block.events.feedbackValues).ravel()[:ntrial][norepeat_indx]
            )
            left_contrast = (
                np.asarray(block.events.contrastLeftValues)
                .ravel()[:ntrial][norepeat_indx]
            )
            right_contrast = (
                np.asarray(block.events.contrastRightValues)
                .ravel()[:ntrial][norepeat_indx]
            )
            # reaction time
            reaction_time = (
                np.asarray(block.events.responseTimes).ravel()[:ntrial]
                - np.asarray(block.events.stimulusOnTimes).ravel()[:ntrial]
            )[norepeat_indx]

            wheel_onset = getWheelOnsetTime(session_root, block)[norepeat_indx]

            T = build_outcome_table(
                left_contrast, right_contrast, response, feedback, reaction_time
            )
            T["wheel_onset"] = wheel_onset

            ratio = np.full((11, 6), np.nan)
            trial_counts = np.zeros((11, 6))  # MATLAB sum over an empty bin is 0
            sum_counts = np.zeros(11)
            rt_median = np.full(11, np.nan)
            for i in range(11):
                T_temp = T[
                    (T.left_contrast == CONTRAST_STIM[i, 0])
                    & (T.right_contrast == CONTRAST_STIM[i, 1])
                ]
                n_i = len(T_temp)
                if n_i:
                    ratio[i, :] = (
                        T_temp[RATIO_COLS].to_numpy(dtype=float).sum(axis=0) / n_i
                    )
                    trial_counts[i, :] = T_temp[RATIO_COLS].to_numpy(dtype=float).sum(
                        axis=0
                    )
                sum_counts[i] = n_i
                sel = T_temp.label.isin(["correct", "falarmL", "falarmR"])
                rt = T_temp.loc[sel, "wheel_onset"].to_numpy(dtype=float)
                if rt.size:
                    rt_median[i] = np.median(rt)  # MATLAB median

            T_trial_counts[(kk, m)] = pd.DataFrame(
                np.column_stack(
                    [
                        np.array(
                            [-1, -0.5, -0.25, -0.125, -0.06, 0, 0.06, 0.125, 0.25, 0.5, 1]
                        ),
                        trial_counts,
                        sum_counts,
                    ]
                ),
                columns=["contrast"] + RATIO_COLS + ["total"],
            )

            # MATLAB T_ratio column indices (1-based): 3 correct, 4 incorrect,
            # 5 reject, 6 falarmL, 7 falarmR
            left_choice_all[:, kk] = np.concatenate(
                [ratio[:5, 3], [ratio[5, 5]], ratio[6:11, 2]]
            )
            right_choice_all[:, kk] = np.concatenate(
                [ratio[:5, 2], [ratio[5, 6]], ratio[6:11, 3]]
            )
            no_go_all[:, kk] = np.concatenate(
                [ratio[:5, 1], [ratio[5, 4]], ratio[6:11, 1]]
            )
            rt_median_all[:, kk] = rt_median
            print(f"getPsychometricCurve: {mn} session {kk + 1}/{sessions}")

        # mean and SEM across sessions (MATLAB omitnan; the SEM keeps the
        # total session count in the denominator, as in the MATLAB code)
        n_cols = rt_median_all.shape[1] or 1
        rt_median_all_mean[:, m] = nanmean(rt_median_all, axis=1)
        rt_median_all_sem[:, m] = nanstd(rt_median_all, axis=1) / np.sqrt(n_cols)
        left_choice_all_mean[:, m] = nanmean(left_choice_all, axis=1)
        left_choice_all_sem[:, m] = nanstd(left_choice_all, axis=1) / np.sqrt(n_cols)
        right_choice_all_mean[:, m] = nanmean(right_choice_all, axis=1)
        right_choice_all_sem[:, m] = nanstd(right_choice_all, axis=1) / np.sqrt(n_cols)
        no_go_all_mean[:, m] = nanmean(no_go_all, axis=1)
        no_go_all_sem[:, m] = nanstd(no_go_all, axis=1) / np.sqrt(n_cols)

    # materialize the (max_sessions, 4) cell array of trial-count tables
    T_trial_counts_all = np.empty((max_sessions, 4), dtype=object)
    for i in range(max_sessions):
        for j in range(4):
            T_trial_counts_all[i, j] = T_trial_counts.get((i, j), np.zeros((0, 0)))

    save_mat(
        save_folder / "Task_performance_across_sessions.mat",
        {
            "T_trial_counts_all": T_trial_counts_all,
            "rt_median_all_mean": rt_median_all_mean,
            "rt_median_all_sem": rt_median_all_sem,
            "left_choice_all_mean": left_choice_all_mean,
            "left_choice_all_sem": left_choice_all_sem,
            "right_choice_all_mean": right_choice_all_mean,
            "right_choice_all_sem": right_choice_all_sem,
            "no_go_all_mean": no_go_all_mean,
            "no_go_all_sem": no_go_all_sem,
        },
    )
