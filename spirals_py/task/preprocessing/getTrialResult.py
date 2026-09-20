import numpy as np
import pandas as pd

from spirals_py.task.plots._task_helpers_s15 import load_block
from spirals_py.task.preprocessing._task_utils import session_dirs
from spirals_py.task.preprocessing.getWheelOnsetTime import getWheelOnsetTime
from spirals_py.task.preprocessing.parse_action_table import parse_action_table


def getTrialResult(T1, data_folder):
    """Translated from task/preprocessing/getTrialResult.m

    Behavior table of all non-repeat trials across the sessions of T1,
    with the first wheel-movement onset time added.

    Adaptation: the MATLAB source looks for the sessions under
    data_folder/task_svd; the data release (and getTrialID.m /
    getPsychometricCurve.m) uses data_folder/task/task_svd, which is
    used here.
    """
    T_all = []
    for kk in range(len(T1)):
        _, session_root = session_dirs(data_folder, T1, kk)
        block = load_block(session_root)
        # get behavior results
        T, _ = parse_action_table(block)
        ntrial = np.size(block.events.endTrialValues)
        norepeatValues = np.asarray(block.events.repeatNumValues).ravel()[:ntrial]
        norepeat_indx = norepeatValues == 1
        wheel_onset = getWheelOnsetTime(session_root, block)[norepeat_indx]
        T = T.copy()
        T["wheel_onset"] = wheel_onset
        T_all.append(T)
    if not T_all:
        return pd.DataFrame()
    return pd.concat(T_all, ignore_index=True)
