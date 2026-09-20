import numpy as np
import pandas as pd

from spirals_py.task.plots._task_helpers_s15 import load_block
from spirals_py.task.preprocessing._task_utils import session_dirs
from spirals_py.task.preprocessing.parse_action_table import parse_action_table


def getTrialID(T1, data_folder):
    """Translated from task/preprocessing/getTrialID.m

    Behavior table of all non-repeat trials with the (1-based) session
    number and within-session trial number prepended
    (MATLAB T = T(:, [13, 14, 1:12])).
    """
    T_all = []
    for kk in range(len(T1)):
        _, session_root = session_dirs(data_folder, T1, kk)
        block = load_block(session_root)
        # get behavior results
        T, _ = parse_action_table(block)
        T = T.copy()
        T.insert(0, "trial", np.arange(1, len(T) + 1))
        T.insert(0, "session", np.full(len(T), kk + 1))
        T_all.append(T)
    if not T_all:
        return pd.DataFrame()
    return pd.concat(T_all, ignore_index=True)
