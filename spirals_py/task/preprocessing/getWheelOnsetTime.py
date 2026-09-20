import numpy as np

from spirals_py.task.plots._task_helpers import (
    correctCounterDiscont,
    load_h5_var,
    tsToT,
)
from spirals_py.task.plots._task_helpers_s15 import getPhotodiodeTime
from spirals_py.task.preprocessing._task_utils import interp1_nan
from spirals_py.task.preprocessing.findWheelMoves3 import findWheelMoves3


def getWheelOnsetTime(session_root, block):
    """Translated from task/preprocessing/getWheelOnsetTime.m

    First wheel-movement onset time relative to each stimulus onset
    (photodiode) time, from the rotary-encoder signal.
    """
    # get photodiode time
    win = [0, 5000]
    allPD2 = getPhotodiodeTime(session_root, win)
    ntrial = np.size(block.events.endTrialValues)
    allPD2 = allPD2[:ntrial]

    # rotaryEncoder
    sigName = "rotaryEncoder"
    pd_sig = load_h5_var(session_root / f"{sigName}_raw.mat", "pd").ravel()
    tlTimes = load_h5_var(session_root / f"{sigName}_timestamps_Timeline.mat", "tlTimes")
    tt = tsToT(tlTimes, pd_sig.size)
    wh = correctCounterDiscont(pd_sig)

    # wheel onset time
    trialWin = [0, 5]
    wheel_onset = np.full(allPD2.size, np.nan)
    trialWin3 = np.arange(0, trialWin[1] + 1e-9, 1 / 200)  # MATLAB 0:1/200:5
    for i in range(allPD2.size):
        wt = allPD2[i] + trialWin3
        wval = interp1_nan(tt, wh, wt)
        mon, _, _, _ = findWheelMoves3(wval, wt, 200)
        if mon.size:
            wheel_onset[i] = mon[0] - allPD2[i]
    return wheel_onset
