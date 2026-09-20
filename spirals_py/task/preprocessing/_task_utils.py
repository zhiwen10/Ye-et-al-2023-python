"""Shared helpers for the task preprocessing pipeline (pipeline6_task).

Sources (MATLAB):
- task/utils/getPhotodiodeTime2.m and task/utils/getPassiveContrasts.m
  (getPhotodiodeTime itself is available as
  ``spirals_py.task.plots._task_helpers_s15.getPhotodiodeTime``)
- task/utils/density_color_plot2.m
- the session prologue and outcome-table body repeated across
  task/preprocessing/*.m (parse_action_table.m, getPsychometricCurve.m,
  getTrialResult.m, getTrialID.m, getTaskSpirals.m, ...)
- affine point transforms and grouped-spiral loading are reused from
  ``spirals_py.spirals.plots._fig1_helpers_s5``
  (transformPointsForward / ismember-rows / find [col,row] semantics)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _index_xy,
    _ismember_rows,
    _load_spirals_grouping,
    _transformPointsForward,
)
from spirals_py.task.plots._task_helpers import load_h5_var, tsToT
from spirals_py.task.plots._task_helpers_s15 import (
    getPhotodiodeTime,
    load_block,
    matlab_round,
    schmittTimes,
)

__all__ = [
    "getPhotodiodeTime",
    "getPhotodiodeTime2",
    "getPassiveContrasts",
    "density_color_plot2",
    "build_outcome_table",
    "load_task_sessions",
    "session_dirs",
    "first_index_after",
    "interp1_nan",
    "load_spirals_grouping",
    "transformPointsForward",
    "ismember_rows",
    "brain_index_from_atlas",
]

RATIO_COLS = ["miss", "correct", "incorrect", "reject", "falarmL", "falarmR"]

# signed contrast bins of the 2AFC task (parse_action_table.m)
CONTRAST_SIGNED = np.array(
    [-1, -0.5, -0.25, -0.125, -0.06, 0, 0.06, 0.125, 0.25, 0.5, 1]
)
CONTRAST_STIM = np.array(
    [
        [1, 0],
        [0.5, 0],
        [0.25, 0],
        [0.125, 0],
        [0.06, 0],
        [0, 0],
        [0, 0.06],
        [0, 0.125],
        [0, 0.25],
        [0, 0.5],
        [0, 1],
    ]
)

TASK_MICE = ["ZYE_0085", "ZYE_0088", "ZYE_0090", "ZYE_0091"]


# ------------------------------------------------------- wheel / photodiode
def getPhotodiodeTime2(session_root, block, win):
    """Translated from task/utils/getPhotodiodeTime2.m

    Unlike getPhotodiodeTime (which cleans the combined up/down flip
    times), this returns the first ``ntrials`` upward flips only, where
    ntrials is taken from ``block.events.endTrialValues``.
    """
    session_root = Path(session_root)
    pd_sig = load_h5_var(session_root / "photodiode_raw.mat", "pd").ravel()
    tlTimes = load_h5_var(session_root / "photodiode_timestamps_Timeline.mat", "tlTimes")
    tt = tsToT(tlTimes, pd_sig.size)

    _, flipsUp, _ = schmittTimes(tt, pd_sig, [0.5, 0.8])
    flipsUp = flipsUp[(flipsUp >= win[0]) & (flipsUp <= win[1])]
    flipsUp = flipsUp[:-1]
    ntrials = np.size(block.events.endTrialValues)
    return flipsUp[:ntrials]


def getPassiveContrasts(T1, data_folder, win):
    """Translated from task/utils/getPassiveContrasts.m

    (ntrials, 2) left/right contrast of every passive-viewing trial in
    the sessions of table T1.
    """
    contrast_all = []
    for kk in range(len(T1)):
        fname, session_root = session_dirs(data_folder, T1, kk)
        block = load_block(session_root)
        allPD2 = getPhotodiodeTime2(session_root, block, win)
        ntrials = np.size(block.events.endTrialValues)
        left_contrast = np.asarray(block.events.contrastLeftValues).ravel()[:ntrials]
        right_contrast = np.asarray(block.events.contrastRightValues).ravel()[:ntrials]
        contrast_all.append(np.column_stack([left_contrast, right_contrast]))
    return np.vstack(contrast_all)


# ------------------------------------------------------- spiral statistics
def density_color_plot2(pwAllRaw, histbin):
    """Translated from task/utils/density_color_plot2.m

    For every unique (x, y) center, count the spirals within a
    +-histbin/2 pixel square window.  Returns an (n, 3) array
    [x, y, count].
    """
    half = histbin / 2
    pwAllRaw = np.asarray(pwAllRaw, dtype=float)
    if pwAllRaw.size == 0:
        return np.zeros((0, 3))
    # MATLAB unique(..., 'rows') returns sorted unique rows
    unique_spirals1 = np.unique(pwAllRaw[:, :2], axis=0)
    counts = np.zeros(unique_spirals1.shape[0], dtype=int)
    x, y = pwAllRaw[:, 0], pwAllRaw[:, 1]
    for k in range(unique_spirals1.shape[0]):
        cx, cy = unique_spirals1[k]
        indx = (
            (x >= cx - half)
            & (x <= cx + half)
            & (y >= cy - half)
            & (y <= cy + half)
        )
        counts[k] = indx.sum()
    return np.column_stack([unique_spirals1, counts.astype(float)])


def load_spirals_grouping(path, min_duration=2):
    """Load archiveCell from a *_spirals_group_fftn.mat file, keeping
    cells with at least *min_duration* frames (getTaskSpirals.m /
    getPassiveSpirals.m).  Returns the concatenated (n, 5) array
    [x y radius direction frame] in MATLAB orientation."""
    cells, _ = _load_spirals_grouping(path, min_duration=min_duration)
    if not cells:
        return np.zeros((0, 5))
    return np.vstack(cells)


def transformPointsForward(tform, x, y):
    """MATLAB transformPointsForward(affine2d(tform), x, y)."""
    return _transformPointsForward(tform, x, y)


def ismember_rows(a, b):
    """MATLAB ismember(a, b, 'rows') logical output."""
    return _ismember_rows(a, b)


def brain_index_from_atlas(projectedAtlas1):
    """MATLAB BW = logical(projectedAtlas1); [row, col] = find(BW);
    brain_index = [col, row] (i.e. [x y] pixel pairs)."""
    return _index_xy(np.asarray(projectedAtlas1).astype(bool))


# ------------------------------------------------------- session handling
def load_task_sessions(data_folder, mn, label="task"):
    """Read task/sessions/[mn].xlsx and keep the sessions of *label*.

    Task sessions are additionally filtered for performance
    (hit_left > 0.7 & hit_right > 0.7); passive sessions are not
    (getTaskSpirals.m / getPassiveSpirals.m).
    """
    T_session = pd.read_excel(Path(data_folder) / "task" / "sessions" / f"{mn}.xlsx")
    T1 = T_session[T_session.label == label]
    if label == "task":
        T1 = T1[(T1.hit_left > 0.7) & (T1.hit_right > 0.7)]
    return T1.reset_index(drop=True)


def session_dirs(data_folder, T1, kk):
    """Session prologue shared by the task preprocessing functions:
    mouse id / date / folder -> (fname, session_root).

    fname is [MouseID]_[yyyymmdd]_[folder] and session_root is
    task/task_svd/fname, as in getTrialID.m / getPsychometricCurve.m /
    getTaskSpirals.m.
    """
    mn = str(T1.MouseID.iloc[kk])
    tda = pd.Timestamp(T1.date.iloc[kk])
    en = int(T1.folder.iloc[kk])
    tdb = tda.strftime("%Y%m%d")
    fname = f"{mn}_{tdb}_{en}"
    session_root = Path(data_folder) / "task" / "task_svd" / fname
    return fname, session_root


# ------------------------------------------------------- small utilities
def first_index_after(t, events):
    """MATLAB find(t - event > 0, 1, 'first') for every event: 1-based
    index of the first sample after each event time."""
    t = np.asarray(t, dtype=float).ravel()
    events = np.atleast_1d(np.asarray(events, dtype=float))
    out = np.empty(events.size, dtype=int)
    for i, ev in enumerate(events):
        indx = np.flatnonzero(t - ev > 0)
        if indx.size == 0:
            raise IndexError(
                f"no sample of t after event {ev:.3f}s (find(t-ev>0,1,'first') is empty)"
            )
        out[i] = indx[0] + 1  # 1-based
    return out


def interp1_nan(x, v, xq):
    """MATLAB interp1 semantics: NaN outside the range of x."""
    x = np.asarray(x, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    xq = np.asarray(xq, dtype=float)
    out = np.interp(xq, x, v)
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out


# ------------------------------------------------------- outcome table
def build_outcome_table(left_contrast, right_contrast, response, feedback, reaction_time):
    """Outcome-indicator table body shared by parse_action_table.m and
    getPsychometricCurve.m.

    Returns a DataFrame with columns left_contrast, right_contrast,
    response, feedback, miss, correct, incorrect, reject, falarmL,
    falarmR, label, reactionTime.  The 0.006 -> 0.06 contrast typo of
    the recorded data is fixed as in the MATLAB code.
    """
    left_contrast = np.asarray(left_contrast, dtype=float).ravel()
    right_contrast = np.asarray(right_contrast, dtype=float).ravel()
    response = np.asarray(response).ravel()
    feedback = np.asarray(feedback).ravel()
    reaction_time = np.asarray(reaction_time, dtype=float).ravel()

    diff = right_contrast - left_contrast
    miss = (diff != 0) & (response == 0)
    correct = (diff != 0) & (feedback == 1)
    incorrect = (diff != 0) & (response != 0) & (feedback == 0)
    reject = (right_contrast == 0) & (left_contrast == 0) & (response == 0)
    falarmL = (right_contrast == 0) & (left_contrast == 0) & (response == 1)
    falarmR = (right_contrast == 0) & (left_contrast == 0) & (response == -1)

    labels = np.full(left_contrast.size, "", dtype=object)
    for i in range(left_contrast.size):
        binary = [miss[i], correct[i], incorrect[i], reject[i], falarmL[i], falarmR[i]]
        for name, flag in zip(RATIO_COLS, binary):
            if flag:
                labels[i] = name
                break

    # fix the 0.006 -> 0.06 contrast typo of the recorded data
    left_contrast[left_contrast == 0.006] = 0.06
    right_contrast[right_contrast == 0.006] = 0.06

    return pd.DataFrame(
        {
            "left_contrast": left_contrast,
            "right_contrast": right_contrast,
            "response": response.astype(float),
            "feedback": feedback.astype(float),
            "miss": miss.astype(float),
            "correct": correct.astype(float),
            "incorrect": incorrect.astype(float),
            "reject": reject.astype(float),
            "falarmL": falarmL.astype(float),
            "falarmR": falarmR.astype(float),
            "label": labels,
            "reactionTime": reaction_time,
        }
    )
