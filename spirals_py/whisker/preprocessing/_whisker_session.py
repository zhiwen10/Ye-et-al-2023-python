"""Shared whisker-stimulus session prologue and spiral-sorting helpers.

Sources (MATLAB), the prologue repeated across whisker/preprocessing/
getWhiskerMeanMaps.m, getSpiralsPrePost.m, getSpiralsPeriStim.m and
getWhiskerSingleTrials.m:
- tables/whisker_stim_all2.xlsx session table
  (MouseID/date/folder/block_folder/label/hemisphere/center1/center2)
- fname = [MouseID]_[yyyymmdd]_[folder] under whisker/task_svd, SVD files
  loaded with utils/loadUVt2.m
- rewardValve stimulus alignment: task/utils/tsToT.m + schmittTimes.m with
  thresholds [0.5 0.8], flipsUp < 2700 s kept
- stimOn = find(t-flipsUp>0,1,'first') and frames_stimOn = frames1+stimOn
- the grouped-spiral loading (durations >= 2) + rfmap tform transform +
  MATLAB round of getSpiralsPrePost.m lines 124-133
- the by-frame sorting of spirals into (nTrials, nCols) cells

Adaptations: the SVD/timeline .mat files of the whisker sessions are
MATLAB v7.3, so they are read with the h5py loaders of
spirals_py.task.plots._task_helpers (loadUVt2_h5 / load_h5_var).  The
*_Block.mat load, the T{kk,7}/T{kk,8} bfd centers and flipsUp_iti of the
MATLAB prologues are unused downstream and are skipped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from spirals_py.task.plots._task_helpers import (
    load_h5_var,
    schmittTimes,
    tsToT,
)
from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.task.preprocessing._task_utils import (
    first_index_after,
    load_spirals_grouping,
    transformPointsForward,
)


def load_whisker_table(data_folder):
    """Read tables/whisker_stim_all2.xlsx (the 5 whisker-stim sessions)."""
    return pd.read_excel(Path(data_folder) / "tables" / "whisker_stim_all2.xlsx")


def whisker_session_dirs(data_folder, T, kk):
    """fname = [MouseID]_[yyyymmdd]_[folder] and its whisker/task_svd root."""
    mn = str(T.MouseID.iloc[kk])
    tda = pd.Timestamp(T.date.iloc[kk])
    en = int(T.folder.iloc[kk])
    fname = f"{mn}_{tda.strftime('%Y%m%d')}_{en}"
    session_root = Path(data_folder) / "whisker" / "task_svd" / fname
    return fname, session_root


def load_whisker_flipsUp(session_root):
    """rewardValve schmitt([0.5 0.8]) up-flip times, keeping flips < 2700 s."""
    session_root = Path(session_root)
    pd_sig = load_h5_var(session_root / "rewardValve_raw.mat", "pd").ravel()
    tlTimes = load_h5_var(
        session_root / "rewardValve_timestamps_Timeline.mat", "tlTimes"
    )
    tt = tsToT(tlTimes, pd_sig.size)
    _, flipsUp, _ = schmittTimes(tt, pd_sig, [0.5, 0.8])
    return flipsUp[flipsUp < 2700]


def whisker_frames_stimOn(t, flipsUp, halfwindow=70):
    """stimOn = find(t-flipsUp>0,1,'first') (1-based) and
    frames_stimOn = frames1 + stimOn with frames1 = -halfwindow:halfwindow
    (column halfwindow+1 holds the stimulus frame, 1-based frame numbers)."""
    stimOn = first_index_after(t, flipsUp)
    frames1 = np.arange(-halfwindow, halfwindow + 1)
    frames_stimOn = frames1[None, :] + stimOn[:, None]
    return stimOn, frames_stimOn


def load_whisker_spirals(data_folder, fname, tform, min_duration=2):
    """Load whisker/spirals_all/spirals_grouping/<fname>_spirals_group_fftn.mat
    (cells with >= min_duration frames), concatenate (cell2mat) and map the
    centers through the rfmap affine2d tform with MATLAB round.

    (The 'sprials_grouping' spelling of the spiral_folder variable in
    getSpiralsWhiskerMeanMaps.m is a typo of 'spirals_grouping'; the
    readers here use the correct name used by getSpiralsPrePost.m.)
    """
    path = (
        Path(data_folder)
        / "whisker"
        / "spirals_all"
        / "spirals_grouping"
        / f"{fname}_spirals_group_fftn.mat"
    )
    pwAll = load_spirals_grouping(path, min_duration=min_duration)
    sx, sy = transformPointsForward(tform, pwAll[:, 0], pwAll[:, 1])
    pwAll[:, 0] = matlab_round(sx)
    pwAll[:, 1] = matlab_round(sy)
    return pwAll


def bin_spirals_by_frame(pwAll, frames_stimOn):
    """Spiral-cell body shared by sortSpirals.m / getSpiralsPrePost.m /
    getWhiskerSingleTrials.m: (nTrials, nCols) object array of the (m, 5)
    spirals whose 1-based frame number (column 5) equals frames_stimOn."""
    by_frame = {}
    for i, fr in enumerate(pwAll[:, 4]):
        by_frame.setdefault(int(fr), []).append(i)
    n_trials, n_cols = frames_stimOn.shape
    spiral_cell = np.empty((n_trials, n_cols), dtype=object)
    for j in range(n_trials):
        for k in range(n_cols):
            ids = by_frame.get(int(frames_stimOn[j, k]))
            spiral_cell[j, k] = pwAll[ids] if ids else np.zeros((0, 5))
    return spiral_cell
