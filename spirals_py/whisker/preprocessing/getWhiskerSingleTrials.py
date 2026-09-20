"""Translated from whisker/preprocessing/getWhiskerSingleTrials.m"""

from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers import imwarp, load_tform, loadUVt2_h5
from spirals_py.task.preprocessing._task_utils import (
    brain_index_from_atlas,
    ismember_rows,
)
from spirals_py.utils.matio import save_mat73

from ._whisker_session import (
    bin_spirals_by_frame,
    load_whisker_flipsUp,
    load_whisker_spirals,
    load_whisker_table,
    whisker_frames_stimOn,
    whisker_session_dirs,
)
from ._whisker_utils import _get_cortex_atlas_path, _load_outline, _select_area


def getWhiskerSingleTrials(data_folder, save_folder):
    """Translated from whisker/preprocessing/getWhiskerSingleTrials.m

    Single session only: MATLAB kk = 4 (0-based 3, ZYE_0092).  Rebuilds
    the (165, 143, 141, nTrials) single-trial maps exactly as in
    getWhiskerMeanMaps (Va segments, warped Ut1, F-order reshape,
    left-hemisphere column flip), then keeps the duration>=2 spirals of
    the right hemisphere (brain mask, left-mouse mirror, then
    indexSSp_right restriction; sensoryArea = [] -> [""] as in
    getSpiralsPeriStim), counts the radius>=60 spirals over frames 71:85
    (0-based columns 70:85, offsets 0..+14) per trial and saves the three
    hardcoded publication trials wf1 = wf(:,:,:, [81, 486, 31]) (1-based)
    as single_trial_maps2.mat (165, 143, 141, 3) in the v7.3 layout read
    by plotWhiskerSingleTrials.

    Dead code skipped: the BW2/indexSSp_left blocks, the *_Block.mat
    load, the bfd centers T{kk,7}/T{kk,8}, flipsUp_iti, the
    mimgtransformed warp and the commented-out trial-selection variants.
    trial_select is computed as in MATLAB but, as there, unused by the
    saved output.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    projectedAtlas1, projectedTemplate1, _ = _load_outline(data_folder)
    brain_index = brain_index_from_atlas(projectedAtlas1.astype(bool))
    _, st = _get_cortex_atlas_path(data_folder)
    # whole right hemisphere (MATLAB sensoryArea = [])
    indexSSp_right = _select_area([""], st, projectedAtlas1, "right")

    T = load_whisker_table(data_folder)
    kk = 3  # MATLAB kk = 4 (ZYE_0092)

    fname, session_root = whisker_session_dirs(data_folder, T, kk)
    U, V, t, mimg = loadUVt2_h5(session_root)
    dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
    U = U / mimg[:, :, None]  # MATLAB U./mimg broadcasts over size(U,3)
    flipsUp = load_whisker_flipsUp(session_root)
    tform = load_tform(data_folder / "whisker" / "rfmap" / f"{fname}.mat")
    stimOn, frames_stimOn = whisker_frames_stimOn(t, flipsUp)

    # Va(:,:,i) = dV(:, indx(i)-70 : indx(i)+70), 1-based indx == stimOn
    Va = np.stack(
        [dV[:, stimOn[i] - 71 : stimOn[i] + 70] for i in range(stimOn.size)],
        axis=2,
    )
    Ut = imwarp(U, tform, projectedTemplate1.shape)
    Ut1 = Ut[::8, ::8, :]
    va1 = Va.reshape(Va.shape[0], Va.shape[1] * Va.shape[2], order="F")
    wf = Ut1.reshape(Ut1.shape[0] * Ut1.shape[1], Ut1.shape[2], order="F") @ va1
    wf = wf.reshape(Ut1.shape[0], Ut1.shape[1], Va.shape[1], stimOn.size, order="F")
    if T.hemisphere.iloc[kk] == "left":
        wf = wf[:, ::-1]

    # load spirals (durations >= 2 frames)
    pwAll = load_whisker_spirals(data_folder, fname, tform)
    pwAll = pwAll[ismember_rows(pwAll[:, :2], brain_index)]
    if T.hemisphere.iloc[kk] == "left":
        pwAll[:, 0] = 1140 - pwAll[:, 0]
        pwAll[:, 3] = -pwAll[:, 3]
    pwAll2 = pwAll[ismember_rows(pwAll[:, :2], indexSSp_right)]

    spiral_cell2 = bin_spirals_by_frame(pwAll2, frames_stimOn)
    # MATLAB frames_post = 71:85 (1-based trial-window columns)
    frames_post = np.arange(70, 85)
    n_trials = spiral_cell2.shape[0]
    spiral_large_count = np.zeros(n_trials)
    for i in range(n_trials):
        spiral_temp = np.vstack([spiral_cell2[i, k] for k in frames_post])
        spiral_large_count[i] = np.sum(spiral_temp[:, 2] >= 60)
    index = spiral_large_count >= 5
    trial_select = np.flatnonzero(index)

    wf1 = wf[:, :, :, [80, 485, 30]]
    save_mat73(save_folder / "single_trial_maps2.mat", {"wf1": wf1})
    print(f"getWhiskerSingleTrials: {fname}, {trial_select.size} trials with >=5 large spirals")
