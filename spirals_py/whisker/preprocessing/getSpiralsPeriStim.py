"""Translated from whisker/preprocessing/getSpiralsPeriStim.m"""

from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers import load_tform, loadUVt2_h5
from spirals_py.task.preprocessing._task_utils import (
    brain_index_from_atlas,
    ismember_rows,
)
from spirals_py.utils.matio import save_mat73

from ._whisker_session import (
    load_whisker_flipsUp,
    load_whisker_spirals,
    load_whisker_table,
    whisker_frames_stimOn,
    whisker_session_dirs,
)
from ._whisker_utils import _get_cortex_atlas_path, _load_outline, _select_area
from .sortSpirals import sortSpirals


def getSpiralsPeriStim(data_folder, save_folder):
    """Translated from whisker/preprocessing/getSpiralsPeriStim.m

    For each of the 5 whisker sessions: load the duration>=2 grouped
    spirals, transform the centers through the rfmap tform and round,
    mask to the brain outline first and mirror left-hemisphere mice
    afterwards (x -> 1140-x, direction negated; note the mask/mirror
    order is the reverse of getSpiralsPrePost), then count the spirals
    of each whole hemisphere in the frames_stimOn [-35, 35] window
    (71 columns, column 36 the stimulus frame) with sortSpirals.  The
    counts are scaled by the 35 Hz frame rate and saved as
    Whisker_spirals_peri_stimulus.mat (spiral_count_sum_left/right,
    (5, 7, 71)) in the v7.3 layout read by plotSpiralRatePeriStimulus.

    MATLAB passes sensoryArea = [] to select_area; startsWith(spath, '')
    matches every entry, so the selection reduces to the whole
    hemisphere (reproduced with [""], as in plotDensityPrePost).

    Dead code skipped: the *_Block.mat load, the U/V SVD segments and
    flipsUp_iti (only t is needed from the session).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    projectedAtlas1, _, _ = _load_outline(data_folder)
    brain_index = brain_index_from_atlas(projectedAtlas1.astype(bool))
    _, st = _get_cortex_atlas_path(data_folder)
    indexSSp_right = _select_area([""], st, projectedAtlas1, "right")
    indexSSp_left = _select_area([""], st, projectedAtlas1, "left")
    T = load_whisker_table(data_folder)

    spiral_count_sum_left = np.zeros((len(T), 7, 71))
    spiral_count_sum_right = np.zeros((len(T), 7, 71))
    for kk in range(len(T)):
        fname, session_root = whisker_session_dirs(data_folder, T, kk)
        _, _, t, _ = loadUVt2_h5(session_root)
        flipsUp = load_whisker_flipsUp(session_root)
        tform = load_tform(data_folder / "whisker" / "rfmap" / f"{fname}.mat")
        _, frames_stimOn = whisker_frames_stimOn(t, flipsUp, halfwindow=35)

        # load spirals (durations >= 2 frames)
        pwAll = load_whisker_spirals(data_folder, fname, tform)
        pwAll = pwAll[ismember_rows(pwAll[:, :2], brain_index)]
        if T.hemisphere.iloc[kk] == "left":
            pwAll[:, 0] = 1140 - pwAll[:, 0]
            pwAll[:, 3] = -pwAll[:, 3]

        spiral_count_sum_left[kk] = sortSpirals(pwAll, indexSSp_left, frames_stimOn)
        spiral_count_sum_right[kk] = sortSpirals(pwAll, indexSSp_right, frames_stimOn)
        print(f"getSpiralsPeriStim: {fname} ({kk + 1}/{len(T)})")

    # * 35 converts per-frame counts to per-second rates
    spiral_count_sum_left = spiral_count_sum_left * 35
    spiral_count_sum_right = spiral_count_sum_right * 35
    save_mat73(
        save_folder / "Whisker_spirals_peri_stimulus.mat",
        {
            "spiral_count_sum_left": spiral_count_sum_left,
            "spiral_count_sum_right": spiral_count_sum_right,
        },
    )
