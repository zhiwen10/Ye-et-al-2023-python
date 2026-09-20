"""Translated from whisker/preprocessing/getSpiralsPrePost.m"""

from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers import load_tform, loadUVt2_h5
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
from ._whisker_utils import _load_outline


def _concat_columns(spiral_cell, c0, c1):
    """MATLAB cat(1, spiral_cell{:, c0+1:c1}) over 1-based columns
    c0+1..c1 (MATLAB column-major cell order: frame column outer)."""
    return np.vstack(
        [
            spiral_cell[j, k]
            for k in range(c0, c1)
            for j in range(spiral_cell.shape[0])
        ]
    )


def getSpiralsPrePost(data_folder, save_folder):
    """Translated from whisker/preprocessing/getSpiralsPrePost.m

    For each of the 5 whisker sessions: load the duration>=2 grouped
    spirals, transform the centers through the rfmap tform and round,
    mirror left-hemisphere mice (x -> 1140-x, direction negated) before
    masking to the brain outline, and bin the spirals into the
    frames_stimOn [-70, 70] window.  The pre (MATLAB columns 67:71,
    0-based 66:71, frames -4..0) and post (columns 72:76, 0-based 71:76,
    frames 1..5) windows are concatenated over trials and mice and saved
    as Whisker_spirals_pre_post.mat (spirals_pre_all / spirals_post_all
    (N, 5) and spirals_pre_cell / spirals_post_cell (5, 1) cells) in the
    v7.3 layout read by plotDensityPrePost.

    Dead code skipped: the indexSSp_right/indexSSp_left/indexMOs/indexMOp
    select_area blocks, the *_Block.mat load, the U/V/dV SVD segments and
    flipsUp_iti (only t is needed from the session).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    projectedAtlas1, _, _ = _load_outline(data_folder)
    brain_index = brain_index_from_atlas(projectedAtlas1.astype(bool))
    T = load_whisker_table(data_folder)

    spirals_pre_cell = np.empty((len(T), 1), dtype=object)
    spirals_post_cell = np.empty((len(T), 1), dtype=object)
    for kk in range(len(T)):
        fname, session_root = whisker_session_dirs(data_folder, T, kk)
        _, _, t, _ = loadUVt2_h5(session_root)
        flipsUp = load_whisker_flipsUp(session_root)
        tform = load_tform(data_folder / "whisker" / "rfmap" / f"{fname}.mat")
        _, frames_stimOn = whisker_frames_stimOn(t, flipsUp)

        # load spirals (durations >= 2 frames)
        pwAll = load_whisker_spirals(data_folder, fname, tform)
        if T.hemisphere.iloc[kk] == "left":
            pwAll[:, 0] = 1140 - pwAll[:, 0]
            pwAll[:, 3] = -pwAll[:, 3]
        pwAll = pwAll[ismember_rows(pwAll[:, :2], brain_index)]

        spiral_cell = bin_spirals_by_frame(pwAll, frames_stimOn)
        spirals_pre_cell[kk, 0] = _concat_columns(spiral_cell, 66, 71)
        spirals_post_cell[kk, 0] = _concat_columns(spiral_cell, 71, 76)
        print(f"getSpiralsPrePost: {fname} ({kk + 1}/{len(T)})")

    spirals_pre_all = np.vstack([c for c in spirals_pre_cell.ravel()])
    spirals_post_all = np.vstack([c for c in spirals_post_cell.ravel()])
    save_mat73(
        save_folder / "Whisker_spirals_pre_post.mat",
        {
            "spirals_pre_all": spirals_pre_all,
            "spirals_post_all": spirals_post_all,
            "spirals_pre_cell": spirals_pre_cell,
            "spirals_post_cell": spirals_post_cell,
        },
    )
