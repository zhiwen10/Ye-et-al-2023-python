from pathlib import Path

import numpy as np

from spirals_py.ephys.plots._prediction_example_utils import (
    _load_outline_mat,
    _load_tform,
    _transform_points_forward,
)
from spirals_py.ephys.utils import get_session_info2
from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.utils.matio import load_mat_cell, load_mat_var, save_mat73


def _ismember_rows_int(A, B):
    """MATLAB ismember(A, B, 'rows') for 1-based integer coordinates."""
    keys_B = {tuple(row) for row in np.asarray(B, dtype=np.int64)}
    return np.array(
        [tuple(row) in keys_B for row in np.asarray(A, dtype=np.int64)]
    )


def _match_hemisphere(spirals_hemi, prediction_hemi, traceAmp):
    """Neighbor-matching rule of getSpiralComparePredict.m for one
    hemisphere: a raw spiral is matched iff exactly one predicted spiral
    of the same frame lies within the +-100 px (x AND y) box.  Returns
    (M, 12) rows [raw_x raw_y radius direction frame | pdt_x pdt_y
    radius direction frame | direction match | 2-8 Hz amp of the frame].
    """
    rows = []
    for i in range(spirals_hemi.shape[0]):
        frame = spirals_hemi[i, 4]
        spiralx = spirals_hemi[i, 0]
        spiraly = spirals_hemi[i, 1]
        spiral_p1 = prediction_hemi[prediction_hemi[:, 4] == frame]
        index1 = np.flatnonzero(
            (spiral_p1[:, 0] >= spiralx - 100)
            & (spiral_p1[:, 0] <= spiralx + 100)
            & (spiral_p1[:, 1] >= spiraly - 100)
            & (spiral_p1[:, 1] <= spiraly + 100)
        )
        spiral_p2 = spiral_p1[index1]
        if spiral_p2.shape[0] == 1:
            rows.append(np.concatenate([spirals_hemi[i, :5], spiral_p2[0]]))
    spiral_match = np.zeros((len(rows), 12))
    for j, r in enumerate(rows):
        spiral_match[j, :10] = r
    if len(rows):
        # check if spiral directions are matching in raw and predicted
        match = (spiral_match[:, 8] - spiral_match[:, 3]) == 0
        spiral_match[:, 10] = match
        # attach 2-8Hz amplitude value for that spiral frame (1-based)
        spiral_match[:, 11] = traceAmp[0, spiral_match[:, 4].astype(int) - 1]
    return spiral_match


def _compare_sessions(T, data_folder, save_folder, predict_subfolder,
                      predicted_suffix, var_suffix, out_name):
    """Shared body of getSpiralComparePredict.m / getSpiralComparePermute.m.

    predict_subfolder/predicted_suffix select the predicted spiral files;
    var_suffix appends '_perm' to the saved variables in the permute
    variant.  The unused timestamp load of the MATLAB files is skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    projectedAtlas1, _projectedTemplate1 = _load_outline_mat(data_folder)

    raw_folder = data_folder / "ephys" / "spirals_raw_fftn"
    predict_folder = data_folder / "ephys" / predict_subfolder
    reg_folder = data_folder / "ephys" / "rf_tform"
    amp_folder = data_folder / "ephys" / "amplitude"

    # index for left and right hemisphere ([col, row], 1-based)
    BW = projectedAtlas1.astype(bool)
    BW_right = BW.copy()
    BW_right[:, : projectedAtlas1.shape[1] // 2] = False
    BW_left = BW.copy()
    BW_left[:, projectedAtlas1.shape[1] // 2 :] = False
    rowL, colL = np.nonzero(BW_left)
    brain_index_left = np.column_stack([colL + 1, rowL + 1])
    rowR, colR = np.nonzero(BW_right)
    brain_index_right = np.column_stack([colR + 1, rowR + 1])

    left_all = []
    right_all = []
    for kk in range(len(T)):
        ops = get_session_info2(T, kk, data_folder)
        fname = ops.fname
        try:
            traceAmp = np.atleast_2d(
                load_mat_var(amp_folder / f"{fname}_amp.mat", "traceAmp")
            )
        except FileNotFoundError:
            print(
                f"WARNING: skipping {fname} "
                "(ephys/amplitude artifact not available)"
            )
            left_all.append(np.zeros((0, 12)))
            right_all.append(np.zeros((0, 12)))
            continue

        archiveCell = load_mat_cell(
            raw_folder / f"{fname}_spirals_group_fftn.mat", "archiveCell"
        ).ravel()
        spiral_length = np.array([np.asarray(c).shape[0] for c in archiveCell])
        spiral_sequence = archiveCell[spiral_length >= 2]
        if len(spiral_sequence):
            spirals_filt = np.vstack(
                [np.atleast_2d(c) for c in spiral_sequence]
            )
        else:
            spirals_filt = np.zeros((0, 5))
        T_tform = _load_tform(reg_folder / f"{fname}_tform.mat")
        if spirals_filt.shape[0]:
            sx, sy = _transform_points_forward(
                T_tform, spirals_filt[:, 0], spirals_filt[:, 1]
            )
            spirals_filt[:, 0] = matlab_round(sx)
            spirals_filt[:, 1] = matlab_round(sy)

        # search for match in left and right hemispheres separately
        liaL = _ismember_rows_int(spirals_filt[:, :2], brain_index_left)
        spirals_left = spirals_filt[liaL]
        liaR = _ismember_rows_int(spirals_filt[:, :2], brain_index_right)
        spirals_right = spirals_filt[liaR]

        # load spirals from prediction
        pwAll = load_mat_var(
            predict_folder / f"{fname}_{predicted_suffix}", "pwAll"
        )
        if pwAll.shape[0]:
            sx1, sy1 = _transform_points_forward(
                T_tform, pwAll[:, 0], pwAll[:, 1]
            )
            pwAll[:, 0] = matlab_round(sx1)
            pwAll[:, 1] = matlab_round(sy1)
        liaL1 = _ismember_rows_int(pwAll[:, :2], brain_index_left)
        spirals_prediction_left = pwAll[liaL1]
        liaR1 = _ismember_rows_int(pwAll[:, :2], brain_index_right)
        spirals_prediction_right = pwAll[liaR1]

        left_all.append(
            _match_hemisphere(spirals_left, spirals_prediction_left, traceAmp)
        )
        right_all.append(
            _match_hemisphere(spirals_right, spirals_prediction_right, traceAmp)
        )
        print(f"spiral compare: {fname} ({kk + 1}/{len(T)})")

    save_mat73(
        save_folder / out_name,
        {
            f"spiral_left_match_all{var_suffix}": np.array([left_all], dtype=object),
            f"spiral_right_match_all{var_suffix}": np.array([right_all], dtype=object),
        },
    )


def getSpiralComparePredict(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getSpiralComparePredict.m

    For each raw grouped spiral (duration >= 2 frames, transformed into
    atlas coordinates and kept inside the hemisphere), finds the unique
    same-frame predicted spiral (from ephys/spirals_predict/
    <fname>_spirals_predicted.mat) within a +-100 px box and records the
    pair plus the direction-match flag and the frame's 2-8 Hz amplitude.
    One file for all sessions: spiral_compare_sessions_neighbor.mat with
    spiral_left_match_all / spiral_right_match_all as (1, nSessions)
    object cells of (M, 12) rows.  traceAmp comes from
    ephys/amplitude/<fname>_amp.mat, a data-release artifact; sessions
    lacking it are skipped with FileNotFoundError.
    """
    _compare_sessions(
        T, data_folder, save_folder,
        predict_subfolder="spirals_predict",
        predicted_suffix="spirals_predicted.mat",
        var_suffix="",
        out_name="spiral_compare_sessions_neighbor.mat",
    )
