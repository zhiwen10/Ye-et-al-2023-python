from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.task.plots._task_helpers import loadUVt2_h5, load_tform
from spirals_py.task.plots._task_helpers_s15 import (
    getPhotodiodeTime,
    load_block,
    load_projectedAtlas1,
    matlab_round,
)
from spirals_py.task.preprocessing._task_io import load_mat_table, save_mat
from spirals_py.task.preprocessing._task_utils import (
    TASK_MICE,
    brain_index_from_atlas,
    first_index_after,
    ismember_rows,
    load_spirals_grouping,
    load_task_sessions,
    session_dirs,
    transformPointsForward,
)
from spirals_py.task.preprocessing.getConcatTrials import getConcatTrials
from spirals_py.utils.paths import out_root, release_twin


def getTaskSpirals(data_folder, save_folder):
    """Translated from task/preprocessing/getTaskSpirals.m

    Organize the detected spirals (task/spirals_all/spirals_grouping,
    durations >= 2 frames) of every non-repeat task trial into
    (nTrials, 141) cells of [-2, 2] s around stimulus onset, after
    transforming the spiral centers into atlas-template coordinates and
    keeping only those inside the cortical mask.  Trial-type subsets
    (concatenated across trials per frame) and trial counts are saved as
    [mouse]_spirals_task_sort.mat.

    The U / V SVD components and the warped Ut of the MATLAB source are
    unused downstream (only t, the block events and the registration
    transform are needed) and are skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    BW = load_projectedAtlas1(data_folder).astype(bool)
    brain_index = brain_index_from_atlas(BW)

    spiral_folder = data_folder / "task" / "spirals_all"

    for mn in TASK_MICE:
        T1 = load_task_sessions(data_folder, mn)
        spiral_rows = []
        for kk in tqdm(range(len(T1)), desc="getTaskSpirals"):
            fname, session_root = session_dirs(data_folder, T1, kk)
            _, _, t, _ = loadUVt2_h5(session_root)
            block = load_block(session_root)
            # get photodiode time (active task)
            win = [0, 5000]
            allPD1 = getPhotodiodeTime(session_root, win)
            ntrials = np.size(block.events.endTrialValues)
            allPD1 = allPD1[:ntrials]
            norepeatValues = np.asarray(block.events.repeatNumValues).ravel()[:ntrials]
            norepeat_indx = norepeatValues == 1
            allPD2 = allPD1[norepeat_indx]

            # load atlas registration
            tform = load_tform(data_folder / "task" / "rfmap" / f"{fname}.mat")

            stimOn = first_index_after(t, allPD2)  # 1-based
            frames1 = np.arange(-70, 71)
            frames_stimOn = frames1[None, :] + stimOn[:, None]  # (nTrials, 141)

            # load spirals (durations >= 2 frames)
            pwAll = load_spirals_grouping(
                spiral_folder / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
                min_duration=2,
            )
            sx, sy = transformPointsForward(tform, pwAll[:, 0], pwAll[:, 1])
            pwAll[:, 0] = matlab_round(sx)
            pwAll[:, 1] = matlab_round(sy)
            keep = ismember_rows(pwAll[:, :2], brain_index)
            pwAll = pwAll[keep]

            # organize spirals by frame
            by_frame = {}
            for i, fr in enumerate(pwAll[:, 4]):
                by_frame.setdefault(int(fr), []).append(i)
            n2 = allPD2.size
            spiral_cell = np.empty((n2, 141), dtype=object)
            for j in range(n2):
                for k in range(141):
                    ids = by_frame.get(int(frames_stimOn[j, k]))
                    spiral_cell[j, k] = pwAll[ids] if ids else np.zeros((0, 5))
            spiral_rows.append(spiral_cell)
            print(f"getTaskSpirals: {fname} ({kk + 1}/{len(T1)})")

        spiral_all = np.vstack(spiral_rows)

        T_all = load_mat_table(
            release_twin(
                out_root() / "task" / "task_outcome" / f"{mn}_task_outcome.mat",
                data_folder,
            ),
            "T_all",
        )

        left = T_all.left_contrast.to_numpy()
        right = T_all.right_contrast.to_numpy()
        lab = T_all.label.to_numpy()

        correct_index_stimL = (lab == "correct") & (left > 0)
        correct_index_stimR = (lab == "correct") & (right > 0)
        incorrect_index_stimL = (lab == "incorrect") & (left > 0)
        incorrect_index_stimR = (lab == "incorrect") & (right > 0)
        miss_index = lab == "miss"

        spiral_correct_L = getConcatTrials(spiral_all[correct_index_stimL, :])
        spiral_correct_R = getConcatTrials(spiral_all[correct_index_stimR, :])
        spiral_incorrect_L = getConcatTrials(spiral_all[incorrect_index_stimL, :])
        spiral_incorrect_R = getConcatTrials(spiral_all[incorrect_index_stimR, :])
        spiral_miss2 = getConcatTrials(spiral_all[miss_index, :])

        save_mat(
            save_folder / f"{mn}_spirals_task_sort.mat",
            {
                "T_all": T_all,
                "spiral_all": spiral_all,
                "spiral_correct_L": np.array(spiral_correct_L, dtype=object).reshape(-1, 1),
                "spiral_correct_R": np.array(spiral_correct_R, dtype=object).reshape(-1, 1),
                "spiral_incorrect_L": np.array(spiral_incorrect_L, dtype=object).reshape(-1, 1),
                "spiral_incorrect_R": np.array(spiral_incorrect_R, dtype=object).reshape(-1, 1),
                "spiral_miss2": np.array(spiral_miss2, dtype=object).reshape(-1, 1),
                "correct_L_trialN": int(correct_index_stimL.sum()),
                "correct_R_trialN": int(correct_index_stimR.sum()),
                "incorrect_L_trialN": int(incorrect_index_stimL.sum()),
                "incorrect_R_trialN": int(incorrect_index_stimR.sum()),
                "miss_trialN": int(miss_index.sum()),
            },
        )
