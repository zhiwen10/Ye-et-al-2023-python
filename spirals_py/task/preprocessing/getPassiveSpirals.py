from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.task.plots._task_helpers_s15 import (
    load_block,
    load_projectedAtlas1,
    matlab_round,
)
from spirals_py.task.preprocessing._task_io import save_mat
from spirals_py.task.preprocessing._task_utils import (
    TASK_MICE,
    brain_index_from_atlas,
    first_index_after,
    getPassiveContrasts,
    getPhotodiodeTime2,
    ismember_rows,
    load_spirals_grouping,
    load_task_sessions,
    session_dirs,
    transformPointsForward,
)
from spirals_py.task.plots._task_helpers import loadUVt2_h5, load_tform


def getPassiveSpirals(data_folder, save_folder):
    """Translated from task/preprocessing/getPassiveSpirals.m

    Same spiral sorting as getTaskSpirals, for the passive-viewing
    sessions (no non-repeat filtering; photodiode times from
    getPhotodiodeTime2), binned by stimulus contrast (high / low / zero)
    and side.  Saved as [mouse]_spirals_passive_sort.mat.

    The U / V SVD components and the warped Ut of the MATLAB source are
    unused downstream and are skipped (the MATLAB sizeTemplate
    [132, 1140] there is a typo of [1320, 1140] and only feeds that
    unused warp).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    BW = load_projectedAtlas1(data_folder).astype(bool)
    brain_index = brain_index_from_atlas(BW)

    spiral_folder = data_folder / "task" / "spirals_all"

    for mn in TASK_MICE:
        T1 = load_task_sessions(data_folder, mn, label="passive")
        spiral_rows = []
        for kk in tqdm(range(len(T1)), desc="getPassiveSpirals"):
            fname, session_root = session_dirs(data_folder, T1, kk)
            _, _, t, _ = loadUVt2_h5(session_root)
            block = load_block(session_root)
            # get photodiode time (passive viewing)
            win = [0, 5000]
            allPD2 = getPhotodiodeTime2(session_root, block, win)

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
            print(f"getPassiveSpirals: {fname} ({kk + 1}/{len(T1)})")

        spiral_all = np.vstack(spiral_rows)

        win = [0, 5000]
        contrast_all = getPassiveContrasts(T1, data_folder, win)

        high_index_stimR = (contrast_all[:, 1] == 1) | (contrast_all[:, 1] == 0.5)
        spiral_high_stimR = spiral_all[high_index_stimR, :]
        low_index_stimR = (contrast_all[:, 1] == 0.25) | (contrast_all[:, 1] == 0.125)
        spiral_low_stimR = spiral_all[low_index_stimR, :]

        high_index_stimL = (contrast_all[:, 0] == 1) | (contrast_all[:, 0] == 0.5)
        spiral_high_stimL = spiral_all[high_index_stimL, :]
        low_index_stimL = (contrast_all[:, 0] == 0.25) | (contrast_all[:, 0] == 0.125)
        spiral_low_stimL = spiral_all[low_index_stimL, :]

        zero_index = (contrast_all[:, 0] == 0) & (contrast_all[:, 1] == 0)
        spiral_zero = spiral_all[zero_index, :]

        save_mat(
            save_folder / f"{mn}_spirals_passive_sort.mat",
            {
                "spiral_all": spiral_all,
                "spiral_high_stimL": spiral_high_stimL,
                "spiral_low_stimL": spiral_low_stimL,
                "spiral_high_stimR": spiral_high_stimR,
                "spiral_low_stimR": spiral_low_stimR,
                "spiral_zero": spiral_zero,
                "spiral_high_stimL_trialN": int(high_index_stimL.sum()),
                "spiral_low_stimL_trialN": int(low_index_stimL.sum()),
                "spiral_high_stimR_trialN": int(high_index_stimR.sum()),
                "spiral_low_stimR_trialN": int(low_index_stimR.sum()),
                "spiral_zero_trialN": int(zero_index.sum()),
            },
        )

