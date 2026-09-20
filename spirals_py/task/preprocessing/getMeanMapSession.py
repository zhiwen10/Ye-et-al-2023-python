from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers import imwarp, loadUVt2_h5, load_tform
from spirals_py.task.plots._task_helpers_s15 import (
    getPhotodiodeTime,
    load_block,
)
from spirals_py.task.preprocessing._task_io import nanmean, save_mat
from spirals_py.task.preprocessing._task_utils import (
    TASK_MICE,
    first_index_after,
    load_task_sessions,
    session_dirs,
)
from spirals_py.task.preprocessing.load_task_table import load_task_table


def getMeanMapSession(data_folder, save_folder):
    """Translated from task/preprocessing/getMeanMapSession.m

    Mean widefield map (derivative of the SVD temporal components,
    warped to the atlas template and downsampled 16x to 83 x 72) around
    stimulus onset ([-2, 2] s, 141 frames) for each trial type
    (correct / incorrect / miss), contrast and mouse, averaged across
    trials and saved per mouse under save_folder/individual as
    [mouse]_mean_map_[label].mat.
    """
    data_folder = Path(data_folder)
    save_folder1 = Path(save_folder) / "individual"
    save_folder1.mkdir(parents=True, exist_ok=True)

    win = [0, 5000]
    trialWin = [-4, 4]
    downscale = 16
    labels = ["correct", "incorrect", "miss"]
    contrasts = np.array(
        [
            [1, 0.5, 0.25, 0.125, 0.06, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0.5, 0.25, 0.125, 0.06, 0],
        ]
    )

    for mn in TASK_MICE:
        T1 = load_task_sessions(data_folder, mn)
        task_outcome = data_folder / "task" / "task_outcome"
        T_all = load_task_table(task_outcome / f"{mn}_task_outcome.mat", "T_all")
        trial_all = load_task_table(task_outcome / f"{mn}_task_trial_ID.mat", "trial_all")

        for label in labels:
            # for each trial type, concatenate all trials across sessions
            n_trials = int((T_all.label == label).sum())
            wf_all = np.zeros((83, 72, 141, n_trials))
            count1 = 0
            for kk in range(len(T1)):
                fname, session_root = session_dirs(data_folder, T1, kk)
                U, V, t, mimg = loadUVt2_h5(session_root)
                dV = np.concatenate(
                    [np.zeros((V.shape[0], 1)), np.diff(V, axis=1)], axis=1
                )
                block = load_block(session_root)
                # get photodiode time
                allPD2 = getPhotodiodeTime(session_root, win)
                # load atlas registration
                tform = load_tform(data_folder / "task" / "rfmap" / f"{fname}.mat")
                sizeTemplate = (1320, 1140)
                Ut = imwarp(U, tform, sizeTemplate)
                mimgt = imwarp(mimg, tform, sizeTemplate)
                mimgt = mimgt[::downscale, ::downscale]
                Ut1 = Ut[::downscale, ::downscale, :50] / mimgt
                dV1 = dV[:50, :].astype(float)

                ntrial = np.size(block.events.endTrialValues)
                norepeatValues = np.asarray(block.events.repeatNumValues).ravel()[
                    :ntrial
                ]
                norepeat_indx = norepeatValues == 1
                allPD2 = allPD2[:ntrial][norepeat_indx]

                # get time index for all events of this trial type
                trial_current = trial_all[trial_all.session == kk + 1]
                allPD3 = allPD2[(trial_current.label == label).to_numpy()]
                n3 = allPD3.size
                if n3:
                    indx = first_index_after(t, allPD3)  # 1-based
                    Va = np.empty((50, 141, n3))
                    for i in range(n3):
                        lo = indx[i] - 71  # MATLAB indx-70 (1-based) -> 0-based
                        hi = indx[i] + 70  # MATLAB indx+70 inclusive
                        if lo < 0 or hi > t.size:
                            raise IndexError(
                                f"trial window [{lo}, {hi}] outside session {fname}"
                            )
                        Va[:, :, i] = dV1[:, lo : hi + 1]
                    # image sequence for the mean trace map
                    Va = Va.reshape(50, 141 * n3, order="F")
                    M = Ut1.reshape(83 * 72, 50, order="F")
                    wf = (M @ Va).reshape(83, 72, 141, n3, order="F")
                else:
                    wf = None

                # concatenate
                if wf is not None:
                    count2 = count1 + n3
                    wf_all[:, :, :, count1:count2] = wf
                    count1 = count1 + n3

            # mean map per contrast (empty bins stay NaN, like MATLAB mean)
            Ta = T_all[T_all.label == label]
            trace_correct_mean = np.full((83, 72, 141, 11), np.nan)
            for ii in range(11):
                index = (
                    (Ta.left_contrast == contrasts[0, ii]).to_numpy()
                    & (Ta.right_contrast == contrasts[1, ii]).to_numpy()
                )
                trace_correct_mean[:, :, :, ii] = nanmean(wf_all[:, :, :, index], axis=3)

            save_mat(
                save_folder1 / f"{mn}_mean_map_{label}.mat",
                {"contrasts": contrasts, "trace_correct_mean": trace_correct_mean},
            )
            print(f"getMeanMapSession: {mn} {label} ({count1} trials)")
