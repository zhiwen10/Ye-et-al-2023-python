"""Translated from whisker/preprocessing/getWhiskerMeanMaps.m"""

from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers import imwarp, load_tform, loadUVt2_h5
from spirals_py.utils.matio import save_mat73

from ._whisker_session import (
    load_whisker_flipsUp,
    load_whisker_table,
    whisker_frames_stimOn,
    whisker_session_dirs,
)
from ._whisker_utils import _load_outline


def getWhiskerMeanMaps(data_folder, save_folder):
    """Translated from whisker/preprocessing/getWhiskerMeanMaps.m

    Per mouse: dV = [0, diff(V)], U = U/mimg, per-trial segments
    Va = dV(:, stimOn-70 : stimOn+70) with 1-based stimOn (MATLAB colon,
    141 columns; python dV[:, stimOn-71 : stimOn+70], so 0-based column
    70 is the stimulus frame), U warped into the atlas template
    (imwarp to the 1320x1140 projected template, then 1:8:end sampling
    -> 165x143) and wf = Ut1 * Va reshaped to (165, 143, 141, nTrials).
    Left-hemisphere mice flip the image columns (flip(wf, 2)).  The trial
    mean per mouse and the mean over mice are saved as
    whisker_spirals_mean_all.mat (wf_mean2, wf_mean_all), in the v7.3
    layout read by the whisker plot modules.

    Dead code skipped: the BW/brain_index/BW2 and indexSSp2 select_area
    blocks, the *_Block.mat load, the T{kk,7}/T{kk,8} bfd centers,
    flipsUp_iti, and the mimgtransformed warp (only projectedAtlas1 /
    projectedTemplate1 shapes of the atlas prologue are used).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    _, projectedTemplate1, _ = _load_outline(data_folder)
    T = load_whisker_table(data_folder)

    wf_mean_all = None
    for kk in range(len(T)):
        fname, session_root = whisker_session_dirs(data_folder, T, kk)
        U, V, t, mimg = loadUVt2_h5(session_root)
        dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
        U = U / mimg[:, :, None]  # MATLAB U./mimg broadcasts over size(U,3)
        flipsUp = load_whisker_flipsUp(session_root)
        tform = load_tform(data_folder / "whisker" / "rfmap" / f"{fname}.mat")
        stimOn, _ = whisker_frames_stimOn(t, flipsUp)

        # Va(:,:,i) = dV(:, indx(i)-70 : indx(i)+70), 1-based indx == stimOn
        Va = np.stack(
            [dV[:, stimOn[i] - 71 : stimOn[i] + 70] for i in range(stimOn.size)],
            axis=2,
        )
        Ut = imwarp(U, tform, projectedTemplate1.shape)
        Ut1 = Ut[::8, ::8, :]
        va1 = Va.reshape(Va.shape[0], Va.shape[1] * Va.shape[2], order="F")
        wf = Ut1.reshape(
            Ut1.shape[0] * Ut1.shape[1], Ut1.shape[2], order="F"
        ) @ va1
        wf = wf.reshape(
            Ut1.shape[0], Ut1.shape[1], Va.shape[1], stimOn.size, order="F"
        )
        if T.hemisphere.iloc[kk] == "left":
            wf = wf[:, ::-1]
        wf_mean = wf.mean(axis=3)
        if wf_mean_all is None:
            wf_mean_all = np.empty(wf_mean.shape + (len(T),))
        wf_mean_all[..., kk] = wf_mean
        print(f"getWhiskerMeanMaps: {fname} ({kk + 1}/{len(T)})")

    wf_mean2 = wf_mean_all.mean(axis=3)
    save_mat73(
        save_folder / "whisker_spirals_mean_all.mat",
        {"wf_mean2": wf_mean2, "wf_mean_all": wf_mean_all},
    )
