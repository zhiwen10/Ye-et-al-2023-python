"""Translated from spirals/preprocessing/getFFTNSpiralsStats.m (input of
Extended Data Fig.3d-e, plotScatterDataVsFftn)."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _load_tform
from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _density_color_plot,
    _index_xy,
    _ismember_rows,
    _session_info,
    _transformPointsForward,
)
from spirals_py.task.plots._task_helpers_s15 import (
    load_projectedAtlas1,
    matlab_round,
)
from spirals_py.utils.matio import load_mat_var, save_mat73
from spirals_py.utils.paths import release_twin


def getFFTNSpiralsStats(T, freq, label, data_folder, save_folder):
    """Translated from spirals/preprocessing/getFFTNSpiralsStats.m

    Per session: loads the spirals detected at *freq* (label 'control' =
    raw data, 'fftn' = 3-D-FFT phase-scrambled data), maps the centers
    through the rf_tform affine2d tform into atlas space, keeps centers
    inside the projectedAtlas1 brain mask and saves, per radius
    (10:10:100), the density histogram as
    <save_folder>/<freq_folder>/<label>_stats/<fname>_density.mat (v7.3)
    with spiral_density (10, 1) cell of (n, 3) [x y count] arrays and
    frame_all (10, 1), read by plotScatterDataVsFftn (_load_density_file).

    The FFTN detection outputs are read from save_folder (written by
    getSpiralDetectionFftnRaw / getSpiralDetectionFftnPermute), falling
    back to the release twin via release_twin.

    Dead code skipped: scale = 1 and the unused td string.  MATLAB never
    resets frame_all between sessions, but every entry is overwritten by
    the current session's frame_count, so frame_all is rebuilt per
    session here.  A radius bin with no spirals stores a (0, 3) cell
    instead of erroring in the min/max of the (here unused) color bounds.
    """
    freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    spirals_folder = save_folder / freq_folder / label

    projectedAtlas1 = load_projectedAtlas1(data_folder)
    brain_index = _index_xy(projectedAtlas1.astype(bool))

    hist_bin = 40
    for kk in tqdm(range(len(T)), desc="getFFTNSpiralsStats"):
        _mn, _tdb, _en, fname = _session_info(T, kk)
        tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
        detection_file = release_twin(spirals_folder / f"{fname}.mat", data_folder)
        pwAll1 = load_mat_var(detection_file, "pwAll1")
        frame_count = load_mat_var(detection_file, "frame_count")
        sx, sy = _transformPointsForward(tform, pwAll1[:, 0], pwAll1[:, 1])
        pwAll1[:, 0] = matlab_round(sx)
        pwAll1[:, 1] = matlab_round(sy)
        keep = _ismember_rows(pwAll1[:, :2], brain_index)
        pwAll1 = pwAll1[keep]

        spiral_density = np.empty((10, 1), dtype=object)
        frame_all = np.zeros(10)
        for count, radius in enumerate(range(10, 101, 10)):
            spirals_temp = pwAll1[pwAll1[:, 2] == radius, :]
            if spirals_temp.shape[0]:
                unique_spirals = _density_color_plot(spirals_temp, hist_bin)
            else:
                unique_spirals = np.zeros((0, 3))
            spiral_density[count, 0] = unique_spirals
            frame_all[count] = np.asarray(frame_count).ravel()[0]
        save_mat73(
            save_folder / freq_folder / f"{label}_stats" / f"{fname}_density.mat",
            {
                "spiral_density": spiral_density,
                "frame_all": frame_all.reshape(-1, 1),
            },
        )
