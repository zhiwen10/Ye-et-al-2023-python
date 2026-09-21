"""Translated from spirals/preprocessing/getFFTNSpiralsMap.m (input of
Extended Data Fig.3c, plotMapDataVsFftn)."""

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


def getFFTNSpiralsMap(T, freq, label, data_folder, save_folder):
    """Translated from spirals/preprocessing/getFFTNSpiralsMap.m

    Concatenates the spirals of all sessions detected at *freq* (label
    'control' = raw data, 'fftn' = 3-D-FFT phase-scrambled data), maps
    the centers through the rf_tform affine2d tform into atlas space,
    keeps centers inside the projectedAtlas1 brain mask and saves, per
    radius (10:10:100), the density histogram unique_spirals (n, 3)
    [x y count] as <save_folder>/<freq_folder>/<label>_map/
    histogram_radius_<r>.mat (v7.3 layout read by plotMapDataVsFftn).

    The FFTN detection outputs (<freq_folder>/<label>/<fname>.mat with
    pwAll1/frame_count) are read from save_folder (written by
    getSpiralDetectionFftnRaw / getSpiralDetectionFftnPermute), falling
    back to the release twin via release_twin.

    Dead code skipped: frame_all accumulates frame_count but is never
    saved (the control_map/total_frames.mat that plotMapDataVsFftn reads
    is not written by this function in MATLAB either), and the loop-final
    round of the already-rounded centers is a no-op.  A radius bin with
    no spirals saves a (0, 3) array instead of erroring in the min/max
    of the (here unused) color bounds.
    """
    freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    spirals_folder = save_folder / freq_folder / label

    projectedAtlas1 = load_projectedAtlas1(data_folder)
    brain_index = _index_xy(projectedAtlas1.astype(bool))

    spirals_all = np.zeros((0, 5))
    for kk in tqdm(range(len(T)), desc="getFFTNSpiralsMap"):
        _mn, _tdb, _en, fname = _session_info(T, kk)
        detection_file = release_twin(spirals_folder / f"{fname}.mat", data_folder)
        pwAll1 = load_mat_var(detection_file, "pwAll1")
        tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
        sx, sy = _transformPointsForward(tform, pwAll1[:, 0], pwAll1[:, 1])
        pwAll1[:, 0] = matlab_round(sx)
        pwAll1[:, 1] = matlab_round(sy)
        spirals_all = np.vstack([spirals_all, pwAll1])

    keep = _ismember_rows(spirals_all[:, :2], brain_index)
    spirals_all = spirals_all[keep]

    hist_bin = 40
    for radius in range(10, 101, 10):
        spirals_temp = spirals_all[spirals_all[:, 2] == radius, :]
        if spirals_temp.shape[0]:
            unique_spirals = _density_color_plot(spirals_temp, hist_bin)
        else:
            unique_spirals = np.zeros((0, 3))
        save_mat73(
            save_folder / freq_folder / f"{label}_map" / f"histogram_radius_{radius}.mat",
            {"unique_spirals": unique_spirals},
        )
