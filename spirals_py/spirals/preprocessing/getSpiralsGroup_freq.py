"""Translated from
revision/frequency_band/preprocessing/getSpiralsGroup_freq.m
(frequency-band revision of Extended Data Fig.3)."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s5 import _session_info
from spirals_py.spirals.preprocessing.spiral_detection import getGroupingAlgorithm
from spirals_py.utils.matio import load_mat_var, save_mat73


def getSpiralsGroup_freq(T, data_folder, save_folder):
    """Translated from
    revision/frequency_band/preprocessing/getSpiralsGroup_freq.m

    For the three frequency bands [0.2 0.5], [0.5 2], [2 8] Hz: load the
    raw detected spirals pwAll of every session from the release folder
    spirals/spirals_freq/raw/<freq_folder>/<fname>_spirals_all.mat, keep
    radius >= 40 pixels, drop duplicate rows, sortrows by frame (col 5)
    and run the spatiotemporal grouping (getGroupingAlgorithm).
    archiveCell is saved as an (M, 1) cell of (m, 5) arrays in
    <save_folder>/<freq_folder>/<fname>_spirals_group_fftn.mat (v7.3),
    the format read by getSpiralDurationRatio_freq /
    getSpiralsDensityLine_freq.

    test_stats of the grouping is computed but unused in MATLAB and not
    saved.  A session with no radius >= 40 spirals saves an empty cell
    (MATLAB would error inside getGroupingAlgorithm).
    """
    freq_all = np.array([[0.2, 0.5], [0.5, 2], [2, 8]])
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    for ifreq in tqdm(range(freq_all.shape[0]), desc="getSpiralsGroup_freq"):
        freq = freq_all[ifreq]
        freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
        save_folder1 = save_folder / freq_folder
        save_folder1.mkdir(parents=True, exist_ok=True)
        for kk in tqdm(range(len(T)), desc="getSpiralsGroup_freq"):
            _mn, _tdb, _en, fname = _session_info(T, kk)
            pwAll = load_mat_var(
                data_folder
                / "spirals"
                / "spirals_freq"
                / "raw"
                / freq_folder
                / f"{fname}_spirals_all.mat",
                "pwAll",
            )
            filteredSpirals = pwAll[pwAll[:, 2] >= 40, :]  # radius > 40 pixels
            filteredSpirals = np.unique(filteredSpirals, axis=0)
            filteredSpirals = filteredSpirals[
                np.argsort(filteredSpirals[:, 4], kind="stable")
            ]  # sortrows(...,5) by frame number
            if filteredSpirals.shape[0]:
                archiveCell, _test_stats = getGroupingAlgorithm(filteredSpirals)
            else:
                archiveCell = []
            if archiveCell:
                cell = np.empty((len(archiveCell), 1), dtype=object)
                for i, c in enumerate(archiveCell):
                    cell[i, 0] = c
            else:
                cell = np.empty((0, 1), dtype=object)
            save_mat73(
                save_folder1 / f"{fname}_spirals_group_fftn.mat",
                {"archiveCell": cell},
            )
