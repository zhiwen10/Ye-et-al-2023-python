"""Translated from
revision/frequency_band/preprocessing/getSpiralDurationRatio_freq.m
(frequency-band revision of Extended Data Fig.3)."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.utils.matio import load_mat_cell, save_mat73
from spirals_py.utils.paths import out_root, release_twin

from .getSpiralGrouping import _session_fname


def getSpiralDurationRatio_freq(T, freq, data_folder, save_folder):
    """Translated from
    revision/frequency_band/preprocessing/getSpiralDurationRatio_freq.m

    Frequency-band variant of getSpiralDurationRatio: per session, count
    the fraction of archived group spirals per sequence length 1..50 for
    the band grouping (spirals/spirals_freq/spirals_fftn/<freq_folder>/
    <fname>_spirals_group_fftn.mat) and for the 10 scrambled groupings
    (spirals_freq/spirals_scrambled/<freq_folder>/<fname>/
    <fname>_scramble_group_<ii>.mat).  Saves N (50, 1), N_ratio (50, 1),
    N_scramble (50, 10), N_ratio_scramble (50, 10) and the scalar
    spirals_total in save_folder/<freq_folder>/<fname>.mat (v7.3);
    ratios use the raw spirals_total denominator for the scrambled
    counts too (as in MATLAB).

    Adaptations (as in getSpiralDurationRatio): spirals_total equals the
    sum of per-cell row counts; an empty archiveCell gives NaN ratios;
    the chained grouping/scrambled inputs are read via release_twin
    (python output tree first, release fallback); the unused datestr and
    per-iteration clear are skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
    save_folder1 = save_folder / freq_folder
    save_folder1.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getSpiralDurationRatio_freq"):
        fname = _session_fname(T, kk)
        # load grouped spiral centers (> 40 pixels radius)
        group_path = release_twin(
            out_root()
            / "spirals"
            / "spirals_freq"
            / "spirals_fftn"
            / freq_folder
            / f"{fname}_spirals_group_fftn.mat",
            data_folder,
        )
        archiveCell = load_mat_cell(group_path, "archiveCell")
        indx2 = np.array(
            [np.size(c, 0) for c in archiveCell.ravel()], dtype=float
        )
        spirals_total = float(indx2.sum())
        N = np.zeros((50, 1))
        for duration in range(1, 51):
            N[duration - 1, 0] = indx2[indx2 == duration].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            N_ratio = N / spirals_total
        # spiral ratio in the permutations
        N_scramble = np.zeros((50, 10))
        for ii in range(1, 11):
            scr_path = release_twin(
                out_root()
                / "spirals"
                / "spirals_freq"
                / "spirals_scrambled"
                / freq_folder
                / fname
                / f"{fname}_scramble_group_{ii}.mat",
                data_folder,
            )
            archiveCell = load_mat_cell(scr_path, "archiveCell")
            indx_scramble = np.array(
                [np.size(c, 0) for c in archiveCell.ravel()], dtype=float
            )
            for duration in range(1, 51):
                N_scramble[duration - 1, ii - 1] = indx_scramble[
                    indx_scramble == duration
                ].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            N_ratio_scramble = N_scramble / spirals_total
        save_mat73(
            save_folder1 / f"{fname}.mat",
            {
                "N": N,
                "N_ratio": N_ratio,
                "N_scramble": N_scramble,
                "N_ratio_scramble": N_ratio_scramble,
                "spirals_total": np.array(spirals_total),
            },
        )
