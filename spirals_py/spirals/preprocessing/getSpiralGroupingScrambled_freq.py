"""Translated from
revision/frequency_band/preprocessing/getSpiralGroupingScrambled_freq.m
(frequency-band revision of Extended Data Fig.3)."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.spirals.preprocessing.spiral_detection import getGroupingAlgorithm
from spirals_py.utils.matio import load_mat_var, save_mat73
from spirals_py.utils.paths import out_root, release_twin

from .getSpiralGrouping import _archive_to_cell, _session_fname


def getSpiralGroupingScrambled_freq(T, freq, data_folder, save_folder):
    """Translated from
    revision/frequency_band/preprocessing/getSpiralGroupingScrambled_freq.m

    Frequency-band variant of getSpiralGroupingScrambled: the frame
    numbers (column 5) of the radius >= 40 spirals of the band-filtered
    detection output (spirals/spirals_freq/raw/<freq_folder>/
    <fname>_spirals_all.mat, variable pwAll) are randomly permuted, the
    rows re-sorted by frame and grouped with getGroupingAlgorithm.  Each
    session is permuted 10x; rep ii saves archiveCell ((M, 1) object
    cell of (m, 5) arrays) and p (MATLAB 1-based randperm, (1, n)
    double) in save_folder/<fname>/<fname>_scramble_group_<ii>.mat
    (v7.3).

    Adaptations (as in getSpiralGroupingScrambled): the MATLAB
    save/load of extensionless names uses explicit .mat; the raw input
    is read via release_twin (python output tree first, release
    fallback); unseeded rng per repetition (MATLAB global randperm); a
    session with no radius >= 40 spirals saves empty cells; tic/toc and
    fprintf progress are dropped (tqdm on the repetition loop); the
    unused test_stats output is skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
    for kk in range(len(T)):
        fname = _session_fname(T, kk)
        # load all spirals of this frequency band
        raw_path = release_twin(
            out_root()
            / "spirals"
            / "spirals_freq"
            / "raw"
            / freq_folder
            / f"{fname}_spirals_all.mat",
            data_folder,
        )
        pwAll = np.asarray(load_mat_var(raw_path, "pwAll"), dtype=float)
        # only use spirals >= 40 pixels radius, based on fftn
        pwAll = pwAll[pwAll[:, 2] >= 40, :]
        for ii in tqdm(range(1, 11), desc="getSpiralGroupingScrambled_freq"):
            rng = np.random.default_rng()
            p = rng.permutation(pwAll.shape[0])
            frame_number = pwAll[:, 4]  # column 5 is the frameN
            pwAll1 = pwAll.copy()
            pwAll1[:, 4] = frame_number[p]
            pwAll1 = pwAll1[np.argsort(pwAll1[:, 4], kind="stable")]
            if pwAll1.shape[0]:
                archiveCell, _test_stats = getGroupingAlgorithm(pwAll1)
            else:
                archiveCell = []
            save_mat73(
                save_folder / fname / f"{fname}_scramble_group_{ii}.mat",
                {
                    "archiveCell": _archive_to_cell(archiveCell),
                    "p": (p + 1).astype(float).reshape(1, -1),
                },
            )
