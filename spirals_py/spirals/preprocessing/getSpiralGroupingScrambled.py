"""Translated from spirals/preprocessing/getSpiralGroupingScrambled.m"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.preprocessing.spiral_detection import getGroupingAlgorithm
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import release_twin

from .getSpiralGrouping import _archive_to_cell, _session_fname


def getSpiralGroupingScrambled(T, data_folder, save_folder):
    """Translated from spirals/preprocessing/getSpiralGroupingScrambled.m

    Permutation control for the spatiotemporal grouping: the frame
    numbers (column 5) of the radius >= 40 spirals are randomly
    permuted (structure of spiral frames kept, sequence scrambled), the
    rows are re-sorted by frame and grouped with the same
    getGroupingAlgorithm.  Each session is permuted 10x; rep ii saves
    archiveCell ((M, 1) object cell of (m, 5) arrays) and p (MATLAB
    1-based randperm, (1, n) double) in
    save_folder/<fname>/<fname>_scramble_group_<ii>.mat (v7.3).

    Adaptations: MATLAB save/load append .mat to the extensionless
    names, done explicitly here; the scrambled files are written under a
    per-session <fname> subfolder of save_folder (the MATLAB caller
    passed the per-session spirals_scrambled/<fname> folder), matching
    the spirals_scrambled/<fname>/ layout getSpiralDurationRatio reads;
    the spirals_raw CSV is read via release_twin on the spirals_raw
    folder next to save_folder (release fallback, as in
    getSpiralGrouping); MATLAB uses the default global rng with no seed
    (randperm), so a fresh np.random.default_rng() is drawn per
    repetition without a fixed seed; a session with no radius >= 40
    spirals saves empty cells (MATLAB would error); tic/toc timing and
    the fprintf progress log are dropped (tqdm on the repetition loop);
    the unused test_stats output is skipped; the MATLAB unique-rows
    dedup of getSpiralGrouping is NOT applied here (the original
    scrambles the filtered rows directly).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in range(len(T)):
        fname = _session_fname(T, kk)
        # load all spirals
        csv_path = release_twin(
            save_folder.parent / "spirals_raw" / f"{fname}_spirals_all.csv",
            data_folder,
        )
        pwAll = pd.read_csv(csv_path).to_numpy(dtype=float)
        # only use spirals >= 40 pixels radius, based on fftn
        pwAll = pwAll[pwAll[:, 2] >= 40, :]
        for ii in tqdm(range(1, 11), desc="getSpiralGroupingScrambled"):
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
            cell = _archive_to_cell(archiveCell)
            save_mat73(
                save_folder / fname / f"{fname}_scramble_group_{ii}.mat",
                {
                    "archiveCell": cell,
                    "p": (p + 1).astype(float).reshape(1, -1),
                },
            )
