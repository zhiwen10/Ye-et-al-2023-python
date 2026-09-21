"""Translated from spirals/preprocessing/getSpiralDurationRatio.m"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.utils.matio import load_mat_cell, save_mat73
from spirals_py.utils.paths import release_twin

from .getSpiralGrouping import _session_fname


def getSpiralDurationRatio(T, data_folder, save_folder):
    """Translated from spirals/preprocessing/getSpiralDurationRatio.m

    Per session: count the fraction of archived group spirals per
    sequence length 1..50 for the raw grouping
    (spirals/spirals_grouping/<fname>_spirals_group_fftn.mat) and for
    the 10 scrambled groupings
    (spirals_scrambled/<fname>/<fname>_scramble_group_<ii>.mat).
    Saves N (50, 1), N_ratio (50, 1), N_scramble (50, 10),
    N_ratio_scramble (50, 10) and the scalar spirals_total in
    save_folder/<fname>.mat (v7.3); ratios use the raw spirals_total
    denominator for the scrambled counts too (as in MATLAB).

    Adaptations: MATLAB size(cell2mat(archiveCell), 1) equals the sum
    of the per-cell row counts, computed as such; a session with an
    empty archiveCell gives spirals_total = 0 and NaN ratios (MATLAB
    0/0 = NaN); the grouping/scrambled inputs (chained intermediates
    the MATLAB joins under data_folder) are read from the folders next
    to save_folder via release_twin, falling back to the release twins
    when the python outputs are missing; the unused td datestr and the
    per-iteration clear are skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getSpiralDurationRatio"):
        fname = _session_fname(T, kk)
        # load grouped spiral centers (> 40 pixels radius)
        group_path = release_twin(
            save_folder.parent / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            data_folder,
        )
        archiveCell = load_mat_cell(group_path, "archiveCell")
        indx2 = np.array(
            [np.size(c, 0) for c in archiveCell.ravel()], dtype=float
        )
        # total number of grouped spirals (size(cell2mat(...), 1))
        spirals_total = float(indx2.sum())
        N = np.zeros((50, 1))
        for duration in range(1, 51):
            # all spirals in clusters with this sequence length
            N[duration - 1, 0] = indx2[indx2 == duration].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            N_ratio = N / spirals_total
        # spiral ratio in the permutations
        N_scramble = np.zeros((50, 10))
        for ii in range(1, 11):
            scr_path = release_twin(
                save_folder.parent
                / "spirals_scrambled"
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
            save_folder / f"{fname}.mat",
            {
                "N": N,
                "N_ratio": N_ratio,
                "N_scramble": N_scramble,
                "N_ratio_scramble": N_ratio_scramble,
                "spirals_total": np.array(spirals_total),
            },
        )
