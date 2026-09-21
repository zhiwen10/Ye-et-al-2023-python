"""Translated from spirals/preprocessing/getSpiralGrouping.m"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.preprocessing.spiral_detection import getGroupingAlgorithm
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import release_twin


def _session_fname(T, kk):
    """Session file name <MouseID>_<yyyymmdd>_<folder> (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tda = pd.to_datetime(T["date"].iloc[kk])
    en = int(T["folder"].iloc[kk])
    tdb = tda.strftime("%Y%m%d")
    return f"{mn}_{tdb}_{en}"


def _archive_to_cell(archiveCell):
    """archiveCell list -> (M, 1) object cell (explicit fill: numpy 2.x
    broadcasts equal-shaped elements in np.array(..., dtype=object))."""
    cell = np.empty((len(archiveCell), 1), dtype=object)
    for j in range(len(archiveCell)):
        cell[j, 0] = archiveCell[j]
    return cell


def getSpiralGrouping(T, data_folder, save_folder):
    """Translated from spirals/preprocessing/getSpiralGrouping.m

    Per session: load the raw detected spirals CSV
    (spirals/spirals_raw/<fname>_spirals_all.csv), keep only spirals
    with radius >= 40 pixels (the MATLAB comment says '> 40' but the
    code keeps '>= 40'; the code is mirrored), deduplicate rows, sort
    by frame number and run the spatiotemporal grouping
    (getGroupingAlgorithm, 30-px/3-frame chaining).  archiveCell is
    saved as an (M, 1) object cell of (m, 5)
    [x y radius direction frame] arrays (1-based frames) in
    <fname>_spirals_group_fftn.mat (v7.3).

    Adaptations: the spirals_raw CSV (a chained detection output the
    MATLAB joins under data_folder) is read from the spirals_raw folder
    next to save_folder via release_twin, so a fully-python rerun reads
    regenerated CSVs from the output tree and falls back to the release
    twin otherwise; a session with no radius >= 40 spirals saves an
    empty (0, 1) cell (MATLAB would error inside getGroupingAlgorithm);
    the unused td datestr and the unused test_stats output are skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getSpiralGrouping"):
        fname = _session_fname(T, kk)
        # load raw detected spirals
        csv_path = release_twin(
            save_folder.parent / "spirals_raw" / f"{fname}_spirals_all.csv",
            data_folder,
        )
        pwAll = pd.read_csv(csv_path).to_numpy(dtype=float)
        # only use spirals with radius >= 40 pixels, based on 3d-fft
        filteredSpirals = pwAll[pwAll[:, 2] >= 40, :]
        # temporal grouping
        filteredSpirals = np.unique(filteredSpirals, axis=0)  # deduplicate
        filteredSpirals = filteredSpirals[
            np.argsort(filteredSpirals[:, 4], kind="stable")
        ]  # sortrows by frame number
        if filteredSpirals.shape[0]:
            archiveCell, _test_stats = getGroupingAlgorithm(filteredSpirals)
        else:
            archiveCell = []
        cell = _archive_to_cell(archiveCell)
        save_mat73(
            save_folder / f"{fname}_spirals_group_fftn.mat",
            {"archiveCell": cell},
        )
