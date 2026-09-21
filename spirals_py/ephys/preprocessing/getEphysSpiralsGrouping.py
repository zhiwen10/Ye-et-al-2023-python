from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.ephys.utils import get_session_info2
from spirals_py.spirals.preprocessing.spiral_detection import getGroupingAlgorithm
from spirals_py.utils.matio import load_mat_var, save_mat73
from spirals_py.utils.paths import out_root, release_twin


def _group_spirals(pwAll):
    """Core of getEphysSpiralsGrouping.m: radius >= 40 filter, unique
    rows, sortrows by frame (col 5), temporal grouping."""
    filteredSpirals = pwAll[pwAll[:, 2] >= 40, :]  # radius > 40 pixels
    filteredSpirals = np.unique(filteredSpirals, axis=0)
    filteredSpirals = filteredSpirals[
        np.argsort(filteredSpirals[:, 4], kind="stable")
    ]
    if filteredSpirals.shape[0]:
        archiveCell, _test_stats = getGroupingAlgorithm(filteredSpirals)
    else:
        archiveCell = []
    return archiveCell


def getEphysSpiralsGrouping(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getEphysSpiralsGrouping.m

    Groups raw spirals by spatiotemporal structure; archiveCell is saved
    as an (M, 1) object cell of (m, 5) arrays in
    <fname>_spirals_group_fftn.mat (v7.3).  Fixes the MATLAB
    'opsnum2str(en)' typo: fname is built from the ops fields
    (get_session_info2's ops.fname).  A session with no radius >= 40
    spirals saves an empty cell (MATLAB would error in
    getGroupingAlgorithm).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getEphysSpiralsGrouping"):
        ops = get_session_info2(T, kk, data_folder)
        fname = ops.fname
        pwAll_path = release_twin(
            out_root() / "ephys" / "spirals_raw" / f"{fname}_spirals.mat",
            data_folder,
        )
        if not pwAll_path.exists():
            print(f"getEphysSpiralsGrouping: no spirals_raw output for {fname} "
                  f"(detection skipped and not in the release), skipping")
            continue
        pwAll = load_mat_var(pwAll_path, "pwAll")
        archiveCell = _group_spirals(pwAll)
        cell = np.empty((max(len(archiveCell), 0), 1), dtype=object)
        for i, c in enumerate(archiveCell):
            cell[i, 0] = c
        save_mat73(
            save_folder / f"{fname}_spirals_group_fftn.mat",
            {"archiveCell": cell},
        )
        print(f"getEphysSpiralsGrouping: {fname} ({kk + 1}/{len(T)})")
