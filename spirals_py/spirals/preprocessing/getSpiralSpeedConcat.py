"""Sort per-session spiral speeds by radius across sessions.

Translated from spirals/preprocessing/getSpiralSpeedConcat.m
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.utils.matio import load_mat_cell, nanmean, save_mat73
from spirals_py.utils.paths import out_root, release_twin


def _session_fname(T, kk):
    """Session file name <MouseID>_<yyyymmdd>_<folder> (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tda = pd.to_datetime(T["date"].iloc[kk])
    en = int(T["folder"].iloc[kk])
    return f"{mn}_{tda.strftime('%Y%m%d')}_{en}"


def _grouped_grid_spirals(data_folder, fname, grid_x, grid_y):
    """Grouped spirals (duration >= 2) kept inside the sampling grid."""
    archiveCell = load_mat_cell(
        release_twin(
            out_root() / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            data_folder,
        ),
        "archiveCell",
    ).ravel()
    indx2 = np.array([np.atleast_2d(c).shape[0] for c in archiveCell])
    groupedCells = [np.asarray(c) for c in archiveCell[indx2 >= 2]]
    filteredSpirals2 = np.vstack(groupedCells) if groupedCells else np.zeros((0, 5))
    index1 = (
        (filteredSpirals2[:, 1] > grid_x[0])
        & (filteredSpirals2[:, 1] < grid_x[1])
        & (filteredSpirals2[:, 0] > grid_y[0])
        & (filteredSpirals2[:, 0] < grid_y[1])
    )
    return filteredSpirals2[index1, :]


def getSpiralSpeedConcat(T, data_folder, save_folder):
    """Translated from spirals/preprocessing/getSpiralSpeedConcat.m

    For every spiral radius 40:10:100 pixels and session, selects the
    per-spiral speed outputs of getSpiralSpeed whose spiral radius
    matches, and collects them into the (nSessions, 7) object cells
    angle_offset / distance_offset of speed_all.mat: each angle_offset
    element is a cell of per-spiral (1, R) mean angular offsets (mean
    over the 12 sampling angles, omitnan), each distance_offset element
    a cell of per-spiral (12, R) distance-offset matrices.  Read back by
    plotSpiralSpeedSummary.

    Notes:
    - Path bugs: the MATLAB source reads the per-session speed files
      from data_folder/spirals_speed (but getSpiralSpeed wrote them to
      data_folder/spirals/spirals_speed) and the grouping archives from
      data_folder/spirals_grouped (the actual folder is
      data_folder/spirals/spirals_grouping).  Here the speed files are
      read from save_folder (the folder getSpiralSpeed writes, with a
      release_twin fallback) and the grouping archives from the
      spirals_grouping folder.
    - The MATLAB loop is hardcoded to 15 sessions; the table length is
      used instead.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    grid_x = (250, 350)
    grid_y = (350, 550)
    nS = len(T)
    angle_offset = np.empty((nS, 7), dtype=object)
    distance_offset = np.empty((nS, 7), dtype=object)
    for count1, radiusi in enumerate(range(40, 101, 10)):
        for kk in tqdm(range(nS), desc="getSpiralSpeedConcat"):
            fname = _session_fname(T, kk)
            speed_file = release_twin(save_folder / f"{fname}.mat", data_folder)
            angle_offset_all = load_mat_cell(speed_file, "angle_offset_all").ravel()
            distance_offset_all = load_mat_cell(speed_file, "distance_offset_all").ravel()
            filteredSpirals2 = _grouped_grid_spirals(data_folder, fname, grid_x, grid_y)
            indx2 = filteredSpirals2[:, 2] == radiusi
            angle_offset_all1 = np.empty(angle_offset_all.shape, dtype=object)
            for i, x in enumerate(angle_offset_all):
                angle_offset_all1[i] = nanmean(np.asarray(x), axis=0)
            angle_offset[kk, count1] = angle_offset_all1[indx2]
            distance_offset[kk, count1] = distance_offset_all[indx2]
    save_mat73(
        save_folder / "speed_all.mat",
        {"angle_offset": angle_offset, "distance_offset": distance_offset},
    )
