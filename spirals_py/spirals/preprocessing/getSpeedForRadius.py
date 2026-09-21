"""Spiral angular velocity and linear speed for one radius class.

Translated from spirals/preprocessing/getSpeedForRadius.m
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.utils.matio import load_mat_cell, nanmean, save_mat73
from spirals_py.utils.paths import release_twin

from spirals_py.spirals.preprocessing.getSpiralSpeedConcat import (
    _grouped_grid_spirals,
    _session_fname,
)


def getSpeedForRadius(T, radius, data_folder, save_folder):
    """Translated from spirals/preprocessing/getSpeedForRadius.m

    Selects the per-spiral speed outputs of getSpiralSpeed whose spiral
    radius equals *radius* pixels (e.g. 50 or 100), pools them across
    sessions and saves speed_<radius>pixels.mat with
    angular_velocity_all (n, R) = mean angular offset x 35 frames/s and
    linear_velocity_all (n, R) = mean distance offset per sampling
    radius (R = floor(radius/8)).  Read back by plotSpeedForRadius.

    Notes:
    - Path adaptation: the MATLAB source reads the per-session speed
      files from data_folder/spirals/spirals_speed; here they are read
      from save_folder (the folder getSpiralSpeed writes in the python
      pipelines) with a release_twin fallback.
    - Dead code skipped: scale1, color_stairs, count1, td and the
      initial angular_velocity_all / linear_velocity_all stubs.
    - The MATLAB loop is hardcoded to 15 sessions; the table length is
      used instead.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    grid_x = (250, 350)
    grid_y = (350, 550)
    angle_offset_all_sessions = []
    distance_offset_all_sessions = []
    for kk in tqdm(range(len(T)), desc="getSpeedForRadius"):
        fname = _session_fname(T, kk)
        speed_file = release_twin(save_folder / f"{fname}.mat", data_folder)
        angle_offset_all = load_mat_cell(speed_file, "angle_offset_all").ravel()
        distance_offset_all = load_mat_cell(speed_file, "distance_offset_all").ravel()
        filteredSpirals2 = _grouped_grid_spirals(data_folder, fname, grid_x, grid_y)
        indx2 = filteredSpirals2[:, 2] == radius
        angle_offset_all1 = np.empty(angle_offset_all.shape, dtype=object)
        for i, x in enumerate(angle_offset_all):
            angle_offset_all1[i] = nanmean(np.asarray(x), axis=0)
        angle_offset_all_sessions.append(angle_offset_all1[indx2])
        distance_offset_all_sessions.append(distance_offset_all[indx2])

    # MATLAB horzcat of the per-session cell selections, then vertcat
    angle_cells = [
        np.asarray(c) for s in angle_offset_all_sessions for c in np.atleast_1d(s)
    ]
    distance_cells = [
        np.asarray(c) for s in distance_offset_all_sessions for c in np.atleast_1d(s)
    ]
    if angle_cells:
        angular_velocity_all = np.vstack([c.reshape(1, -1) for c in angle_cells]) * 35
        linear_velocity_all = np.vstack(
            [nanmean(c, axis=0) for c in distance_cells]
        )
    else:
        angular_velocity_all = np.zeros((0, 0))
        linear_velocity_all = np.zeros((0, 0))
    save_mat73(
        save_folder / f"speed_{radius}pixels.mat",
        {
            "angular_velocity_all": angular_velocity_all,
            "linear_velocity_all": linear_velocity_all,
        },
    )
