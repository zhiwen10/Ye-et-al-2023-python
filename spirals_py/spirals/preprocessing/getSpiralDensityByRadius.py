"""Translated from spirals/preprocessing/getSpiralDensityByRadius.m

Density maps (40-pixel bins) of the spirals grouped by detection
radius (40:10:100 pixels), combined over all sessions (all grouped
spirals, duration >= 1) and saved as histogram_<r>radius.mat
(Extended Data Fig.6ab).

Notes:
- The frame_total normalization used by the consumer
  (plotSpiralDensityByRadius) comes from
  spirals_density_duration/frame_total.mat written by
  getSpiralDensityByDuration (MATLAB reads no frame counts here).
- The horizontal_cortex_atlas_50um / get_cortex_atlas_path / root1 /
  ctx loads of the source are dead code (projectedAtlas1 comes from
  isocortex_horizontal_projection_outline.mat, which overwrites it).
- MATLAB filters spirals_all(:,3) (radius, the 3rd of
  [x y radius direction frame]) == radius_i by exact float equality;
  the same is done here.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _load_tform
from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _density_color_plot,
    _index_xy,
    _ismember_rows,
    _load_spirals_grouping,
    _transformPointsForward,
)
from spirals_py.spirals.preprocessing.getSpiralDensityMap import _session_name
from spirals_py.task.plots._task_helpers_s15 import (
    load_projectedAtlas1,
    matlab_round,
)
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import out_root, release_twin


def getSpiralDensityByRadius(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    projectedAtlas1 = load_projectedAtlas1(data_folder)
    brain_index = _index_xy(projectedAtlas1.astype(bool))

    spirals_all = np.zeros((0, 5))
    for kk in tqdm(range(len(T)), desc="getSpiralDensityByRadius"):
        fname = _session_name(T, kk)

        grouping_file = release_twin(
            out_root() / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            data_folder,
        )
        cells, _ = _load_spirals_grouping(grouping_file, min_duration=1)
        filteredSpirals = np.vstack(cells) if cells else np.zeros((0, 5))

        tform = _load_tform(
            release_twin(out_root() / "spirals" / "rf_tform" / f"{fname}_tform.mat", data_folder)
        )
        sx, sy = _transformPointsForward(tform, filteredSpirals[:, 0], filteredSpirals[:, 1])
        filteredSpirals[:, 0] = matlab_round(sx)
        filteredSpirals[:, 1] = matlab_round(sy)

        spirals_all = np.vstack([spirals_all, filteredSpirals])

    spirals_all[:, :2] = matlab_round(spirals_all[:, :2])  # no-op
    lia = _ismember_rows(spirals_all[:, :2], brain_index)
    spirals_all = spirals_all[lia, :]

    radius = np.arange(40, 101, 10)  # 40:10:100
    hist_bin = 40
    for radius_i in radius:
        spirals_temp = spirals_all[spirals_all[:, 2] == radius_i, :]

        if spirals_temp.shape[0]:
            unique_spirals = _density_color_plot(spirals_temp, hist_bin)
        else:
            unique_spirals = np.zeros((0, 3))
        save_mat73(
            save_folder / f"histogram_{radius_i}radius.mat",
            {"unique_spirals": unique_spirals},
        )
