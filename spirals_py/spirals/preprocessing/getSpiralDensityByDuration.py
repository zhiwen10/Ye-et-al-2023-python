"""Translated from spirals/preprocessing/getSpiralDensityByDuration.m

Density maps (40-pixel bins) of the spirals grouped by duration in
frames (7 = >= 7), combined over all sessions and saved as
histogram_<d>frame.mat (Extended Data Fig.6cd).

Notes:
- Bug fix: MATLAB never writes the frame_total that the consumers
  (plotSpiralDensityByDuration / plotSpiralDensityByRadius) read from
  spirals_density_duration/frame_total.mat; it is computed here (total
  frame count over the sessions, as in getSpiralDensityMap) and saved
  once.
- MATLAB reloads the grouping/tform files inside the duration x session
  double loop; here each session is loaded once up front (identical
  output).
- The horizontal_cortex_atlas_50um / get_cortex_atlas_path / root1 /
  ctx loads of the source are dead code (projectedAtlas1 comes from
  isocortex_horizontal_projection_outline.mat, which overwrites it).
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


def getSpiralDensityByDuration(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    projectedAtlas1 = load_projectedAtlas1(data_folder)
    brain_index = _index_xy(projectedAtlas1.astype(bool))

    frame_total = 0.0
    sessions = []
    for kk in tqdm(range(len(T)), desc="getSpiralDensityByDuration"):
        fname = _session_name(T, kk)

        grouping_file = release_twin(
            out_root() / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            data_folder,
        )
        cells, durations = _load_spirals_grouping(grouping_file, min_duration=1)
        tform = _load_tform(
            release_twin(out_root() / "spirals" / "rf_tform" / f"{fname}_tform.mat", data_folder)
        )

        t = np.load(
            data_folder / "spirals" / "svd" / fname / "svdTemporalComponents_corr.timestamps.npy"
        )
        frame_total = frame_total + np.atleast_1d(t.squeeze()).size

        sessions.append((cells, durations, tform))

    hist_bin = 40
    duration = np.arange(7, 0, -1)  # 7:-1:1
    for duration_temp in duration:
        parts = []
        for cells, durations, tform in sessions:
            indx2 = durations == duration_temp
            if duration_temp == 7:
                indx2 = durations >= 7
            groupedCells = [c for c, m in zip(cells, indx2) if m]
            filteredSpirals = np.vstack(groupedCells) if groupedCells else np.zeros((0, 5))

            sx, sy = _transformPointsForward(tform, filteredSpirals[:, 0], filteredSpirals[:, 1])
            filteredSpirals[:, 0] = matlab_round(sx)
            filteredSpirals[:, 1] = matlab_round(sy)
            parts.append(filteredSpirals)

        spirals_all = np.vstack(parts)
        spirals_all[:, :2] = matlab_round(spirals_all[:, :2])  # no-op
        lia = _ismember_rows(spirals_all[:, :2], brain_index)
        spirals_all = spirals_all[lia, :]

        if spirals_all.shape[0]:
            unique_spirals = _density_color_plot(spirals_all, hist_bin)
        else:
            unique_spirals = np.zeros((0, 3))
        save_mat73(
            save_folder / f"histogram_{duration_temp}frame.mat",
            {"unique_spirals": unique_spirals},
        )

    save_mat73(save_folder / "frame_total.mat", {"frame_total": frame_total})
