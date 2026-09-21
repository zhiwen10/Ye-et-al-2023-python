"""Translated from spirals/preprocessing/getSpiralDensityMap.m

Transforms the grouped spirals (duration >= 2 frames) of every session
into the atlas frame, keeps the centers inside the brain boundary and
saves the combined 40-pixel density map as histogram_40pixels.mat
(Fig 1e).

Notes:
- MATLAB loads tables/horizontal_cortex_atlas_50um.mat first, but the
  following load of isocortex_horizontal_projection_outline.mat
  overwrites projectedAtlas1, so only the outline file is read here;
  the get_cortex_atlas_path/root1/ctx/scale/td loads of the source are
  dead code.
- Bug fix: MATLAB computes frame_total but saves only unique_spirals;
  the consumers (plotSpiralDensityAllSessions /
  plotSpiralDensitySessionsMeanSEM) read frame_total from
  histogram_40pixels.mat, so it is saved here as well.
- The spirals_grouping / rf_tform inputs are chained pipeline outputs:
  they are resolved through release_twin (python output tree first,
  release twin as fallback).
- MATLAB round() is half-away-from-zero (matlab_round).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _load_tform
from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _density_color_plot,
    _index_xy,
    _ismember_rows,
    _load_spirals_grouping,
    _transformPointsForward,
)
from spirals_py.task.plots._task_helpers_s15 import (
    load_projectedAtlas1,
    matlab_round,
)
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import out_root, release_twin


def _session_name(T, kk):
    """[MouseID]_[yyyymmdd]_[folder] of session kk (0-based)."""
    mn = str(T["MouseID"].iloc[kk])
    tdb = pd.Timestamp(T["date"].iloc[kk]).strftime("%Y%m%d")
    en = int(T["folder"].iloc[kk])
    return f"{mn}_{tdb}_{en}"


def getSpiralDensityMap(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # atlas brain horizontal projection and outline (10um resolution)
    projectedAtlas1 = load_projectedAtlas1(data_folder)
    brain_index = _index_xy(projectedAtlas1.astype(bool))  # [x y] pairs

    spirals_all = np.zeros((0, 5))
    frame_total = 0.0
    for kk in tqdm(range(len(T)), desc="getSpiralDensityMap"):
        fname = _session_name(T, kk)

        # read how many total frames
        t = np.load(
            data_folder / "spirals" / "svd" / fname / "svdTemporalComponents_corr.timestamps.npy"
        )
        nframe = np.atleast_1d(t.squeeze()).size

        grouping_file = release_twin(
            out_root() / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            data_folder,
        )
        cells, _ = _load_spirals_grouping(grouping_file, min_duration=2)
        filteredSpirals = np.vstack(cells) if cells else np.zeros((0, 5))

        tform = _load_tform(
            release_twin(out_root() / "spirals" / "rf_tform" / f"{fname}_tform.mat", data_folder)
        )
        sx, sy = _transformPointsForward(tform, filteredSpirals[:, 0], filteredSpirals[:, 1])
        filteredSpirals[:, 0] = matlab_round(sx)
        filteredSpirals[:, 1] = matlab_round(sy)

        spirals_all = np.vstack([spirals_all, filteredSpirals])
        frame_total = frame_total + nframe

    spirals_all[:, :2] = matlab_round(spirals_all[:, :2])  # no-op (already rounded)
    lia = _ismember_rows(spirals_all[:, :2], brain_index)
    spirals_all = spirals_all[lia, :]

    hist_bin = 40
    if spirals_all.shape[0]:
        unique_spirals = _density_color_plot(spirals_all, hist_bin)
    else:
        unique_spirals = np.zeros((0, 3))  # MATLAB density_color_plot errors on empty input
    save_mat73(
        save_folder / "histogram_40pixels.mat",
        {"unique_spirals": unique_spirals, "frame_total": frame_total},
    )
