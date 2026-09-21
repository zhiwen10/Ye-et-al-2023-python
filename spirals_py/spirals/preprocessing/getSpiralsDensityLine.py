"""Translated from spirals/preprocessing/getSpiralsDensityLine.m

Per-session spiral density (spirals/(mm^2*s), duration >= 2 frames,
40-pixel bins, no brain-mask filtering) interpolated along the fixed
cortical line x = [105,520], y = [780,225] and saved as
spiralDensityLinePerSession.mat (Extended Data Fig.7d).

Notes:
- MATLAB loops kk = 1:15 (the fixed session count); here all rows of T
  are used.
- The outline load / BW / brain_index / 1140x1320 meshgrid of the
  source feed no computation (brain_index is never used to filter) and
  are dropped; cell_count is dead code.
- MATLAB builds a scatteredInterpolant over the full 1140x1320 grid
  and then samples the line points; scipy's LinearNDInterpolator gives
  identical values when evaluated directly at those points (plotSpiral
  DensityByDuration uses the same adaptation).  NaN outside the convex
  hull is kept, as in MATLAB.
- The line points are reused from _fig1_helpers_s5._line_points
  (interp1 + MATLAB round; the interpolated values are never exact
  halves so numpy/ MATLAB rounding agree).
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _load_tform
from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _density_color_plot,
    _line_points,
    _load_spirals_grouping,
    _transformPointsForward,
)
from spirals_py.spirals.preprocessing.getSpiralDensityMap import _session_name
from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import out_root, release_twin


def getSpiralsDensityLine(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    pixSize = 0.01  # mm/pix after registration
    pixArea = pixSize**2

    points = _line_points()  # draw a line
    count_sample = np.full((points.shape[0], len(T)), np.nan)
    for kk in tqdm(range(len(T)), desc="getSpiralsDensityLine"):
        fname = _session_name(T, kk)

        # read how many total frames
        t = np.load(
            data_folder / "spirals" / "svd" / fname / "svdTemporalComponents_corr.timestamps.npy"
        )
        nframe = np.atleast_1d(t.squeeze()).size

        tform = _load_tform(
            release_twin(out_root() / "spirals" / "rf_tform" / f"{fname}_tform.mat", data_folder)
        )
        grouping_file = release_twin(
            out_root() / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            data_folder,
        )
        cells, _ = _load_spirals_grouping(grouping_file, min_duration=2)
        filteredSpirals = np.vstack(cells) if cells else np.zeros((0, 5))

        sx, sy = _transformPointsForward(tform, filteredSpirals[:, 0], filteredSpirals[:, 1])
        filteredSpirals[:, 0] = matlab_round(sx)
        filteredSpirals[:, 1] = matlab_round(sy)

        hist_bin = 40
        unique_spirals = _density_color_plot(filteredSpirals, hist_bin)
        unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)  # counts/mm^2
        unique_spirals_unit = unique_spirals_unit / nframe * 35  # spirals/(mm^2*s)

        # interp histogram counts along the line
        F = LinearNDInterpolator(unique_spirals[:, :2], unique_spirals_unit)
        count_sample[:, kk] = F(points[:, 0], points[:, 1])

    save_mat73(save_folder / "spiralDensityLinePerSession.mat", {"count_sample": count_sample})
