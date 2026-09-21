"""Translated from axons/preprocessing/getSpiralsPhaseMap.m

Phase maps (current + next frame) of the spirals detected in the SSp
ROI of the 15 spiral-imaging sessions: for every spiral with radius >=
70, direction 1 and 800 <= x <= 900, 500 <= y <= 650 (full-resolution
atlas coordinates), the 2-8 Hz instantaneous phase is reconstructed
over frameStart-35 : frameStart+135, the 35 filter-pad frames are
trimmed off each side, and the first two frames are kept.  All maps
are normalized to a common phase at pixel (70, 95) via
angle(exp(1i * (phase - phase_ref))) and saved as the single variable
spiral_phase_all_norm (165, 143, 2, n) in
<fname>_mean_flow2.mat (v7.3 layout via save_mat73).

Bug fix: the MATLAB source indexes dV(:, frameStart-35 : frameEnd+35)
with no bounds check, which errors on real sessions whose spirals
start within the first 35 frames or run past the end of dV; as in
getSpiralsPhaseMapLeft.m, such spirals are skipped here.

Skipped dead code of the MATLAB source: BW1, mimg and the outline /
maskPath loads that feed no computation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _imwarp_row, _load_tform
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap4
from spirals_py.spirals.utils import loadUVt1
from spirals_py.task.plots._task_helpers_s15 import (
    load_projectedAtlas1,
    matlab_round,
)
from spirals_py.task.preprocessing._task_utils import (
    brain_index_from_atlas,
    ismember_rows,
    load_spirals_grouping,
    transformPointsForward,
)
from spirals_py.utils.matio import save_mat73


def _session_phase_maps(T, data_folder, kk, radius, direction, roi, ref_pixel):
    """Per-session body shared by getSpiralsPhaseMap{,2,Left}.m: warp U
    into the atlas frame at 1/8 resolution, transform and filter the
    grouped spirals, and return (fname, spiral_phase_all_norm)."""
    data_folder = Path(data_folder)
    row = T.iloc[kk]
    mn = str(row["MouseID"])
    tdb = pd.Timestamp(row["date"]).strftime("%Y%m%d")
    en = int(row["folder"])
    fname = f"{mn}_{tdb}_{en}"

    # load SVD
    session_root = data_folder / "spirals" / "svd" / fname
    U, V, t, mimg = loadUVt1(session_root)
    dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])

    tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")

    projectedAtlas1 = load_projectedAtlas1(data_folder)
    Utransformed = _imwarp_row(U, tform, projectedAtlas1.shape, stride=8)

    # transform spirals to atlas space
    filteredSpirals = load_spirals_grouping(
        data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
        min_duration=2,
    )
    sx, sy = transformPointsForward(tform, filteredSpirals[:, 0], filteredSpirals[:, 1])
    filteredSpirals[:, 0] = matlab_round(sx)
    filteredSpirals[:, 1] = matlab_round(sy)
    keep = ismember_rows(
        filteredSpirals[:, :2], brain_index_from_atlas(projectedAtlas1.astype(bool))
    )
    filteredSpirals = filteredSpirals[keep]

    # only use spirals >= radius, one hemisphere, within the ROI
    filteredSpirals1 = filteredSpirals[filteredSpirals[:, 2] >= radius]
    filteredSpirals1 = filteredSpirals1[filteredSpirals1[:, 3] == direction]
    xlo, xhi, ylo, yhi = roi
    location_index = (
        (filteredSpirals1[:, 0] >= xlo)
        & (filteredSpirals1[:, 0] <= xhi)
        & (filteredSpirals1[:, 1] >= ylo)
        & (filteredSpirals1[:, 1] <= yhi)
    )
    filteredSpirals1 = filteredSpirals1[location_index]

    # get phase maps (current + next frame) for each spiral
    params = {"lowpass": 0, "gsmooth": 0}
    rate = 1
    pad = int(35 / rate)
    maps = []
    for i in range(filteredSpirals1.shape[0]):
        frameStart = int(filteredSpirals1[i, 4])
        frameEnd = frameStart + 100
        frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)  # extra 2*35 frames before filter data
        if frameTemp[0] > 0 and frameTemp[-1] < dV.shape[1]:
            dV1 = dV[:, frameTemp - 1]
            tracePhase1 = spiralPhaseMap4(Utransformed, dV1, t, params, rate)[2]
            tracePhase1 = tracePhase1[
                :, :, pad : tracePhase1.shape[2] - pad
            ]  # reduce 2*35 frames after filter data
            maps.append(tracePhase1[:, :, :2])
    if maps:
        spiral_phase_all = np.stack(maps, axis=3)
    else:
        spiral_phase_all = np.zeros(Utransformed.shape[:2] + (2, 0))

    # normalize all spiral phase maps to the same value at the reference pixel
    spiral_phase_all_norm = np.zeros_like(spiral_phase_all)
    for i in range(spiral_phase_all.shape[3]):
        spiral_phase_temp = spiral_phase_all[:, :, :, i]
        spiral_phase_temp1 = spiral_phase_temp - spiral_phase_temp[
            ref_pixel[0] - 1, ref_pixel[1] - 1
        ]
        spiral_phase_all_norm[:, :, :, i] = np.angle(np.exp(1j * spiral_phase_temp1))
    return fname, spiral_phase_all_norm


def getSpiralsPhaseMap(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for kk in tqdm(range(15), desc="getSpiralsPhaseMap"):
        fname, spiral_phase_all_norm = _session_phase_maps(
            T, data_folder, kk, radius=70, direction=1,
            roi=(800, 900, 500, 650), ref_pixel=(70, 95),
        )
        save_mat73(
            save_folder / f"{fname}_mean_flow2.mat",
            {"spiral_phase_all_norm": spiral_phase_all_norm},
        )
        print(f"getSpiralsPhaseMap: {fname} ({kk + 1}/15)")
