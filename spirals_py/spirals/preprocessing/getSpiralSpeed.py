"""Per-session spiral speed from grouped spiral trajectories.

Translated from spirals/preprocessing/getSpiralSpeed.m
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _padZeros
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.spirals.utils import loadUVt1, wrapAngle
from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.utils.matio import load_mat_cell, save_mat73
from spirals_py.utils.paths import out_root, release_twin


def _session_fname(T, kk):
    """Session file name <MouseID>_<yyyymmdd>_<folder> (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tda = pd.to_datetime(T["date"].iloc[kk])
    en = int(T["folder"].iloc[kk])
    return f"{mn}_{tda.strftime('%Y%m%d')}_{en}"


def _get_angular_speed(filteredSpirals3, tracePhase_padded, scale, halfpadding):
    """Translated from spirals/utils/get_anglular_speed.m (sic).

    For every spiral row: sample the phase map of two consecutive padded
    frames on 12 angles x floor(radius/scale) concentric circles around
    the (scaled, padded) spiral center, unwrap each circle with
    wrapAngle, and take the frame-to-frame phase offset (wrapTo2Pi, then
    values > pi shifted by -pi as in MATLAB) plus the arc-length
    distance offset.  Returns lists of (12, R) arrays, one per spiral
    row.
    """
    pixSize = 3.45 / 1000 / 0.6 * 3  # mm / pix
    angle = np.arange(np.pi / 6, 2 * np.pi + 1e-12, np.pi / 6)  # 12 samples
    angle_offset_all = []
    distance_offset_all = []
    for iframe in range(filteredSpirals3.shape[0]):
        frame = int(filteredSpirals3[iframe, 4])  # 1-based frame number
        spiral_radius = filteredSpirals3[iframe, 2] / scale
        center = matlab_round(
            [filteredSpirals3[iframe, 1] / scale, filteredSpirals3[iframe, 0] / scale]
        ) + halfpadding
        frame1 = tracePhase_padded[:, :, frame - 1]
        frame2 = tracePhase_padded[:, :, frame]
        nrad = int(np.floor(spiral_radius))  # MATLAB for radius = 1:1:spiral_radius
        phase_circle1 = np.empty((angle.size, nrad))
        phase_circle2 = np.empty((angle.size, nrad))
        for j in range(nrad):
            radius = j + 1
            x = (center[1] + matlab_round(radius * np.cos(angle))).astype(int)
            y = (center[0] + matlab_round(radius * np.sin(angle))).astype(int)
            phase_circle1[:, j] = frame1[y - 1, x - 1]  # MATLAB 1-based indexing
            phase_circle2[:, j] = frame2[y - 1, x - 1]
        phase_circle1b = np.column_stack(
            [wrapAngle(phase_circle1[:, j]) for j in range(nrad)]
        )
        phase_circle2b = np.column_stack(
            [wrapAngle(phase_circle2[:, j]) for j in range(nrad)]
        )
        for pc in (phase_circle1b, phase_circle2b):
            pc[np.abs(pc) < 0.0001] = np.nan
            pc[np.abs(pc - 2 * np.pi) < 0.0001] = np.nan
        angle_offset = phase_circle2b - phase_circle1b
        angle_offset = np.mod(angle_offset, 2 * np.pi)  # wrapTo2Pi
        angle_offset[angle_offset > np.pi] -= np.pi
        radius_all = np.arange(1, nrad + 1)
        distance_offset = (
            angle_offset / (2 * np.pi) * (2 * np.pi * radius_all[None, :] * pixSize * scale)
        ) * 35
        angle_offset_all.append(angle_offset)
        distance_offset_all.append(distance_offset)
    return angle_offset_all, distance_offset_all


def _col_cell(cells):
    """MATLAB column cell {n x 1} from a list of arrays."""
    out = np.empty((len(cells), 1), dtype=object)
    for i, c in enumerate(cells):
        out[i, 0] = c
    return out


def getSpiralSpeed(T, data_folder, save_folder):
    """Translated from spirals/preprocessing/getSpiralSpeed.m

    Per session: rebuild the 2-8 Hz phase movie from the SVD components
    (decimated by scale1=8, first 50 components), pad it spatially by
    halfpadding=15, and measure the angular/linear phase offset across
    consecutive frames on concentric circles around every grouped spiral
    (duration >= 2 frames, center inside the 250<x<350 / 350<y<550 grid).
    Saves <fname>.mat with the (n x 1) object cells angle_offset_all /
    distance_offset_all of per-spiral (12, R) arrays into save_folder.

    Notes:
    - Bug: freq and rate are undefined in the MATLAB source (they leak
      from the pipeline workspace); set to [2, 8] and 1 as in
      getAmpIndex.m and pipeline1_spirals.m.
    - Bug: the MATLAB source permutes tracePhase_raw with [2,3,1] after
      spiralPhaseMap_freq, which already returns the (x, y, frame)
      orientation; the extra permute would move frames to dim 2 and make
      padZeros allocate an impossibly large array.  The released figures
      must stem from an older helper returning (frame, x, y), so the
      extra permute is dropped here.
    - Dead code skipped: the horizontal_cortex_atlas_50um.mat /
      isocortex_horizontal_projection_outline.mat / structure-tree
      loads (BW, brain_index unused), the rf_tform load (tform unused),
      the Fs = 35 constant, and the trailing per-spiral mean summaries
      (angle_offset_all1/2, distance_offset_all1/2, mean_angle_offset,
      mean_angle_all, mean_distance_offset) that are computed but never
      saved.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    params = {"lowpass": 0, "gsmooth": 0}  # 0 is bandpass filter between 2-8Hz
    freq = [2, 8]
    rate = 1
    scale1 = 8
    padding = 30
    halfpadding = padding // 2
    grid_x = (250, 350)
    grid_y = (350, 550)
    for kk in tqdm(range(len(T)), desc="getSpiralSpeed"):
        fname = _session_fname(T, kk)
        session_root = data_folder / "spirals" / "svd" / fname
        U, V, t, mimg = loadUVt1(session_root)  # load U, V, t
        dV = np.column_stack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])

        archiveCell = load_mat_cell(
            release_twin(
                out_root()
                / "spirals"
                / "spirals_grouping"
                / f"{fname}_spirals_group_fftn.mat",
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
        filteredSpirals3 = filteredSpirals2[index1, :]

        U1 = U[::scale1, ::scale1, :50]
        _, _, tracePhase_raw = spiralPhaseMap_freq(U1, dV[:50, :], t, params, freq, rate)
        tracePhase_padded = _padZeros(tracePhase_raw, halfpadding)

        angle_offset_all, distance_offset_all = _get_angular_speed(
            filteredSpirals3, tracePhase_padded, scale1, halfpadding
        )
        save_mat73(
            save_folder / f"{fname}.mat",
            {
                "angle_offset_all": _col_cell(angle_offset_all),
                "distance_offset_all": _col_cell(distance_offset_all),
            },
        )
