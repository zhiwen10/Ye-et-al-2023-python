"""Bin the sync/spirality indices by motion energy.

Translated from spirals/preprocessing/getMotionEnergyIndex.m
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots.plotSpiralSyncIndex import _imwarp, _load_tform
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap4
from spirals_py.spirals.utils import loadUVt1
from spirals_py.task.plots._task_helpers_s15 import load_projectedAtlas1
from spirals_py.utils.matio import load_mat_var, nanmean, save_mat73
from spirals_py.utils.paths import out_root, release_twin


def _session_fname(T, kk):
    """Session file name <MouseID>_<yyyymmdd>_<folder> (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tda = pd.to_datetime(T["date"].iloc[kk])
    en = int(T["folder"].iloc[kk])
    return f"{mn}_{tda.strftime('%Y%m%d')}_{en}"


def getMotionEnergyIndex(data_folder, save_folder):
    """Translated from spirals/preprocessing/getMotionEnergyIndex.m

    Bin the per-frame synchrony and spirality indices (Extended Data
    Fig. 8) by motion-energy bins (edges pre-determined from motion
    energy histograms across sessions) and save them as
    energy_sync_bin_0_0.01_all.mat with edges, sync, spirality (per
    frame and session) and index_sort / sync_sort / spirality_sort (mean
    index per bin and session, NaN for bins with <= 10 frames).  Read
    back by plotMotionEnergyIndex.

    Notes:
    - Bug: the MATLAB signature has no T, but the body uses
      T.MouseID/T.date/T.folder leaked from the base workspace (it would
      error as written); the session table is read from
      tables/spiralSessions3.xlsx here.
    - The MATLAB session loop is hardcoded to [1:6, 9:15] (no motion
      energy files for sessions 7 and 8); here all table rows are tried
      and sessions whose spirals_index/<fname>_motion_energy.mat is
      missing are skipped, which yields the same 13 sessions on the
      release data.
    - Dead code skipped: the <fname>_amp.mat load (none of its variables
      is used), mimgt (warped to the unused 1320 x 1140 template),
      trace2d1, index_unity, the commented-out motion-energy filtering
      block, and scale.
    - MATLAB grows sync/spirality/energy_all dynamically across
      sessions (zero-filled to the longest session window); the columns
      are zero-padded to the common length here.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    projectedAtlas1 = load_projectedAtlas1(data_folder)
    BW = projectedAtlas1.astype(bool)[::8, ::8]
    T = pd.read_excel(data_folder / "tables" / "spiralSessions3.xlsx")
    params = {"downscale": 8, "lowpass": 0, "gsmooth": 0}
    rate = 1
    tStart, tEnd = 900, 1100  # find spirals between time tStart:tEnd

    sync_cols = []
    spirality_cols = []
    energy_cols = []
    for kk in tqdm(range(len(T)), desc="getMotionEnergyIndex"):
        fname = _session_fname(T, kk)
        try:
            energy_file = release_twin(
                out_root() / "spirals" / "spirals_index" / f"{fname}_motion_energy.mat",
                data_folder,
            )
            image_energy2 = load_mat_var(energy_file, "image_energy2").ravel()
        except (FileNotFoundError, OSError):
            print(f"WARNING: no motion energy for {fname}, skipping")
            continue

        session_root = data_folder / "spirals" / "svd" / fname
        U, V, t, mimg = loadUVt1(session_root)
        dV = np.column_stack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
        tform = _load_tform(
            release_twin(
                out_root() / "spirals" / "rf_tform" / f"{fname}_tform.mat", data_folder
            )
        )
        Utransformed = _imwarp(U, tform, projectedAtlas1.shape, step=params["downscale"])
        mimgtransformed = _imwarp(
            mimg, tform, projectedAtlas1.shape, step=params["downscale"]
        )

        frameStart = int(np.flatnonzero(t > tStart)[0]) + 1
        frameEnd = int(np.flatnonzero(t > tEnd)[0]) + 1
        # extra 2*35 frames before/after the filter margin
        frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)
        dV1 = dV[:, frameTemp - 1]
        _, traceAmp1, tracePhase1 = spiralPhaseMap4(Utransformed, dV1, t, params, rate)
        trim = 35 // rate
        tracePhase1 = tracePhase1[:, :, trim : tracePhase1.shape[2] - trim]
        with np.errstate(invalid="ignore"):  # 0/0 outside the atlas -> NaN, as in MATLAB
            traceAmp1 = traceAmp1[:, :, trim : traceAmp1.shape[2] - trim] / (
                mimgtransformed[:, :, None]
            )

        cols = tracePhase1.shape[0]
        rows = tracePhase1.shape[1]
        xs = np.ceil(np.linspace(-rows / 4, rows / 4, int(np.ceil(rows / 2))))
        ys = np.ceil(np.linspace(-cols / 2, cols / 2, cols))
        Xs, Ys = np.meshgrid(xs, ys)
        anglein = np.arctan2(Ys, Xs)  # expected angle for centered spiral
        BW_half = BW[:, 71:]  # MATLAB 72:end
        pixel_count = BW_half.sum()

        n_frames = tracePhase1.shape[2]
        sync1 = np.empty(n_frames)
        spirality1 = np.empty(n_frames)
        for frame in range(n_frames):
            phase_hemi = tracePhase1[:, 71:, frame].copy()  # MATLAB (:,72:end,frame)
            phase_hemi[~BW_half] = np.nan
            sync1[frame] = np.abs(np.nansum(np.exp(1j * phase_hemi))) / pixel_count
            phase_diff = phase_hemi - anglein
            spirality1[frame] = np.abs(np.nansum(np.exp(1j * phase_diff))) / pixel_count
        sync_cols.append(sync1)
        spirality_cols.append(spirality1)
        energy_cols.append(image_energy2[frameStart - 1 : frameEnd])

    n_frames = max(len(c) for c in sync_cols)
    n_sess = len(sync_cols)
    sync = np.zeros((n_frames, n_sess))
    spirality = np.zeros((n_frames, n_sess))
    energy_all = np.zeros((n_frames, n_sess))
    for kk in range(n_sess):
        sync[: len(sync_cols[kk]), kk] = sync_cols[kk]
        spirality[: len(spirality_cols[kk]), kk] = spirality_cols[kk]
        energy_all[: len(energy_cols[kk]), kk] = energy_cols[kk]

    edges = np.arange(0, 1000000 + 1, 50000)  # MATLAB 0:50000:1000000
    index_sort = np.full((edges.size - 1, n_sess), np.nan)
    sync_sort = np.full((edges.size - 1, n_sess), np.nan)
    spirality_sort = np.full((edges.size - 1, n_sess), np.nan)
    for kk in range(n_sess):
        sync1 = sync[:, kk]
        spirality1 = spirality[:, kk]
        energy1 = energy_all[:, kk]
        index_unity1 = np.sqrt(sync1**2 + spirality1**2)
        for i in range(edges.size - 1):
            a = np.flatnonzero((energy1 >= edges[i]) & (energy1 < edges[i + 1]))
            if a.size > 10:
                sync_sort[i, kk] = np.mean(sync1[a])
                spirality_sort[i, kk] = np.mean(spirality1[a])
                index_sort[i, kk] = np.mean(index_unity1[a])
    save_mat73(
        save_folder / "energy_sync_bin_0_0.01_all.mat",
        {
            "edges": edges.reshape(1, -1),
            "sync": sync,
            "spirality": spirality,
            "index_sort": index_sort,
            "sync_sort": sync_sort,
            "spirality_sort": spirality_sort,
        },
    )
