"""Bin the sync/spirality indices by 2-8 Hz amplitude.

Translated from spirals/preprocessing/getAmpIndex.m
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots.plotSpiralSyncIndex import _imwarp, _load_tform
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.spirals.utils import loadUVt1
from spirals_py.task.plots._task_helpers_s15 import load_projectedAtlas1
from spirals_py.utils.matio import nanmean, save_mat73
from spirals_py.utils.paths import out_root, release_twin


def _session_fname(T, kk):
    """Session file name <MouseID>_<yyyymmdd>_<folder> (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tda = pd.to_datetime(T["date"].iloc[kk])
    en = int(T["folder"].iloc[kk])
    return f"{mn}_{tda.strftime('%Y%m%d')}_{en}"


def getAmpIndex(data_folder, save_folder):
    """Translated from spirals/preprocessing/getAmpIndex.m

    Bin the per-frame synchrony, spirality and 2-8 Hz amplitude indices
    (Extended Data Fig. 8) by amplitude bins (edges pre-determined from
    amplitude histograms across sessions) and save them as
    amp_sync_bin_0_0.01_all.mat with edges, sync, spirality, amp_mean
    (per frame and session) and index_sort / sync_sort /
    spirality_sort (mean index per bin and session, NaN for bins with
    <= 2000 frames).  Read back by plotAmpIndex.

    Notes:
    - Bug: like getMotionEnergyIndex.m, the MATLAB signature has no T
      but the body uses T.MouseID/T.date/T.folder leaked from the base
      workspace; the session table is read from
      tables/spiralSessions3.xlsx here.
    - Dead code skipped: mimgt (warped to the unused 1320 x 1140
      template), trace2d1, index_unity, the histcounts counts N, scale,
      and the MATLAB round of td.
    - MATLAB grows sync/spirality/amp_mean dynamically across sessions
      (zero-filled to the longest session window); the columns are
      zero-padded to the common length here.
    - The MATLAB loops are hardcoded to 15 sessions; the table length is
      used instead.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    projectedAtlas1 = load_projectedAtlas1(data_folder)
    BW = projectedAtlas1.astype(bool)[::8, ::8]
    T = pd.read_excel(data_folder / "tables" / "spiralSessions3.xlsx")
    params = {"downscale": 8, "lowpass": 0, "gsmooth": 0}
    rate = 1
    freq = [2, 8]
    tStart, tEnd = 200, 1200  # find spirals between time tStart:tEnd

    sync_cols = []
    spirality_cols = []
    amp_cols = []
    for kk in tqdm(range(len(T)), desc="getAmpIndex"):
        fname = _session_fname(T, kk)
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
        _, traceAmp1, tracePhase1 = spiralPhaseMap_freq(
            Utransformed, dV1, t, params, freq, rate
        )
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
        BW_half = BW[:, 71:]  # MATLAB 72:end, right hemisphere only
        pixel_count = BW_half.sum()

        n_frames = tracePhase1.shape[2]
        sync1 = np.empty(n_frames)
        spirality1 = np.empty(n_frames)
        amp_mean1 = np.empty(n_frames)
        for frame in range(n_frames):
            phase_hemi = tracePhase1[:, 71:, frame].copy()  # MATLAB (:,72:end,frame), right hemisphere only
            phase_hemi[~BW_half] = np.nan
            sync1[frame] = np.abs(np.nansum(np.exp(1j * phase_hemi))) / pixel_count
            phase_diff = phase_hemi - anglein
            spirality1[frame] = np.abs(np.nansum(np.exp(1j * phase_diff))) / pixel_count
            amp_hemi = traceAmp1[:, 71:, frame].copy()  # MATLAB (:,72:end,frame)
            amp_hemi[~BW_half] = np.nan
            amp_mean1[frame] = nanmean(amp_hemi)
        sync_cols.append(sync1)
        spirality_cols.append(spirality1)
        amp_cols.append(amp_mean1)

    n_frames = max(len(c) for c in sync_cols)
    n_sess = len(sync_cols)
    sync = np.zeros((n_frames, n_sess))
    spirality = np.zeros((n_frames, n_sess))
    amp_mean = np.zeros((n_frames, n_sess))
    for kk in range(n_sess):
        sync[: len(sync_cols[kk]), kk] = sync_cols[kk]
        spirality[: len(spirality_cols[kk]), kk] = spirality_cols[kk]
        amp_mean[: len(amp_cols[kk]), kk] = amp_cols[kk]

    edges = np.arange(0, 0.01 + 0.0005 / 2, 0.0005)  # MATLAB 0:0.0005:0.01
    index_sort = np.full((edges.size - 1, n_sess), np.nan)
    sync_sort = np.full((edges.size - 1, n_sess), np.nan)
    spirality_sort = np.full((edges.size - 1, n_sess), np.nan)
    for kk in range(n_sess):
        sync1 = sync[:, kk]
        spirality1 = spirality[:, kk]
        amp_mean1 = amp_mean[:, kk]
        index_unity1 = np.sqrt(sync1**2 + spirality1**2)
        for i in range(edges.size - 1):
            a = np.flatnonzero((amp_mean1 >= edges[i]) & (amp_mean1 < edges[i + 1]))
            if a.size > 2000:
                sync_sort[i, kk] = np.mean(sync1[a])
                spirality_sort[i, kk] = np.mean(spirality1[a])
                index_sort[i, kk] = np.mean(index_unity1[a])
    save_mat73(
        save_folder / "amp_sync_bin_0_0.01_all.mat",
        {
            "edges": edges.reshape(1, -1),
            "sync": sync,
            "spirality": spirality,
            "index_sort": index_sort,
            "sync_sort": sync_sort,
            "spirality_sort": spirality_sort,
            "amp_mean": amp_mean,
        },
    )
