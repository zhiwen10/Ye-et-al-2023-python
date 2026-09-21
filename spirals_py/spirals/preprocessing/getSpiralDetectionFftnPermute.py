"""Translated from spirals/preprocessing/getSpiralDetectionFftnPermute.m

Spiral detection on the 3-D-FFT phase-scrambled control movie within
the square fft ROI: same session loop, loading and cropping as
getSpiralDetectionFftnRaw, but the movie of each batch is the
phase-randomized reconstruction of spiralPhaseMap_fftn_freq
(spiralDetectionAlgorithmFftn.m, translated here as
_spiralDetectionAlgorithmFftn).  Output is saved as
save_folder/<freq_folder>/fftn/<fname>.mat (v7.3) with pwAll1 (centers
mapped back to full-frame 1-based coordinates), roi_ml, roi_ap and
frame_count.

The MATLAB source runs a single phase-randomized pass per session (no
repetition loop).  rng (numpy Generator) replaces the MATLAB randperm
stream of spiralPhaseMap_fftn_freq so the scramble is reproducible.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _padZeros
from spirals_py.spirals.preprocessing.getSpiralDetectionFftnRaw import (
    _setSpiralDetectionParamsFFTN,
)
from spirals_py.spirals.preprocessing.spiral_detection import detect_padded_frames
from spirals_py.spirals.preprocessing.spiralPhaseMap_fftn_freq import (
    spiralPhaseMap_fftn_freq,
)
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.matio import load_mat_var, save_mat73


def _spiralDetectionAlgorithmFftn(U1, dV, t, params, freq, rate, rng=None):
    """Translated from
    spirals/utils/spirals_detection/spiralDetectionAlgorithmFftn.m

    Same batched detection as spiralDetectionAlgorithm (reusing
    detect_padded_frames) but the movie of each epoch is filtered
    through the 3-D-FFT phase-scrambled spiralPhaseMap_fftn_freq.
    Returns pwAll (N, 5) [x y radius direction frame] with unpadded
    1-based frame numbers.
    """
    pwAll = np.zeros((0, 5))
    frameRange = params["frameRange"]
    for kkk in range(len(frameRange) - 1):  # MATLAB 1:numel(frameRange)-1
        tic = time.time()
        frameStart = frameRange[kkk]
        frameEnd = frameStart + params["epochL"] - 1
        # extra 2*35 frames before filter data
        frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)
        dV1 = dV[:50, frameTemp - 1]
        t1 = t[frameTemp - 1]
        # filter data at a pre-defined frequency range
        _, _, tracePhase1 = spiralPhaseMap_fftn_freq(
            U1, dV1, t1, params, freq, rate, rng=rng
        )
        # reduce 2*35 frames after filter data
        tracePhase1 = tracePhase1[:, :, 35 : tracePhase1.shape[2] - 35]
        # pad tracephase with edge zeros
        tracePhase = _padZeros(tracePhase1, params["halfpadding"])
        pwAll = np.vstack([pwAll, detect_padded_frames(tracePhase, params, frameStart)])
        print(
            f"spiralDetectionAlgorithmFftn: frames {frameStart}/"
            f"{params['frameN1']}; {time.time() - tic:.1f} s"
        )
    if pwAll.size:
        # recalculate spiral 2d coordinates without padding
        pwAll[:, :2] = pwAll[:, :2] - params["halfpadding"]
    return pwAll


def getSpiralDetectionFftnPermute(T, freq, data_folder, save_folder, rng=None):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
    for kk in tqdm(range(len(T)), desc="getSpiralDetectionFftnPermute"):
        row = T.iloc[kk]
        mn = str(row["MouseID"])
        tda = pd.Timestamp(row["date"])
        en = int(row["folder"])
        td = tda.strftime("%Y-%m-%d")
        tdb = tda.strftime("%Y%m%d")
        # read svd components (U,V,t) from processed data folder
        subfolder = f"{mn}_{tdb}_{en}"
        session_root = data_folder / "spirals" / "svd" / subfolder
        U, V, t, mimg = loadUVt1(session_root)
        dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
        # set params for detection
        rate = 1  # set to 1, if no upsampling in time
        fname = f"{mn}_{tdb}_{en}"
        roi_path = data_folder / "spirals" / "fft_roi" / f"{fname}_roi.mat"
        roi_ap = load_mat_var(roi_path, "roi_ap").ravel().astype(int)
        roi_ml = load_mat_var(roi_path, "roi_ml").ravel().astype(int)
        # set detection parameters, with square ROI
        params = _setSpiralDetectionParamsFFTN(mimg, t, roi_ap, roi_ml)
        # only use U space within square ROI
        U2 = U[roi_ap[0] - 1 : roi_ap[1], roi_ml[0] - 1 : roi_ml[1], :]
        U1 = U2[:: params["downscale"], :: params["downscale"], :50]
        # same detection algorithm with FFTN
        pwAll = _spiralDetectionAlgorithmFftn(U1, dV, t, params, freq, rate, rng=rng)
        pwAll1 = pwAll.copy()
        pwAll1[:, 0] = roi_ml[0] + pwAll1[:, 0] - 1
        pwAll1[:, 1] = roi_ap[0] + pwAll1[:, 1] - 1
        frame_count = t.size
        save_mat73(
            save_folder / freq_folder / "fftn" / f"{fname}.mat",
            {
                "pwAll1": pwAll1,
                "roi_ml": roi_ml.astype(float).reshape(1, 2),
                "roi_ap": roi_ap.astype(float).reshape(1, 2),
                "frame_count": float(frame_count),
            },
        )
        print(f"getSpiralDetectionFftnPermute: {fname} ({kk + 1}/{len(T)})")
