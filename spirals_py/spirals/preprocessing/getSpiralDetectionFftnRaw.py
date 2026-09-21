"""Translated from spirals/preprocessing/getSpiralDetectionFftnRaw.m

Spiral detection on the raw (unscrambled) movie restricted to the
square fft ROI: loads the SVD components from
data_folder/spirals/svd/<fname> and roi_ap / roi_ml from
data_folder/spirals/fft_roi/<fname>_roi.mat, runs
spiralDetectionAlgorithm on the roi_ap x roi_ml crop and saves pwAll1
(centers mapped back to full-frame 1-based coordinates), roi_ml,
roi_ap and frame_count as
save_folder/<freq_folder>/control/<fname>.mat (v7.3).

spirals/utils/spirals_detection/setSpiralDetectionParamsFFTN.m is
translated here as _setSpiralDetectionParamsFFTN and shared with
getSpiralDetectionFftnPermute.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import _padZeros
from spirals_py.spirals.plots.plotSpiralTimeSeries3d import _matlab_round
from spirals_py.spirals.preprocessing.spiral_detection import spiralDetectionAlgorithm
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.matio import load_mat_var, save_mat73


def _setSpiralDetectionParamsFFTN(mimg, t, roi_ap, roi_ml):
    """Translated from
    spirals/utils/spirals_detection/setSpiralDetectionParamsFFTN.m

    Same detection parameters as _setSpiralDetectionParams, but sized to
    the roi_ap x roi_ml crop of the mean image and with the coarse
    search grids masked to the nonzero padded mean image.  Two MATLAB
    quirks are reproduced exactly: the mask is indexed as
    mimg2_bw(xx(i), yy(i)) (xx used as the row index) and the roi grids
    are assigned swapped (xxRoi = yy(tf), yyRoi = xx(tf)).
    """
    params = {}
    params["downscale"] = 1
    params["lowpass"] = 0
    params["Fs"] = 35
    params["halfpadding"] = 120
    params["padding"] = 2 * params["halfpadding"]
    params["th"] = np.arange(1, 361, 36)
    params["rs"] = np.arange(10, 21, 5)
    params["gridsize"] = 10
    params["spiralRange"] = np.linspace(-np.pi, np.pi, 5)
    params["gsmooth"] = 0
    params["epochL"] = 100
    params["nt"] = np.size(t)
    params["frameN1"] = int(_matlab_round((params["nt"] - 70) / 100) * 100)
    params["frameRange"] = np.arange(36, 36 + params["frameN1"], params["epochL"])
    params["dThreshold"] = 15
    params["rsRCheck"] = np.arange(10, 101, 10)
    params["pgridx"], params["pgridy"] = np.meshgrid(
        np.arange(21 - 10, 21 + 10 + 1), np.arange(21 - 10, 21 + 10 + 1)
    )
    # generate coarse search grids with zeros padded at the edges
    mimg = mimg[roi_ap[0] - 1 : roi_ap[1], roi_ml[0] - 1 : roi_ml[1]]
    params["xsize"] = mimg.shape[0]
    params["ysize"] = mimg.shape[1]
    xsizePadded = params["xsize"] + params["padding"]
    ysizePadded = params["ysize"] + params["padding"]
    xx, yy = np.meshgrid(
        np.arange(
            np.min(params["rs"]) + 1,
            xsizePadded - np.min(params["rs"]) - 1 + 1,
            params["gridsize"],
        ),
        np.arange(
            np.min(params["rs"]) + 1,
            ysizePadded - np.min(params["rs"]) - 1 + 1,
            params["gridsize"],
        ),
    )
    params["xx"] = xx
    params["yy"] = yy
    # apply mask, this helps speed up spiral detection later
    mimg1 = mimg[:: params["downscale"], :: params["downscale"]]
    mimg2 = _padZeros(mimg1, params["halfpadding"])
    mimg2_bw = mimg2 != 0
    xxf = xx.ravel(order="F").astype(int)
    yyf = yy.ravel(order="F").astype(int)
    tf = mimg2_bw[xxf - 1, yyf - 1]
    params["xxRoi"] = yyf[tf]
    params["yyRoi"] = xxf[tf]
    return params


def getSpiralDetectionFftnRaw(T, freq, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
    for kk in tqdm(range(len(T)), desc="getSpiralDetectionFftnRaw"):
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
        # same detection algorithm as the main one
        pwAll = spiralDetectionAlgorithm(U1, dV, t, params, freq, rate)
        pwAll1 = pwAll.copy()
        pwAll1[:, 0] = roi_ml[0] + pwAll1[:, 0] - 1
        pwAll1[:, 1] = roi_ap[0] + pwAll1[:, 1] - 1
        frame_count = t.size
        save_mat73(
            save_folder / freq_folder / "control" / f"{fname}.mat",
            {
                "pwAll1": pwAll1,
                "roi_ml": roi_ml.astype(float).reshape(1, 2),
                "roi_ap": roi_ap.astype(float).reshape(1, 2),
                "frame_count": float(frame_count),
            },
        )
        print(f"getSpiralDetectionFftnRaw: {fname} ({kk + 1}/{len(T)})")
