"""Translated from spirals/preprocessing/getSpiralDetection.m

Iterates through all sessions in T for spiral detection: loads the SVD
components from data_folder/spirals/svd/<fname>, band-pass filters
(freq = [2, 8] Hz) the U * dV movie and runs the batched detection
chain of spiralDetectionAlgorithm within the full-brain ROI of
data_folder/spirals/full_roi/<fname>_roi.mat.  The detected spirals
pwAll (N, 5) [x, y, radius, direction, frame] (1-based frames) are
written as <fname>_spirals_all.csv with the MATLAB variable names.

Adaptations: the MATLAB polyshape `roi` loaded from full_roi is read
with _load_roi_vertices + _inROI (repo convention, see getSpiralsRaw);
writetable/array2table are replaced by pandas to_csv.  The unused td /
mimg variables of the MATLAB source are kept only as dead code.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _inROI,
    _load_roi_vertices,
    _setSpiralDetectionParams,
)
from spirals_py.spirals.preprocessing.spiral_detection import spiralDetectionAlgorithm
from spirals_py.spirals.utils import loadUVt1


def getSpiralDetection(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getSpiralDetection"):
        row = T.iloc[kk]
        mn = str(row["MouseID"])
        tda = pd.Timestamp(row["date"])
        en = int(row["folder"])
        td = tda.strftime("%Y-%m-%d")
        tdb = tda.strftime("%Y%m%d")
        # read svd components (U,V,t) from processed data folder
        subfolder = f"{mn}_{tdb}_{en}"
        session_root = data_folder / "spirals" / "svd" / subfolder
        U, V, t, _mimg = loadUVt1(session_root)
        dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
        # set params for detection
        freq = [2, 8]  # data filtering frequency range
        rate = 1  # set to 1, if no upsampling in time
        params = _setSpiralDetectionParams(U, t)
        # import the brain mask roi from the roi folder and detect spirals
        fname1 = f"{mn}_{tdb}_{en}_roi"
        roi = _load_roi_vertices(data_folder / "spirals" / "full_roi" / f"{fname1}.mat")
        tf = _inROI(roi, params["xx"].ravel(), params["yy"].ravel())
        # only use the grids that inside the roi to save time
        params["xxRoi"] = params["xx"].ravel()[tf]
        params["yyRoi"] = params["yy"].ravel()[tf]
        U1 = U[:: params["downscale"], :: params["downscale"], :50]
        # main spiral detection algorithm
        pwAll = spiralDetectionAlgorithm(U1, dV, t, params, freq, rate)
        fname = f"{mn}_{tdb}_{en}_spirals_all.csv"
        T1 = pd.DataFrame(
            pwAll,
            columns=[
                "spiral_center_x",
                "spiral_center_y",
                "spiral_radius",
                "spiral_direction",
                "spiral_frame",
            ],
        )
        T1.to_csv(save_folder / fname, index=False)
        print(f"getSpiralDetection: {subfolder} ({kk + 1}/{len(T)})")
