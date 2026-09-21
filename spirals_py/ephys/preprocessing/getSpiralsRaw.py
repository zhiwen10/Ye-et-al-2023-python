from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.ephys.utils import get_session_info2
from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _inROI,
    _load_roi_vertices,
    _setSpiralDetectionParams,
)
from spirals_py.spirals.preprocessing.spiral_detection import spiralDetectionAlgorithm
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import out_root, release_twin


def _spiral_detection_session(ops, data_folder, dV, save_folder, out_name):
    """Shared body of getSpiralsRaw.m / getSpiralsPrediction.m /
    getSpiralsPredictionPermute.m: ROI-masked spiral detection on the dV
    movie, pwAll (N, 5) [x, y, radius, direction, frame] (1-based frames)
    saved as out_name (v7.3)."""
    data_folder = Path(data_folder)
    U, V, t, mimg = loadUVt1(ops.session_root)
    freq = [2, 8]  # data filtering frequency range
    rate = 1  # set to 1, if no upsampling in time
    params = _setSpiralDetectionParams(U, t)
    # only detect spirals within ROI
    roi = _load_roi_vertices(
        release_twin(out_root() / "ephys" / "roi" / f"{ops.fname}_roi.mat", data_folder)
    )
    tf = _inROI(roi, params["xx"].ravel(), params["yy"].ravel())
    # only use the grids inside the roi to save time
    params["xxRoi"] = params["xx"].ravel()[tf]
    params["yyRoi"] = params["yy"].ravel()[tf]
    U1 = U[:: params["downscale"], :: params["downscale"], :50]
    pwAll = spiralDetectionAlgorithm(U1, dV, t, params, freq, rate)
    save_mat73(Path(save_folder) / out_name, {"pwAll": pwAll})
    return pwAll


def getSpiralsRaw(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getSpiralsRaw.m

    Detects widefield spirals within the brain ROI of the raw dV movie
    (dV = [0, diff(V)]); pwAll is saved as <fname>_spirals.mat.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getSpiralsRaw"):
        ops = get_session_info2(T, kk, data_folder)
        _U, V, _t, _mimg = loadUVt1(ops.session_root)
        dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
        _spiral_detection_session(
            ops, data_folder, dV, save_folder, f"{ops.fname}_spirals.mat"
        )
        print(f"getSpiralsRaw: {ops.fname} ({kk + 1}/{len(T)})")
