from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.ephys.preprocessing.getSpiralsRaw import _spiral_detection_session
from spirals_py.ephys.utils import get_session_info2
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.matio import load_mat_var
from spirals_py.utils.paths import out_root, release_twin


def getSpiralsPrediction(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getSpiralsPrediction.m

    Same as getSpiralsRaw but the movie is the predicted dV loaded from
    ephys/dv_prediction/<fname>_dv_predict.mat (load_mat_var handles the
    v7.3 layout); pwAll is saved as <fname>_spirals_predicted.mat.

    Bug fixed: the MATLAB file clears dV, loads a file that only
    contains dV_predict / explained_var_all, and then passes the cleared
    (undefined) dV to spiralDetectionAlgorithm — a latent error.  The
    predicted dV_predict is passed here, which is the evident intent.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getSpiralsPrediction"):
        ops = get_session_info2(T, kk, data_folder)
        fname = ops.fname
        dV_predict = load_mat_var(
            release_twin(
                out_root() / "ephys" / "dv_prediction" / f"{fname}_dv_predict.mat",
                data_folder,
            ),
            "dV_predict",
        )
        _spiral_detection_session(
            ops, data_folder, dV_predict, save_folder,
            f"{fname}_spirals_predicted.mat",
        )
        print(f"getSpiralsPrediction: {fname} ({kk + 1}/{len(T)})")
