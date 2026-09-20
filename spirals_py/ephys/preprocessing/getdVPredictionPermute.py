from pathlib import Path

import numpy as np

from spirals_py.ephys.preprocessing.getdVPrediction import _dV_prediction_session


def getdVPredictionPermute(T, data_folder, save_folder, rng=None):
    """Translated from ephys/preprocessing/getdVPredictionPermute.m

    Same as getdVPrediction, but the TEST regressor rows are permuted in
    each prediction direction (MUA_std_even(perm_indx,:,:) when
    predicting even epochs, MUA_std_odd2(perm_indx,:,:) when predicting
    odd); output goes to <fname>_dv_predict.mat in save_folder (the
    pipeline4 ephys/dv_permute folder).  rng (numpy Generator) replaces
    MATLAB's global randperm stream.
    """
    if rng is None:
        rng = np.random.default_rng()
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in range(len(T)):
        fname = _dV_prediction_session(
            T, kk, data_folder, save_folder, permute=True, rng=rng
        )
        print(f"getdVPredictionPermute: {fname} ({kk + 1}/{len(T)})")
