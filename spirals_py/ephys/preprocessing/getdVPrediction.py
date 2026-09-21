from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.ephys.preprocessing._regression import (
    get_variance_explained,
    mua_prediction_full,
)
from spirals_py.ephys.utils import (
    get_MUA_bin,
    get_session_info2,
    get_wf2ephysT2,
    loadKSdir2,
)
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.matio import save_mat73


def _dV_prediction_session(T, kk, data_folder, save_folder, permute=False, rng=None):
    """Shared body of ephys/preprocessing/getdVPrediction.m and
    getdVPredictionPermute.m (kk is a 0-based row of T).

    With permute=True the TEST regressor rows are permuted in each
    direction: MUA_std_even(perm_indx,:,:) when predicting even epochs
    and MUA_std_odd2(perm_indx,:,:) when predicting odd epochs (training
    regressors are left intact, as in the MATLAB original).
    """
    ops = get_session_info2(T, kk, data_folder)
    U, V, t, mimg = loadUVt1(ops.session_root)
    dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
    sp = loadKSdir2(ops.session_root)
    _syncTL, _syncProbe, WF2ephysT1 = get_wf2ephysT2(ops, t)
    WF2ephysT = WF2ephysT1[~np.isnan(WF2ephysT1)]
    dV1 = np.asarray(dV[:, ~np.isnan(WF2ephysT1)], dtype=float)
    MUA_std = get_MUA_bin(sp, WF2ephysT)
    dV1 = np.asarray(dV1[:50, :], dtype=float)
    fname = ops.fname

    kernel_t = [-0.5, 0.5]
    scale = 4
    Ut = U[::scale, ::scale, :50]

    sample_length = dV1.shape[1]
    epochN = 10
    epochSize = sample_length // epochN

    if permute:
        perm_indx = rng.permutation(MUA_std.shape[0])

    # separate into 5 odd and 5 even series, use odd to predict even
    odd_sel = [slice(2 * i * epochSize, (2 * i + 1) * epochSize) for i in range(5)]
    even_sel = [slice((2 * i + 1) * epochSize, (2 * i + 2) * epochSize) for i in range(5)]
    MUA_std_odd = np.concatenate([MUA_std[:, s] for s in odd_sel], axis=1)
    dV1_odd = np.concatenate([dV1[:, s] for s in odd_sel], axis=1)
    MUA_std_even = np.stack([MUA_std[:, s] for s in even_sel], axis=2)
    dV1_even = np.stack([dV1[:, s] for s in even_sel], axis=2)
    if permute:
        MUA_std_even = MUA_std_even[perm_indx]
    dV_predict_even, _k_cv = mua_prediction_full(
        dV1_odd, MUA_std_odd, dV1_even, MUA_std_even, kernel_t
    )

    # separate into 5 odd and 5 even series, use even to predict odd
    MUA_std_even2 = np.concatenate([MUA_std[:, s] for s in even_sel], axis=1)
    dV1_even2 = np.concatenate([dV1[:, s] for s in even_sel], axis=1)
    MUA_std_odd2 = np.stack([MUA_std[:, s] for s in odd_sel], axis=2)
    dV1_odd2 = np.stack([dV1[:, s] for s in odd_sel], axis=2)
    if permute:
        MUA_std_odd2 = MUA_std_odd2[perm_indx]
    dV_predict_odd, _k_cv = mua_prediction_full(
        dV1_even2, MUA_std_even2, dV1_odd2, MUA_std_odd2, kernel_t
    )

    # fully reconstruct the entire duration from predicted odd/even epochs
    parts = []
    for i in range(10):
        if i % 2 == 0:
            parts.append(dV_predict_odd[:, :, i // 2])
        else:
            parts.append(dV_predict_even[:, :, (i - 1) // 2])
    dV_predict = np.concatenate(parts, axis=1)

    # variance explained, averaging across the 10 epochs
    dV1 = dV1[:, : dV_predict.shape[1]]
    epoch_sel = [slice(i * epochSize, (i + 1) * epochSize) for i in range(10)]
    dV1_epochs = np.stack([dV1[:, s] for s in epoch_sel], axis=2)
    dV_predict_epochs = np.stack([dV_predict[:, s] for s in epoch_sel], axis=2)
    explained_var_all = get_variance_explained(Ut, dV1_epochs, dV_predict_epochs)
    explained_var_all[explained_var_all < 0] = 0
    save_mat73(
        Path(save_folder) / f"{fname}_dv_predict.mat",
        {"dV_predict": dV_predict, "explained_var_all": explained_var_all},
    )
    return fname


def getdVPrediction(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getdVPrediction.m

    Per session (T rows, kk 0-based -> get_session_info2), predicts dV
    from MUA by alternating odd/even 10-epoch cross-validation and saves
    <fname>_dv_predict.mat (v7.3, read back by plotVarMap /
    _prediction_example_utils with h5py + full transpose) holding
    dV_predict (50, nFrames) and explained_var_all (x/4, y/4, 10)
    clipped >= 0.  The unused area/V1/t1 blocks of the MATLAB file are
    skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getdVPrediction"):
        fname = _dV_prediction_session(T, kk, data_folder, save_folder)
        print(f"getdVPrediction: {fname} ({kk + 1}/{len(T)})")
