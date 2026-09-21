from pathlib import Path

import numpy as np

from spirals_py.ephys.plots._prediction_example_utils import (
    _imwarp,
    _load_outline_mat,
    _load_tform,
)
from spirals_py.ephys.preprocessing._regression import (
    get_prediction,
    get_variance_explained,
)
from spirals_py.ephys.utils import (
    get_MUA_bin,
    get_session_info2,
    get_wf2ephysT2,
    loadKSdir2,
)
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.matio import nanmean, save_mat


def getVarOrderedByNeuron(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getVarOrderedByNeuron.m

    Pass 1: per-unit short-kernel ([-0.2, 0.2] s) predictions ->
    explained_var (x/4, y/4, nUnits), each map registered with
    ephys/rf_tform_4x/<fname>_tform_4x.mat (_load_tform, which falls
    back to the _v73 sibling for v7/MCOS files), NaN-masked outside the
    atlas BW and averaged -> mean_var; units sorted descending -> B, I
    (I kept 1-based as in MATLAB).  Pass 2: cumulative unit sets
    k = 1..nUnits -> explained_var_all1, mean_var2.  Saves
    <fname>_explained_var.mat (v5; only mean_var2 is consumed
    downstream).  The unused V1/spike-count blocks of the MATLAB file
    are skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    projectedAtlas1, projectedTemplate1 = _load_outline_mat(data_folder)
    BW = projectedAtlas1.astype(bool)

    for kk in range(len(T)):
        ops = get_session_info2(T, kk, data_folder)
        fname = ops.fname
        U, V, t, mimg = loadUVt1(ops.session_root)
        dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])
        sp = loadKSdir2(ops.session_root)
        _syncTL, _syncProbe, WF2ephysT1 = get_wf2ephysT2(ops, t)
        WF2ephysT = WF2ephysT1[~np.isnan(WF2ephysT1)]
        dV1 = np.asarray(dV[:, ~np.isnan(WF2ephysT1)], dtype=float)
        MUA_std = get_MUA_bin(sp, WF2ephysT)
        spike_n = MUA_std.shape[0]
        dV1 = np.asarray(dV1[:50, :], dtype=float)

        # no registration first
        scale = 4
        Ut = U[::scale, ::scale, :50]

        # first predict using single units, order them by contribution
        explained_var_all = np.full(Ut.shape[:2] + (spike_n,), np.nan)
        for n_select in range(spike_n):
            MUA_std1 = MUA_std[n_select : n_select + 1, :]
            dV_raw, dV_predict, _epoch_indx = get_prediction(dV1, MUA_std1)
            explained_var = get_variance_explained(Ut, dV_raw, dV_predict)
            explained_var_all[:, :, n_select] = nanmean(explained_var, axis=2)

        T4 = _load_tform(
            data_folder / "ephys" / "rf_tform_4x" / f"{fname}_tform_4x.mat"
        )
        sizeTemplate = projectedTemplate1.shape
        var_reg = _imwarp(explained_var_all, T4, sizeTemplate)
        var_reg1 = var_reg.copy()
        var_reg1[~BW, :] = np.nan  # MATLAB temp(~BW) = nan per slice
        mean_var = nanmean(var_reg1, axis=(0, 1))
        order = np.argsort(-mean_var, kind="stable")
        B = mean_var[order]
        I = order + 1  # MATLAB 1-based sort indices

        # increase number of units incrementally by contribution
        explained_var_all1 = np.full(Ut.shape[:2] + (spike_n,), np.nan)
        for n_select in range(spike_n):
            MUA_std1 = MUA_std[order[: n_select + 1], :]
            dV_raw, dV_predict, _epoch_indx = get_prediction(dV1, MUA_std1)
            explained_var1 = get_variance_explained(Ut, dV_raw, dV_predict)
            explained_var_all1[:, :, n_select] = nanmean(explained_var1, axis=2)

        var_reg2 = _imwarp(explained_var_all1, T4, sizeTemplate)
        var_reg3 = var_reg2.copy()
        var_reg3[~BW, :] = np.nan
        mean_var2 = nanmean(var_reg3, axis=(0, 1))

        save_mat(
            save_folder / f"{fname}_explained_var.mat",
            {
                "I": I,
                "B": B,
                "mean_var": mean_var,
                "explained_var_all1": explained_var_all1,
                "mean_var2": mean_var2,
            },
        )
        print(f"getVarOrderedByNeuron: {fname} ({kk + 1}/{len(T)})")
