"""Translated from spirals_mirror/preprocessing/getReducedRankRegressionAP.m
(pipeline3_spirals_mirror.m, Figure 3c/h-j)."""
from pathlib import Path

import numpy as np

from spirals_py.spirals.utils import loadUVt1
from spirals_py.spirals_mirror.plots._helpers import (
    _get_cortex_atlas_path,
    _imwarp,
    _load_atlas_tables,
    _load_tform,
    _zscore_rows,
)
from spirals_py.spirals_mirror.plots.plotCortexDivision import (
    FRONTAL_AREA_PATHS,
    SENSORY_AREA_PATHS,
)
from spirals_py.spirals_mirror.preprocessing._regression_utils import (
    CanonCor2,
    _session_fname,
    select_area_indices,
    sseExplainedCal,
)
from spirals_py.spirals_mirror.utils import redoSVD
from spirals_py.utils.matio import save_mat73

SCALE = 8
N_TRAIN = 40000
N_SVD_TRAIN = 58000
N_TEST_END = 60000
N_RANK = 50


def getReducedRankRegressionAP(T, data_folder, save_folder, dense=False):
    """Translated from spirals_mirror/preprocessing/getReducedRankRegressionAP.m

    For each of the 15 sessions: warp the first 50 SVD components onto the
    atlas template, redo the SVD separately within the right-hemisphere
    sensory and frontal (MO) masks, run the CanonCor2 reduced-rank
    regression from the MO components to the sensory components on frames
    1:40000, and save <fname>-AP.mat with explained_var5 (50,1) (ranks
    1..50, evaluated on test frames 40001:60000 via sseExplainedCal), a, b
    and the kernel k1_real (nMO_sel x nSSp_sel) plus indexMO / indexSSp
    (column-major 1-based linear index vectors into the 165x143 grid; +1 of
    the 0-based select_area_indices output), from which kernel_full2 can be
    reconstructed without the dense tensor.

    dense=True additionally writes kernel_full2 (165x143x165x143, ~4.5 GB
    as MATLAB v7.3 doubles); dense=False skips it.

    Dead code skipped: dV (never used), the point / nameList / maskPath
    definitions, the mimg warp (mimg1 / mimgT2 are only used for the
    kernel_full2 shape, taken from the downsampled projectedAtlas1
    instead) and BW_empty. The gpuArray branches run on CPU. The MATLAB
    areaPath(8) gap (never assigned) just drops that entry from
    sensoryArea.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    _, _, projectedAtlas1, projectedTemplate1 = _load_atlas_tables(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    spath = st["structure_id_path"].astype(str)

    n_sessions = len(T)
    for kk in range(n_sessions):
        fname = _session_fname(T, kk)
        session_root = data_folder / "spirals" / "svd" / fname
        U, V, t, mimg = loadUVt1(session_root)
        tform = _load_tform(
            data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat"
        )

        U = U[:, :, :N_RANK]
        V = V[:N_RANK, :]

        Utransformed = _imwarp(U, tform, projectedTemplate1.shape)

        # mask and Kernel regression map for SSp and MO
        hemi = "right"
        indexSSp, UselectedSSp = select_area_indices(
            SENSORY_AREA_PATHS, spath, Utransformed, projectedAtlas1, hemi, SCALE
        )
        Unew_SSp, Vnew_SSp, _ = redoSVD(UselectedSSp, V[:, :N_SVD_TRAIN])
        USSp = Unew_SSp[:, :N_RANK]
        VSSp = Vnew_SSp[:N_RANK, :]

        indexMO, UselectedMO = select_area_indices(
            FRONTAL_AREA_PATHS, spath, Utransformed, projectedAtlas1, hemi, SCALE
        )
        Unew_MO, Vnew_MO, _ = redoSVD(UselectedMO, V[:, :N_SVD_TRAIN])
        UMO = Unew_MO[:, :N_RANK]
        VMO = Vnew_MO[:N_RANK, :]

        grid_shape = projectedAtlas1[::SCALE, ::SCALE].shape

        # prepare regressor and signal for regression
        regressor1 = _zscore_rows(VMO)
        regressor = regressor1[:, :N_TRAIN]
        signal1 = USSp @ VSSp[:, :N_TRAIN]

        # reduced rank regression regression
        a, b, R2 = CanonCor2(signal1.T, regressor.T)

        kk1 = b[:, :N_RANK] @ a[:, :N_RANK].T
        k1_real = UMO @ kk1

        # prepare test regressor
        data = UselectedMO @ V[:, N_TRAIN:N_TEST_END]
        # subtract mean as we did before
        data = data - data.mean(axis=1, keepdims=True)
        regressor_test = Unew_MO.T @ data
        regressor_test_z = _zscore_rows(regressor_test)

        traceSSp2 = UselectedSSp @ V[:, N_TRAIN:N_TEST_END]
        explained_var5 = np.zeros((N_RANK, 1))
        for n in range(1, N_RANK + 1):
            traceSSp_predict = (
                regressor_test_z[:N_RANK, :].T @ b[:, :n] @ a[:, :n].T
            )
            traceSSp_predict = traceSSp_predict.T
            explained_var5[n - 1, 0] = sseExplainedCal(
                traceSSp2.ravel(order="F")[None, :],
                traceSSp_predict.ravel(order="F")[None, :],
            )[0]

        variables = {
            "explained_var5": explained_var5,
            "a": a,
            "b": b,
            "k1_real": k1_real,
            "indexMO": (indexMO + 1).reshape(-1, 1),
            "indexSSp": (indexSSp + 1).reshape(-1, 1),
        }
        if dense:
            kernel_full2 = np.zeros(
                (grid_shape[0] * grid_shape[1], grid_shape[0] * grid_shape[1])
            )
            kernel_full2[np.ix_(indexMO, indexSSp)] = k1_real
            kernel_full2 = kernel_full2.reshape(
                (*grid_shape, *grid_shape), order="F"
            )
            variables["kernel_full2"] = kernel_full2

        save_mat73(save_folder / f"{fname}-AP.mat", variables)
        print(f"getReducedRankRegressionAP: {fname} ({kk + 1}/{n_sessions})")
