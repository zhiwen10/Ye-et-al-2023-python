"""Translated from spirals_mirror/preprocessing/getReducedRankRegressionHEMI.m
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


def getReducedRankRegressionHEMI(T, data_folder, save_folder, dense=False):
    """Translated from spirals_mirror/preprocessing/getReducedRankRegressionHEMI.m

    Same reduced-rank regression as getReducedRankRegressionAP but between
    the left- and right-hemisphere sensory areas: the regressor is built
    from the left components and the signal from the right ones;
    k1_real = Uleft @ kk1 (nLeft_sel x nRight_sel) is scattered from the
    left sensory index into the right sensory columns. Saves
    <fname>-hemi.mat with explained_var5 (50,1), a, b and the compact
    kernel k1_real / indexMO / indexSSp (column-major 1-based linear
    indices into the 165x143 grid; in this file indexMO is the left
    sensory source index and indexSSp the right sensory target index).
    kernel_full2 is only written when dense=True (see
    getReducedRankRegressionAP).

    Dead code skipped: dV, the point / nameList / maskPath / root1 /
    areaName definitions, the mimg warp (mimgT2 only supplies the
    kernel_full2 shape, taken from the downsampled projectedAtlas1
    instead), BW_empty / BW_left / BW_right (never saved) and the unused
    td. The gpuArray branches run on CPU.
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

        # mask and Kernel regression map for left and right
        hemi = "right"
        indexright, UselectedRight = select_area_indices(
            SENSORY_AREA_PATHS, spath, Utransformed, projectedAtlas1, hemi, SCALE
        )
        Unew_right, Vnew_right, _ = redoSVD(UselectedRight, V[:, :N_SVD_TRAIN])
        Uright = Unew_right[:, :N_RANK]
        Vright = Vnew_right[:N_RANK, :]

        hemi = "left"
        indexleft, UselectedLeft = select_area_indices(
            SENSORY_AREA_PATHS, spath, Utransformed, projectedAtlas1, hemi, SCALE
        )
        Unew_left, Vnew_left, _ = redoSVD(UselectedLeft, V[:, :N_SVD_TRAIN])
        Uleft = Unew_left[:, :N_RANK]
        Vleft = Vnew_left[:N_RANK, :]

        grid_shape = projectedAtlas1[::SCALE, ::SCALE].shape

        # prepare regressor and signal for regression
        regressor1 = _zscore_rows(Vleft)
        regressor = regressor1[:, :N_TRAIN]
        signal1 = Uright @ Vright[:, :N_TRAIN]

        # regression
        a, b, R2 = CanonCor2(signal1.T, regressor.T)

        kk1 = b[:, :N_RANK] @ a[:, :N_RANK].T
        k1_real = Uleft @ kk1

        # prepare test regressor
        data = UselectedLeft @ V[:, N_TRAIN:N_TEST_END]
        # subtract mean as we did before
        data = data - data.mean(axis=1, keepdims=True)
        regressor_test = Unew_left.T @ data
        regressor_test_z = _zscore_rows(regressor_test)

        traceSSp2 = UselectedRight @ V[:, N_TRAIN:N_TEST_END]
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
            "indexMO": (indexleft + 1).reshape(-1, 1),
            "indexSSp": (indexright + 1).reshape(-1, 1),
        }
        if dense:
            kernel_full2 = np.zeros(
                (grid_shape[0] * grid_shape[1], grid_shape[0] * grid_shape[1])
            )
            kernel_full2[np.ix_(indexleft, indexright)] = k1_real
            kernel_full2 = kernel_full2.reshape(
                (*grid_shape, *grid_shape), order="F"
            )
            variables["kernel_full2"] = kernel_full2

        save_mat73(save_folder / f"{fname}-hemi.mat", variables)
        print(f"getReducedRankRegressionHEMI: {fname} ({kk + 1}/{n_sessions})")
