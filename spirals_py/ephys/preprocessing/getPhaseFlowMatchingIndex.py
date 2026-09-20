from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from spirals_py.ephys.preprocessing._regression import compare_flow1, get_wf_mua2
from spirals_py.ephys.utils import get_session_info2
from spirals_py.spirals.plots._fig1_helpers_s1 import _load_roi_vertices
from spirals_py.utils.matio import load_mat_var, save_mat


def _roi_bw(roi, mimg_shape, scale=4):
    """MATLAB poly2mask(roi.Position(:,1), roi.Position(:,2), M + 240,
    N + 240) rasterization (adaptation: matplotlib Path.contains_points
    on pixel centers of the padded grid), cropped [120:-120] then
    [::scale] as in getPhaseFlowMatchingIndex.m."""
    M = mimg_shape[0] + 240
    N = mimg_shape[1] + 240
    yy, xx = np.mgrid[0:M, 0:N]
    pts = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    bw = MplPath(np.asarray(roi, dtype=float)).contains_points(pts).reshape(M, N)
    bw = bw[120:-120, 120:-120]
    return bw[::scale, ::scale]


def getPhaseFlowMatchingIndex(T, data_folder, save_folder, rng=None):
    """Translated from ephys/preprocessing/getPhaseFlowMatchingIndex.m

    Per session: BW = (mean(explained_var_all, axis=2) >= 0.1) AND the
    rasterized ROI mask on the padded grid (cropped/dowsampled to the
    Ut resolution); compare_flow1 then produces 21 repetitions (real +
    20 permuted-MUA) of amplitude-binned phase/flow circular stats.
    Saves <fname>.mat (v5; consumer _load_mat_var reads it with a
    scipy + h5py fallback) with N (21, 20), edges (21, 21), phase_mu /
    phase_var / flow_mu / flow_var (21, 20) and traceAmp_mean
    ((len - 1), 1).  The stray MATLAB `save folder` side-effect line is
    skipped.
    """
    if rng is None:
        rng = np.random.default_rng()
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in range(len(T)):
        ops = get_session_info2(T, kk, data_folder)
        fname = ops.fname
        explained_var_all = load_mat_var(
            data_folder / "ephys" / "dv_prediction" / f"{fname}_dv_predict.mat",
            "explained_var_all",
        )
        roi = _load_roi_vertices(
            data_folder / "ephys" / "roi" / f"{fname}_roi.mat"
        )
        Ut, mimg, V1, dV1, MUA_std = get_wf_mua2(ops)
        scale = 4
        mimg1 = mimg[::scale, ::scale]

        explained_var_all[explained_var_all < 0] = 0
        var_mean = explained_var_all.mean(axis=2)
        BW1 = var_mean >= 0.1
        bw = _roi_bw(roi, mimg.shape, scale)
        BW = BW1 & bw

        len_ = 5000  # sample size to use
        (N, edges, phase_mu, phase_var, flow_mu, flow_var, traceAmp_mean) = (
            compare_flow1(Ut, mimg1, dV1, MUA_std, BW, len_, rng=rng)
        )
        save_mat(
            save_folder / f"{fname}.mat",
            {
                "N": N,
                "edges": edges,
                "phase_mu": phase_mu,
                "phase_var": phase_var,
                "flow_mu": flow_mu,
                "flow_var": flow_var,
                "traceAmp_mean": traceAmp_mean.reshape(-1, 1),
            },
        )
        print(f"getPhaseFlowMatchingIndex: {fname} ({kk + 1}/{len(T)})")
