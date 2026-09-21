"""Translated from spirals_mirror/preprocessing/getMapsSession.m
(pipeline3_spirals_mirror.m, Extended Data Figure 11 a-b)."""
from pathlib import Path

import numpy as np

from spirals_py.spirals_mirror.plots._helpers import (
    POINT8,
    _imwarp,
    _load_atlas_tables,
    _load_tform,
)
from spirals_py.spirals_mirror.preprocessing._regression_utils import (
    _session_fname,
    load_kernel_slice,
)
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import out_root, release_twin

SCALE = 8


def getMapsSession(T, data_folder, save_folder):
    """Translated from spirals_mirror/preprocessing/getMapsSession.m

    Per session: warp the mean image onto the atlas template and downsample
    1:8:end, and read the kernel_full2 slices at the 8 hardcoded example
    points; colorIntensity = (TheColorImage / max).^2 (SQUARED here, unlike
    the raw slices saved by getExampleKernelAP/HEMI, mirroring the MATLAB).
    Saves kernelMaps_allSessions_AP.mat and kernelMaps_allSessions_hemi.mat
    with colorIntensityAll (165, 143, 8, 15) and mimgtransformed2All
    (165, 143, 15).

    mimgtransformed2 is computed with the stride argument of _imwarp, which
    equals the MATLAB full imwarp followed by (1:scale:end, 1:scale:end)
    slicing. Dead code skipped: mimg1 / mimgT2 / BW (only its shape is
    used), count and the unused td / point / nameList definitions. Kernel
    slices are read via load_kernel_slice (dense kernel_full2 if present,
    else the compact k1_real / indexMO / indexSSp variables).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    _, _, projectedAtlas1, projectedTemplate1 = _load_atlas_tables(data_folder)

    scale = SCALE
    grid_shape = projectedAtlas1[::scale, ::scale].shape
    n_sessions = len(T)

    for suffix, subfolder, kind in (
        ("-AP.mat", "regression_ap", "AP"),
        ("-hemi.mat", "regression_hemi", "hemi"),
    ):
        colorIntensityAll = np.zeros(grid_shape + (8, n_sessions))
        mimgtransformed2All = np.zeros(grid_shape + (n_sessions,))
        for kk in range(n_sessions):
            fname = _session_fname(T, kk)
            # load atlas transformation matrix tform
            tform = _load_tform(
                data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat"
            )
            # read svd components from processed data folder
            session_root = data_folder / "spirals" / "svd" / fname
            mimg = np.load(session_root / "meanImage.npy")
            path = release_twin(
                out_root() / "spirals_mirror" / subfolder / f"{fname}{suffix}",
                data_folder,
            )

            mimgtransformed2 = _imwarp(
                mimg, tform, projectedTemplate1.shape, stride=scale
            )
            mimgtransformed2All[:, :, kk] = mimgtransformed2
            for kkk in range(8):
                TheColorImage = load_kernel_slice(
                    path,
                    int(POINT8[kkk, 0]),
                    int(POINT8[kkk, 1]),
                    shape=grid_shape,
                )
                maxI = TheColorImage.max()
                colorIntensity = TheColorImage / maxI
                colorIntensity = colorIntensity**2
                colorIntensityAll[:, :, kkk, kk] = colorIntensity

        save_mat73(
            save_folder / f"kernelMaps_allSessions_{kind}.mat",
            {
                "colorIntensityAll": colorIntensityAll,
                "mimgtransformed2All": mimgtransformed2All,
            },
        )
