"""Translated from spirals_mirror/preprocessing/getExampleKernelAP.m
(pipeline3_spirals_mirror.m, Figure 3h/j-k)."""
from pathlib import Path

import numpy as np

from spirals_py.spirals_mirror.plots._helpers import POINT8
from spirals_py.spirals_mirror.preprocessing._regression_utils import (
    _session_fname,
    load_kernel_slice,
)
from spirals_py.utils.matio import save_mat73
from spirals_py.utils.paths import out_root, release_twin


def getExampleKernelAP(T, data_folder, save_folder):
    """Translated from spirals_mirror/preprocessing/getExampleKernelAP.m

    Saves TheColorImage_all (165, 143, 8, 15), the RAW (unsquared,
    unnormalized) kernel_full2 slices at the 8 hardcoded example points of
    every session, to AP_weights_8points_allsessions.mat.

    Bug fixed: the MATLAB original saves the singular
    'AP_weight_8points_allsessions.mat', while plotKernelMapsAP /
    plotMatchingIndexAP read the plural 'AP_weights_8points_allsessions.mat'
    (matching the data release); the plural name is written here. The dead
    atlas loads of the original (only spath / st are ever used, and not
    even those) are skipped. kernel slices are read via load_kernel_slice
    (dense kernel_full2 if present, else the compact k1_real / indexMO /
    indexSSp variables).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    n_sessions = len(T)
    TheColorImage_all = None
    for kk in range(n_sessions):
        fname = _session_fname(T, kk)
        path = release_twin(
            out_root() / "spirals_mirror" / "regression_ap" / f"{fname}-AP.mat",
            data_folder,
        )
        # loop through 8 example points
        for kkk in range(8):
            TheColorImage = load_kernel_slice(
                path, int(POINT8[kkk, 0]), int(POINT8[kkk, 1])
            )
            if TheColorImage_all is None:
                TheColorImage_all = np.zeros(
                    TheColorImage.shape + (8, n_sessions)
                )
            TheColorImage_all[:, :, kkk, kk] = TheColorImage

    save_mat73(
        save_folder / "AP_weights_8points_allsessions.mat",
        {"TheColorImage_all": TheColorImage_all},
    )
