"""Translated from spirals_mirror/preprocessing/getExampleKernelHEMI.m
(pipeline3_spirals_mirror.m, Figure 3h/j-k)."""
from pathlib import Path

import numpy as np

from spirals_py.spirals_mirror.plots._helpers import POINT8
from spirals_py.spirals_mirror.preprocessing._regression_utils import (
    _session_fname,
    load_kernel_slice,
)
from spirals_py.utils.matio import save_mat73


def getExampleKernelHEMI(T, data_folder, save_folder):
    """Translated from spirals_mirror/preprocessing/getExampleKernelHEMI.m

    Saves TheColorImage_all (165, 143, 8, 15), the RAW (unsquared,
    unnormalized) kernel_full2 slices at the 8 hardcoded example points of
    every session, to hemi_weights_8points_allsessions.mat.

    Bug fixed: the MATLAB original saves the singular
    'hemi_weight_8points_allsessions.mat', while plotKernelMapsHEMI /
    plotMatchingIndexHEMI read the plural
    'hemi_weights_8points_allsessions.mat' (matching the data release);
    the plural name is written here. The dead atlas loads of the original
    are skipped. Kernel slices are read via load_kernel_slice (dense
    kernel_full2 if present, else the compact k1_real / indexMO / indexSSp
    variables, where indexMO is the left sensory source index and indexSSp
    the right sensory target index).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    n_sessions = len(T)
    TheColorImage_all = None
    for kk in range(n_sessions):
        fname = _session_fname(T, kk)
        path = (
            data_folder
            / "spirals_mirror"
            / "regression_hemi"
            / f"{fname}-hemi.mat"
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
        save_folder / "hemi_weights_8points_allsessions.mat",
        {"TheColorImage_all": TheColorImage_all},
    )
