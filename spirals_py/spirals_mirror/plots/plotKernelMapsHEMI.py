"""Translated from spirals_mirror/plots/plotKernelMapsHEMI.m (Figure 3h)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals_mirror.plots._helpers import (
    POINT8,
    _bw_mask,
    _colorcet,
    _get_cortex_atlas_path,
    _imresize,
    _imshow_color_overlay,
    _load_atlas_tables,
    _load_h5_var,
    _select_area,
)
from spirals_py.spirals_mirror.plots.plotCortexDivision import (
    FRONTAL_AREA_PATHS,
    SENSORY_AREA_PATHS,
)
from spirals_py.utils.atlas import plotOutline

SCALE = 8
SCALE3 = 5 / 8


def plotKernelMapsHEMI(data_folder, save_folder):
    """Translated from spirals_mirror/plots/plotKernelMapsHEMI.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # load atlas brain horizontal projection and outline
    atlas1, template1, projectedAtlas1, projectedTemplate1 = _load_atlas_tables(
        data_folder
    )
    maskPath, st = _get_cortex_atlas_path(data_folder)

    mi = data_folder / "spirals_mirror" / "matching_index"
    # MATLAB (165, 143, 8, 15)
    TheColorImage_all = _load_h5_var(
        mi / "hemi_weights_8points_allsessions.mat", "TheColorImage_all"
    )
    # MATLAB (264, 228, 8)
    intensity_all = _load_h5_var(mi / "axon_intensity_all_hemi.mat", "intensity_all")
    injection_intensity = _load_h5_var(
        mi / "axon_intensity_all_injection.mat", "injection_intensity"
    )
    target_hw = TheColorImage_all.shape[:2]
    intensity_all1 = np.stack(
        [_imresize(intensity_all[:, :, i], target_hw) for i in range(8)], axis=2
    )
    injection_intensity1 = np.stack(
        [_imresize(injection_intensity[:, :, i], target_hw) for i in range(8)], axis=2
    )

    spath = st["structure_id_path"].astype(str)

    # mask and Kernel regression map for right and left
    U0 = np.zeros_like(projectedAtlas1)
    indexright, _ = _select_area(
        SENSORY_AREA_PATHS, spath, None, projectedAtlas1, "right", SCALE
    )
    indexleft, _ = _select_area(
        SENSORY_AREA_PATHS, spath, None, projectedAtlas1, "left", SCALE
    )
    BW_left = _bw_mask(indexleft)
    BW_right = _bw_mask(indexright)
    # right SSp and MO index
    indexSSp, _ = _select_area(
        SENSORY_AREA_PATHS, spath, None, projectedAtlas1, "right", SCALE
    )
    indexMO, _ = _select_area(
        FRONTAL_AREA_PATHS, spath, None, projectedAtlas1, "right", SCALE
    )
    BW_MO = _bw_mask(indexMO)
    BW_SSp = _bw_mask(indexSSp)

    color1 = _colorcet(9)
    gcamp_mean = TheColorImage_all.mean(axis=3)

    h3h = plt.figure(figsize=(5, 4))

    # SSp mask with the 8 example points
    ax = h3h.add_subplot(1, 4, 1)
    ax.imshow(template1, cmap="gray", alpha=BW_SSp)
    for kkk in range(8):
        ax.scatter(
            POINT8[kkk, 1] - 1,
            POINT8[kkk, 0] - 1,
            s=24,
            color=color1[kkk],
            edgecolors="none",
        )
    plotOutline([maskPath[3]], st, atlas1, "right", SCALE3, "w", ax=ax)
    plotOutline([maskPath[4]], st, atlas1, "right", SCALE3, "w", ax=ax)
    plotOutline(maskPath[5:11], st, atlas1, "right", SCALE3, "w", ax=ax)
    plotOutline(maskPath[3:11], st, atlas1, "right", SCALE3, "k", ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    # mean kernel maps on left hemisphere
    ax = h3h.add_subplot(1, 4, 2)
    ax.imshow(template1, cmap="gray", alpha=BW_left)
    for kkk in range(8):
        _imshow_color_overlay(ax, gcamp_mean[:, :, kkk], color1[kkk])
    plotOutline([maskPath[3]], st, atlas1, "left", SCALE3, "w", ax=ax)
    plotOutline([maskPath[4]], st, atlas1, "left", SCALE3, "w", ax=ax)
    plotOutline(maskPath[5:11], st, atlas1, "left", SCALE3, "w", ax=ax)
    plotOutline(maskPath[3:11], st, atlas1, "left", SCALE3, "k", ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    # injection sites on right SSp
    ax = h3h.add_subplot(1, 4, 3)
    ax.imshow(template1, cmap="gray", alpha=BW_SSp)
    for k in range(8):
        _imshow_color_overlay(
            ax, injection_intensity1[:, :, k], color1[k], square_normalize=False
        )
    plotOutline([maskPath[3]], st, atlas1, "right", SCALE3, "w", ax=ax)
    plotOutline([maskPath[4]], st, atlas1, "right", SCALE3, "w", ax=ax)
    plotOutline(maskPath[5:11], st, atlas1, "right", SCALE3, "w", ax=ax)
    plotOutline(maskPath[3:11], st, atlas1, "right", SCALE3, "k", ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    # axon projection intensity on left hemisphere
    ax = h3h.add_subplot(1, 4, 4)
    ax.imshow(template1, cmap="gray", alpha=BW_left)
    for kkk in range(8):
        _imshow_color_overlay(
            ax, intensity_all1[:, :, kkk], color1[kkk], square_normalize=False
        )
    plotOutline([maskPath[3]], st, atlas1, "left", SCALE3, "w", ax=ax)
    plotOutline([maskPath[4]], st, atlas1, "left", SCALE3, "w", ax=ax)
    plotOutline(maskPath[5:11], st, atlas1, "left", SCALE3, "w", ax=ax)
    plotOutline(maskPath[3:11], st, atlas1, "left", SCALE3, "k", ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    h3h.savefig(save_folder / "Fig3h_hemi_mean_activity_axon.png", bbox_inches="tight")
    h3h.savefig(save_folder / "Fig3h_hemi_mean_activity_axon.pdf", bbox_inches="tight")
    return h3h
