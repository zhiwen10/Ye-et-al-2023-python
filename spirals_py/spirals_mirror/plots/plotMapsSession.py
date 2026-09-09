"""Translated from spirals_mirror/plots/plotMapsSession.m
(Extended Data Figure 11 a-b)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals_mirror.plots._helpers import (
    POINT8,
    _bw_mask,
    _colorcet,
    _get_cortex_atlas_path,
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


def plotMapsSession(T, data_folder, save_folder):
    """Translated from spirals_mirror/plots/plotMapsSession.m

    T is the pandas session table (MouseID column is used for titles)."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # load atlas brain horizontal projection and outline
    atlas1, template1, projectedAtlas1, projectedTemplate1 = _load_atlas_tables(
        data_folder
    )
    maskPath, st = _get_cortex_atlas_path(data_folder)
    spath = st["structure_id_path"].astype(str)

    # mask and Kernel regression map for right and left
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
    rk = data_folder / "spirals_mirror" / "regression_kernels"

    # MO: AP kernel maps across sessions
    colorIntensityAll = _load_h5_var(
        rk / "kernelMaps_allSessions_AP.mat", "colorIntensityAll"
    )  # MATLAB (165, 143, 8, 15)
    mimgtransformed2All = _load_h5_var(
        rk / "kernelMaps_allSessions_AP.mat", "mimgtransformed2All"
    )  # MATLAB (165, 143, 15)

    hs11a = plt.figure(figsize=(6, 8))

    ax = hs11a.add_subplot(4, 4, 1)
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

    for kk in range(15):
        ax = hs11a.add_subplot(4, 4, kk + 2)
        ax.imshow(mimgtransformed2All[:, :, kk], cmap="gray", alpha=BW_MO)
        for kkk in range(8):
            _imshow_color_overlay(
                ax, colorIntensityAll[:, :, kkk, kk], color1[kkk],
                square_normalize=False,
            )
        plotOutline(maskPath[0:3], st, atlas1, "right", SCALE3, "k", ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(T["MouseID"].iloc[kk], fontsize=6)

    hs11a.savefig(save_folder / "FigS11a_map_AP.png", bbox_inches="tight")
    hs11a.savefig(save_folder / "FigS11a_map_AP.pdf", bbox_inches="tight")

    # LEFT TO RIGHT: hemi kernel maps across sessions
    colorIntensityAll = _load_h5_var(
        rk / "kernelMaps_allSessions_hemi.mat", "colorIntensityAll"
    )
    mimgtransformed2All = _load_h5_var(
        rk / "kernelMaps_allSessions_hemi.mat", "mimgtransformed2All"
    )

    hs11b = plt.figure(figsize=(6, 8))

    ax = hs11b.add_subplot(4, 4, 1)
    ax.imshow(template1, cmap="gray", alpha=BW_right)
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

    for kk in range(15):
        ax = hs11b.add_subplot(4, 4, kk + 2)
        ax.imshow(mimgtransformed2All[:, :, kk], cmap="gray", alpha=BW_left)
        for kkk in range(8):
            _imshow_color_overlay(
                ax, colorIntensityAll[:, :, kkk, kk], color1[kkk],
                square_normalize=False,
            )
        plotOutline([maskPath[3]], st, atlas1, "left", SCALE3, "w", ax=ax)
        plotOutline([maskPath[4]], st, atlas1, "left", SCALE3, "w", ax=ax)
        plotOutline(maskPath[5:11], st, atlas1, "left", SCALE3, "w", ax=ax)
        plotOutline(maskPath[3:11], st, atlas1, "left", SCALE3, "k", ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(T["MouseID"].iloc[kk], fontsize=6)

    hs11b.savefig(save_folder / "FigS11b_map_hemi.png", bbox_inches="tight")
    hs11b.savefig(save_folder / "FigS11b_map_hemi.pdf", bbox_inches="tight")
    return hs11a, hs11b
