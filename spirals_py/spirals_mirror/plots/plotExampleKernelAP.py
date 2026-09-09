"""Translated from spirals_mirror/plots/plotExampleKernelAP.m (Figures 3e-g).

Adaptations (documented deviations from the MATLAB source), shared with
plotExampleKernelHEMI:
- MATLAB `kk1 = regressor1'\\signal1` is dimensionally inconsistent as
  written; the equivalent least-squares solution used by the companion
  preprocessing (getReducedRankRegressionAP.m, kk1 = b*a' from CanonCor2) is
  computed via normal equations (see _helpers._build_regression).
- kernel_full/kernel_full2 is never materialized; the two point-indexed
  slices the figure uses are computed lazily (_kernel_slice).
- rawAll (computed but unused in MATLAB) is skipped.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spirals_py.spirals.utils import loadUVt1
from spirals_py.spirals_mirror.plots._helpers import (
    _build_regression,
    _colorcet,
    _get_cortex_atlas_path,
    _imwarp,
    _kernel_slice,
    _load_atlas_tables,
    _load_tform,
    _matlab_round,
    _select_area,
    _bw_mask,
    _imshow_color_overlay,
)
from spirals_py.spirals_mirror.plots.plotCortexDivision import (
    FRONTAL_AREA_PATHS,
    SENSORY_AREA_PATHS,
)
from spirals_py.spirals_mirror.utils import redoSVD
from spirals_py.utils.atlas import plotOutline

SCALE = 8
SCALE3 = 5 / 8


def plotExampleKernelAP(data_folder, save_folder):
    """Translated from spirals_mirror/plots/plotExampleKernelAP.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # load atlas brain horizontal projection and outline
    atlas1, template1, projectedAtlas1, projectedTemplate1 = _load_atlas_tables(
        data_folder
    )
    maskPath, st = _get_cortex_atlas_path(data_folder)

    mn = "ZYE_0012"
    td = "2020-10-16"
    tdb = pd.Timestamp(td).strftime("%Y%m%d")
    en = 5
    subfolder = f"{mn}_{tdb}_{en}"
    session_root = data_folder / "spirals" / "svd" / subfolder
    U, V, t, mimg = loadUVt1(session_root)
    U = U[:, :, :50]
    V = V[:50, :]
    # registration
    tform = _load_tform(data_folder / "tables" / f"{subfolder}_tform.mat")
    Utransformed = _imwarp(U, tform, projectedTemplate1.shape)
    mimgtransformed = _imwarp(mimg, tform, projectedTemplate1.shape)

    spath = st["structure_id_path"].astype(str)

    # right SSp index
    indexSSp, UselectedSSp = _select_area(
        SENSORY_AREA_PATHS, spath, Utransformed, projectedAtlas1, "right", SCALE
    )
    Unew_SSp, Vnew_SSp, _ = redoSVD(UselectedSSp, V[:, :58000])
    USSp = Unew_SSp[:, :50]
    VSSp = Vnew_SSp[:50, :]
    # MO index
    indexMO, UselectedMO = _select_area(
        FRONTAL_AREA_PATHS, spath, Utransformed, projectedAtlas1, "right", SCALE
    )
    Unew_MO, Vnew_MO, _ = redoSVD(UselectedMO, V[:, :58000])
    UMO = Unew_MO[:, :50]
    VMO = Vnew_MO[:50, :]

    BW_MO = _bw_mask(indexMO)
    BW_SSp = _bw_mask(indexSSp)

    # prepare regressor and signal for regression
    kk1, k1_real, regressor_test_z = _build_regression(
        UMO, VMO, USSp, VSSp, UselectedMO, Unew_MO, V[:, 77999:]
    )
    # prediction
    predicted_signals = kk1.T @ regressor_test_z[:50, :]

    def predicted_trace(fr, fc):
        """squeeze(traceSSp_predict2(round(fr), round(fc), :)) lazily."""
        pr = int(_matlab_round(fr))
        pc = int(_matlab_round(fc))
        lin0 = (pc - 1) * BW_SSp.shape[0] + (pr - 1)
        j = int(np.searchsorted(indexSSp, lin0))
        if j >= indexSSp.size or indexSSp[j] != lin0:
            return np.zeros(predicted_signals.shape[1])
        return predicted_signals[j]

    # example points (1-based row/col; VISp here is [110,104] as in the .m)
    point = np.zeros((8, 2), dtype=int)
    point[5] = [83, 93]  # SSp-tr
    point[4] = [73, 96]  # SSp-ul
    point[3] = [65, 105]  # SSp-ll
    point[2] = [55, 118]  # SSp-m
    point[1] = [69, 122]  # SSp-n
    point[0] = [84, 117]  # SSp-bfd
    point[7] = [110, 104]  # VISp
    point[6] = [97, 79]  # RSP

    # target pixels in the full-resolution warped image (1-based row/col)
    pixel = np.zeros((3, 2), dtype=int)
    pixel[0] = np.array([115, 105]) * 8  # V1
    pixel[1] = np.array([78, 118]) * 8  # S1
    pixel[2] = np.array([89, 78]) * 8  # RSP

    color1 = _colorcet(9)
    Vraw = V[:, 77999:]
    h3eg = plt.figure(figsize=(9, 4))

    for block, (kkk, i) in enumerate([(8, 1), (1, 2)]):
        pr, pc = int(point[kkk - 1, 0]), int(point[kkk - 1, 1])
        # kernel map on MO
        ax = h3eg.add_subplot(2, 3, 1 + 3 * block)
        ax.imshow(template1, cmap="gray", alpha=BW_MO)
        TheColorImage = _kernel_slice(
            k1_real, indexMO, indexSSp, BW_MO.shape, pr, pc
        )
        maxI = TheColorImage.max()
        row, col = np.unravel_index(
            np.argmax(TheColorImage.ravel(order="F")), TheColorImage.shape, order="F"
        )
        _imshow_color_overlay(ax, TheColorImage, color1[kkk - 1])
        ax.scatter(col, row, s=24, c="k", edgecolors="none")
        plotOutline(maskPath[0:3], st, atlas1, "right", SCALE3, "k", ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")

        # target pixel on SSp
        ax = h3eg.add_subplot(2, 3, 2 + 3 * block)
        ax.imshow(template1, cmap="gray", alpha=BW_SSp)
        ax.scatter(
            pc - 1,
            pr - 1,
            s=24,
            color=color1[kkk - 1],
            edgecolors="none",
        )
        plotOutline([maskPath[3]], st, atlas1, "right", SCALE3, "w", ax=ax)
        plotOutline([maskPath[4]], st, atlas1, "right", SCALE3, "w", ax=ax)
        plotOutline(maskPath[5:11], st, atlas1, "right", SCALE3, "w", ax=ax)
        plotOutline(maskPath[3:11], st, atlas1, "right", SCALE3, "k", ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")

        # traces: predicted / raw at target / raw at kernel peak
        ax = h3eg.add_subplot(2, 3, 3 + 3 * block)
        frow, fcol = int(pixel[i - 1, 0]), int(pixel[i - 1, 1])
        mval = mimgtransformed[frow - 1, fcol - 1]
        trace_raw1 = (
            Utransformed[frow - 1, fcol - 1, :].astype(np.float64) @ Vraw / mval
        )  # sensory
        trace_raw3 = (
            Utransformed[int(row) * SCALE - 1, int(col) * SCALE - 1, :].astype(
                np.float64
            )
            @ Vraw
            / mval
        )  # left hemisphere
        trace_predict = predicted_trace(frow / SCALE, fcol / SCALE) / mval  # predicted
        tt = np.arange(1, trace_predict.size + 1) / 35
        ax.plot(tt, trace_predict + 0.1, color=color1[kkk - 1])
        ax.plot(tt, trace_raw1, color=[0, 0, 0])
        ax.plot(tt, trace_raw3 + 0.2, color=color1[kkk - 1])
        ax.set_xlim(50, 60)
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(50, 61, 2))
        ax.set_xticklabels([str(v) for v in range(0, 11, 2)])
        ax.set_ylim(-0.05, 0.25)

    h3eg.savefig(save_folder / "Fig3eg_example_traces_MO2SSp.png", bbox_inches="tight")
    h3eg.savefig(save_folder / "Fig3eg_example_traces_MO2SSp.pdf", bbox_inches="tight")
    return h3eg
