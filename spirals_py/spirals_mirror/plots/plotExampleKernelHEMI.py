"""Translated from spirals_mirror/plots/plotExampleKernelHEMI.m (Figures 3d-f).

Adaptations (documented deviations from the MATLAB source):
- MATLAB `kk1 = regressor'\\signal1` is dimensionally inconsistent as written
  (regressor' is frames x 50, signal1 is npixels x frames). The equivalent
  least-squares solution used by the companion preprocessing
  (getReducedRankRegressionHEMI.m, kk1 = b*a' from CanonCor2, i.e.
  signal1 ~= kk1' * regressor) is computed instead via normal equations,
  without materializing signal1 (npixels x frames).
- kernel_full/kernel_full2 (a 4.5-GB dense tensor in MATLAB) is never
  materialized; the two (point-indexed) slices that the figure uses are
  computed lazily and are mathematically identical (_kernel_slice).
- rawAll (Utr*V(:,78000:end)) is computed in MATLAB but never used; skipped.
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
    _select_area,
    _bw_mask,
    _imshow_color_overlay,
    _zscore_rows,
)
from spirals_py.spirals_mirror.plots.plotCortexDivision import SENSORY_AREA_PATHS
from spirals_py.spirals_mirror.utils import redoSVD
from spirals_py.utils.atlas import plotOutline

SCALE = 8
SCALE3 = 5 / 8


def plotExampleKernelHEMI(data_folder, save_folder):
    """Translated from spirals_mirror/plots/plotExampleKernelHEMI.m"""
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

    # mask and Kernel regression map for right
    indexright, UselectedRight = _select_area(
        SENSORY_AREA_PATHS, spath, Utransformed, projectedAtlas1, "right", SCALE
    )
    Unew_right, Vnew_right, _ = redoSVD(UselectedRight, V[:, :58000])
    Uright = Unew_right[:, :50]
    Vright = Vnew_right[:50, :]
    # mask and Kernel regression map for left
    indexleft, UselectedLeft = _select_area(
        SENSORY_AREA_PATHS, spath, Utransformed, projectedAtlas1, "left", SCALE
    )
    Unew_left, Vnew_left, _ = redoSVD(UselectedLeft, V[:, :58000])
    Uleft = Unew_left[:, :50]
    Vleft = Vnew_left[:50, :]

    BW_left = _bw_mask(indexleft)
    BW_right = _bw_mask(indexright)

    # prepare regressor and signal for regression
    kk1, k1_real, regressor_test_z = _build_regression(
        Uleft, Vleft, Uright, Vright, UselectedLeft, Unew_left, V[:, 77999:]
    )
    # prediction
    predicted_signals = kk1.T @ regressor_test_z[:50, :]

    def predicted_trace(pr, pc):
        """squeeze(traceSSp_predict2(pr, pc, :)) lazily (see module docstring)."""
        lin0 = (pc - 1) * BW_left.shape[0] + (pr - 1)
        j = int(np.searchsorted(indexright, lin0))
        if j >= indexright.size or indexright[j] != lin0:
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

    color1 = _colorcet(9)
    Vraw = V[:, 77999:]
    h3df = plt.figure(figsize=(9, 4))

    for block, kkk in enumerate([8, 1]):  # MATLAB kkk = 8 then kkk = 1
        pr, pc = int(point[kkk - 1, 0]), int(point[kkk - 1, 1])
        # kernel map on left hemisphere
        ax = h3df.add_subplot(2, 3, 1 + 3 * block)
        ax.imshow(template1, cmap="gray", alpha=BW_left)
        TheColorImage = _kernel_slice(
            k1_real, indexleft, indexright, BW_left.shape, pr, pc
        )
        maxI = TheColorImage.max()
        row, col = np.unravel_index(
            np.argmax(TheColorImage.ravel(order="F")), TheColorImage.shape, order="F"
        )
        _imshow_color_overlay(ax, TheColorImage, color1[kkk - 1])
        ax.scatter(col, row, s=24, c="k", edgecolors="none")
        plotOutline([maskPath[3]], st, atlas1, "left", SCALE3, "w", ax=ax)
        plotOutline([maskPath[4]], st, atlas1, "left", SCALE3, "w", ax=ax)
        plotOutline(maskPath[5:11], st, atlas1, "left", SCALE3, "w", ax=ax)
        plotOutline(maskPath[3:11], st, atlas1, "left", SCALE3, "k", ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")

        # target pixel on right hemisphere
        ax = h3df.add_subplot(2, 3, 2 + 3 * block)
        ax.imshow(template1, cmap="gray", alpha=BW_right)
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
        ax = h3df.add_subplot(2, 3, 3 + 3 * block)
        mval = mimgtransformed[pr * SCALE - 1, pc * SCALE - 1]
        trace_raw1 = (
            Utransformed[pr * SCALE - 1, pc * SCALE - 1, :].astype(np.float64)
            @ Vraw
            / mval
        )  # sensory
        trace_raw3 = (
            Utransformed[int(row) * SCALE - 1, int(col) * SCALE - 1, :].astype(
                np.float64
            )
            @ Vraw
            / mval
        )  # left hemisphere
        trace_predict = predicted_trace(pr, pc) / mval  # predicted
        tt = np.arange(1, trace_predict.size + 1) / 35
        ax.plot(tt, trace_predict + 0.1, color=color1[kkk - 1])
        ax.plot(tt, trace_raw1, color=[0, 0, 0])
        ax.plot(tt, trace_raw3 + 0.2, color=color1[kkk - 1])
        ax.set_xlim(50, 60)
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.arange(50, 61, 2))
        ax.set_xticklabels([str(v) for v in range(0, 11, 2)])
        ax.set_ylim(-0.05, 0.25)

    h3df.savefig(
        save_folder / "Fig3df_example_traces_left2right.png", bbox_inches="tight"
    )
    h3df.savefig(
        save_folder / "Fig3df_example_traces_left2right.pdf", bbox_inches="tight"
    )
    return h3df
