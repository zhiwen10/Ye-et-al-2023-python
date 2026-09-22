"""Translated from axons/plots/plotAxonOrientationMO5.m

Plots axon orientation bias of all neurons in the left hemisphere relative
to MOp (Extended Data Fig. 10a-d).
"""

from pathlib import Path

import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from spirals_py.axons.plots.plotAxonFlowMatch2 import (
    MASK_PATHS,
    _get_cortex_atlas_path,
)
from spirals_py.axons.plots.plotBiasHitogram import (
    CENTER_MOP,
    _getAngleHistogramStats,
    _plotAngleStairsShuffle,
)
from spirals_py.utils.atlas import plotOutline
from spirals_py.utils.circular import circ_mtest, watsons_U2_perm_test

CENTER = (377, 428)  # MOp
CENTER2 = (244, 542)  # SSp-un
POINT1 = (340, 475)
POINT2 = (500, 250)


def _colorcet_C06(N=256):
    """colorcet('C06','N',N) via the Python colorcet package (CET_C6)."""
    base = np.array([to_rgb(c) for c in colorcet.palette["CET_C6"]])
    if N >= base.shape[0]:
        return base
    xi = np.arange(N) * (base.shape[0] - 1) / (N - 1)
    return np.stack(
        [np.interp(xi, np.arange(base.shape[0]), base[:, j]) for j in range(3)], axis=1
    )


def _interp_colors(x1, cmap_arr, v):
    """interp1(x1, cmap_arr, v) per color channel."""
    return np.stack(
        [np.interp(v, x1, cmap_arr[:, j]) for j in range(3)], axis=-1
    )


def _plot_outlines(ax, maskPath, st, atlas1, hemi, scale3):
    """The repeated plotOutline block from plotAxonOrientationMO5.m"""
    lineColor = "k"
    lineColor1 = "w"
    for i in list(range(0, 15)) + list(range(18, 22)):  # MATLAB [1:15,19:22]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor1, ax=ax)
    for i in list(range(0, 6)) + [13, 14, 18]:  # MATLAB [1:6,14,15,19]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor, ax=ax)
    for i in range(15, 18):  # MATLAB [16:18]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor, ax=ax)
    for i in [13, 14]:  # MATLAB [14:15]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor, ax=ax)


def _plot_bias_lines(ax, T1, color2, scale1=15):
    x1 = np.linspace(-90, 90, 180)
    colora3 = np.nan_to_num(_interp_colors(x1, color2, T1["bias_angle"].to_numpy()))
    soma_center1 = T1[["soma_center_1", "soma_center_2"]].to_numpy()
    axon_bias = T1[["axon_bias_1", "axon_bias_2"]].to_numpy()
    pc_ratio = T1["pc_ratio"].to_numpy()
    for i in range(len(T1)):
        ax.plot(
            [
                soma_center1[i, 0] - axon_bias[i, 0] * scale1 * pc_ratio[i],
                soma_center1[i, 0] + axon_bias[i, 0] * scale1 * pc_ratio[i],
            ],
            [
                soma_center1[i, 1] - axon_bias[i, 1] * scale1 * pc_ratio[i],
                soma_center1[i, 1] + axon_bias[i, 1] * scale1 * pc_ratio[i],
            ],
            color=colora3[i],
            linewidth=1,
        )


def plotAxonOrientationMO5(data_folder, save_folder, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    with h5py.File(data_folder / "tables" / "horizontal_cortex_atlas_50um.mat") as f:
        # v7.3, stored transposed: MATLAB array is (264, 228)
        atlas1 = np.array(f["atlas1"]).T
    maskPath, st = _get_cortex_atlas_path(data_folder)
    maskPath = MASK_PATHS

    # loaded in MATLAB but unused downstream
    with h5py.File(data_folder / "revision" / "axons" / "MO_roi.mat") as f:
        _roi = np.array(f["roi"])

    color2 = _colorcet_C06(180)
    hemi = "left"
    scale3 = 5

    hs10ad, axes = plt.subplots(1, 4, figsize=(12, 4))

    # --- subplot 1: all MO cells, bias relative to MOp ---
    ax1 = axes[0]
    T1 = pd.read_csv(data_folder / "revision" / "axons" / "Axon_bias_all_cells_MO.csv")
    _plot_outlines(ax1, maskPath, st, atlas1, hemi, scale3)
    ax1.set_axis_off()
    ax1.set_aspect("equal")
    _plot_bias_lines(ax1, T1, color2)
    ax1.scatter(*CENTER, s=36, marker="*", c="k")
    ax1.scatter(*CENTER2, s=36, marker="*", c="k")
    ax1.invert_yaxis()  # MATLAB YDir reverse

    # --- subplot 2: MO2 cells with border line ---
    ax2 = axes[1]
    T1 = pd.read_csv(data_folder / "revision" / "axons" / "Axon_bias_all_cells_MO2.csv")
    _plot_outlines(ax2, maskPath, st, atlas1, hemi, scale3)
    ax2.set_axis_off()
    ax2.set_aspect("equal")
    _plot_bias_lines(ax2, T1, color2)
    ax2.scatter(*CENTER, s=36, marker="*", c="k")
    ax2.plot(
        [POINT1[0], POINT2[0]], [POINT1[1], POINT2[1]], "--k", linewidth=1
    )
    ax2.invert_yaxis()

    # --- subplot 3: bias angle histogram with shuffle ---
    ax4 = axes[2]
    soma_center1 = T1[["soma_center_1", "soma_center_2"]].to_numpy()
    axon_bias = T1[["axon_bias_1", "axon_bias_2"]].to_numpy()
    edges = np.arange(0, 190, 10)
    N1, _ = np.histogram(T1["center_bias_angle"], bins=edges)
    mean_bias = T1["center_bias_angle"].mean()
    orthog_vector = soma_center1 - np.asarray(CENTER, dtype=float)
    n = axon_bias.shape[0]
    N_rp = np.zeros((100, edges.size - 1))
    for k in range(100):
        rp_indx = rng.permutation(n)
        vector_all1 = axon_bias[rp_indx]
        cross_z = np.abs(
            vector_all1[:, 0] * orthog_vector[:, 1]
            - vector_all1[:, 1] * orthog_vector[:, 0]
        )
        dot = vector_all1[:, 0] * orthog_vector[:, 0] + vector_all1[:, 1] * orthog_vector[:, 1]
        ang_diff2 = np.rad2deg(np.arctan2(cross_z, dot))
        N_rp[k], _ = np.histogram(ang_diff2, bins=edges)
    ax4 = _plotAngleStairsShuffle(ax4, edges, N1, N_rp)
    ax4.axvline(mean_bias, color="r")
    ax4.set_xlim(0, 180)
    ax4.set_ylim(0, 15)

    # --- subplot 4: angle histograms on either side of the border ---
    slope = (POINT1[1] - POINT2[1]) / (POINT1[0] - POINT2[0])
    intercept = POINT1[1] - slope * POINT1[0]
    x = T1["soma_center_1"].to_numpy()
    y = T1["soma_center_2"].to_numpy()
    border = y - (x * slope + intercept)
    left_roi = border < 0
    right_roi = border >= 0

    axon_vector = T1[["axon_bias_1", "axon_bias_2"]].to_numpy()
    theta = 90
    vr = (axon_vector[:, 0] + 1j * axon_vector[:, 1]) * np.exp(-1j * theta * np.pi / 180)
    angle1 = np.round(np.rad2deg(np.arctan(vr.imag / vr.real))).astype(int) + 1
    angle1[angle1 > 90] = 90  # one cell came out at 91; round to 90 for colors
    angle2a = angle1[left_roi]
    angle2b = angle1[right_roi]

    ax5 = axes[3]
    counts_a, edges_a = np.histogram(angle2a, bins=10)
    hist2a = ax5.stairs(counts_a, edges_a, fill=False, edgecolor="k")
    # MATLAB BinEdges(Values==max): shorter logical index takes BinEdges(1:10)
    angle2a_max = edges_a[:-1][counts_a == counts_a.max()].mean()
    ax5.axvline(angle2a_max + 5, color="k", linestyle="--")
    counts_b, edges_b = np.histogram(angle2b, bins=10)
    hist2b = ax5.stairs(counts_b, edges_b, fill=False, edgecolor="r")
    angle2b_max = edges_b[:-1][counts_b == counts_b.max()]
    for v in np.atleast_1d(angle2b_max):
        ax5.axvline(v + 5, color="k", linestyle="--")
    ax5.set_xlim(-90, 90)
    ax5.set_ylim(0, 30)
    ax5.set_xticks([-90, -45, 0, 45, 90])
    ax5.set_xticklabels(["-pi/2", "-pi/4", "0", "pi/4", "pi/2"])

    angle_all = np.array([angle2a_max + 6, angle2b_max[0] + 6])
    angle_all2 = angle_all + 90 - 180
    x1 = np.linspace(-90, 90, 180)
    colora3 = np.nan_to_num(_interp_colors(x1, color2, angle_all2))
    hist2a.set_edgecolor(colora3[0])
    hist2b.set_edgecolor(colora3[1])

    # circular statistics (getAngleHistogramStatsMO.m uses the MOp center)
    angle_diff_real, angle_diff_perm, edges, N1, N_rp = _getAngleHistogramStats(
        T1, rng, center=CENTER_MOP
    )
    h, mu, ul, ll = circ_mtest(angle_diff_real, np.deg2rad(90))
    mean_deg = np.rad2deg([mu, ll, ul])
    p, _, _ = watsons_U2_perm_test(angle_diff_real, angle_diff_perm[0], 100, rng=rng)
    print(f"circ_mtest vs 90 deg: h={h}, mean={mean_deg[0]:.2f} deg "
          f"(CI {mean_deg[1]:.2f}-{mean_deg[2]:.2f})")
    print(f"Watson's U2 permutation test (real vs shuffle): p={p:.4f}")

    hs10ad.savefig(save_folder / "FigS10ad_circularbias_MO2.pdf", bbox_inches="tight")
    return hs10ad
