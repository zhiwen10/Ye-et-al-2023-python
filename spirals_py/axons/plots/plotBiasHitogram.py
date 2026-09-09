"""Translated from axons/plots/plotBiasHitogram.m

Plots the axon bias angle histogram with shuffle distribution (Fig. 2d)
and runs circular statistics (circ_mtest, Watson's U2 permutation test).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spirals_py.utils.circular import circ_mtest, watsons_U2_perm_test

CENTER_SSPUN = (244, 542)  # SSp-un
CENTER_MOP = (377, 428)  # MOp (used by getAngleHistogramStatsMO.m)


def _getAngleHistogramStats(T1, rng, center=CENTER_SSPUN):
    """Translated from axons/utils/getAngleHistogramStats.m.

    With center=CENTER_MOP this matches axons/utils/getAngleHistogramStatsMO.m.
    """
    soma_center1 = T1[["soma_center_1", "soma_center_2"]].to_numpy()
    edges = np.arange(0, 190, 10)
    N1, _ = np.histogram(T1["center_bias_angle"], bins=edges)
    orthog_vector = soma_center1 - np.asarray(center, dtype=float)
    angle_diff_real = np.deg2rad(T1["center_bias_angle"].to_numpy())
    axon_bias = T1[["axon_bias_1", "axon_bias_2"]].to_numpy()
    n = axon_bias.shape[0]
    angle_diff_perm = np.zeros((100, n))
    N_rp = np.zeros((100, edges.size - 1))
    for k in range(100):
        rp_indx = rng.permutation(n)
        vector_all1 = axon_bias[rp_indx]
        # atan2(norm(cross(u,v)), dot(u,v)) for 2-D vectors embedded in 3-D
        cross_z = np.abs(
            vector_all1[:, 0] * orthog_vector[:, 1]
            - vector_all1[:, 1] * orthog_vector[:, 0]
        )
        dot = vector_all1[:, 0] * orthog_vector[:, 0] + vector_all1[:, 1] * orthog_vector[:, 1]
        ang_diff1 = np.arctan2(cross_z, dot)
        angle_diff_perm[k] = ang_diff1
        N_rp[k], _ = np.histogram(np.rad2deg(ang_diff1), bins=edges)
    return angle_diff_real, angle_diff_perm, edges, N1, N_rp


def _plotAngleStairsShuffle(ax, edges, N1, N_rp):
    """Translated from axons/utils/plotAngleStairsShuffle.m"""
    mean_N_rp = N_rp.mean(axis=0)
    std_N_rp = N_rp.std(axis=0, ddof=1)  # MATLAB std default normalization
    ax.step(edges[:-1], N1, where="post", color="k", label="data")
    ax.step(edges[:-1], mean_N_rp, where="post", color=(0.5, 0.5, 0.5),
            label="shuffle \u00b1 std")

    uE = mean_N_rp + std_N_rp
    lE = mean_N_rp - std_N_rp
    uE_new = np.empty(2 * uE.size - 1)
    uE_new[0::2] = uE
    uE_new[1::2] = uE[:-1]
    lE_new = np.empty(2 * lE.size - 1)
    lE_new[0::2] = lE
    lE_new[1::2] = lE[:-1]
    xu = edges[:-1]
    xu_new = np.empty(2 * xu.size - 1)
    xu_new[0::2] = xu
    xu_new[1::2] = xu[1:]

    xP = np.concatenate([xu_new, xu_new[::-1]])
    yP = np.concatenate([lE_new, uE_new[::-1]])
    ax.fill(xP, yP, facecolor=(0.5, 0.5, 0.5), edgecolor="none", alpha=0.5)

    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(["0", "1/4*pi", "1/2*pi", "3/4*pi", "pi"])
    ax.set_xlabel("angle bins")
    ax.set_ylabel("bin counts")
    ax.legend()
    return ax


def plotBiasHitogram(data_folder, save_folder, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    T = pd.read_csv(data_folder / "axons" / "Axon_bias_all_cells.csv")

    h2d, ax1 = plt.subplots(figsize=(4, 4))
    soma_center1 = T[["soma_center_1", "soma_center_2"]].to_numpy()
    axon_bias = T[["axon_bias_1", "axon_bias_2"]].to_numpy()
    edges = np.arange(0, 190, 10)
    N1, _ = np.histogram(T["center_bias_angle"], bins=edges)
    mean_bias = T["center_bias_angle"].mean()
    # shuffle
    orthog_vector = soma_center1 - np.asarray(CENTER_SSPUN, dtype=float)
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

    ax1 = _plotAngleStairsShuffle(ax1, edges, N1, N_rp)
    ax1.axvline(mean_bias, color="r")
    ax1.set_xlim(0, 180)

    # circular statistics
    angle_diff_real, angle_diff_perm, edges, N1, N_rp = _getAngleHistogramStats(T, rng)
    h, mu, ul, ll = circ_mtest(angle_diff_real, np.deg2rad(90))
    mean_deg = np.rad2deg([mu, ll, ul])
    p, _, _ = watsons_U2_perm_test(angle_diff_real, angle_diff_perm[0], 100, rng=rng)
    print(f"circ_mtest vs 90 deg: h={h}, mean={mean_deg[0]:.2f} deg "
          f"(CI {mean_deg[1]:.2f}-{mean_deg[2]:.2f})")
    print(f"Watson's U2 permutation test (real vs shuffle): p={p:.4f}")

    h2d.savefig(save_folder / "Fig2d_axon_bias_histogram.pdf", bbox_inches="tight")
    return h2d
