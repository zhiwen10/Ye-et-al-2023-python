"""Translated from axons/plots/plotAxonOrientation.m

Plots axon orientation bias of all sensory neurons (Fig. 2a-c).
"""

from pathlib import Path

import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Polygon

from spirals_py.utils.atlas import overlayOutlines

CENTER = (244, 542)  # SSp-un


def _colorcet_C06(N=256):
    """colorcet('C06','N',N) via the Python colorcet package (CET_C6)."""
    base = np.array([to_rgb(c) for c in colorcet.palette["CET_C6"]])
    if N >= base.shape[0]:
        return base
    xi = np.arange(N) * (base.shape[0] - 1) / (N - 1)
    return np.stack(
        [np.interp(xi, np.arange(base.shape[0]), base[:, j]) for j in range(3)], axis=1
    )


def _load_outline_coords(path):
    """Load the coords struct array from the v7.3 isocortex outline .mat."""
    coords = []
    with h5py.File(path) as f:
        xs, ys = f["coords/x"], f["coords/y"]
        for i in range(xs.shape[0]):
            coords.append(
                {
                    "x": np.array(f[xs[i, 0]]).ravel(),
                    "y": np.array(f[ys[i, 0]]).ravel(),
                }
            )
    return coords


def _load_example_cells(path):
    """Load example_12_cells.mat (v7.3); cell arrays become lists of (N,3) arrays."""
    with h5py.File(path) as f:

        def cells(key):
            d = f[key]
            return [np.array(f[d[i, 0]]).T for i in range(d.shape[0])]

        return cells("axon_current_all1"), cells("axon_current1"), cells("soma_current_2d1")


def _interp_colors(x1, cmap_arr, v):
    """interp1(x1, cmap_arr, v) per color channel."""
    return np.stack(
        [np.interp(v, x1, cmap_arr[:, j]) for j in range(3)], axis=-1
    )


def plotAxonOrientation(data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    coords = _load_outline_coords(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )
    T1 = pd.read_csv(data_folder / "axons" / "Axon_bias_all_cells.csv")
    axon_current_all1, axon_current1, soma_current_2d1 = _load_example_cells(
        data_folder / "axons" / "example_12_cells.mat"
    )

    scale = 1
    scale1 = 15
    color2 = _colorcet_C06(180)
    center = CENTER

    h2ac, axes = plt.subplots(1, 4, figsize=(12, 4))

    # --- subplot 1: outline with zoom box ---
    ax = axes[0]
    overlayOutlines(coords, scale, (0.8, 0.8, 0.8), ax=ax)
    ax.scatter(*center, s=36, marker="*", c="k")
    ax.plot([200, 400], [1100, 1100], "k")
    ax.text(300, 1150, "2 mm")
    v = np.array([[100, 400], [100, 800], [500, 800], [500, 400]]) / scale
    ax.add_patch(Polygon(v, closed=True, edgecolor="k", facecolor="none", linewidth=1))
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(0, 600)
    ax.set_ylim(1200, 0)  # MATLAB YDir reverse + ylim([0,1200])

    # --- subplot 2: single example cell with SVD axes ---
    ax = axes[1]
    overlayOutlines(coords, scale, (0.8, 0.8, 0.8), ax=ax)
    soma_center1 = None
    for icell in range(1):
        axon_current_all = axon_current_all1[icell]
        axon_current = axon_current1[icell]
        soma_current_2d = soma_current_2d1[icell]

        axon_current_2d = axon_current[:, [2, 0]].astype(float)
        axon_center = axon_current_2d.mean(axis=0)
        axon_current_2d = axon_current_2d - axon_center
        soma_center1 = soma_current_2d[:, [2, 0]].ravel().astype(float)
        sample_n = axon_current_2d.shape[0]
        U, S, _ = np.linalg.svd(axon_current_2d.T, full_matrices=False)
        # MATLAB: round(rad2deg(atan(U(2,1)/U(1,1))))+91 as 1-based index
        angle1 = int(round(np.rad2deg(np.arctan(U[1, 0] / U[0, 0])))) + 90
        angle1 = min(max(angle1, 0), 179)
        ax.plot(
            axon_current_all[:, 2],
            axon_current_all[:, 0],
            ".",
            color=color2[angle1],
            markersize=1,
        )
        ax.scatter(axon_current[:, 2], axon_current[:, 0], s=1, c=[(0.5, 0.5, 0.5)])
        for j in range(2):
            ax.plot(
                [
                    soma_center1[0] - U[0, j] * S[j] / np.sqrt(sample_n),
                    soma_center1[0] + U[0, j] * S[j] / np.sqrt(sample_n),
                ],
                [
                    soma_center1[1] - U[1, j] * S[j] / np.sqrt(sample_n),
                    soma_center1[1] + U[1, j] * S[j] / np.sqrt(sample_n),
                ],
                color="k",
                linewidth=1,
            )
    ax.scatter(*center, s=36, marker="*", c="k")
    ax.plot(
        [soma_center1[0], center[0]],
        [soma_center1[1], center[1]],
        "--",
        color="k",
    )
    ax.plot([150, 250], [800, 800], "k")
    ax.text(150, 850, "1 mm")
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(100, 500)
    ax.set_ylim(800, 400)

    # --- subplot 3: all 12 example cells ---
    ax = axes[2]
    overlayOutlines(coords, scale, (0.8, 0.8, 0.8), ax=ax)
    ax.scatter(*center, s=36, marker="*", c="k")
    for icell in list(range(0, 8)) + list(range(9, 12)):  # MATLAB [1:8,10:12]
        axon_current_all = axon_current_all1[icell]
        axon_current = axon_current1[icell]
        soma_current_2d = soma_current_2d1[icell]

        axon_current_2d = axon_current[:, [2, 0]].astype(float)
        axon_center = axon_current_2d.mean(axis=0)
        axon_current_2d = axon_current_2d - axon_center
        soma_center1 = soma_current_2d[:, [2, 0]].ravel().astype(float)
        U, S, _ = np.linalg.svd(axon_current_2d.T, full_matrices=False)
        angle1 = int(round(np.rad2deg(np.arctan(U[1, 0] / U[0, 0])))) + 90
        angle1 = min(max(angle1, 0), 179)
        polarity3 = S[0] / S[1] - 1
        ax.plot(
            axon_current_all[:, 2],
            axon_current_all[:, 0],
            ".",
            color=color2[angle1],
            markersize=1,
        )
        ax.scatter(axon_current[:, 2], axon_current[:, 0], s=1, c=[(0.5, 0.5, 0.5)])
        ax.plot(
            [
                soma_center1[0] - U[0, 0] * scale1 * polarity3,
                soma_center1[0] + U[0, 0] * scale1 * polarity3,
            ],
            [
                soma_center1[1] - U[1, 0] * scale1 * polarity3,
                soma_center1[1] + U[1, 0] * scale1 * polarity3,
            ],
            color="k",
            linewidth=1,
        )
    ax.plot([200, 400], [1100, 1100], "k")
    ax.text(300, 1150, "2 mm")
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(0, 600)
    ax.set_ylim(1200, 0)

    # --- subplot 4: bias vectors of all cells ---
    ax = axes[3]
    x1 = np.linspace(-90, 90, 180)
    colora3 = _interp_colors(x1, color2, T1["bias_angle"].to_numpy())
    soma_center1 = T1[["soma_center_1", "soma_center_2"]].to_numpy()
    axon_bias = T1[["axon_bias_1", "axon_bias_2"]].to_numpy()
    pc_ratio = T1["pc_ratio"].to_numpy()
    cell_n = len(T1)
    overlayOutlines(coords, scale, (0.8, 0.8, 0.8), ax=ax)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(0, 600)
    ax.set_ylim(1200, 0)
    for i in range(cell_n):
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
    ax.scatter(*center, s=36, marker="*", c="k")
    cmap = ListedColormap(color2[::-1])  # colormap(flipud(color2))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = h2ac.colorbar(sm, ax=ax, fraction=0.046 * 0.5, pad=0.04)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["-pi/2", "0", "pi/2"])

    h2ac.savefig(save_folder / "Fig2ac_axon_bias_all.pdf", bbox_inches="tight")
    return h2ac
