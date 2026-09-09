"""Translated from spirals/plots/plotSpiralDensityByRadius.m (Extended Data
Fig.6ab): spiral density maps for spirals grouped by detection radius
(top) and the density profile along a fixed cortical line (bottom).

MATLAB evaluates its scatteredInterpolant over the full 1140x1320 grid
and then samples the line points; scipy's LinearNDInterpolator gives
identical values when evaluated directly at those points (NaN outside
the convex hull, where MATLAB linearly extrapolates, is treated as 0).
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from spirals_py.spirals.plots._fig1_helpers_s5 import _line_points
from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.io import load_outline_coords_h5


def plotSpiralDensityByRadius(data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralDensityByRadius.m

    Returns the figure handle hs6ab.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    coords = load_outline_coords_h5(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )  # 10um resolution
    pixSize = 0.01  # mm/pix
    pixArea = pixSize**2

    with h5py.File(
        data_folder / "spirals" / "spirals_density_duration" / "frame_total.mat", "r"
    ) as f:
        frame_total = float(np.asarray(f["frame_total"]).ravel()[0])

    radius = np.arange(40, 101, 10)
    hist_bin = 40
    points = _line_points()
    hs6ab = plt.figure(figsize=(13, 7))
    for k in range(7):
        radius_i = radius[k]
        with h5py.File(
            data_folder / "spirals" / "spirals_density_radius" / f"histogram_{radius_i}radius.mat",
            "r",
        ) as f:
            unique_spirals = np.asarray(f["unique_spirals"]).T  # load histogram counts
        unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)  # counts/mm^2
        unique_spirals_unit = unique_spirals_unit / frame_total * 35  # spirals/(mm^2*s)

        ax1 = hs6ab.add_subplot(2, 7, k + 1)
        # MATLAB scatter without caxis -> colors span the data min..max
        sc = ax1.scatter(
            unique_spirals[:, 0],
            unique_spirals[:, 1],
            s=3,
            c=unique_spirals_unit,
            cmap="hot",
            vmin=unique_spirals_unit.min(),
            vmax=unique_spirals_unit.max(),
        )
        overlayOutlines(coords, 1, ax=ax1)
        ax1.invert_yaxis()  # MATLAB set(gca,'Ydir','reverse')
        ax1.set_aspect("equal")
        ax1.set_xlim(0, 1140)
        ax1.set_ylim(1320, 0)
        ax1.set_xticks(np.arange(0, 1001, 200))
        ax1.set_xticklabels([])
        ax1.set_yticks(np.arange(0, 1201, 200))
        ax1.set_yticklabels([])
        cmax = unique_spirals_unit.max()
        cb1 = hs6ab.colorbar(sc, ax=ax1, ticks=[0, cmax])
        cb1.ax.set_yticklabels(["0", f"{round(cmax * 10) / 10:g}"])
        ax1.set_title(f"{radius_i} mm")

        # draw a line
        ax1.scatter(points[:, 0], points[:, 1], s=5, c="g")

        # interp histogram counts along the line
        F = LinearNDInterpolator(unique_spirals[:, :2], unique_spirals_unit)
        count_sample = F(points[:, 0], points[:, 1])
        count_sample = np.nan_to_num(count_sample, nan=0.0)
        count_sample[count_sample < 0] = 0

        ax2 = hs6ab.add_subplot(2, 7, 7 + k + 1)
        points1 = points - points[0, :]
        points_line = np.linalg.norm(points1, axis=1)  # MATLAB vecnorm(...,2,2)
        ax2.plot(points_line, count_sample)
        ax2.set_xlim(0, 800)
        ax2.set_xticks([0, 200, 400, 600, 800])
        ax2.set_xticklabels(["0", "2", "4", "6", "8"])
        ax2.set_ylabel("sprials/(mm^2*s)")

    hs6ab.savefig(save_folder / "Figs6ab_spirals_density_radius.png", bbox_inches="tight")
    plt.show()
    return hs6ab
