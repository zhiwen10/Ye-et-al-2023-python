"""Translated from spirals/plots/plotSpiralDensityByDuration.m (Extended
Data Fig.6cd): spiral density maps for spirals grouped by duration in
frames (top) and the density profile along a fixed cortical line
(bottom).

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


def plotSpiralDensityByDuration(data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralDensityByDuration.m

    Returns the figure handle hs6cd.
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

    duration = np.arange(1, 8)
    duration_t = np.round(duration / 35 * 1000)  # frame count -> ms
    hist_bin = 40
    points = _line_points()
    hs6cd = plt.figure(figsize=(13, 7))
    for k in range(7):
        duration_i = duration[k]
        with h5py.File(
            data_folder
            / "spirals"
            / "spirals_density_duration"
            / f"histogram_{duration_i}frame.mat",
            "r",
        ) as f:
            unique_spirals = np.asarray(f["unique_spirals"]).T  # load histogram counts
        unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)  # counts/mm^2
        unique_spirals_unit = unique_spirals_unit / frame_total * 35  # spirals/(mm^2*s)

        ax1 = hs6cd.add_subplot(2, 7, k + 1)
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
        cb1 = hs6cd.colorbar(sc, ax=ax1, ticks=[0, cmax])
        cb1.ax.set_yticklabels(["0", f"{round(cmax * 10) / 10:g}"])
        ax1.set_title(f"{duration_t[k]:g}ms")  # MATLAB num2str drops the .0

        # draw a line
        ax1.scatter(points[:, 0], points[:, 1], s=5, c="g")

        # interp histogram counts along the line
        F = LinearNDInterpolator(unique_spirals[:, :2], unique_spirals_unit)
        count_sample = F(points[:, 0], points[:, 1])
        count_sample = np.nan_to_num(count_sample, nan=0.0)
        count_sample[count_sample < 0] = 0

        ax2 = hs6cd.add_subplot(2, 7, 7 + k + 1)
        points1 = points - points[0, :]
        points_line = np.linalg.norm(points1, axis=1)  # MATLAB vecnorm(...,2,2)
        ax2.plot(points_line, count_sample)
        ax2.set_xlim(0, 800)
        ax2.set_xticks([0, 200, 400, 600, 800])
        ax2.set_xticklabels(["0", "2", "4", "6", "8"])
        ax2.set_ylabel("sprials/(mm^2*s)")

    hs6cd.savefig(save_folder / "Figs6cd_spirals_density_duration.png", bbox_inches="tight")
    plt.show()
    return hs6cd
