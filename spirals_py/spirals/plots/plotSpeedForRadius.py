"""Translated from spirals/plots/plotSpeedForRadius.m (Extended Data Fig.9g-j / k-n).

Spiral angular velocity and linear speed vs sampling radius for all spirals
of a given radius class (speed_50pixels.mat / speed_100pixels.mat), with
density-colored scatter, mean curve and distributions of peak values.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s3 import scatter_kde
from spirals_py.spirals.plots.plotSpiralTimeSeries3d import _matlab_round


def _xticks_mm(ax):
    ax.set_xticks(np.arange(0, 1.61, 0.2))
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6"])


def plotSpeedForRadius(radius, data_folder, save_folder):
    """Translated from spirals/plots/plotSpeedForRadius.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(
        data_folder / "spirals" / "spirals_speed" / f"speed_{radius}pixels.mat", "r"
    ) as f:
        angular_velocity_all = np.asarray(f["angular_velocity_all"]).T
        linear_velocity_all = np.asarray(f["linear_velocity_all"]).T

    spiral_n = angular_velocity_all.shape[0]
    h9gj = plt.figure(figsize=(11, 2.5))
    rng = np.random.default_rng()
    p = rng.choice(spiral_n, 1000, replace=False)  # MATLAB randperm(spiral_n,1000)
    scale1 = 8
    pixSize = 3.45 / 1000 / 0.6 * 3  # mm / pix
    sampling_radius = np.arange(1, angular_velocity_all.shape[1] + 1) * scale1 * pixSize
    sampling_radius1 = np.tile(sampling_radius, (1000, 1))

    ax1 = h9gj.add_subplot(1, 4, 1)
    sc1 = scatter_kde(ax1, sampling_radius1.ravel(),
                      angular_velocity_all[p, :].ravel(), marker_size=6)
    sc1.set_cmap("viridis")  # MATLAB parula
    mean_angle_offset_all = np.nanmean(angular_velocity_all, axis=0)
    ax1.plot(sampling_radius, mean_angle_offset_all, "-o", color=[1, 0, 0],
             markersize=3, markerfacecolor=[1, 0, 0], markeredgecolor="none")
    ax1.set_xlim(0, 1.6)
    ax1.set_ylim(0, 100)
    ax1.set_yticks(np.arange(0, 101, 20))
    _xticks_mm(ax1)
    ax1.set_xlabel("Radius (mm)")
    ax1.set_ylabel("Angular velocity (rad/s)")

    ax2 = h9gj.add_subplot(1, 4, 2)
    edges = np.arange(0, 101, 2)  # MATLAB 0:2:100
    angular_velocity_all_max = angular_velocity_all[:, -1]
    N1, _ = np.histogram(angular_velocity_all_max, bins=edges)
    max_angle = angular_velocity_all_max.mean()
    ax2.step(edges[:-1], N1, where="post")
    ax2.axvline(max_angle, color="r")
    ax2.text(max_angle, 10, f"{_matlab_round(max_angle * 10) / 10:g}")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Angular velocity (rad/s)")
    ax2.set_ylabel("spiral counts")

    ax3 = h9gj.add_subplot(1, 4, 3)
    sc3 = scatter_kde(ax3, sampling_radius1.ravel(),
                      linear_velocity_all[p, :].ravel(), marker_size=6)
    sc3.set_cmap("viridis")
    mean_distance_offset_all = np.nanmean(linear_velocity_all, axis=0)
    ax3.plot(sampling_radius, mean_distance_offset_all, "-o", color=[1, 0, 0],
             markersize=3, markerfacecolor=[1, 0, 0], markeredgecolor="none")
    ax3.set_xlim(0, 1.6)
    ax3.set_ylim(0, 100)
    ax3.set_yticks(np.arange(0, 101, 20))
    _xticks_mm(ax3)
    ax3.set_xlabel("Radius (mm)")
    ax3.set_ylabel("Speed(mm/s)")

    ax4 = h9gj.add_subplot(1, 4, 4)
    linear_velocity_all_max = linear_velocity_all[:, -1]
    max_distance = linear_velocity_all_max.mean()
    edges2 = edges * radius * pixSize  # angular edges converted to linear
    ax4.step(edges2[:-1], N1, where="post")
    ax4.axvline(max_distance, color="r")
    ax4.set_xlim(0, 100)
    ax4.text(max_distance, 10, f"{_matlab_round(max_distance * 10) / 10:g}")
    ax4.set_xlabel("Speed(mm/s)")
    ax4.set_ylabel("spiral counts")

    h9gj.savefig(save_folder / f"Figs9gj_spiralSpeed_{radius}pixels.png")
    plt.show()
    return h9gj
