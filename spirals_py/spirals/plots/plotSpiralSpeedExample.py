from pathlib import Path

import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.spirals.utils import loadUVt1, wrapAngle


def _matlab_round(x):
    """MATLAB round (half away from zero) for positive values."""
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def plotSpiralSpeedExample(data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralSpeedExample.m

    Illustration of the spiral speed calculation for one example spiral
    (ZYE_0012, frame 58768): circular phase sampling around the spiral
    center on two consecutive frames, angular and linear velocity vs radius.

    Adaptation: MATLAB computes the 2-8Hz phase map over the whole recording
    (5440 x 84592 matrix). Here the bandpass/Hilbert transform is computed on
    a +/-1500 frame window around the target frames; filtfilt edge effects
    decay within ~50 frames, so values at the window center are identical to
    full-recording computation (verified numerically, max diff < 1e-9).
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    mn, tdb, en = "ZYE_0012", "20201016", 5
    fname = f"{mn}_{tdb}_{en}"
    session_root = data_folder / "spirals" / "svd" / fname
    U, V, t, mimg = loadUVt1(session_root)
    dV = np.column_stack([np.zeros(V.shape[0]), np.diff(V, axis=1)])

    with h5py.File(data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat", "r") as f:
        refs = f["archiveCell"][()].flat
        cells = [np.asarray(f[r]).T for r in refs]  # HDF5 (5, n) -> MATLAB (n, 5)
    spiral_duration = np.array([c.shape[0] for c in cells])
    groupedCells = [c for c, d in zip(cells, spiral_duration) if d >= 2]
    filteredSpirals = np.vstack(groupedCells)

    grid_x = [250, 350]
    grid_y = [350, 550]
    index1 = ((filteredSpirals[:, 1] > grid_x[0]) & (filteredSpirals[:, 1] < grid_x[1])
              & (filteredSpirals[:, 0] > grid_y[0]) & (filteredSpirals[:, 0] < grid_y[1])
              & (filteredSpirals[:, 2] > 50))
    filteredSpirals2 = filteredSpirals[index1, :]

    scale = 8
    params = {"lowpass": 0, "gsmooth": 0}
    rate = 1
    freq = [2, 8]
    U1 = U[::scale, ::scale, :50]

    frame = 58768
    W = 1500  # half-window around the target frames (see docstring)
    f0 = frame - 1  # 0-based index of MATLAB `frame`
    dV_win = dV[:50, f0 - W:f0 + W + 2]
    _, _, tracePhase_raw = spiralPhaseMap_freq(U1, dV_win, t, params, freq, rate)
    frame1 = tracePhase_raw[:, :, W]
    frame2 = tracePhase_raw[:, :, W + 1]

    iframe_all = np.flatnonzero(filteredSpirals2[:, 4] == frame)
    iframe = iframe_all[0]
    pixSize = 3.45 / 1000 / 0.6 * 3  # mm / pix
    spiral_radius = filteredSpirals2[iframe, 2] / scale
    center = _matlab_round([filteredSpirals2[iframe, 1] / scale,
                            filteredSpirals2[iframe, 0] / scale])

    angle = np.arange(np.pi / 6, 2 * np.pi + 1e-12, np.pi / 6)  # 12 samples
    radii = np.arange(1, int(spiral_radius) + 1)  # MATLAB 1:1:spiral_radius
    x = np.empty((angle.size, radii.size))
    y = np.empty_like(x)
    phase_circle1 = np.empty_like(x)
    phase_circle2 = np.empty_like(x)
    for j, radius in enumerate(radii):
        x[:, j] = center[1] + radius * np.cos(angle)
        y[:, j] = center[0] + radius * np.sin(angle)
        rr = np.clip(_matlab_round(y[:, j]).astype(int) - 1, 0, frame1.shape[0] - 1)
        cc = np.clip(_matlab_round(x[:, j]).astype(int) - 1, 0, frame1.shape[1] - 1)
        phase_circle1[:, j] = frame1[rr, cc]
        phase_circle2[:, j] = frame2[rr, cc]

    phase_circle1b = np.stack([wrapAngle(phase_circle1[:, j]) for j in range(radii.size)], axis=1)
    phase_circle2b = np.stack([wrapAngle(phase_circle2[:, j]) for j in range(radii.size)], axis=1)
    for pc in (phase_circle1b, phase_circle2b):
        pc[np.abs(pc) < 0.0001] = np.nan
        pc[np.abs(pc - 2 * np.pi) < 0.0001] = np.nan
    angle_offset = phase_circle2b - phase_circle1b
    angle_offset = np.mod(angle_offset, 2 * np.pi)  # wrapTo2Pi
    angle_offset[angle_offset > np.pi] -= np.pi
    mean_angle_offset = np.mean(angle_offset, axis=0)  # NaN-propagating, as in MATLAB
    angular_velocity = angle_offset * 35
    mean_angular_velocity = mean_angle_offset * 35

    radius_all = radii
    radius_all1 = radius_all * pixSize * scale
    kk = radius_all1.size - 1  # 0-based index of MATLAB kk = numel(radius_all1)

    cgreys = plt.get_cmap("Greys").resampled(radii.size + 4)

    h9af = plt.figure(figsize=(11, 8))

    for col, (fr_img, marker, fr_num) in enumerate(
            [(frame1, "^", frame), (frame2, "x", frame + 1)]):
        ax = h9af.add_subplot(2, 4, col + 1)
        ax.imshow(fr_img, cmap=colorcet.cm["CET_C6"])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.scatter(center[1], center[0], 12, c="w")
        ax.scatter(x[:, kk], y[:, kk], 24, marker=marker, c="k")
        for k, angle1 in enumerate(angle):
            x1 = center[1] + (spiral_radius + 2) * np.cos(angle1)
            y1 = center[0] + (spiral_radius + 2) * np.sin(angle1)
            ax.text(x1, y1, str(k + 1), fontsize=10)
        ax.set_title(f"frame{fr_num}")
        if col == 0:
            n1 = round(2 / (scale * pixSize))
            ax.plot([10, 10 + n1], [60, 60], "k", linewidth=2)

    ax3 = h9af.add_subplot(2, 4, 3)
    idx = np.arange(1, angle.size + 1)
    ax3.scatter(idx, phase_circle1b[:, kk], 24, marker="^",
                c=[[0.5, 0.5, 0.5]])
    ax3.scatter(idx, phase_circle2b[:, kk], 24, marker="x", c="k")
    for a_i, b_i, c_i in zip(idx, phase_circle1b[:, kk], phase_circle2b[:, kk]):
        ax3.plot([a_i, a_i], [b_i, c_i], "r")
    ax3.set_ylabel("phase angle wrapped")
    ax3.set_xticks(np.arange(1, 13))
    ax3.set_ylim([-2.5, 6])
    ax3.set_xlabel("circular sampling points")

    ax4 = h9af.add_subplot(2, 4, 4)
    diff1 = phase_circle2b[:, kk] - phase_circle1b[:, kk]
    ax4.scatter(idx, diff1, 12, c="k")
    ax4.set_ylabel("Angular diff")
    ax4.set_xticks(np.arange(1, 13))
    ax4.set_ylim([15 / 35, 40 / 35])
    ax4.set_xlabel("circular sampling points")

    # concentric circles over full frame and zoom
    angle2 = np.arange(0, 2 * np.pi + 1e-12, np.pi / 36)
    x1c = center[1] + np.outer(np.cos(angle2), radii)
    y1c = center[0] + np.outer(np.sin(angle2), radii)

    ax7 = h9af.add_subplot(2, 4, 5)
    ax7.imshow(frame1, cmap=colorcet.cm["CET_C6"])
    ax7.set_aspect("equal")
    ax7.set_axis_off()
    ax7.scatter(center[1], center[0], 12, c="w")
    for j in range(radii.size):
        ax7.plot(x1c[:, j], y1c[:, j], color=cgreys(j + 4), linewidth=0.5)
    v = np.array([[362, 200], [512, 200], [512, 350], [362, 350]]) / scale
    ax7.add_patch(Polygon(v, closed=True, facecolor="none", edgecolor="k", linewidth=1))
    ax7.set_title(f"frame{frame}")

    ax72 = h9af.add_subplot(2, 4, 6)
    ax72.imshow(frame1, cmap=colorcet.cm["CET_C6"])
    ax72.set_aspect("equal")
    ax72.set_axis_off()
    ax72.scatter(center[1], center[0], 12, c="w")
    for j in range(radii.size):
        ax72.plot(x1c[:, j], y1c[:, j], color=cgreys(j + 4), linewidth=0.5)
        ax72.scatter(x[:, j], y[:, j], 24, color=cgreys(j + 4))
    ax72.set_ylim([350 / scale, 200 / scale])  # reverse YDir as in imagesc
    ax72.set_xlim([362 / scale, 512 / scale])
    n2 = round(0.5 / (scale * pixSize))
    ax72.plot([362 / scale + 1, 362 / scale + 1 + n2],
              [350 / scale - 1, 350 / scale - 1], "k", linewidth=2)

    ax8 = h9af.add_subplot(2, 4, 7)
    for j in range(radii.size):
        # MATLAB plot(scalar, vector) broadcasts x; replicate that here
        ax8.plot(np.full(angle.size, radius_all1[j]), angular_velocity[:, j], "-o",
                 color=cgreys(j + 4),
                 markersize=3, markerfacecolor=cgreys(j + 4), markeredgecolor="none")
    ax8.plot(radius_all1, mean_angular_velocity, "-o", color=[1, 0, 0],
             markersize=3, markerfacecolor=[1, 0, 0], markeredgecolor="none")
    ax8.set_xlabel("radius (mm)")
    ax8.set_ylabel("angluar velocity(rad/s)")
    ax8.set_xlim([0, 1.6])
    ax8.set_xticks(np.arange(0, 1.61, 0.2))

    distance_offset = np.empty_like(angle_offset)
    for j in range(radii.size):
        distance_offset[:, j] = (angle_offset[:, j] / (2 * np.pi)
                                 * (2 * np.pi * radius_all[j] * pixSize * scale)) * 35
    ax9 = h9af.add_subplot(2, 4, 8)
    for j in range(radii.size):
        # MATLAB plot(scalar, vector) broadcasts x; replicate that here
        ax9.plot(np.full(angle.size, radius_all1[j]), distance_offset[:, j], "-o",
                 color=cgreys(j + 4),
                 markersize=3, markerfacecolor=cgreys(j + 4), markeredgecolor="none")
    mean_distance_offset = np.mean(distance_offset, axis=0)
    ax9.plot(radius_all1, mean_distance_offset, "-o", color=[1, 0, 0],
             markersize=3, markerfacecolor=[1, 0, 0], markeredgecolor="none")
    ax9.set_xlabel("radius (mm)")
    ax9.set_ylabel("velocity (mm/s)")
    ax9.set_xlim([0, 1.6])
    ax9.set_xticks(np.arange(0, 1.61, 0.2))

    h9af.tight_layout()
    h9af.savefig(save_folder / f"Figs9af_spiral_speed_frame{frame}.pdf", bbox_inches="tight")
    plt.show()
    return h9af
