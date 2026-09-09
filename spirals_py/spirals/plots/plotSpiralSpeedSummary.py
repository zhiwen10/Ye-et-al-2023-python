"""Translated from spirals/plots/plotSpiralSpeedSummary.m

Plots spiral angular and linear speeds (mean +/- STD across spirals) for
spiral radii 40:10:100 pixels, from spirals/spirals_speed/speed_all.mat.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def plotSpiralSpeedSummary(T, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralSpeedSummary.m

    T is accepted for signature compatibility (unused, as in MATLAB).
    speed_all.mat is v7.3: angle_offset/distance_offset are MATLAB (15, 7)
    cell arrays (session x radius); each cell holds a cell array of per-spiral
    numeric arrays. HDF5 stores the outer cells transposed as (7, 15) and the
    inner arrays transposed as well; both are transposed back here.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    speed_file = data_folder / "spirals" / "spirals_speed" / "speed_all.mat"
    angular_all = []
    distance_all = []
    with h5py.File(speed_file, "r") as f:
        ang_refs = f["angle_offset"][()]  # (7, 15) -> MATLAB (15, 7)
        dist_refs = f["distance_offset"][()]
        for count1 in range(7):  # radius index (MATLAB column count1+1)
            # horzcat(angle_offset{:,count1}): per-spiral arrays of all sessions
            angle_vecs = []
            dist_mats = []
            for i in range(ang_refs.shape[1]):  # sessions (MATLAB rows)
                for ref in f[ang_refs[count1, i]][()].flat:
                    angle_vecs.append(np.asarray(f[ref]).T.ravel())  # (1, L)
                for ref in f[dist_refs[count1, i]][()].flat:
                    dist_mats.append(np.asarray(f[ref]).T)  # (M, L)
            angular_velocity_temp = np.vstack(angle_vecs) * 35
            distance_offset_all1 = np.stack(
                [np.nanmean(m, axis=0) for m in dist_mats]  # mean(x,1,'omitnan')
            )

            angular_velocity_all_max = angular_velocity_temp[:, -1]
            angular_all.append(
                [
                    angular_velocity_all_max.mean(),
                    angular_velocity_all_max.std(ddof=1),
                    angular_velocity_all_max.std(ddof=1) / angular_velocity_all_max.size**0.5,
                ]
            )
            linear_velocity_all_max = distance_offset_all1[:, -1]
            distance_all.append(
                [
                    linear_velocity_all_max.mean(),
                    linear_velocity_all_max.std(ddof=1),
                    linear_velocity_all_max.std(ddof=1) / linear_velocity_all_max.size**0.5,
                ]
            )

    angular_all = np.array(angular_all).T  # 3 x 7, like MATLAB angular_all(:,count1)
    distance_all = np.array(distance_all).T

    h1fg = plt.figure(figsize=(5, 3))
    pixSize = 3.45 / 1000 / 0.6 * 3  # mm / pix
    x_value = np.arange(40, 101, 10) * pixSize
    t1 = [0.5, 1.0, 1.5, 2.0]

    ax1 = h1fg.add_subplot(1, 2, 1)
    ax1.errorbar(x_value, angular_all[0], yerr=angular_all[1], color="r", capsize=8)
    ax1.set_xlim(0.5, 2)
    ax1.set_ylim(0, 80)
    ax1.set_xticks(t1, [str(v) for v in t1])
    ax1.set_xlabel("Spiral radius (mm)")
    ax1.set_ylabel("Angular speed (rad/s)")

    ax2 = h1fg.add_subplot(1, 2, 2)
    ax2.errorbar(x_value, distance_all[0], yerr=distance_all[1], color="r", capsize=8)
    ax2.set_xlim(0.5, 2)
    ax2.set_ylim(0, 80)
    ax2.set_xticks(t1, [str(v) for v in t1])
    ax2.set_xlabel("Spiral radius (mm)")
    ax2.set_ylabel("Linear speed (mm/s)")

    h1fg.tight_layout()
    h1fg.savefig(save_folder / "Fig1fg_spiral_speed_all.pdf")
    plt.show()
    return h1fg
