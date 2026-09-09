"""Translated from spirals/plots/plotSpiralDensitySessionsMeanSEM.m
(Extended Data Fig.7cd): the all-sessions spiral density map (left) and
the mean +/- SEM density profile across the 15 sessions along a fixed
cortical line (right)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

from spirals_py.spirals.plots._fig1_helpers_s5 import _line_points
from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.io import load_outline_coords_h5
from spirals_py.utils.plotting import shadedErrorBar


def plotSpiralDensitySessionsMeanSEM(T, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralDensitySessionsMeanSEM.m

    Returns the figure handle hs7cd.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    coords = load_outline_coords_h5(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )  # 10um resolution

    with h5py.File(
        data_folder / "spirals" / "spirals_density" / "spiralDensityLinePerSession.mat", "r"
    ) as f:
        count_sample = np.asarray(f["count_sample"]).T  # MATLAB (416, 15)

    pixSize = 0.01  # mm/pix after registration
    pixArea = pixSize**2
    with h5py.File(data_folder / "spirals" / "spirals_density" / "histogram_40pixels.mat", "r") as f:
        unique_spirals = np.asarray(f["unique_spirals"]).T
        frame_total = float(np.asarray(f["frame_total"]).ravel()[0])

    count_sample[count_sample < 0] = 0
    max_density = count_sample.max(axis=0)
    mean_g7 = max_density[0:11].mean()
    std_g7 = max_density[0:11].std(ddof=1)
    mean_g8 = max_density[11:15].mean()
    std_g8 = max_density[11:15].std(ddof=1)
    _, p = ttest_ind(max_density[0:11], max_density[11:15])
    print(
        f"ttest2 peak density g7 vs g8: p = {p:.4g} "
        f"(g7 {mean_g7:.3f}+/-{std_g7:.3f}, g8 {mean_g8:.3f}+/-{std_g8:.3f})"
    )

    session_total = 15
    mean_spirals = count_sample.mean(axis=1)
    std_spirals = count_sample.std(axis=1, ddof=1) / np.sqrt(session_total)

    hs7cd = plt.figure(figsize=(5.6, 4.2))
    hist_bin = 40
    unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)  # counts/mm^2
    unique_spirals_unit = unique_spirals_unit / frame_total * 35  # spirals/(mm^2*s)

    ax1 = hs7cd.add_subplot(1, 2, 1)
    cmax = unique_spirals_unit.max()
    sc = ax1.scatter(
        unique_spirals[:, 0],
        unique_spirals[:, 1],
        s=3,
        c=unique_spirals_unit,
        cmap="hot",
        vmin=0,
        vmax=cmax,  # caxis([0,cmax])
    )
    overlayOutlines(coords, 1, ax=ax1)
    ax1.invert_yaxis()  # MATLAB set(gca,'Ydir','reverse')
    ax1.set_aspect("equal")
    ax1.set_xlim(0, 1140)
    ax1.set_ylim(1320, 0)
    xtick = np.arange(0, 1001, 200)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels([str(v) for v in xtick])
    ytick = np.arange(0, 1201, 200)
    ax1.set_yticks(ytick)
    ax1.set_yticklabels([str(v) for v in ytick])
    ax1.tick_params(labelsize=8)
    cb1 = hs7cd.colorbar(sc, ax=ax1, ticks=[0, cmax])
    cb1.ax.set_yticklabels(["0", f"{round(cmax * 10) / 10:g}"])

    # draw a line
    points = _line_points()
    ax1.scatter(points[:, 0], points[:, 1], s=8, c="g")

    ax2 = hs7cd.add_subplot(1, 2, 2)
    points1 = points - points[0, :]
    points_line = np.linalg.norm(points1, axis=1)  # MATLAB vecnorm(...,2,2)
    shadedErrorBar(points_line, mean_spirals, std_spirals, lineProps="-g", ax=ax2)
    ax2.set_xlim(0, 800)
    ax2.set_ylim(0, 2)
    ax2.set_yticks([0, 0.5, 1, 1.5, 2])
    ax2.set_yticklabels(["0", "0.5", "1", "1.5", "2"])
    ax2.set_xticks([0, 200, 400, 600, 800])
    ax2.set_xticklabels(["0", "2", "4", "6", "8"])
    ax2.set_ylabel("sprials/(mm^2*s)")

    hs7cd.savefig(save_folder / "Figs7cd_spirals_density_sessions.png", bbox_inches="tight")
    plt.show()
    return hs7cd
