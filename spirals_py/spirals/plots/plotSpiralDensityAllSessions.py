"""Translated from spirals/plots/plotSpiralDensityAllSessions.m

Plots the spiral density map combined over all sessions
(spirals/(mm^2*s) per 40x40 pixel bin of the 10 um template).
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.io import load_outline_coords_h5


def plotSpiralDensityAllSessions(data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralDensityAllSessions.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    coords = load_outline_coords_h5(outline_file)

    pixSize = 0.01  # mm/pix after registration
    pixArea = pixSize**2
    hist_file = data_folder / "spirals" / "spirals_density" / "histogram_40pixels.mat"
    with h5py.File(hist_file, "r") as f:
        unique_spirals = np.asarray(f["unique_spirals"]).T  # (N, 3) MATLAB orientation
        frame_total = float(np.asarray(f["frame_total"]).ravel()[0])

    hist_bin = 40
    unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)  # counts/mm^2
    unique_spirals_unit = unique_spirals_unit / frame_total * 35  # spirals/(mm^2*s)
    unique_spirals[:, 2] = unique_spirals_unit

    h1e = plt.figure(figsize=(5, 6))
    ax1 = h1e.add_subplot(1, 1, 1)
    cmax = unique_spirals_unit.max()
    sc = ax1.scatter(
        unique_spirals[:, 0],
        unique_spirals[:, 1],
        s=3,
        c=unique_spirals[:, 2],
        cmap="hot",
        vmin=0,
        vmax=cmax,
        rasterized=True,  # keeps the PDF at a reasonable size (adaptation)
    )
    scale2 = 1
    overlayOutlines(coords, scale2, ax=ax1)
    ax1.invert_yaxis()  # MATLAB set(gca,'Ydir','reverse')
    ax1.set_aspect("equal")
    ax1.set_xlim(0, 1140)
    ax1.set_ylim(1320, 0)
    ax1.set_xticks(np.arange(0, 1001, 200))
    ax1.set_yticks(np.arange(0, 1201, 200))
    cb1 = h1e.colorbar(sc, ax=ax1)
    cb1.set_ticks([0, cmax])
    cb1.set_ticklabels(["0", str(round(cmax * 10) / 10)])

    h1e.tight_layout()
    h1e.savefig(save_folder / "Fig1e_spiral_histogram.pdf")
    plt.show()
    return h1e
