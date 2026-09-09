"""Translated from spirals/plots/plotSpiralsBySession2.m (Extended Data
Fig.5a2): spiral density maps registered to the atlas, one panel per
session (spirals/(mm^2*s) per 40x40 pixel bin)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _density_color_plot,
    _index_xy,
    _ismember_rows,
    _load_spirals_grouping,
    _session_info,
    _transformPointsForward,
)
from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_tform
from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.io import load_outline_coords_h5


def plotSpiralsBySession2(T, session_rows, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralsBySession2.m

    T is the session table (pandas DataFrame read from spiralSessions3.xlsx);
    session_rows is a 0-based list of table rows (MATLAB 1:6 -> [0..5]).
    Returns the figure handle hs5a2.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # atlas brain horizontal projection and outline
    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    coords = load_outline_coords_h5(outline_file)
    with h5py.File(outline_file, "r") as f:
        projectedAtlas1 = f["projectedAtlas1"][()].T
    BW = projectedAtlas1.astype(bool)  # atlas brain boundary binary mask
    brain_index = _index_xy(BW)  # [x y] positions within the brain boundary

    # set params
    hist_bin = 40  # bin size (pixels) for spiral density estimation
    pixSize = 0.01  # pixel resolution mm/pix
    pixArea = pixSize**2  # 2d-pixel resolution mm^2/pix^2
    scale2 = 1  # no need to scale for atlas outline here

    hs5a2 = plt.figure(figsize=(11, 4.5))
    for count1, kk in enumerate(session_rows):
        # session info
        mn, tdb, en, fname = _session_info(T, kk)

        # read how many total frames
        t = np.load(
            data_folder / "spirals" / "svd" / fname / "svdTemporalComponents_corr.timestamps.npy"
        )
        frame_all = t.size

        # load spiral centers (>40 pixels radius) and the atlas tform
        cells, _ = _load_spirals_grouping(
            data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
            min_duration=2,  # only spiral sequences with >= 2 consecutive frames
        )
        filteredSpirals = np.vstack(cells)
        tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")

        # transform spirals to atlas space and keep those inside the brain boundary
        sx, sy = _transformPointsForward(tform, filteredSpirals[:, 0], filteredSpirals[:, 1])
        filteredSpirals[:, 0] = np.round(sx)
        filteredSpirals[:, 1] = np.round(sy)
        lia = _ismember_rows(filteredSpirals[:, :2], brain_index)
        filteredSpirals = filteredSpirals[lia, :]

        # histogram based density estimate
        unique_spirals = _density_color_plot(filteredSpirals, hist_bin)
        unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)  # counts/mm^2
        unique_spirals_unit = unique_spirals_unit / frame_all * 35  # spirals/(mm^2*s)

        ax1 = hs5a2.add_subplot(1, 6, count1 + 1)
        sc = ax1.scatter(
            unique_spirals[:, 0],
            unique_spirals[:, 1],
            s=1,
            c=unique_spirals_unit,
            cmap="hot",
            vmin=0,
            vmax=1.0,  # caxis([0,max_c])
        )
        overlayOutlines(coords, scale2, ax=ax1)
        ax1.invert_yaxis()  # MATLAB set(gca,'Ydir','reverse')
        ax1.set_aspect("equal")
        ax1.set_axis_off()
        ax1.set_xlim(0, 1140)
        ax1.set_ylim(1320, 0)
        cb0 = hs5a2.colorbar(sc, ax=ax1, ticks=[0, 1.0])
        cb0.ax.set_yticklabels(["0", "1"])

    fig_name = f"FigS5a2_spirals_by_session_{session_rows[0]}_{session_rows[-1]}"
    hs5a2.savefig(save_folder / f"{fig_name}.png", bbox_inches="tight")
    plt.show()
    return hs5a2
