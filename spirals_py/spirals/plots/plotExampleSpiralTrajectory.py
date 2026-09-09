"""Translated from spirals/plots/plotExampleSpiralTrajectory.m (Extended
Data Fig.7a): example grouped spiral sequences (>= 7 consecutive frames)
of session table row 7 (MATLAB kk = 7 -> 0-based row 6), colored by frame
within each sequence.

The MATLAB original also loads horizontal_cortex_atlas_50um.mat and
builds several atlas masks from the structure tree that the figure
itself does not use; those steps are omitted here.
flipud(cbrewer2('seq',...)) is approximated with the equivalent reversed
matplotlib ColorBrewer sequential map (same adaptation as
plotSpiralSyncIndex).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s5 import (
    _load_spirals_grouping,
    _seq_colormap,
    _session_info,
    _transformPointsForward,
)
from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_tform
from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.io import load_outline_coords_h5


def plotExampleSpiralTrajectory(T, data_folder, save_folder):
    """Translated from spirals/plots/plotExampleSpiralTrajectory.m

    Returns the figure handle hs7a.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # atlas brain horizontal projection and outline
    coords = load_outline_coords_h5(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )

    kk = 6  # MATLAB kk = 7
    mn, tdb, en, fname = _session_info(T, kk)

    tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
    cells, durations = _load_spirals_grouping(
        data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat",
        min_duration=7,  # spiral sequences with >= 7 consecutive frames
    )

    colorn = int(durations.max() + 5)
    color_cw = _seq_colormap("Blues", colorn)  # flipud(cbrewer2('seq','Blues',colorn))
    color_ccw = _seq_colormap("OrRd", colorn)
    scale = 1

    hs7a = plt.figure()
    ax = hs7a.add_subplot(1, 1, 1)
    overlayOutlines(coords, scale, color=[0.8, 0.8, 0.8], ax=ax)
    ax.invert_yaxis()  # MATLAB set(gca,'YDir','reverse')
    for pwAllEpoch1 in cells:
        u, v = _transformPointsForward(tform, pwAllEpoch1[:, 0], pwAllEpoch1[:, 1])
        if pwAllEpoch1[0, 3] == 0:
            cc = color_cw
        else:
            cc = color_ccw
        for k in range(u.size - 1):
            ax.plot([u[k], u[k + 1]], [v[k], v[k + 1]], color=cc[k + 5 - 1], linewidth=1.5)
        ax.scatter([u[0]], [v[0]], s=18, facecolor=cc[6 - 1], edgecolor="none")
        ax.set_aspect("equal")
        ax.set_axis_off()

    hs7a.savefig(save_folder / "Figs7a_spirals_trajectory.png", bbox_inches="tight")
    plt.show()
    return hs7a
