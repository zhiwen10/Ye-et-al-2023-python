"""Translated from spirals/plots/plotSpiralsBySession1.m (Extended Data
Fig.5a1): receptive-field sign maps registered to the atlas, one panel per
session.

The MATLAB original also loads the svd timestamps and the spiral grouping
file (unused by the figure) and warps mimg (unused); those loads are
omitted here.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s5 import _imwarp_nearest, _session_info
from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_tform
from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.colormaps import colormap_RedWhiteBlue
from spirals_py.utils.io import load_outline_coords_h5


def plotSpiralsBySession1(T, session_rows, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralsBySession1.m

    T is the session table (pandas DataFrame read from spiralSessions3.xlsx);
    session_rows is a 0-based list of table rows (MATLAB 1:6 -> [0..5]).
    Returns the figure handle hs5a1.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # atlas brain horizontal projection and outline
    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    coords = load_outline_coords_h5(outline_file)
    with h5py.File(outline_file, "r") as f:
        projectedAtlas1 = f["projectedAtlas1"][()].T
        projectedTemplate1 = f["projectedTemplate1"][()].T
    BW = projectedAtlas1.astype(bool)  # atlas brain boundary binary mask
    scale = 1  # no need to scale for atlas outline here

    hs5a1 = plt.figure(figsize=(11, 4.5))
    for count1, kk in enumerate(session_rows):
        # session info
        mn, tdb, en, fname = _session_info(T, kk)

        # load atlas transformation matrix and receptive field signMap
        tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
        with h5py.File(data_folder / "spirals" / "rf_signmap" / f"{fname}_signMap.mat", "r") as f:
            signMap = np.asarray(f["signMap"]).T
        # transform signMap image to atlas space (imwarp with the atlas
        # OutputView; MATLAB default nearest interpolation)
        signMaptransformed = _imwarp_nearest(signMap, tform, projectedTemplate1.shape)

        # plot registered sign map with brain outline
        ax0 = hs5a1.add_subplot(1, 6, count1 + 1)
        # MATLAB imagesc autoscales with min/max (NaN-ignoring) and leaves
        # NaN pixels undrawn; AlphaData = BW clips to the brain boundary
        im = ax0.imshow(
            np.ma.masked_where(~BW | ~np.isfinite(signMaptransformed), signMaptransformed),
            cmap=colormap_RedWhiteBlue(),
            vmin=np.nanmin(signMaptransformed),
            vmax=np.nanmax(signMaptransformed),
        )
        ax0.set_facecolor("w")
        overlayOutlines(coords, scale, "k", ax=ax0)
        ax0.set_aspect("equal")
        ax0.set_axis_off()
        cb00 = hs5a1.colorbar(im, ax=ax0, ticks=[-1, 0, 1])
        cb00.ax.set_yticklabels(["-1", "0", "1"])
        ax0.set_title(fname)

    fig_name = f"FigS5a1_rfmap_by_session_{session_rows[0]}_{session_rows[-1]}"
    hs5a1.savefig(save_folder / f"{fig_name}.png", bbox_inches="tight")
    plt.show()
    return hs5a1
