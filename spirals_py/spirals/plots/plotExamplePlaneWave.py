from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from spirals_py.spirals.plots.plotExamplePlaneWaveSeries2 import (
    _phase_quiver,
    _prepare_flow,
)
from spirals_py.utils.atlas import plotOutline


def plotExamplePlaneWave(data_folder, save_folder):
    """Translated from revision/plane_wave/plots/plotExamplePlaneWave.m

    Example plane-wave frame with flow field, masked to SSp left, SSp right,
    MO right and SSp right (as in the MATLAB source) with area outlines.
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_folder / "revision" / "plane_wave" / "area_mask.mat", "r") as f:
        BW_SSp_left = f["BW_SSp_left"][()].T.astype(bool)
        BW_SSp_right = f["BW_SSp_right"][()].T.astype(bool)
        BW_MO_left = f["BW_MO_left"][()].T.astype(bool)
        BW_MO_right = f["BW_MO_right"][()].T.astype(bool)

    d = _prepare_flow(data_folder)
    tracePhase1t = d["tracePhase1t"]
    vxRawt, vyRawt = d["vxRawt"], d["vyRawt"]
    maskPath, st, atlas1 = d["maskPath"], d["st"], d["atlas1"]

    frame = 11
    scale3 = 5 / 8
    # MATLAB: cat(3, BW_SSp_left, BW_SSp_right, BW_MO_right, BW_SSp_right)
    BW_all = [BW_SSp_left, BW_SSp_right, BW_MO_right, BW_SSp_right]
    hemi_all = ["left", "right", "right", "right"]

    hs8n = plt.figure(figsize=(9, 7))
    for kk in range(4):
        BW2 = BW_all[kk]
        hemi = hemi_all[kk]
        fr = frame - 1  # 0-based
        ax1 = hs8n.add_subplot(2, 2, kk + 1)
        _phase_quiver(ax1, tracePhase1t[fr], vxRawt[:, :, fr], vyRawt[:, :, fr],
                      BW2, skip=6, zoom_scale=3, lw=1)
        if kk in (0, 1, 3):
            plotOutline([maskPath[3]], st, atlas1, hemi, scale3, "w", ax=ax1)
            plotOutline([maskPath[4]], st, atlas1, hemi, scale3, "w", ax=ax1)
            plotOutline(maskPath[5:11], st, atlas1, hemi, scale3, "w", ax=ax1)
            plotOutline(maskPath[3:11], st, atlas1, hemi, scale3, "k", ax=ax1)
        elif kk == 2:
            plotOutline(maskPath[0:3], st, atlas1, hemi, scale3, "k", ax=ax1)
        ax1.set_aspect("equal")
        ax1.set_axis_off()

    hs8n.savefig(save_folder / "FigS8n_wave_symmetry_example.pdf", bbox_inches="tight")
    plt.show()
    return hs8n
