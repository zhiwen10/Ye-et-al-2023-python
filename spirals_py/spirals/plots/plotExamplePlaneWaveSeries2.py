from pathlib import Path

import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from spirals_py.spirals.plots.plotMotionEnergyAmpX import _get_cortex_atlas_path
from spirals_py.spirals.plots.plotSpiralSyncIndex import (
    _imwarp,
    _load_projected_atlas,
    _load_tform,
    _session_info,
)
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.atlas import plotOutline
from spirals_py.utils.optical_flow import HS_flowfield


def _subplottight(fig, n, m, i):
    """Translated from utils/subplottight.m (i is 1-based, column-major)."""
    c = (i - 1) % m + 1
    r = (i - 1) // m + 1
    return fig.add_axes([(c - 1) / m, 1 - r / n, 1 / m, 1 / n])


def _load_atlas1(data_folder):
    """horizontal_cortex_atlas_50um.mat (v7.3); returns atlas1 in MATLAB
    orientation (264 x 228)."""
    path = Path(data_folder) / "tables" / "horizontal_cortex_atlas_50um.mat"
    with h5py.File(path, "r") as f:
        return f["atlas1"][()].T


def _prepare_flow(data_folder):
    """Shared prep for the Extended Data Fig.8 plane-wave panels (MATLAB
    plotExamplePlaneWaveSeries2.m / plotExamplePlaneWave.m /
    plotBorderPlaneWave.m): 2-8Hz phase maps for epoch 990*35:992*35 of the
    ZYE_0060 session, Horn-Schunck flow of the phase movie in raw space,
    everything warped into the 8x-downscaled atlas space."""
    data_folder = Path(data_folder)
    projectedAtlas1, _ = _load_projected_atlas(data_folder)
    atlas1 = _load_atlas1(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    BW = projectedAtlas1.astype(bool)

    params = {"downscale": 8, "lowpass": 0, "gsmooth": 0}
    rate = 1
    freq = [2, 8]

    T = pd.read_excel(data_folder / "tables" / "spiralSessions3.xlsx")
    kk = 4  # MATLAB kk = 5 (ZYE_0060)
    mn, tdb, en, fname = _session_info(T, kk)
    session_root = data_folder / "spirals" / "svd" / fname
    U, V, t, mimg = loadUVt1(session_root)
    dV = np.column_stack([np.zeros(V.shape[0]), np.diff(V, axis=1)])

    tform = _load_tform(data_folder / "spirals" / "rf_tform_8x" / f"{fname}_tform_8x.mat")
    # flow field from unregistered frames first, then transform, to avoid
    # interpolation problems with circular phase
    ds = params["downscale"]
    U1 = U[::ds, ::ds, :].astype(np.float64)
    mimg1 = mimg[::ds, ::ds].astype(np.float64)
    U1 = U1 / mimg1[:, :, None]

    epoch0 = 990 * 35 - 1  # MATLAB 990*35:992*35, 1-based inclusive
    epoch1 = 992 * 35
    dV_epoch = dV[:50, epoch0:epoch1]
    _, _, tracePhase1 = spiralPhaseMap_freq(U1[:, :, :50], dV_epoch, t, params, freq, rate)
    tracePhase1 = np.transpose(tracePhase1, (2, 0, 1))  # frames first
    vxRaw, vyRaw = HS_flowfield(tracePhase1, False)
    vxRaw1 = np.transpose(vxRaw, (1, 2, 0))
    vyRaw1 = np.transpose(vyRaw, (1, 2, 0))

    BW1 = BW[::ds, ::ds]
    vxRawt = _imwarp(vxRaw1, tform, BW1.shape)
    vyRawt = _imwarp(vyRaw1, tform, BW1.shape)
    # phase map from atlas-transformed U space
    Ut = _imwarp(U1[:, :, :50], tform, BW1.shape)
    _, traceAmp1t, tracePhase1t = spiralPhaseMap_freq(Ut, dV_epoch, t, params, freq, rate)
    tracePhase1t = np.transpose(tracePhase1t, (2, 0, 1))  # frames first

    return {
        "tracePhase1t": tracePhase1t,
        "traceAmp1t": traceAmp1t,
        "vxRawt": vxRawt,
        "vyRawt": vyRawt,
        "BW1": BW1,
        "maskPath": maskPath,
        "st": st,
        "atlas1": atlas1,
    }


def _phase_quiver(ax, phase_frame, vx, vy, BW, skip, zoom_scale, lw=0.5):
    """Phase map with masked, subsampled quiver arrows (MATLAB imagesc +
    quiver(...,'autoScale','off'))."""
    ax.imshow(np.ma.masked_where(~BW, phase_frame), cmap=colorcet.cm["CET_C6"])
    rs = np.arange(0, vx.shape[0], skip)
    cs = np.arange(0, vx.shape[1], skip)
    uu = vx[np.ix_(rs, cs)] * zoom_scale
    vv = vy[np.ix_(rs, cs)] * zoom_scale
    mask = BW[np.ix_(rs, cs)]
    uu[~mask] = np.nan
    vv[~mask] = np.nan
    CC, RR = np.meshgrid(cs, rs)
    ax.quiver(CC, RR, uu, vv, angles="xy", scale_units="xy", scale=1,
              color="k", linewidths=lw)
    ax.set_aspect("equal")
    ax.set_axis_off()


def plotExamplePlaneWaveSeries2(data_folder, save_folder):
    """Translated from revision/plane_wave/plots/plotExamplePlaneWaveSeries2.m

    Example plane-wave series: 10 consecutive phase frames (top), with flow
    field (middle), and zoomed into the rectangle ROI (bottom).
    Returns the figure handle."""
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    d = _prepare_flow(data_folder)
    tracePhase1t = d["tracePhase1t"]
    vxRawt, vyRawt = d["vxRawt"], d["vyRawt"]
    BW1, maskPath, st, atlas1 = d["BW1"], d["maskPath"], d["st"], d["atlas1"]

    frame = 5
    scale3 = 5 / 8
    a, b = 40, 90   # height
    c, wd = 20, 70  # width
    v1 = np.array([[c, a], [wd, a], [wd, b], [c, b]])

    hs8k = plt.figure(figsize=(9, 4))
    for i in range(1, 11):
        fr = frame + i - 1  # 0-based index of MATLAB frame+i
        ax1 = _subplottight(hs8k, 3, 10, i)
        ax1.imshow(np.ma.masked_where(~BW1, tracePhase1t[fr]), cmap=colorcet.cm["CET_C6"])
        ax1.set_aspect("equal")
        ax1.set_axis_off()
        plotOutline(maskPath[0:11], st, atlas1, None, scale3, "k", ax=ax1)

        ax2 = _subplottight(hs8k, 3, 10, 10 + i)
        _phase_quiver(ax2, tracePhase1t[fr], vxRawt[:, :, fr], vyRawt[:, :, fr],
                      BW1, skip=8, zoom_scale=5, lw=0.5)
        plotOutline(maskPath[0:11], st, atlas1, None, scale3, "k", ax=ax2)
        ax2.add_patch(Polygon(v1, closed=True, facecolor="none", edgecolor="k", linewidth=1))

        ax3 = _subplottight(hs8k, 3, 10, 20 + i)
        _phase_quiver(ax3, tracePhase1t[fr], vxRawt[:, :, fr], vyRawt[:, :, fr],
                      BW1, skip=5, zoom_scale=3, lw=1)
        plotOutline(maskPath[0:11], st, atlas1, None, scale3, "k", ax=ax3)
        # MATLAB axis has reverse YDir (imagesc); ylim [a b] shows a at top
        ax3.set_ylim([b, a])
        ax3.set_xlim([c, wd])

    hs8k.savefig(save_folder / "FigS8k_example_plane_wave.pdf", bbox_inches="tight")
    plt.show()
    return hs8k
