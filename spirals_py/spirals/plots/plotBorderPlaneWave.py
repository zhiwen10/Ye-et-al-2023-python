from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots.plotExamplePlaneWaveSeries2 import (
    _phase_quiver,
    _prepare_flow,
)
from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_h5var
from spirals_py.utils.atlas import plotOutline


def plotBorderPlaneWave(data_folder, save_folder):
    """Translated from revision/plane_wave/plots/plotBorderPlaneWave.m

    Left: example plane-wave frame at the SSp/MO border with the polygon ROI.
    Right: polar histogram of mean flow angles across 15 sessions.
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    d = _prepare_flow(data_folder)
    tracePhase1t = d["tracePhase1t"]
    vxRawt, vyRawt = d["vxRawt"], d["vyRawt"]
    BW1, maskPath, st, atlas1 = d["BW1"], d["maskPath"], d["st"], d["atlas1"]

    frame = 11
    scale3 = 5 / 8
    fr = frame - 1  # 0-based

    hs8p = plt.figure(figsize=(9, 7))
    ax1 = hs8p.add_subplot(1, 2, 1)
    vxRaw1a = vxRawt[:, :, fr]
    vyRaw1a = vyRawt[:, :, fr]
    _phase_quiver(ax1, tracePhase1t[fr], vxRaw1a, vyRaw1a, BW1,
                  skip=6, zoom_scale=3, lw=1)
    for hemi in ["left", "right"]:
        plotOutline([maskPath[3]], st, atlas1, hemi, scale3, "k", ax=ax1)
        plotOutline([maskPath[4]], st, atlas1, hemi, scale3, "k", ax=ax1)
        plotOutline(maskPath[5:11], st, atlas1, hemi, scale3, "k", ax=ax1)
        plotOutline(maskPath[3:11], st, atlas1, hemi, scale3, "k", ax=ax1)
        plotOutline(maskPath[0:3], st, atlas1, hemi, scale3, "k", ax=ax1)
    ax1.set_aspect("equal")
    ax1.set_axis_off()

    with h5py.File(data_folder / "revision" / "plane_wave" / "border_roi2.mat", "r") as f:
        bwroi = f["bwroi"][()].T.astype(bool)
        # images.roi.Polygon object: vertex positions live in the #refs# struct
        positions = None
        for name in f["#refs#"]:
            g = f["#refs#"][name]
            if not isinstance(g, h5py.Group) and g.dtype == np.float64 and g.shape == (2, 4):
                positions = g[()].T  # MATLAB (4, 2) [x, y]
    if positions is None:
        raise ValueError("could not locate polygon Position in border_roi2.mat")
    # mean flow angle in the polygon ROI (computed in MATLAB, not plotted)
    vxRaw3_mean = vxRaw1a[bwroi].mean()
    vyRaw3_mean = vyRaw1a[bwroi].mean()
    print(f"ROI mean flow: vx={vxRaw3_mean:.4f}, vy={vyRaw3_mean:.4f}")
    positions = np.vstack([positions, positions[0]])
    ax1.plot(positions[:, 0], positions[:, 1], "r", linewidth=2)

    with h5py.File(data_folder / "revision" / "plane_wave" / "angle_mean_all3.mat", "r") as f:
        vxy_all = _load_h5var(f, "vxy_all")  # MATLAB (15, 10500)
    angle_mean1 = np.angle(vxy_all)
    amp_mean1 = np.abs(vxy_all)
    threshold = 0.6
    edges = np.arange(-np.pi, np.pi + np.pi / 12, np.pi / 12)
    N1 = np.empty((15, edges.size - 1))
    for i in range(15):
        angle_mean_filt = angle_mean1[i, amp_mean1[i] > threshold]
        N, _ = np.histogram(angle_mean_filt, bins=edges)
        N1[i] = N / angle_mean_filt.size
    N_mean = N1.mean(axis=0)
    N_sem = N1.std(axis=0, ddof=1) / np.sqrt(15)

    ax2 = hs8p.add_subplot(1, 2, 2, projection="polar")
    centers = edges[:-1] + np.pi / 24
    width = np.pi / 12
    ax2.bar(centers, N_mean + N_sem, width=width, facecolor=[0.7, 0.7, 0.7], edgecolor="k")
    ax2.bar(centers, N_mean, width=width, facecolor=[0.7, 0.7, 0.7], edgecolor="k")
    ax2.bar(centers, N_mean - N_sem, width=width, facecolor=[0.2, 0.2, 0.2], edgecolor="k")
    ax2.set_theta_direction(-1)  # MATLAB ThetaDir = 'clockwise'

    hs8p.savefig(save_folder / "FigS8p_border_planar_wave2.pdf", bbox_inches="tight")
    plt.show()
    return hs8p
