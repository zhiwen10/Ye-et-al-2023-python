"""Translated from spirals/plots/plotExampleDataVsFft.m (Extended Data Fig.3ab).

Example epoch (ZYE_0012) of 2-8 Hz band-passed data vs the 3-D-FFT
phase-scrambled control: dF/F and phase frames warped to the atlas.
"""

from pathlib import Path

import colorcet as cc
import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s3 import _session_strings, subplottight
from spirals_py.spirals.plots.plotSpiralTimeSeries3d import (
    _get_cortex_atlas_path,
    _imwarp,
    _load_atlas_data,
    _load_tform,
)
from spirals_py.spirals.preprocessing.spiralPhaseMap_fftn_freq import (
    spiralPhaseMap_fftn_freq,
)
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.atlas import plotOutline


def plotExampleDataVsFft(T, data_folder, save_folder):
    """Translated from spirals/plots/plotExampleDataVsFft.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1, coords, projectedAtlas1 = _load_atlas_data(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    kk = 6  # MATLAB kk = 7 (ZYE_0012)
    mn, td, tdb, en = _session_strings(T, kk)
    fname = f"{mn}_{tdb}_{en}"

    tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
    with h5py.File(data_folder / "spirals" / "fft_roi" / f"{fname}_roi.mat", "r") as f:
        roi_ap = np.asarray(f["roi_ap"]).ravel().astype(int)
        roi_ml = np.asarray(f["roi_ml"]).ravel().astype(int)

    session_root = data_folder / "spirals" / "svd" / fname
    U, V, t, mimg = loadUVt1(session_root)
    dV = np.hstack([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)])

    U1 = U / mimg[:, :, None]  # MATLAB U./mimg broadcasts over the 3rd dim
    U1 = U1[roi_ap[0] - 1 : roi_ap[1], roi_ml[0] - 1 : roi_ml[1], :50]

    params = {"downscale": 1, "lowpass": 0, "gsmooth": 0}
    rate = 1
    freq = [2, 8]

    tStart, tEnd = 1681, 1683
    frameStart = int(np.argmax(t > tStart)) + 1  # MATLAB 1-based
    frameEnd = int(np.argmax(t > tEnd)) + 1
    frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)
    dV1 = dV[:50, frameTemp - 1]

    trace2d1, traceAmp1, tracePhase1 = spiralPhaseMap_freq(
        U1, dV1, t, params, freq, rate
    )
    trace2d1 = trace2d1[:, :, 35:-35]
    tracePhase1 = tracePhase1[:, :, 35:-35]

    trace2d1_fft, traceAmp1_fft, tracePhase1_fft = spiralPhaseMap_fftn_freq(
        U1, dV1, t, params, freq, rate
    )
    trace2d1_fft = trace2d1_fft[:, :, 35:-35]
    tracePhase1_fft = tracePhase1_fft[:, :, 35:-35]

    n_frames = trace2d1.shape[2]
    trace2 = np.zeros((512, 512, n_frames))
    trace2[149:512, :, :] = trace2d1
    trace2_fft = np.zeros((512, 512, n_frames))
    trace2_fft[149:512, :, :] = trace2d1_fft
    tracePhase2 = np.zeros((512, 512, n_frames))
    tracePhase2[149:512, :, :] = tracePhase1
    tracePhase2_fft = np.zeros((512, 512, n_frames))
    tracePhase2_fft[149:512, :, :] = tracePhase1_fft

    out_shape = projectedAtlas1.shape
    trace3 = _imwarp(trace2, tform, out_shape)
    trace3_fft = _imwarp(trace2_fft, tform, out_shape)
    tracePhase3 = _imwarp(tracePhase2, tform, out_shape)
    tracePhase3_fft = _imwarp(tracePhase2_fft, tform, out_shape)

    BW1 = projectedAtlas1.astype(bool)
    mask1 = np.zeros((512, 512))
    mask1[149:512, :] = 1
    mask2 = _imwarp(mask1, tform, out_shape)
    BW = BW1 & (mask2 != 0)

    dff = trace3 * 100
    dff_fft = trace3_fft * 100
    raw_min = dff_fft.min()
    raw_max = dff_fft.max()
    scale3 = 5
    startframe = 5

    def _panel(fig, panel_idx, img, cmap, vmin, vmax):
        ax = subplottight(4, 11, panel_idx, fig)
        im = ax.imshow(
            img, cmap=cmap, vmin=vmin, vmax=vmax, alpha=BW.astype(float)
        )
        for sl in (slice(0, 3), slice(3, 4), slice(4, 5), slice(5, 11), slice(0, 11)):
            plotOutline(maskPath[sl], st, atlas1, [], scale3, ax=ax)
        ax.set_aspect("equal")
        ax.set_axis_off()
        return ax, im

    cmap_c06 = cc.cm["CET_C6"]
    hs3ab = plt.figure(figsize=(9, 9))
    for i in range(1, 11):
        frame = i + startframe - 1
        _panel(hs3ab, i, dff[:, :, frame], "viridis", raw_min, raw_max)
        phase_img = tracePhase3[:, :, frame]
        _panel(hs3ab, i + 11, phase_img, cmap_c06, phase_img.min(), phase_img.max())
        _panel(hs3ab, i + 22, dff_fft[:, :, frame], "viridis", raw_min, raw_max)
        phase_fft_img = tracePhase3_fft[:, :, frame]
        _panel(hs3ab, i + 33, phase_fft_img, cmap_c06, phase_fft_img.min(), phase_fft_img.max())

    frame = 11 + startframe - 1
    ax1_11, im1 = _panel(hs3ab, 11, dff[:, :, frame], "viridis", raw_min, raw_max)
    cbax1 = hs3ab.add_axes([0.972, 0.76, 0.010, 0.22])  # beside panel 11
    hs3ab.colorbar(im1, cax=cbax1)
    ax3_11, im3 = _panel(hs3ab, 33, dff_fft[:, :, frame], "viridis", raw_min, raw_max)
    cbax3 = hs3ab.add_axes([0.972, 0.26, 0.010, 0.22])  # beside panel 33
    hs3ab.colorbar(im3, cax=cbax3)

    hs3ab.savefig(save_folder / "FigS3ab_control_fftn_example.png")
    plt.show()
    return hs3ab
