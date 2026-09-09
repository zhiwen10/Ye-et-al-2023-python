from pathlib import Path

import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq
from spirals_py.spirals.utils import loadUVt1
from spirals_py.utils.circular import circ_mean, circ_r


def _load_projected_atlas(data_folder):
    """Load projectedAtlas1/projectedTemplate1 from
    tables/isocortex_horizontal_projection_outline.mat (v7.3, 10um).
    h5py arrays are transposed back to MATLAB orientation."""
    path = Path(data_folder) / "tables" / "isocortex_horizontal_projection_outline.mat"
    with h5py.File(path, "r") as f:
        projectedAtlas1 = f["projectedAtlas1"][()].T
        projectedTemplate1 = f["projectedTemplate1"][()].T
    return projectedAtlas1, projectedTemplate1


def _load_tform(path):
    """Load an affine2d saved in a v7.3 .mat file; returns the 3x3 MATLAB
    transformation matrix T (used as [x y 1] * T)."""
    with h5py.File(path, "r") as f:
        for name in f["#refs#"]:
            g = f["#refs#"][name]
            if isinstance(g, h5py.Group) and "TransformationMatrix" in g:
                return g["TransformationMatrix"][()].T
    raise ValueError(f"no TransformationMatrix found in {path}")


def _imwarp(A, T, out_shape, step=1):
    """MATLAB imwarp(A, affine2d(T), 'OutputView', imref2d(out_shape)) with
    bilinear interpolation and fill value 0. A may be 2D or 3D (plane-wise).
    step>1 evaluates only output pixels 1:step:end (identical to imwarp
    followed by decimation, at a fraction of the cost)."""
    A = np.asarray(A, dtype=np.float64)
    M, N = out_shape
    rows = np.arange(0, M, step)
    cols = np.arange(0, N, step)
    # world coordinates of output pixel centers (1-based intrinsic coords)
    xw, yw = np.meshgrid(cols + 1.0, rows + 1.0)  # x = col, y = row
    ones = np.ones_like(xw)
    # MATLAB inverse mapping: [xin yin 1] = [xout yout 1] * inv(T)
    Tinv = np.linalg.inv(T)
    xin = xw * Tinv[0, 0] + yw * Tinv[1, 0] + Tinv[2, 0]
    yin = xw * Tinv[0, 1] + yw * Tinv[1, 1] + Tinv[2, 1]
    coords = np.array([yin - 1.0, xin - 1.0])  # 0-based (row, col)
    if A.ndim == 2:
        return map_coordinates(A, coords, order=1, mode="constant", cval=0.0)
    out = np.empty((rows.size, cols.size, A.shape[2]))
    for k in range(A.shape[2]):
        out[:, :, k] = map_coordinates(A[:, :, k], coords, order=1, mode="constant", cval=0.0)
    return out


def _load_h5var(f, name):
    """Read one variable from an open v7.3 .mat file (h5py), transposed back
    to MATLAB orientation; compound real/imag datasets become complex."""
    arr = f[name][()]
    if arr.dtype.names and "real" in arr.dtype.names:
        arr = arr["real"] + 1j * arr["imag"]
    return arr.T


def _session_info(T, kk):
    """Session strings from the spiralSessions3 table (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tdb = pd.Timestamp(T["date"].iloc[kk]).strftime("%Y%m%d")
    en = int(T["folder"].iloc[kk])
    fname = f"{mn}_{tdb}_{en}"
    return mn, tdb, en, fname


def _compute_indices(tracePhase1, traceAmp1, BW_half, anglein):
    """sync/spirality/amp per frame over the selected hemisphere."""
    n_frames = tracePhase1.shape[2]
    pixel_count = int(BW_half.sum())
    sync1 = np.empty(n_frames)
    spirality1 = np.empty(n_frames)
    amp_mean1 = np.empty(n_frames)
    for frame in range(n_frames):
        phase_hemi = tracePhase1[:, :, frame].copy()
        phase_hemi[~BW_half] = np.nan
        sync1[frame] = np.abs(np.nansum(np.exp(1j * phase_hemi))) / pixel_count
        phase_diff = phase_hemi - anglein
        spirality1[frame] = np.abs(np.nansum(np.exp(1j * phase_diff))) / pixel_count
        amp_hemi = traceAmp1[:, :, frame].copy()
        amp_hemi[~BW_half] = np.nan
        amp_mean1[frame] = np.nanmean(amp_hemi)
    return sync1, spirality1, amp_mean1


def _phase_panel(ax, data, BW_half):
    ax.imshow(np.ma.masked_where(~BW_half, data), cmap=colorcet.cm["CET_C6"],
              vmin=-np.pi, vmax=np.pi)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _quiver_panel(ax, angles, label, rng):
    cvariance = circ_r(angles)
    cmean = circ_mean(angles)
    cmean_cV = np.exp(1j * cmean) * cvariance
    x1 = np.real(np.exp(1j * angles))
    y1 = np.imag(np.exp(1j * angles))
    p = rng.permutation(x1.size)[:500]
    x2, y2 = x1[p], y1[p]
    ax.quiver(np.zeros_like(x2), np.zeros_like(y2), x2, y2,
              angles="xy", scale_units="xy", scale=1, color="k")
    ax.quiver([0.0], [0.0], [np.real(cmean_cV)], [np.imag(cmean_cV)],
              angles="xy", scale_units="xy", scale=1, color="r")
    ax.set_aspect("equal")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.text(-1, 0.9, f"{label} = {np.round(cvariance * 10) / 10}")


def plotSpiralSyncIndex(T, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralSyncIndex.m

    Plots example time series (hs8e), example frames with sync/spiral index
    (hs8ad), and the sync-vs-spirality scatter colored by 2-8Hz amplitude
    (hs8f) for session ZYE_0060 (table row 5).
    Returns (hs8e, hs8ad, hs8f) figure handles."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()

    projectedAtlas1, _ = _load_projected_atlas(data_folder)

    kk = 4  # MATLAB kk = 5 (ZYE_0060)
    mn, tdb, en, fname = _session_info(T, kk)
    session_root = data_folder / "spirals" / "svd" / fname
    U, V, t, mimg = loadUVt1(session_root)
    dV = np.column_stack([np.zeros(V.shape[0]), np.diff(V, axis=1)])

    tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")
    with h5py.File(data_folder / "spirals" / "spirals_index" / f"{fname}_motion_energy.mat", "r") as f:
        image_energy2 = f["image_energy2"][()].ravel()

    params = {"downscale": 8, "lowpass": 0, "gsmooth": 0}
    rate = 1
    freq = [2, 8]

    # imwarp to the 10um atlas output view, evaluated directly on the
    # 1:8 decimated grid (same values as imwarp + decimation in MATLAB)
    Utransformed = _imwarp(U, tform, projectedAtlas1.shape, step=params["downscale"])
    mimgtransformed = _imwarp(mimg, tform, projectedAtlas1.shape, step=params["downscale"])
    BW = projectedAtlas1.astype(bool)[:: params["downscale"], :: params["downscale"]]

    # spiral grouping is loaded in MATLAB but not used downstream in this
    # function; reproduce the duration filter for parity
    with h5py.File(data_folder / "spirals" / "spirals_grouping" / f"{fname}_spirals_group_fftn.mat", "r") as f:
        refs = f["archiveCell"][()].flat
        spiral_duration = np.array([f[r].shape[1] for r in refs])  # cells are (5, n) in HDF5
    groupedCells_n = int((spiral_duration >= 15).sum())
    print(f"{fname}: {groupedCells_n} spiral groups with duration >= 15 frames")

    hemi_select = slice(71, 143)  # MATLAB 72:143
    cols = Utransformed.shape[0]
    rows = Utransformed.shape[1]
    xs = np.ceil(np.linspace(-rows / 4, rows / 4, int(np.ceil(rows / 2))))
    ys = np.ceil(np.linspace(-cols / 2, cols / 2, cols))
    Xs, Ys = np.meshgrid(xs, ys)
    anglein = np.arctan2(Ys, Xs)  # expected angle for a centered spiral
    BW_half = BW[:, hemi_select]

    def run_epoch(frameStart, frameEnd):
        # MATLAB 1-based frameStart:frameEnd inclusive
        s0 = frameStart - 36  # 0-based start of frameStart-35
        s1 = frameEnd + 35    # 0-based exclusive end of frameEnd+35
        dV1 = dV[:, s0:s1]
        trace2d1, traceAmp1, tracePhase1 = spiralPhaseMap_freq(
            Utransformed, dV1, t, params, freq, rate)
        trim = 35 // rate
        trace2d1 = trace2d1[:, :, trim:-trim] / mimgtransformed[:, :, None]
        tracePhase1 = tracePhase1[:, :, trim:-trim]
        traceAmp1 = traceAmp1[:, :, trim:-trim] / mimgtransformed[:, :, None]
        phase_hemi = tracePhase1[:, hemi_select, :]
        amp_hemi = traceAmp1[:, hemi_select, :]
        sync1, spirality1, amp_mean1 = _compute_indices(phase_hemi, amp_hemi, BW_half, anglein)
        index_unity1 = np.sqrt(sync1**2 + spirality1**2)
        motion_energy = image_energy2[frameStart - 1:frameEnd]
        return tracePhase1, sync1, spirality1, amp_mean1, index_unity1, motion_energy

    # ---- example time series (hs8e) ----
    frameStart = 72700
    frameEnd = frameStart + 351
    tracePhase1, sync1, spirality1, amp_mean1, index_unity1, motion_energy = run_epoch(frameStart, frameEnd)

    trace1 = Utransformed[99, 99, :50] @ dV[:50, :]  # MATLAB (100,100)
    trace1 = trace1 / mimgtransformed[99, 99]

    frame1, frame2 = 42, 70
    t2 = amp_mean1.shape[0]
    tt = np.arange(1, t2 + 1) / 35
    hs8e, axes = plt.subplots(6, 1, figsize=(8, 10))
    panels = [
        (trace1[frameStart - 1:frameEnd], None),
        (amp_mean1, "3-6Hz amp"),
        (motion_energy, "motion energy"),
        (sync1, "sync index"),
        (spirality1, "spirarity index"),
        (index_unity1, "sum index"),
    ]
    for ax, (y, ylab) in zip(axes, panels):
        ax.plot(tt, y)
        ax.axvline(frame1 / 35, linestyle="--", color="k")
        ax.axvline(frame2 / 35, linestyle="--", color="k")
        if ylab:
            ax.set_ylabel(ylab)
        if ylab in ("sync index", "spirarity index", "sum index"):
            ax.set_ylim([0, 1])
    hs8e.tight_layout()
    hs8e.savefig(save_folder / "Figs8e_example_time_series.pdf", bbox_inches="tight")

    # ---- example frames for index (hs8ad) ----
    hs8ad = plt.figure(figsize=(9, 3.6))
    for row, fr in enumerate([frame1, frame2]):
        f0 = fr - 1  # 0-based frame index
        phasemap1 = tracePhase1[:, hemi_select, f0].copy()
        phasemap1[~BW_half] = np.nan

        _phase_panel(plt.subplot(2, 6, row * 6 + 1), phasemap1, BW_half)
        angle_diff3 = phasemap1[~np.isnan(phasemap1)]
        _quiver_panel(plt.subplot(2, 6, row * 6 + 2), angle_diff3, "syncIndex", rng)

        _phase_panel(plt.subplot(2, 6, row * 6 + 3), phasemap1, BW_half)
        _phase_panel(plt.subplot(2, 6, row * 6 + 4), anglein, BW_half)

        ax3 = plt.subplot(2, 6, row * 6 + 5)
        angle_diff1 = np.angle(np.exp(1j * (phasemap1 - anglein)))  # wrapToPi
        _phase_panel(ax3, angle_diff1, BW_half)

        angle_diff2 = angle_diff1[~np.isnan(angle_diff1)]
        _quiver_panel(plt.subplot(2, 6, row * 6 + 6), angle_diff2, "spiralIndex", rng)
    hs8ad.tight_layout()
    hs8ad.savefig(save_folder / "Figs8ad_example_index.pdf", bbox_inches="tight")

    # ---- sync vs spirality relationship (hs8f) ----
    frameStart = 72700
    frameEnd = frameStart + 3501
    _, sync1, spirality1, amp_mean1, _, _ = run_epoch(frameStart, frameEnd)

    color2 = plt.get_cmap("RdBu_r").resampled(32)  # flipud(cbrewer2('div','RdBu',32))
    hs8f, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(sync1, spirality1, 16, np.log10(amp_mean1), cmap=color2)
    th = np.arange(0, np.pi / 2 + np.pi / 100, np.pi / 100)
    ax.plot(np.cos(th), np.sin(th), "--k")
    ax.set_xlabel("synchrony")
    ax.set_ylabel("spirality")
    ax.set_aspect("equal")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    hs8f.colorbar(sc)
    hs8f.tight_layout()
    hs8f.savefig(save_folder / "Figs8f_sync_spirality_amp.pdf", bbox_inches="tight")

    plt.show()
    return hs8e, hs8ad, hs8f
