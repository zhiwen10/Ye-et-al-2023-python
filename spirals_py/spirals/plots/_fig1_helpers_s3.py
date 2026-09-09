"""Shared private helpers for the Extended Data Fig.3 / Fig.4 / Fig.9 modules.

Translations of MATLAB utilities used by plotExampleDataVsFft,
plotMapDataVsFftn, plotScatterDataVsFftn, plotLFPspirals and
plotSpeedForRadius: subplottight, scatter_kde, get_ssp_index, lfpPhasemap,
makeProbePlot, colorSites (spirals/utils and utils folders of the MATLAB
repo).
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import gaussian_kde

from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_projected_atlas


def subplottight(n, m, i, fig=None):
    """Translated from utils/subplottight.m (i is 1-based, column-major over
    an m-column x n-row grid, as in MATLAB ind2sub)."""
    if fig is None:
        fig = plt.gcf()
    c = (i - 1) % m + 1
    r = (i - 1) // m + 1
    return fig.add_axes([(c - 1) / m, 1 - r / n, 1 / m, 1 / n])


def scatter_kde(ax, x, y, marker_size=6):
    """Translated from utils/scatter_kde.m (2-D kernel-density colored
    scatter; scipy gaussian_kde approximates MATLAB ksdensity)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    c = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    return ax.scatter(x, y, s=marker_size, c=c)


def _session_strings(T, kk):
    """Session strings from the spiralSessions3 table (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tda = T["date"].iloc[kk]
    en = T["folder"].iloc[kk]
    td = pd.Timestamp(tda).strftime("%Y-%m-%d")
    tdb = pd.Timestamp(tda).strftime("%Y%m%d")
    en = float(en)
    en_str = str(int(en)) if en.is_integer() else str(en)
    return mn, td, tdb, en_str


def get_ssp_index(data_folder):
    """Translated from spirals/utils/get_ssp_index.m

    Returns an (N, 2) integer array of [col, row] positions of the right
    hemisphere SSp area in the 10 um projected atlas (1320 x 1140).
    """
    projectedAtlas1, _ = _load_projected_atlas(data_folder)
    st = pd.read_csv(
        Path(data_folder) / "tables" / "structure_tree_safe_2017.csv"
    )
    spath = st["structure_id_path"].astype(str)

    # MATLAB st.index = 0-based row number, which is what projectedAtlas1
    # stores (loadStructureTree prepends index = 0:numel-1)
    spath2 = spath.str.startswith("/997/8/567/").to_numpy()
    idFilt = np.flatnonzero(spath2)
    atlas = projectedAtlas1.copy()
    atlas[~np.isin(atlas, idFilt)] = 0

    sensory_area = "/997/8/567/688/695/315/453/322/"  # SSp
    spath3 = spath.str.startswith(sensory_area).to_numpy()
    idFilt1 = np.flatnonzero(spath3)
    atlas[~np.isin(atlas, idFilt1)] = 0

    atlas[:, : atlas.shape[1] // 2] = 0  # hemi = 'right'

    rows, cols = np.nonzero(atlas)
    return np.column_stack([cols, rows])  # MATLAB brain_index = [col, row]


def _ismember_rows_int(xy, index2d):
    """MATLAB ismember(A, B, 'rows') for integral coordinate pairs."""
    key_a = (xy[:, 0].astype(np.int64) << 32) + xy[:, 1].astype(np.int64)
    key_b = (index2d[:, 0].astype(np.int64) << 32) + index2d[:, 1].astype(
        np.int64
    )
    return np.isin(key_a, key_b)


def _load_density_file(path):
    """Load a v7.3 *_density.mat: returns (list of (N, 3) spiral_density
    cells, frame_all vector)."""
    cells = []
    with h5py.File(path, "r") as f:
        frames = np.asarray(f["frame_all"]).ravel()
        refs = f["spiral_density"][()]
        for ref in refs.flat:
            cells.append(np.asarray(f[ref]).T)
    return cells, frames


def lfpPhasemap(ops, epochT, Fs):
    """Translated from spirals/utils/lfpPhasemap.m

    ops: dict-like with fproc1 (path to int16 binary, 384 channels),
    fs, NT, Nbatch. Returns (traceEphys, tracePhase, traceAmp), each
    (nSamples, 384).
    """
    lfp_file = Path(ops["fproc1"])
    fsize = lfp_file.stat().st_size // 384 // 2  # int16 samples per channel
    lfp = np.fromfile(lfp_file, dtype=np.int16)
    lfp = lfp.reshape(fsize, 384).T.astype(float)  # (384, fsize)

    step = (1 / ops["fs"]) * 300
    tmax = (ops["NT"] * ops["Nbatch"] - 1) / ops["fs"]
    t2 = np.arange(int(np.floor(tmax / step)) + 1) * step

    indx1 = int(np.argmin(np.abs(t2 - epochT[0])))
    indx2 = int(np.argmin(np.abs(t2 - epochT[1])))
    ephys_indx2 = np.arange(indx1 - 100, indx2 + 100 + 1)  # 0-based, padded
    mean_trace = lfp[:, ephys_indx2]

    rate = 1 / Fs
    nq = int(np.floor((epochT[1] - epochT[0]) / rate + 1e-9)) + 1
    tq1 = epochT[0] + np.arange(nq) * rate
    mean_trace1 = interp1d(
        t2[ephys_indx2], mean_trace.T, axis=0, kind="linear"
    )(tq1)  # (nq, 384)

    trace_ephys1 = mean_trace1 - mean_trace1.mean(axis=0, keepdims=True)
    b1, a1 = butter(2, np.array([2, 8], dtype=float) / (Fs / 2), btype="bandpass")
    trace_ephys = filtfilt(b1, a1, trace_ephys1, axis=0)
    trace_hilbert = hilbert(trace_ephys, axis=0)
    return trace_ephys, np.angle(trace_hilbert), np.abs(trace_hilbert)


def makeProbePlot(ax, cm, ss):
    """Translated from spirals/utils/FillGridPlot/makeProbePlot.m

    cm: dict with xcoords / ycoords arrays. Returns the list of square patch
    handles (probe sites)."""
    probe_sites = []
    sq = ss * (np.array([[0, 0], [0, 1], [1, 1], [1, 0]]) - 0.5)
    for q in range(len(cm["xcoords"])):
        site = Polygon(
            sq + np.array([cm["xcoords"][q], cm["ycoords"][q]]),
            closed=True,
            facecolor="b",
            edgecolor="none",
        )
        ax.add_patch(site)
        probe_sites.append(site)
    ax.autoscale_view()
    return probe_sites


def colorSites(probeSites, data, colMap, cax):
    """Translated from spirals/utils/FillGridPlot/colorSites.m

    colMap: (n, 3) LUT; data values are clamped to cax and mapped through
    the LUT."""
    data = np.asarray(data, dtype=float).ravel()
    col_vals = np.linspace(cax[0], cax[1], colMap.shape[0])
    data = np.clip(data, cax[0], cax[1])
    colors = np.empty((data.size, 3))
    for c in range(3):
        colors[:, c] = np.interp(data, col_vals, colMap[:, c])
    for n, site in enumerate(probeSites):
        site.set_facecolor(colors[n])
