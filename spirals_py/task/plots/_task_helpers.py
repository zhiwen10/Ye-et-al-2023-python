"""Private helpers shared by the FigS14/FigS15 task plot translations.

Sources (MATLAB):
- utils/loadUVt2.m (h5py variant: the on-disk SVD files are MATLAB v7.3,
  which scipy.io.loadmat cannot read)
- task/utils/tsToT.m, task/utils/correctCounterDiscont.m,
  task/utils/computeVelocity2.m, task/utils/getPhotodiodeTime.m
- dependencies/spikes/analysis/helpers/schmitt.m, schmittTimes.m, myGaussWin.m
- dependencies/cbrewer2/colorbrewer.mat (ColorBrewer values, embedded)
- dependencies/colorcet/cetc6.mat (cyclic C06 map, embedded)
- axons/utils/get_cortex_atlas_path.m (also used by task plots)
- task/preprocessing/taskTrace_upsample.m
- utils/subplottight.m

MATLAB v7.3 .mat files are read with h5py; numeric arrays are transposed
back to MATLAB orientation (HDF5 stores them with reversed axes).
"""

import base64
import zlib
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt, hilbert


# ---------------------------------------------------------------- v7.3 io
def load_h5_var(path, varname):
    """Load a numeric variable from a MATLAB v7.3 file, transposed back to
    MATLAB orientation."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[varname][()]).T


def loadUVt2_h5(expRoot):
    """v7.3/HDF5 variant of utils/loadUVt2.m (see spirals_py.utils.io.loadUVt2)."""
    expRoot = Path(expRoot)
    U = load_h5_var(expRoot / "svdSpatialComponents.mat", "U")
    mimg = load_h5_var(expRoot / "meanImage.mat", "mimg")
    V = load_h5_var(expRoot / "svdTemporalComponents_corr.mat", "V")
    t = load_h5_var(expRoot / "svdTemporalComponents_corr_timestamps.mat", "t")
    t = np.atleast_1d(t.squeeze())

    if t.size > V.shape[1]:
        t = t[: V.shape[1]]
    elif t.size < V.shape[1]:
        V = V[:, : t.size]

    return U, V, t, mimg


def _decode_char(dset):
    return dset[()].tobytes().decode("utf-16-le").rstrip("\x00")


def read_mcos_tables(path):
    """Decode MATLAB v7.3 MCOS `table` objects stored in #subsystem#/MCOS.

    Returns a list of pandas DataFrames in the order the objects appear in
    the MCOS heap (which matches MATLAB's column-major reference order).
    """
    tables = []
    with h5py.File(path, "r") as f:
        mcos = f["#subsystem#/MCOS"][()]
        flat = list(mcos.flat)
        for i, r in enumerate(flat):
            o = f[r]
            if not (isinstance(o, h5py.Dataset) and o.dtype == object and 1 < o.size < 50):
                continue
            items = o[()]
            names = []
            ok = True
            for x in items.flat:
                d = f[x]
                if not (
                    isinstance(d, h5py.Dataset)
                    and d.dtype == np.uint16
                    and d.attrs.get("MATLAB_int_decode", 0) == 2
                ):
                    ok = False
                    break
                names.append(_decode_char(d))
            if not ok or not names:
                continue
            # the table's data cell sits 5 heap entries before its varnames cell
            if i < 5:
                continue
            dc = f[flat[i - 5]]
            if not (
                isinstance(dc, h5py.Dataset)
                and dc.dtype == object
                and dc[()].size == len(names)
            ):
                continue
            cols = []
            for x in dc[()].flat:
                d = f[x]
                if d.dtype == object:
                    cols.append([_decode_char(f[y]) for y in d[()].flat])
                else:
                    cols.append(np.asarray(d[()]).T.reshape(-1))
            tables.append(pd.DataFrame(dict(zip(names, cols))))
    return tables


def read_table_cell_array(path, varname):
    """Read a v7.3 cell array of tables. Returns (empty_mask, tables) where
    empty_mask has MATLAB orientation and tables are the decoded non-empty
    tables in MATLAB column-major cell order."""
    with h5py.File(path, "r") as f:
        cells = f[varname][()]
        empty_mask = np.array(
            [f[r].attrs.get("MATLAB_empty", 0) == 1 for r in cells.flat]
        ).reshape(cells.shape).T
    return empty_mask, read_mcos_tables(path)


# ------------------------------------------------------- registration
def load_tform(path):
    """Read the affine2d transformation matrix T (MATLAB orientation) from a
    rfmap .mat file."""
    with h5py.File(path, "r") as f:
        for k in f["#refs#"]:
            g = f["#refs#"][k]
            if isinstance(g, h5py.Group) and "TransformationMatrix" in g:
                return np.asarray(g["TransformationMatrix"][()]).T
    raise KeyError(f"no affine2d TransformationMatrix found in {path}")


def imwarp(img, T, out_shape, fill=0.0):
    """MATLAB imwarp(img, affine2d(T), 'OutputView', imref2d(out_shape)).

    T is the 3x3 affine2d matrix in MATLAB orientation ([x y 1] * T maps
    input intrinsic coords to output coords; imref2d with unit pixel extent
    makes output world coords equal to output pixel coords).
    """
    A = T.T  # column-vector convention: p_out = A @ p_in
    invA = np.linalg.inv(A)
    # MATLAB pixel centers are 1-based, cv2's are 0-based; fold the shift
    # into the output->input mapping handed to cv2.warpAffine. M maps
    # output pixels to input pixels, so WARP_INVERSE_MAP must be set
    # (otherwise cv2 inverts M again and the warp is applied backwards).
    M = np.zeros((2, 3))
    M[:, :2] = invA[:2, :2]
    M[:, 2] = invA[:2, :2] @ [1.0, 1.0] + invA[:2, 2] - [1.0, 1.0]
    h, w = out_shape
    if img.ndim == 2:
        return cv2.warpAffine(
            np.ascontiguousarray(img.astype(np.float64)),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=fill,
        )
    out = np.zeros((h, w, img.shape[2]))
    for k in range(img.shape[2]):
        out[:, :, k] = cv2.warpAffine(
            np.ascontiguousarray(img[:, :, k].astype(np.float64)),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=fill,
        )
    return out


# ------------------------------------------------------- wheel / photodiode
def tsToT(ts, numSamps):
    """Translated from task/utils/tsToT.m"""
    ts = np.asarray(ts)
    return np.interp(np.arange(numSamps), ts[:, 0], ts[:, 1])


def _schmitt(x, low, high):
    """Translated from dependencies/spikes/analysis/helpers/schmitt.m
    (single-output form: y takes values -1/+1)."""
    x = np.asarray(x, dtype=float)
    c = (x > high).astype(float) - (x < low).astype(float)
    c[1:] = c[1:] * (c[1:] != c[:-1])
    t = np.flatnonzero(c)
    if t.size > 1:
        dup = np.where(c[t[1:]] == c[t[:-1]])[0] + 1
        t = np.delete(t, dup)
    y = np.zeros_like(c)
    if t.size:
        y[t] = 2 * c[t]
        y[t[0]] = c[t[0]]
        y = np.cumsum(y)
    return y


def schmittTimes(t, sig, thresh):
    """Translated from dependencies/spikes/analysis/helpers/schmittTimes.m

    MATLAB allows a logical index shorter than the indexed array (trailing
    elements ignored); numpy raises IndexError, so truncate to the mask size
    (same fix as _task_helpers_s15.schmittTimes)."""
    t = np.asarray(t, dtype=float).ravel()
    sig = np.asarray(sig, dtype=float).ravel()
    schmittSig = _schmitt(sig, thresh[0], thresh[1])
    down = (schmittSig[:-1] == 1) & (schmittSig[1:] == -1)
    up = (schmittSig[:-1] == -1) & (schmittSig[1:] == 1)
    flipsDown = t[: down.size][down]
    flipsUp = t[: up.size][up]
    flipTimes = np.sort(np.concatenate([flipsUp, flipsDown]))
    return flipTimes, flipsUp, flipsDown


def getPhotodiodeTime(session_root, win):
    """Translated from task/utils/getPhotodiodeTime.m (v7.3 files via h5py)."""
    session_root = Path(session_root)
    pd_sig = load_h5_var(session_root / "photodiode_raw.mat", "pd").ravel()
    tlTimes = load_h5_var(session_root / "photodiode_timestamps_Timeline.mat", "tlTimes")
    tt = tsToT(tlTimes, pd_sig.size)

    allPD, flipsUp, flipsDown = schmittTimes(tt, pd_sig, [0.5, 0.8])
    flipsUp = flipsUp[(flipsUp >= win[0]) & (flipsUp <= win[1])]
    flipsUp = flipsUp[:-1]

    dff_allPD = np.diff(allPD)
    allPD1 = allPD.copy()
    indx = dff_allPD < 0.6
    indx = np.concatenate([[True], indx])  # MATLAB: indx = [1; indx]
    allPD1 = allPD1[~indx]
    return allPD1


def correctCounterDiscont(pos):
    """Translated from task/utils/correctCounterDiscont.m"""
    pos = np.asarray(pos, dtype=float).ravel()
    posDiff = np.diff(pos)
    posDiff[posDiff > 2**31] = posDiff[posDiff > 2**31] - 2**32
    posDiff[posDiff < -(2**31)] = posDiff[posDiff < -(2**31)] + 2**32
    posOut = np.cumsum(np.concatenate([[0.0], posDiff]))
    return posOut + pos[0]


def _gausswin(nech, a=2.5):
    """Translated from task/preprocessing/gausswin.m"""
    x = np.linspace(-1, 1, int(nech))
    return np.exp(-0.5 * (a * x) ** 2)


def _myGaussWin(stdev, Fs):
    """Translated from dependencies/spikes/analysis/helpers/myGaussWin.m"""
    stdevSamps = round(stdev * Fs)
    gw = _gausswin(stdevSamps * 6, 3)
    return gw / gw.sum()


def computeVelocity2(pos, smoothSize, Fs):
    """Translated from task/utils/computeVelocity2.m"""
    smoothWin = _myGaussWin(smoothSize, Fs)
    pos = np.asarray(pos, dtype=float).ravel()
    vel = np.concatenate([[0.0], np.convolve(np.diff(pos), smoothWin, mode="same")]) * Fs
    return vel


# ------------------------------------------------------- atlas paths
def get_cortex_atlas_path(data_folder):
    """Translated from axons/utils/get_cortex_atlas_path.m"""
    st = pd.read_csv(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    maskPath = [
        "/997/8/567/688/695/315/500/985/",  # MOp
        "/997/8/567/688/695/315/500/993/",  # MOs
        "/997/8/567/688/695/315/31/",  # ACA
        "/997/8/567/688/695/315/453/378/",  # SS2
        "/997/8/567/688/695/315/453/322/",  # SSp
        "/997/8/567/688/695/315/247/",  # AUD
        "/997/8/567/688/695/315/669/",  # VIS
        "/997/8/567/688/695/315/254/",  # RSP
        "/997/8/567/688/695/315/22",  # VISa
        "/997/8/567/688/695/315/541/",  # TEa
        "/997/8/567/688/695/315/677/",  # VISC
    ]
    return maskPath, st


# ------------------------------------------------------- colormaps
_CBREWER = {
    ("Reds", 5): [
        [254, 229, 217],
        [252, 174, 145],
        [251, 106, 74],
        [222, 45, 38],
        [165, 15, 21],
    ],
    ("Greys", 5): [
        [247, 247, 247],
        [204, 204, 204],
        [150, 150, 150],
        [99, 99, 99],
        [37, 37, 37],
    ],
    ("YlOrRd", 9): [
        [255, 255, 204],
        [255, 237, 160],
        [254, 217, 118],
        [254, 178, 76],
        [253, 141, 60],
        [252, 78, 42],
        [227, 26, 28],
        [189, 0, 38],
        [128, 0, 38],
    ],
}


def cbrewer2_seq(name, n):
    """ColorBrewer sequential palette as an (n, 3) float array in [0, 1].
    Values match dependencies/cbrewer2/colorbrewer.mat."""
    return np.asarray(_CBREWER[(name, n)], dtype=float) / 255.0


_C06_B64 = (
    "eNpl2Hk8VPv/B/CUKBRSUlSya6FSQvRqk7SgkFSiSCVXUZYsFSpZQkllK0slkhQR7ZYxsmTNvs6YGcY4okLkfod7f879PZq/5vF8"
    "zOOc+XzOe/scl9MRQoUDvVi8ftQh53gxTj8ILhQyyYG5w6l1cr29cI01uHBetgT6MZcVcq9lYfGNrb9VenpBC14/vFKrFFUf5MKV"
    "GRmoKB3blM3phY+GsOKqmDLYHHZ0NHF4iahnb+mX+nsRUSvqZy9dDqe2GqcTCmkI01ger/CrF2JeGz9fF6zA4r1xOdlDT2GZGZO5"
    "nZeAiXpK0gqBSsR9vxo6MJaIlYrcbyIErFiiCUzxKtz+EeXfteUhsiknXl5aTCDj8+5Tv1dWY+qKfrZfXSyeW6ZJZq4kMGPal3d+"
    "u2pw9K+gmL450Zjp19LRrkNgDvffWNh/xXSDTO8NRncxqH2p4YkBAfkXD5u8gmvBW3i8qXpTGEoZjhIllgTeKLm3WafVIbFUY9be"
    "qyHYquPFO3aWe98NDyWkK+rRaEgZWuXhD+vtbh4iPgRUPyUJyxEN8P34OU430xdGoxLnam8RGPj8IzNCoAkrM85FTm/wQqaooYta"
    "AoG++enyT2Wa8TVF7b2qijN+1PPUzksn4O1m8cFqfQvc14vP5TtuB+USj+hTuQRe9hdbBe1oheZWx8Ko0f1Y4WvnK1ZB4G6osLv6"
    "8jZM+fczOlYaKNZKQGBg8YnINNKv7CqMPtFDoP9CgXHI2nbSheUP9g8ROB29R29KFun/fPowGHKVJaLR8YffrODRr8v600VHz8hv"
    "UKf96ZpeOpbpf3q1DaVxpyr9D98oXR6RkfSnL5riaPZcpvPP378wCCci/vQoasl0FWHGH94mklzd4/2nfy6QOFD47U+PGTCW4j/C"
    "/MOpgl19IhTSk36Fl/kxCcjWy1YLKbMmPftRTejrQgK6O9MvSF0jfSP/HVmbWAIhHVLO55tIt4v/orfAkcC1JdGGi5Z3TXr491UZ"
    "WpoEOLUxvdqOpLcNnjkSPdSLm+V3zimnkj71uHFqX2ovhhwGKPrtpLfmh0bvPtyLlRXmtqIC3ZM+UBTQ3j2lFzNuuc4+rkT6g+fm"
    "KpIxHFzYVH97uTbpq5WqTjqt5sDM0mLeE13S5csCnvi/78GqOQ1OKv9x9jodJSXdHtxhfmo230B6pOzyD0IUNjov9bsS/7mvheF5"
    "i6HNbMRb84e/nk36Sj7hqvfZ3bh1vpQexSHXpe17bWymSjeIA/ZhARTSz8VbXL39oAtJDQ4+ZyL/s29vNb7VzO6CVOTLKmE70quu"
    "cBStPVlwMjRzXaFO+r7jEq6+LCbcLMMFw/8mn5e43Z6BYGMmFFXX8HygkK7Tz8+tSAyUnZ22EUGk71AT33NCkYGNcp7EGSPSNVa6"
    "VZiFdkLq6F7RRDHSVXPndlYO0TFSmn9+6Vcy3s6WXa9ssaKj3OZv7Vn3SE/kzZc7TKVBzTBVOdSc9F0pQds3qtIw9FBs5uOFpFP5"
    "Kgseh3cgXcgxJbCRAW6x9rwtOh+hdb2LY0fa0aIxXokY+Mt0Tj2/+X6YmG0sHrBqh2LT8oH9Fgzs+GRMDXe3Rev8u8kZBW1Qvxkh"
    "eHgxA61tgmFJPo6QUlj4sE25DXw1fIrxrZ3gNbsw9WiiG4wtPhYLSbbiltwMs01xnUjpuFEie+kysn/3cyOxGfYKrfbnrDsxdfvd"
    "9X6nr4JbPJfMSmzCtqH5m60VOuEdvjXQnhmI+f0zHB7aN4IlnjPHu5uOBdws3vroJhpumYo6rGnArwj3Zpk0Ov6J1zvQvS9noTZc"
    "hx1jVm3ZrnRk9b01sS+JgtzRVgOZ3Frc0ZlpX7aZex3nZ+8MTeMQPWSn5xj8Fdb9fPeGZ9PhOLFhj3BYi6FtbVkDW5Eldd9aabhq"
    "2c3co5uMrosGmy+qV2N8lQde0ZCyX2F0H55D7GLSg/XzqmCkqRA4EkKDxs+e7N17XmJ5s8WJrWMVkOobPnrlLA3ZLerGSxe/wjYJ"
    "3PAYKMfZVwaHg81oaHFWpz9Y8BpjMZwKyd4v+CnokpepS8PZGaZ2Pze9wZjnoTST+WWIrylaELCBBu8X772LbryHHv3N7RybElTt"
    "OSZyQJOGj0tNkuqnfUKS6PjOfEb4MT2aEWi4bP3lfnF1LuadsVk87FsEx5KXit57aNBbtMhM9k0+6sbbugUVlfRjQinWNPgZPbIV"
    "zqDANFbc1+9UIZ5o0W1fe9MwkZ4fqGAXGBlkPKWg1/1UwJREGmh/Bz8x4t6vdtVWmr80BYeOcFdaRcO51/s96gRLYW73rvVkYQHE"
    "hJWinKbTES3+lrV8xxcs9PtlURdZgOlrRdW1deiYP7Gwcpyx/Z7Qfpvr80w/VF6ggzlfnqX6vQL3X6jWu6UVIEdUfJ5jDh13990N"
    "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"
)


def colorcet_c06():
    """colorcet('C06') cyclic colormap, embedded from
    dependencies/colorcet/cetc6.mat."""
    raw = zlib.decompress(base64.b64decode(_C06_B64))
    arr = np.frombuffer(raw, dtype=np.float64).reshape(256, 3)
    return LinearSegmentedColormap.from_list("colorcet_C06", arr)


# ------------------------------------------------------- trace upsampling
def taskTrace_upsample(trace_mean_correct, freq, rate):
    """Translated from task/preprocessing/taskTrace_upsample.m"""
    x, y = trace_mean_correct.shape[0], trace_mean_correct.shape[1]
    meanTrace = trace_mean_correct.reshape(x * y, trace_mean_correct.shape[2])

    tsize = meanTrace.shape[1]
    Fs = 35 / rate
    tq = np.arange(1, tsize + 1e-9, rate)  # MATLAB 1:rate:tsize
    xp = np.arange(1, tsize + 1)
    meanTrace = np.stack([np.interp(tq, xp, row) for row in meanTrace], axis=0)

    # take care of nan pixels after registration before filtering
    indx1 = ~np.isnan(meanTrace[:, 0])
    trace_mean4 = meanTrace[indx1, :]
    meanTrace2 = trace_mean4 - trace_mean4.mean(axis=1, keepdims=True)
    f1, f2 = butter(2, np.asarray(freq, dtype=float) / (Fs / 2), "bandpass")
    traceFilt = filtfilt(f1, f2, meanTrace2, axis=1)
    traceHilbert = hilbert(traceFilt, axis=1)
    tracePhase = np.angle(traceHilbert)
    tracePhase2 = np.full(meanTrace.shape, np.nan)
    traceFilt2 = np.full(meanTrace.shape, np.nan)
    tracePhase2[indx1, :] = tracePhase
    traceFilt2[indx1, :] = traceFilt

    meanTrace = meanTrace - meanTrace.mean(axis=1, keepdims=True)
    trace_mean3 = meanTrace.reshape(x, y, meanTrace.shape[1])
    traceFilt3 = traceFilt2.reshape(x, y, traceFilt2.shape[1])
    tracePhase3 = tracePhase2.reshape(x, y, tracePhase2.shape[1])
    return trace_mean3, traceFilt3, tracePhase3


# ------------------------------------------------------- plotting
def subplottight(n, m, i, fig=None):
    """Translated from utils/subplottight.m (i is 1-based, column-major over
    an (m, n) grid, as in MATLAB ind2sub)."""
    if fig is None:
        fig = plt.gcf()
    c = (i - 1) % m + 1
    r = (i - 1) // m + 1
    return fig.add_axes([(c - 1) / m, 1 - r / n, 1 / m, 1 / n])
