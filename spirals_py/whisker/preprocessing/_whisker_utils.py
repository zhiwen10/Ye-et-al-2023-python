"""Private helpers shared by the whisker-area plot translations.

Sources (MATLAB):
- spirals/utils/get_cortex_atlas_path.m
- spirals/utils/select_area.m
- utils/subplottight.m
- dependencies/colorcet/colorcet.m ('C06' cyclic map, LUT in cetc6_lut.csv)
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from skimage.transform import resize

_MASK_PATHS = [
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

# SSp subregion paths appended by plotMeanTracePhase7.m (maskPath{12..18})
SSP_SUB_PATHS = [
    "/997/8/567/688/695/315/453/322/329/",  # SSp-bfd
    "/997/8/567/688/695/315/453/322/353/",  # SSp-n
    "/997/8/567/688/695/315/453/322/337/",  # SSp-ll
    "/997/8/567/688/695/315/453/322/345/",  # SSp-m
    "/997/8/567/688/695/315/453/322/369/",  # SSp-ul
    "/997/8/567/688/695/315/453/322/361/",  # SSp-tr
    "/997/8/567/688/695/315/453/322/182305689/",  # SSp-un
]


def _load_atlas_50um(data_folder):
    """atlas1 from tables/horizontal_cortex_atlas_50um.mat (v7.3; transposed
    back to MATLAB [row, col] orientation)."""
    fp = Path(data_folder) / "tables" / "horizontal_cortex_atlas_50um.mat"
    with h5py.File(fp, "r") as f:
        return f["atlas1"][:].T


def _load_outline(data_folder):
    """projectedAtlas1, projectedTemplate1, coords from
    tables/isocortex_horizontal_projection_outline.mat (v7.3, 10um)."""
    fp = Path(data_folder) / "tables" / "isocortex_horizontal_projection_outline.mat"
    with h5py.File(fp, "r") as f:
        projectedAtlas1 = f["projectedAtlas1"][:].T
        projectedTemplate1 = f["projectedTemplate1"][:].T
        cg = f["coords"]
        xrefs = cg["x"][:].ravel()
        yrefs = cg["y"][:].ravel()
        coords = [
            {"x": f[xr][:].ravel(), "y": f[yr][:].ravel()} for xr, yr in zip(xrefs, yrefs)
        ]
    return projectedAtlas1, projectedTemplate1, coords


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m"""
    st = pd.read_csv(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    return list(_MASK_PATHS), st


def _select_area(areaPaths, st, projectedAtlas1, hemi, scale=1):
    """Translated from spirals/utils/select_area.m

    Returns Nx2 array of [col, row] (i.e. [x, y]) pixel pairs, matching the
    MATLAB [row,col] = ind2sub(...); index = [col,row] output. Only the atlas
    selection is kept; the Uselected SVD output is unused by the plots.
    MATLAB st.index is the 0-based structure-tree row number (loadStructureTree).
    """
    spath = st["structure_id_path"].astype(str)
    sel = np.zeros(len(st), dtype=bool)
    for p in areaPaths:
        # MATLAB startsWith(spath, p); an empty p matches everything, as in MATLAB
        sel |= spath.str.startswith(p).to_numpy()
    idFilt1 = np.flatnonzero(sel)
    atlas = projectedAtlas1.copy()
    atlas[~np.isin(atlas, idFilt1)] = 0
    if hemi == "right":
        atlas[:, : atlas.shape[1] // 2] = 0
    elif hemi == "left":
        atlas[:, atlas.shape[1] // 2 :] = 0
    atlas2 = atlas[::scale, ::scale]
    rows, cols = np.nonzero(atlas2)
    return np.column_stack([cols, rows])


def _subplottight(fig, n, m, i):
    """Translated from utils/subplottight.m (i is 1-based)."""
    c = (i - 1) % m
    r = (i - 1) // m
    return fig.add_axes([c / m, 1 - (r + 1) / n, 1 / m, 1 / n])


def _colorcet_c06():
    """colorcet('C06') cyclic colormap; LUT extracted from
    dependencies/colorcet/colorcet.m case {'C6','C06'}."""
    lut = np.loadtxt(Path(__file__).parent / "cetc6_lut.csv", delimiter=",")
    return LinearSegmentedColormap.from_list("colorcet_C06", lut)


def _imresize(img, out_hw):
    """MATLAB imresize(img, [h, w]) on the first two dimensions: bicubic
    (order=3), antialiasing only when shrinking."""
    out_hw = (int(out_hw[0]), int(out_hw[1]))
    trailing = img.shape[2:]
    out = np.empty(out_hw + trailing, dtype=float)
    shrink = (out_hw[0] < img.shape[0]) or (out_hw[1] < img.shape[1])
    indices = np.ndindex(*trailing) if trailing else [()]
    for idx in indices:
        sl = (slice(None), slice(None)) + idx
        out[sl] = resize(
            img[sl].astype(float),
            out_hw,
            order=3,
            anti_aliasing=shrink,
            preserve_range=True,
        )
    return out


def _bandpass_phase(wf, fs=35):
    """Bandpass (2-8 Hz, butter order 2, filtfilt) each pixel time series of
    wf [rows, cols, frames] and return (meanTrace2, tracePhase), both
    [rows, cols, frames]. Columns are processed in chunks to bound memory."""
    nr, nc, nf = wf.shape
    wf1 = wf.reshape(nr * nc, nf)
    meanTrace = (wf1 - wf1.mean(axis=1, keepdims=True)).T.astype(float)  # frames x pixels
    b, a = signal.butter(2, [2 / (fs / 2), 8 / (fs / 2)], btype="bandpass")
    chunk = 20000
    filt = np.empty_like(meanTrace)
    phase = np.empty_like(meanTrace)
    for s in range(0, meanTrace.shape[1], chunk):
        blk = meanTrace[:, s : s + chunk]
        # MATLAB filtfilt default padlen = 3*(max(len(a),len(b))-1) = 12
        filt[:, s : s + chunk] = signal.filtfilt(b, a, blk, axis=0, padlen=12)
        phase[:, s : s + chunk] = np.angle(signal.hilbert(filt[:, s : s + chunk], axis=0))
    meanTrace2 = filt.T.reshape(nr, nc, nf)
    tracePhase = phase.T.reshape(nr, nc, nf)
    return meanTrace2, tracePhase
