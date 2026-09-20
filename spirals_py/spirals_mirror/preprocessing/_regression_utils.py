"""Shared helpers for the spirals_mirror preprocessing translations
(getReducedRankRegressionAP/HEMI, getExampleKernelAP/HEMI, getAxonMap*,
getMapsSession).

Translated from MATLAB sources in YE-et-al-2023-spirals:
- dependencies/spikes/analysis/helpers/CanonCor2.m (plain-matrix version)
- spirals_mirror/utils/sseExplainedCal.m
- spirals/utils/select_area.m (as select_area_indices)
- create3dMask (missing from the MATLAB repository, reconstructed)
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spirals_py.utils.matio import nanmean

# downscaled kernel grid of the spirals_mirror pipeline
# (projectedAtlas1(1:8:end, 1:8:end))
KERNEL_GRID_SHAPE = (165, 143)


def CanonCor2(Y, X):
    """Translated from dependencies/spikes/analysis/helpers/CanonCor2.m
    (the plain-matrix version; the cell-array variant in
    spirals_mirror/utils/CanonCor2.m is not translated).

    Reduced-rank / CCA-style regression of the dependent variable Y on the
    independent variable X. X is (nSamples, nX) and Y is (nSamples, nY)
    (MATLAB cov([X, Y]) treats rows as observations).

    Returns (a, b, R2): the rank-n approximation of Y is
    Y ~ X @ b[:, :n] @ a[:, :n].T, and R2[n] is the fraction of the total
    variance of Y explained by the nth projection (MATLAB returns a column
    vector; a 1-D array is returned here). The gpuArray inputs of the
    spirals_mirror pipelines are plain CPU arrays here.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    XSize = X.shape[1]

    BigCov = np.cov(np.hstack([X, Y]), rowvar=False, ddof=1)
    CXX = BigCov[:XSize, :XSize]
    CYX = BigCov[XSize:, :XSize]

    eps = 1e-7
    CXX = CXX + eps * np.eye(XSize)  # prevents imaginary results in some cases

    vals, vecs = np.linalg.eigh(CXX)
    CXXMH = (vecs / np.sqrt(vals)) @ vecs.T  # CXX ^ -0.5, symmetric

    M = CYX @ CXXMH

    d, s, c = np.linalg.svd(M, full_matrices=False)  # MATLAB [d, s, c]

    b = CXXMH @ c.T
    a = d * s  # d * diag(s)

    R2 = s**2 / Y.var(axis=0, ddof=1).sum()
    return a, b, R2


def sseExplainedCal(signals, predicted_signals):
    """Translated from spirals_mirror/utils/sseExplainedCal.m

    signal = channel x tsample, predicted_signal = channel x tsample;
    returns 1 - SSE/SST per channel (SST about the per-channel nanmean).
    The preprocessing pipelines flatten the traces to a single channel, so
    the result is a scalar there.
    """
    signals = np.asarray(signals, dtype=np.float64)
    predicted_signals = np.asarray(predicted_signals, dtype=np.float64)
    sse_residual = ((signals - predicted_signals) ** 2).sum(axis=1)
    sse_total = ((signals - nanmean(signals, axis=1, keepdims=True)) ** 2).sum(
        axis=1
    )
    return 1 - sse_residual / sse_total


def create3dMask(structure_id_path_prefix, st, atlas_3d):
    """create3dMask is missing from the MATLAB repository (only ever called
    as create3dMask(ctx, st, atlas) in spirals_mirror/preprocessing/
    getAxonMap*.m); reconstructed as a logical volume of the 3-D atlas IDs
    (st['id']) of the structure-tree rows whose structure_id_path starts
    with structure_id_path_prefix (e.g. '/997/8/567/688/' isocortex).

    Note the 3-D annotation volume stores structure IDs, while the 2-D
    projectedAtlas1 stores 0-based structure-tree row numbers (see
    select_area_indices).
    """
    spath = st["structure_id_path"].astype(str)
    rows = np.flatnonzero(
        spath.str.startswith(structure_id_path_prefix).to_numpy()
    )
    ids = st["id"].iloc[rows].to_numpy()
    return np.isin(np.asarray(atlas_3d), ids)


def select_area_indices(area_paths, spath, Utransformed, projectedAtlas1, hemi, scale):
    """Translated from spirals/utils/select_area.m (semantics identical to
    spirals_py/spirals_mirror/plots/plotCortexDivision.py::_select_area).

    Finds the atlas pixels belonging to the areas whose structure_id_path
    starts with any of area_paths, restricted to one hemisphere, on the
    scale-downsampled projectedAtlas1 grid.

    Returns (index, Uselected): index is the 0-based column-major linear
    index into the downsampled grid (MATLAB find() on the column-major
    matrix, minus 1), consistent with the k1_real scatter; Uselected is
    Utransformed downsampled and column-major reshaped, with rows indexed
    by index (zeros (n, 1) when Utransformed is not 3-D, matching the
    2-D zeros(size(projectedAtlas1)) the getAxonMap* originals pass).
    hemi='right' keeps 1-based columns size/2+1:end, hemi='left' keeps
    1:size/2.

    Dead code skipped: the projectedTemplate1 zeroing and the
    plotOverlaidAtlas call of the MATLAB original (their results are never
    used), and the unused st/coords arguments (st.index == the 0-based row
    number of spath, matching loadStructureTree.m).
    """
    mask = np.zeros(len(spath), dtype=bool)
    for p in area_paths:
        mask |= spath.str.startswith(p).to_numpy()
    idFilt = np.flatnonzero(mask)
    pa = projectedAtlas1.copy()
    pa[~np.isin(pa, idFilt)] = 0
    if hemi == "right":
        pa[:, : pa.shape[1] // 2] = 0
    elif hemi == "left":
        pa[:, pa.shape[1] // 2 :] = 0
    pa2 = pa[::scale, ::scale]
    index = np.flatnonzero(pa2.ravel(order="F"))
    if Utransformed is not None and Utransformed.ndim == 3:
        UregDown = Utransformed[::scale, ::scale, :]
        UregDown1d = UregDown.reshape(-1, UregDown.shape[2], order="F")
        Uselected = UregDown1d[index, :]
    else:
        Uselected = np.zeros((index.size, 1), dtype=np.float32)
    return index, Uselected


def _session_fname(T, kk):
    """fname = [T.MouseID{kk} '_' datestr(T.date(kk),'yyyymmdd') '_'
    num2str(T.folder{kk})] as built by every spirals_mirror preprocessing
    .m file (T is the pandas session table)."""
    mn = T["MouseID"].iloc[kk]
    tda = T["date"].iloc[kk]
    en = T["folder"].iloc[kk]
    tdb = pd.Timestamp(tda).strftime("%Y%m%d")
    return f"{mn}_{tdb}_{int(en)}"


def _kernel_slice_from_compact(k1_real, index_from, index_to, shape, pr, pc):
    """squeeze(kernel_full2(:,:,pr,pc)) rebuilt from the compact kernel
    variables (kernel_temp2(index_from) = k1_real(:, j);
    kernel_full2(:, index_to(j)) = kernel_temp2). index_from/index_to are
    0-based column-major linear indices; pr, pc are 1-based (row, col)."""
    img = np.zeros(shape[0] * shape[1])
    lin0 = (pc - 1) * shape[0] + (pr - 1)
    j = int(np.searchsorted(index_to, lin0))
    if j < index_to.size and index_to[j] == lin0:
        img[index_from] = k1_real[:, j]
    return img.reshape(shape, order="F")


def load_kernel_slice(path, pr, pc, shape=KERNEL_GRID_SHAPE):
    """squeeze(kernel_full2(:,:,pr,pc)) from a <fname>-AP/-hemi.mat file
    written by getReducedRankRegressionAP/HEMI (or by MATLAB).

    Reads the dense kernel_full2 dataset if present (only the requested
    slice is read from disk); otherwise rebuilds the slice from the compact
    k1_real / indexMO / indexSSp variables (indexMO = source rows index,
    indexSSp = target columns index, saved 1-based column-major linear
    indices; in the -hemi files they are the left/right sensory indices).
    pr, pc are 1-based (row, col) on the kernel grid; shape is the grid
    used when kernel_full2 is absent (165 x 143 in this pipeline).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "kernel_full2" in f:
            return np.asarray(f["kernel_full2"][pc - 1, pr - 1]).T
        k1_real = np.asarray(f["k1_real"]).T
        index_from = np.asarray(f["indexMO"]).ravel().astype(np.int64) - 1
        index_to = np.asarray(f["indexSSp"]).ravel().astype(np.int64) - 1
    return _kernel_slice_from_compact(k1_real, index_from, index_to, shape, pr, pc)
