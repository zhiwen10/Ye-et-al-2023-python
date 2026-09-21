"""Shared private helpers for the FigS5/FigS6/FigS7 plot translations
(spirals/plots/plotSpiralsBySession1/2.m, plotSpiralDensityByRadius/
Duration.m, plotExampleSpiralTrajectory.m, plotSpiralDirectionRatio2.m,
plotSpiralDensitySessionsMeanSEM.m, plotSpiralsSymmetryRatio.m).

Helper sources translated here:
- spirals/utils/density_color_plot.m (counts only; the unused scolor output
  and remapColor.m are omitted)
- spirals/utils/get_cortex_atlas_path.m
- spirals/utils/select_area.m (returns a boolean mask instead of find()
  linear indices; the Uselected SVD output is unused by these plots)
- spirals/utils/UniqueMatchPoints.m
- spirals/utils/getSymmetryRatio.m
- the drawpolygon ROIs stored (as MCOS objects) in
  spirals/spirals_symmetry/roiSelection.mat, incl. a point-in-polygon
  replacement for MATLAB's inROI
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.ndimage import uniform_filter

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


def _session_info(T, kk):
    """Session strings from the spiralSessions3 table (kk is 0-based)."""
    mn = T["MouseID"].iloc[kk]
    tda = pd.to_datetime(T["date"].iloc[kk])
    en = int(T["folder"].iloc[kk])
    tdb = tda.strftime("%Y%m%d")
    fname = f"{mn}_{tdb}_{en}"
    return mn, tdb, en, fname


def _load_spirals_grouping(path, min_duration=1):
    """Load archiveCell from a *_spirals_group_fftn.mat file (v7.3 or v5).

    Returns (cells, durations): the cell matrices with duration >=
    min_duration in MATLAB (n, 5) orientation [x y r dir frame], and the
    durations of all cells.
    """
    from spirals_py.utils.matio import is_v73

    if not is_v73(path):
        # v5 files (e.g. the whisker release grouping archives) store the
        # cell directly in scipy-readable form
        from spirals_py.utils.matio import load_mat_cell

        arr = load_mat_cell(path, "archiveCell").ravel()
        durations = np.array([np.size(c, 0) for c in arr], dtype=int)
        cells = [np.atleast_2d(c) for c in arr[durations >= min_duration]]
        return cells, durations
    with h5py.File(path, "r") as f:
        refs = f["archiveCell"][()].ravel()
        durations = np.empty(refs.size, dtype=int)
        for i, r in enumerate(refs):
            durations[i] = f[r].shape[1]  # cells are (5, n) in HDF5
        cells = [np.asarray(f[r]).T for r in refs[durations >= min_duration]]
    return cells, durations


def _transformPointsForward(Tmat, x, y):
    """MATLAB transformPointsForward(affine2d(T), x, y): [x' y' 1] = [x y 1]*T."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    xp = x * Tmat[0, 0] + y * Tmat[1, 0] + Tmat[2, 0]
    yp = x * Tmat[0, 1] + y * Tmat[1, 1] + Tmat[2, 1]
    return xp, yp


def _ismember_rows(A, B):
    """MATLAB ismember(A, B, 'rows') logical output."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    dt = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    return np.isin(A.view(dt).ravel(), B.view(dt).ravel())


def _index_xy(mask):
    """MATLAB [row,col] = find(mask); index = [col,row] (i.e. [x y] pairs)."""
    rows, cols = np.nonzero(mask)
    return np.column_stack([cols, rows])


def _density_color_plot(pwAllRaw, histbin):
    """Translated from spirals/utils/density_color_plot.m (unique_spirals1
    output only: unique x/y rows with the number of points within a
    histbin x histbin box centred on each row).

    Coordinates must be integral pixels (all callers round first), so the
    box count is computed exactly with a zero-padded box filter.
    """
    half = histbin / 2
    xy = np.asarray(pwAllRaw[:, :2], dtype=float)
    if not np.allclose(xy, np.round(xy)):
        raise ValueError("density_color_plot expects integer pixel coordinates")
    unique_spirals1 = np.unique(xy, axis=0)  # MATLAB unique(...,'rows')
    xi = unique_spirals1[:, 0].astype(np.int64)
    yi = unique_spirals1[:, 1].astype(np.int64)
    allx = xy[:, 0].astype(np.int64)
    ally = xy[:, 1].astype(np.int64)
    xmin, ymin = allx.min(), ally.min()
    grid = np.zeros((ally.max() - ymin + 1, allx.max() - xmin + 1))
    np.add.at(grid, (ally - ymin, allx - xmin), 1.0)
    k = int(np.floor(half)) * 2 + 1  # integer offsets within +/-half
    counts = uniform_filter(grid, size=k, mode="constant", cval=0.0) * (k * k)
    out = np.zeros((unique_spirals1.shape[0], 3))
    out[:, :2] = unique_spirals1
    out[:, 2] = counts[yi - ymin, xi - xmin]
    return out


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m"""
    st = pd.read_csv(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    return list(_MASK_PATHS), st


def _select_area(sensoryArea, st, projectedAtlas1, hemi, scale=1):
    """Translated from spirals/utils/select_area.m.

    Returns the logical mask of the selected areas/hemisphere (MATLAB
    returns find() linear indices used to index a mask; scale=1 here).
    MATLAB st.index is the 1-based structure-tree row number.
    """
    spath = st["structure_id_path"].astype(str)
    sel = np.zeros(len(st), dtype=bool)
    for p in sensoryArea:
        sel |= spath.str.startswith(p).to_numpy()
    idFilt1 = np.flatnonzero(sel) + 1
    atlas = projectedAtlas1.copy()
    atlas[~np.isin(atlas, idFilt1)] = 0
    if hemi == "right":
        atlas[:, : atlas.shape[1] // 2] = 0
    elif hemi == "left":
        atlas[:, atlas.shape[1] // 2 :] = 0
    return atlas[::scale, ::scale].astype(bool)


def _seq_colormap(name, n):
    """flipud(cbrewer2('seq', name, n)) approximated by sampling the
    equivalent ColorBrewer matplotlib sequential map and reversing it
    (same adaptation as plotSpiralSyncIndex for 'div','RdBu')."""
    cmap = plt.get_cmap(name)
    return cmap(np.linspace(0.0, 1.0, int(n)))[::-1]


def _imwarp_nearest(A, Tmat, out_shape):
    """MATLAB imwarp(A, affine2d(T), 'OutputView', imref2d(out_shape))
    with the default 'nearest' interpolation and fill value 0 (used for
    the rf sign maps; plotSpiralSyncIndex._imwarp is the bilinear
    variant used there for continuous SVD maps)."""
    from scipy.ndimage import map_coordinates

    A = np.asarray(A, dtype=np.float64)
    M, N = out_shape
    rows = np.arange(M)
    cols = np.arange(N)
    xw, yw = np.meshgrid(cols + 1.0, rows + 1.0)  # 1-based intrinsic coords
    # MATLAB inverse mapping: [xin yin 1] = [xout yout 1] * inv(T)
    Tinv = np.linalg.inv(Tmat)
    xin = xw * Tinv[0, 0] + yw * Tinv[1, 0] + Tinv[2, 0]
    yin = xw * Tinv[0, 1] + yw * Tinv[1, 1] + Tinv[2, 1]
    coords = np.array([yin - 1.0, xin - 1.0])  # 0-based (row, col)
    return map_coordinates(A, coords, order=0, mode="constant", cval=0.0)


def _line_points():
    """The fixed profile line used by the density plots:
    x = [105,520], y = [780,225] sampled at every integer x."""
    xq = np.arange(105, 521)
    vq = np.interp(xq, [105, 520], [780, 225])
    return np.column_stack([xq, np.round(vq)])


def _load_rois(path):
    """Load the drawpolygon ROIs from roiSelection.mat (v7.3 with MCOS
    objects, which h5py/scipy cannot deserialize directly).

    Each variable (roiM1L, roiSL, roiM1R, roiSR, roiM2L, roiM2R) is a
    uint32 MCOS header whose 5th element is the object id; the Polygon
    Position (N x 2 vertices) datasets appear in the #subsystem#/MCOS
    cell in object-id order (one 5-dataset group per polygon, 1-dataset
    groups for non-polygon objects). The id -> polygon mapping reproduced
    here was additionally verified geometrically (M2 rostral/medial,
    M1 middle, S1 caudal/lateral; left x < 570, right x > 570).
    """
    names = ["roiM1L", "roiSL", "roiM1R", "roiSR", "roiM2L", "roiM2R"]
    with h5py.File(path, "r") as f:
        ids = {name: int(f[name][()].ravel()[4]) for name in names}
        refs = f["#subsystem#/MCOS"][()].ravel()
        blocks = [f[r] for r in refs]  # dataset handles (shapes only)
        positions = []
        i = 0
        while i < len(blocks):
            if blocks[i].dtype == np.float64 and blocks[i].ndim == 2 and blocks[i].shape[0] == 2:
                positions.append(np.asarray(blocks[i]).T)
                i += 5
            else:
                i += 1
    polygon_ids = sorted(ids.values())
    if len(positions) != len(polygon_ids):
        raise ValueError(f"expected {len(polygon_ids)} ROI polygons, found {len(positions)}")
    id_to_poly = dict(zip(polygon_ids, positions))
    return {name: id_to_poly[ids[name]] for name in names}


def _inROI(poly, x, y):
    """MATLAB inROI(roi, x, y) for a polygon ROI (point-in-polygon test)."""
    pts = np.column_stack([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
    return MplPath(poly).contains_points(pts)


def _uniqueMatchPoints(points1, points2):
    """Translated from spirals/utils/UniqueMatchPoints.m.

    Matches points1/points2 rows that share a unique frame number (last
    column) appearing exactly once in both arrays. DirDiff == 0 marks
    pairs with opposite direction (last-but-one column), as in MATLAB.
    """
    points1 = np.atleast_2d(np.asarray(points1, dtype=float))
    points2 = np.atleast_2d(np.asarray(points2, dtype=float))
    if points1.shape[0] > points2.shape[0]:
        points1, points2 = points2, points1
    # points2 with larger size; last column has frame info
    frames2 = points2[:, -1]
    C, ic = np.unique(frames2, return_inverse=True)
    counts = np.bincount(ic)
    singlePoints2Frames = C[counts == 1]
    fa = np.isin(singlePoints2Frames, points1[:, -1])
    matchFrames = singlePoints2Frames[fa]
    match1 = []
    match2 = []
    for frame in matchFrames:
        pTemp1 = np.flatnonzero(points1[:, -1] == frame)
        pTemp2 = np.flatnonzero(points2[:, -1] == frame)
        if pTemp1.size == 1 and pTemp2.size == 1:
            match1.append(points1[pTemp1[0]])
            match2.append(points2[pTemp2[0]])
    ncol = points1.shape[1]
    MatchPoint1 = np.array(match1).reshape(-1, ncol)
    MatchPoint2 = np.array(match2).reshape(-1, ncol)
    DirPoints1Invert = (~MatchPoint1[:, -2].astype(bool)).astype(float)
    DirDiff = DirPoints1Invert - MatchPoint2[:, -2]
    return MatchPoint1, MatchPoint2, DirDiff


def _getSymmetryRatio(roiM1L, roiSL, roiM1R, roiSR, roiM2L, roiM2R, spiralAll):
    """Translated from spirals/utils/getSymmetryRatio.m

    Returns the 7x1 matching-ratio vector for
    [M1-S1-left, M1-S1-right, S1-left-right, M1-left-right, M2-left-right,
    M1-M2-left, M1-M2-right]. The cbrewer2 color bookkeeping of the
    original only feeds the (unreturned) color vectors and is omitted.
    """
    spiralAll = np.asarray(spiralAll, dtype=float)

    def points_of(roi):
        tf = _inROI(roi, spiralAll[:, 0], spiralAll[:, 1])
        return spiralAll[tf, :]

    pointsM1L, pointsSL = points_of(roiM1L), points_of(roiSL)
    pointsM1R, pointsSR = points_of(roiM1R), points_of(roiSR)
    pointsM2L, pointsM2R = points_of(roiM2L), points_of(roiM2R)

    # left M1 and S1
    _, _, DirDiffa = _uniqueMatchPoints(pointsM1L, pointsSL)
    colora = (DirDiffa != 0).astype(float)

    # right M1 and S1
    _, _, DirDiffb = _uniqueMatchPoints(pointsM1R, pointsSR)
    colorb = (DirDiffb != 0).astype(float)

    # S1 left and right (only mirror-symmetric frame pairs)
    fullXSize = 1200
    symetryCriteria = 100
    MatchPoint5, MatchPoint6, DirDiffc = _uniqueMatchPoints(pointsSL, pointsSR)
    xcheck1 = MatchPoint5[:, 0] + MatchPoint6[:, 0]
    ycheck1 = MatchPoint5[:, 1] - MatchPoint6[:, 1]
    ind = np.flatnonzero(
        (xcheck1 >= fullXSize - symetryCriteria) & (xcheck1 <= fullXSize + symetryCriteria)
        & (ycheck1 >= -symetryCriteria) & (ycheck1 <= symetryCriteria)
    )
    DirDiffc = DirDiffc[ind]
    colorc = (DirDiffc != 0).astype(float)

    # M1 left and right
    _, _, DirDiffd = _uniqueMatchPoints(pointsM1L, pointsM1R)
    colord = (DirDiffd != 0).astype(float)

    # M2 left and right
    _, _, DirDiffe = _uniqueMatchPoints(pointsM2L, pointsM2R)
    colore = (DirDiffe != 0).astype(float)

    # left M1 and M2
    _, _, DirDifff = _uniqueMatchPoints(pointsM1L, pointsM2L)
    colorf = (DirDifff != 0).astype(float)

    # right M1 and M2
    _, _, DirDiffg = _uniqueMatchPoints(pointsM1R, pointsM2R)
    colorg = (DirDiffg != 0).astype(float)

    def ratio(color):
        color = np.asarray(color)
        return np.divide(np.sum(color == 0), color.size, dtype=float) if color.size else np.nan

    LM1Ssum = ratio(colora)
    RM1Ssum = ratio(colorb)
    LRSsum = ratio(colorc)
    LRM1sum = ratio(colord)
    LRM2sum = ratio(colore)
    LM12sum = ratio(colorf)
    RM12sum = ratio(colorg)
    return np.array([LM1Ssum, RM1Ssum, LRSsum, LRM1sum, LRM2sum, LM12sum, RM12sum])
