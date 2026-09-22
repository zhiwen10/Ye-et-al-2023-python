import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure
from statsmodels.nonparametric.smoothers_lowess import lowess


def _get_xy(c):
    # coords entries are mat_struct objects (or dicts) with fields x and y
    if hasattr(c, "x"):
        return np.asarray(c.x).ravel(), np.asarray(c.y).ravel()
    return np.asarray(c["x"]).ravel(), np.asarray(c["y"]).ravel()


def overlayOutlines(coords, scale, color="k", ax=None):
    """Translated from spirals/utils/spirals_detection/overlayOutlines.m
    (identical to axons/utils/overlayOutlines.m)"""
    if ax is None:
        ax = plt.gca()
    for q in range(len(coords)):
        cx, cy = _get_xy(coords[q])
        ax.plot(cx / scale, cy / scale, linewidth=0.5, color=color)
    return ax


def filterProjectedAtlas(projectedAtlas, projectedTemplate, dfolder=None):
    if dfolder is None:
        from spirals_py.utils.paths import data_root

        dfolder = data_root() / "tables"
    """Translated from utils/filterProjectedAtlas.m"""
    st = pd.read_csv(Path(dfolder) / "structure_tree_safe_2017.csv")
    spath = st["structure_id_path"].astype(str)
    # only select cortex in the atlas
    projectedAtlas1 = projectedAtlas.copy()
    projectedTemplate1 = projectedTemplate.copy()
    spath2 = spath.str.startswith("/997/8/567/688/695/315/")
    # MATLAB st.index is the 1-based row number in the structure tree
    idFilt = np.flatnonzero(spath2.to_numpy()) + 1
    Lia = np.isin(projectedAtlas, idFilt)
    projectedAtlas1[~Lia] = 0
    projectedTemplate1[~Lia] = 0
    return projectedAtlas1, projectedTemplate1


def makeSmoothCoords(c):
    """Translated from spirals/utils/makeSmoothCoords.m

    c is a contourc-style 2 x N matrix (header column [level; n] per contour).
    """
    coords = []
    ii = 0
    buff = 10
    while ii < c.shape[1]:
        n = int(c[1, ii])
        if n >= 20:
            x = c[0, ii + 1 : ii + 1 + n]
            y = c[1, ii + 1 : ii + 1 + n]
            x = np.concatenate([x[-buff - 1 :], x, x[:buff]])  # buffer makes ends meet
            y = np.concatenate([y[-buff - 1 :], y, y[:buff]])
            # MATLAB smooth(x,25,'loess'): lowess over a 25-point span
            idx = np.arange(x.size)
            x = lowess(x, idx, frac=25 / x.size, return_sorted=False)
            y = lowess(y, idx, frac=25 / y.size, return_sorted=False)
            x = np.concatenate([x[buff : -buff - 1], [x[buff]]])
            y = np.concatenate([y[buff : -buff - 1], [y[buff]]])
            coords.append({"x": x, "y": y})
        ii = ii + n + 1
    return coords


def _contourc(area):
    """contourc(double(area > 0), [0.5 0.5]) via skimage; returns 2 x N matrix."""
    contours = measure.find_contours((area > 0).astype(float), 0.5)
    parts = []
    for cnt in contours:
        # find_contours returns (row, col); contourc uses x = col, y = row
        x = cnt[:, 1]
        y = cnt[:, 0]
        header = np.array([[0.5], [x.size]])
        parts.append(np.hstack([header, np.vstack([x, y])]))
    if not parts:
        return np.zeros((2, 0))
    return np.hstack(parts)


def _area_mask(areaPath, st, section, hemi):
    indx = []
    spath = st["structure_id_path"].astype(str)
    for p in areaPath:
        indx.extend(np.flatnonzero(spath.str.contains(re.escape(p), na=False)))
    idAll = st["id"].iloc[indx].to_numpy(dtype=float)
    area = np.isin(section.astype(float), idAll).astype(float)
    if hemi == "left":
        area[:, section.shape[1] // 2 :] = 0
    elif hemi == "right":
        area[:, : section.shape[1] // 2] = 0
    return area


def plotOutline(areaPath, st, section, hemi, scale, lineColor="w", lineWidth=1, ax=None):
    """Translated from axons/utils/plotOutline.m"""
    if ax is None:
        ax = plt.gca()
    area = _area_mask(areaPath, st, section, hemi)
    c1 = _contourc(area)
    coordsReg1 = makeSmoothCoords(c1)
    for c in coordsReg1:
        ax.plot(c["x"] * scale, c["y"] * scale, color=lineColor, linewidth=lineWidth)
    return ax


def plotOutline2(areaPath, st, section, hemi, scale, lineColor="w", ax=None):
    """Translated from utils/plotOutline2.m"""
    if ax is None:
        ax = plt.gca()
    area = _area_mask(areaPath, st, section, hemi)
    c1 = _contourc(area)
    coordsReg1 = makeSmoothCoords(c1)
    for c in coordsReg1:
        ax.plot(c["x"] * scale, c["y"] * scale, color=lineColor, linewidth=2)
    return ax
