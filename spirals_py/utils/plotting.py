import matplotlib.pyplot as plt
import numpy as np


def _apply_columnwise(fun, y):
    try:
        return np.asarray(fun(y, axis=0), dtype=float).ravel()
    except TypeError:
        return np.asarray(fun(y), dtype=float).ravel()


def shadedErrorBar(x, y, errBar, lineProps="-k", transparent=True, patchSaturation=0.2, ax=None):
    """Translated from utils/shadedErrorBar.m

    errBar: vector (symmetric), 2 x N array (asymmetric), or a tuple of two
    callables (line statistic, error statistic) applied to the rows of y.
    lineProps: a matplotlib format string (e.g. '-k') or a dict of plot kwargs.
    Returns a dict with handles: mainLine, patch, edge.
    """
    if ax is None:
        ax = plt.gca()

    if callable(errBar[0] if isinstance(errBar, (tuple, list)) else None):
        fun1, fun2 = errBar
        # MATLAB applies the function handles column-wise to y
        errBar = _apply_columnwise(fun2, y)
        y = _apply_columnwise(fun1, y)
    else:
        y = np.asarray(y, dtype=float).ravel()

    if x is None or len(x) == 0:
        x = np.arange(1, y.size + 1)
    else:
        x = np.asarray(x, dtype=float).ravel()

    errBar = np.asarray(errBar, dtype=float)
    if errBar.ndim == 1:
        errBar = np.tile(errBar.ravel(), (2, 1))
    else:
        if 2 not in errBar.shape:
            raise ValueError("errBar has the wrong size")
        if errBar.shape[1] == 2 and errBar.shape[0] != 2:
            errBar = errBar.T
        elif errBar.shape[0] != 2:
            errBar = errBar.T

    if x.size != errBar.shape[1]:
        raise ValueError("length(x) must equal length(errBar)")

    if isinstance(lineProps, dict):
        mainLine = ax.plot(x, y, **lineProps)[0]
    else:
        mainLine = ax.plot(x, y, lineProps)[0]

    mainLineColor = np.array(plt.matplotlib.colors.to_rgb(mainLine.get_color()))
    edgeColor = mainLineColor + (1 - mainLineColor) * 0.55

    if transparent:
        faceAlpha = patchSaturation
        patchColor = mainLineColor
    else:
        faceAlpha = 1
        patchColor = mainLineColor + (1 - mainLineColor) * (1 - patchSaturation)

    uE = y + errBar[0]
    lE = y - errBar[1]

    xP = np.concatenate([x, x[::-1]])
    yP = np.concatenate([lE, uE[::-1]])
    keep = ~np.isnan(yP)
    xP = xP[keep]
    yP = yP[keep]

    patch = ax.fill(xP, yP, facecolor=patchColor, edgecolor="none", alpha=faceAlpha)[0]

    edge1 = ax.plot(x, lE, "-", color=edgeColor)[0]
    edge2 = ax.plot(x, uE, "-", color=edgeColor)[0]

    mainLine.set_zorder(max(mainLine.get_zorder(), patch.get_zorder() + 1))

    return {"mainLine": mainLine, "patch": patch, "edge": (edge1, edge2)}
