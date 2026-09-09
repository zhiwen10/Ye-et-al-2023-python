from pathlib import Path

import numpy as np


def loadUVt1(expRoot):
    """Translated from spirals/utils/spirals_detection/loadUVt1.m"""
    expRoot = Path(expRoot)
    U = np.load(expRoot / "svdSpatialComponents.npy")
    mimg = np.load(expRoot / "meanImage.npy")
    V = np.load(expRoot / "svdTemporalComponents_corr.npy")
    t = np.load(expRoot / "svdTemporalComponents_corr.timestamps.npy")
    t = np.atleast_1d(t.squeeze())

    if t.size > V.shape[1]:
        t = t[: V.shape[1]]
    elif t.size < V.shape[1]:
        V = V[:, : t.size]

    return U, V, t, mimg


def wrapAngle(ph):
    """Translated from spirals/utils/wrapAngle.m"""
    ph = np.asarray(ph, dtype=float).ravel()
    # MATLAB angdiff: consecutive differences wrapped to [-pi, pi]
    phdiff = np.angle(np.exp(1j * np.diff(ph)))
    ph2 = np.concatenate([[ph[0]], ph[0] + np.cumsum(phdiff)])
    return ph2
