from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _load_single_var(path):
    """Load a .mat file that contains a single data variable."""
    m = loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected one variable in {path}, found {keys}")
    return m[keys[0]]


def loadUVt2(expRoot):
    """Translated from utils/loadUVt2.m"""
    expRoot = Path(expRoot)
    U = _load_single_var(expRoot / "svdSpatialComponents.mat")
    mimg = _load_single_var(expRoot / "meanImage.mat")
    V = _load_single_var(expRoot / "svdTemporalComponents_corr.mat")
    t = _load_single_var(expRoot / "svdTemporalComponents_corr_timestamps.mat")
    t = np.atleast_1d(t.squeeze())

    if t.size > V.shape[1]:
        t = t[: V.shape[1]]
    elif t.size < V.shape[1]:
        V = V[:, : t.size]

    return U, V, t, mimg


def loadStructureTree(path):
    """Translated from dependencies/spikes (loadStructureTree.m); returns a DataFrame."""
    return pd.read_csv(path)


def load_cell_array_h5(path, varname):
    """Load a MATLAB v7.3 (HDF5) cell array of numeric matrices as a list of
    ndarrays. Arrays are transposed back to MATLAB orientation.
    (Adaptation: scipy.io.loadmat cannot read v7.3 files.)"""
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        refs = f[varname][()]
        for ref in refs.flat:
            out.append(np.asarray(f[ref]).T)
    return out


def load_outline_coords_h5(path, varname="coords"):
    """Load a MATLAB v7.3 struct array with fields x/y (e.g.
    isocortex_horizontal_projection_outline.mat) as a list of dicts.
    (Adaptation: scipy.io.loadmat cannot read v7.3 files.)"""
    import h5py

    coords = []
    with h5py.File(path, "r") as f:
        g = f[varname]
        xs = g["x"][()]
        ys = g["y"][()]
        for rx, ry in zip(xs.flat, ys.flat):
            coords.append(
                {
                    "x": np.asarray(f[rx]).ravel(),
                    "y": np.asarray(f[ry]).ravel(),
                }
            )
    return coords
