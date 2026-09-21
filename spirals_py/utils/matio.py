"""MATLAB .mat input/output helpers shared by the preprocessing pipelines
(``pipeline2_axons`` … ``pipeline6_task``).

Two output formats are used, mirroring the two kinds of consumers in this
repository:

- :func:`save_mat73` writes MATLAB v7.3-style HDF5 files whose on-disk
  layout matches the paper's data release (numeric arrays stored with
  reversed axes, cell arrays as datasets of HDF5 object references,
  elements under ``#refs#``).  The h5py-based readers of the figure
  notebooks (``load_task_freq_arrays``, ``plotCorrectSpiralDensity``,
  ``plot_spiral_pre_post``, ``plotMeanMapsAll``, ``plotVarMap``,
  ``plotWhiskerMeanMap``, ...) read these files exactly like the release
  files they regenerate.  Files written by plain h5py carry
  ``MATLAB_class`` attributes and are MATLAB-loadable on a best-effort
  basis (MATLAB tables/MCOS cannot be written from Python).
- :func:`save_mat` writes MATLAB v5 files with ``scipy.io.savemat`` for
  outputs that contain trial tables.  Tables are stored as plain structs
  of column arrays (cell-of-char for string columns);
  :func:`load_task_table` and ``read_table_cell_array`` fall back to
  this format automatically, so the figure functions keep working on
  regenerated outputs.

All loaders (:func:`load_mat_var`, :func:`load_mat_cell`,
:func:`load_mat_table`) accept both the v7.3 release files and the files
written by these pipelines.
"""

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def is_v73(path):
    """True if *path* is a MATLAB v7.3 (HDF5) .mat file."""
    with open(path, "rb") as fh:
        return fh.read(8) == _HDF5_SIGNATURE


# ----------------------------------------------------------- v7.3 writing
def _matlab_class(a):
    if a.dtype == np.float64:
        return b"double"
    if a.dtype == np.float32:
        return b"single"
    if a.dtype == np.uint16:
        return b"char"
    if a.dtype == bool:
        return b"logical"
    return b"double"


def _write_numeric(group, name, a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    # HDF5 stores MATLAB arrays with reversed axes
    dset = group.create_dataset(name, data=a.T)
    dset.attrs["MATLAB_class"] = np.bytes_("double")
    return dset


def _write_char(group, name, s):
    # MATLAB char row vector (1, n): stored as (n, 1) uint16
    codes = np.fromiter((ord(c) for c in str(s)), dtype=np.uint16).reshape(-1, 1)
    dset = group.create_dataset(name, data=codes)
    dset.attrs["MATLAB_class"] = np.bytes_("char")
    dset.attrs["MATLAB_int_decode"] = np.uint32(2)
    return dset


def _write_cell(group, name, val, ctr):
    val = np.asarray(val, dtype=object)
    shape = val.shape
    if val.size == 0:
        dset = group.create_dataset(name, shape=shape[::-1], dtype=h5py.ref_dtype)
        dset.attrs["MATLAB_class"] = np.bytes_("cell")
        dset.attrs["MATLAB_empty"] = np.uint32(1)
        return dset
    refs = np.empty(shape[::-1], dtype=h5py.ref_dtype)
    for idx in np.ndindex(*shape):
        refs[idx[::-1]] = _write_element(group, val[idx], ctr).ref
    dset = group.create_dataset(name, data=refs)
    dset.attrs["MATLAB_class"] = np.bytes_("cell")
    return dset


def _write_element(group, item, ctr):
    """Write one cell element into *group* and return its dataset."""
    name = f"r{next(ctr)}"
    if isinstance(item, np.ndarray) and item.dtype == object:
        return _write_cell(group, name, item, ctr)
    if isinstance(item, (str, bytes)):
        return _write_char(group, name, item)
    if item is None:
        return _write_numeric(group, name, np.zeros((0, 0)))
    item = np.asarray(item)
    if item.dtype.kind in "US":
        return _write_char(group, name, str(item))
    return _write_numeric(group, name, item)


def save_mat73(path, variables):
    """Save numeric arrays / cell arrays as a v7.3-style HDF5 .mat file.

    Layout matches the data release: numeric datasets are stored with
    reversed axes (readers apply ``.T`` / ``transpose(3, 2, 1, 0)``),
    cell arrays are datasets of object references in reversed shape.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ctr = iter(range(10**9))
    with h5py.File(path, "w") as f:
        refs_group = f.create_group("#refs#")
        for name, val in variables.items():
            if isinstance(val, np.ndarray) and val.dtype == object:
                _write_cell(f, name, val, ctr)
            elif isinstance(val, (str, bytes)):
                _write_char(f, name, val)
            else:
                _write_numeric(f, name, val)


# ----------------------------------------------------------- v5 writing
def df_to_struct(df):
    """DataFrame -> dict of column arrays for ``scipy.io.savemat``.

    The result is stored as a MATLAB struct with one field per column
    (adaptation: MATLAB tables/MCOS objects cannot be written from
    Python).  Numeric columns become (n, 1) doubles; string columns
    become cell arrays of char.
    """
    out = {}
    for col in df.columns:
        v = df[col].to_numpy()
        if v.dtype == object:
            out[col] = np.array([str(x) for x in v], dtype=object)
        else:
            out[col] = np.asarray(v, dtype=float).reshape(-1, 1)
    return out


def save_mat(path, variables):
    """Save variables (DataFrames become structs of columns) as a MATLAB
    v5 .mat file via ``scipy.io.savemat``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        k: df_to_struct(v) if isinstance(v, pd.DataFrame) else v
        for k, v in variables.items()
    }
    savemat(path, out, do_compression=True)


# ----------------------------------------------------------- reading
def _char_to_str(x):
    """Recursively decode MATLAB char content read back by scipy."""
    a = np.asarray(x)
    if a.dtype.kind in "US":
        return str(a.ravel()[0]) if a.size else ""
    if a.dtype == object:
        if a.size == 0:
            return ""
        return _char_to_str(a.ravel()[0])
    if a.dtype == np.uint16:
        return "".join(chr(int(c)) for c in a.ravel())
    return str(x)


def struct_to_df(struct):
    """MATLAB struct (as read by scipy loadmat) -> DataFrame.

    Inverse of :func:`df_to_struct`: numeric fields are raveled,
    cell-of-char fields become object arrays of str.
    """
    struct = np.asarray(struct)
    if struct.dtype.names is None:
        raise ValueError("not a MATLAB struct")
    cols = {}
    for name in struct.dtype.names:
        v = struct[name].reshape(-1)[0]
        v = np.asarray(v)
        if v.dtype.kind in "US" or v.dtype == object or v.dtype == np.uint16:
            cols[name] = np.array(
                [_char_to_str(e) for e in np.atleast_1d(v).ravel()], dtype=object
            )
        else:
            cols[name] = np.asarray(v, dtype=float).ravel()
    return pd.DataFrame(cols)


def load_mat_var(path, varname):
    """Load a numeric variable in MATLAB orientation from a v5 or v7.3
    .mat file (v7.3 datasets are stored with reversed axes)."""
    path = Path(path)
    if is_v73(path):
        with h5py.File(path, "r") as f:
            return np.asarray(f[varname]).T
    m = loadmat(path)
    if varname not in m:
        raise KeyError(f"{varname} not found in {path}")
    return m[varname]


def load_mat_vars(path, varnames):
    """Load several numeric variables (see :func:`load_mat_var`)."""
    path = Path(path)
    if is_v73(path):
        with h5py.File(path, "r") as f:
            return {v: np.asarray(f[v]).T for v in varnames}
    m = loadmat(path)
    return {v: m[v] for v in varnames}


def _read_ref(f, r):
    d = f[r]
    if isinstance(d, h5py.Dataset) and d.dtype == object:  # nested cell
        refs = d[()]
        shape = d.shape[::-1]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            out[idx] = _read_ref(f, refs[idx[::-1]])
        return out
    cls = d.attrs.get("MATLAB_class", b"")
    if cls == b"char":
        return _char_to_str(np.asarray(d).T)
    return np.asarray(d).T


def _load_cell_v73(path, varname):
    with h5py.File(path, "r") as f:
        dset = f[varname]
        if dset.attrs.get("MATLAB_empty", 0) == 1:
            return np.empty(dset.shape[::-1], dtype=object)
        refs = dset[()]
        shape = dset.shape[::-1]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            out[idx] = _read_ref(f, refs[idx[::-1]])
        return out


def load_mat_cell(path, varname):
    """Load a cell array from a v5 or v7.3 .mat file as an object ndarray
    in MATLAB orientation.

    Elements are ndarrays (numeric cells), nested object arrays (cells of
    cells) or str (char cells), matching what ``scipy.io.loadmat``
    returns for v5 files written by :func:`save_mat`.
    """
    path = Path(path)
    if is_v73(path):
        return _load_cell_v73(path, varname)
    m = loadmat(path)
    if varname not in m:
        raise KeyError(f"{varname} not found in {path}")
    arr = m[varname]
    if arr.dtype == object:
        for idx in np.ndindex(*arr.shape):
            el = arr[idx]
            if isinstance(el, np.ndarray) and el.dtype.names is not None:
                continue  # struct element, keep as read
            if isinstance(el, np.ndarray) and el.dtype.kind in "US":
                arr[idx] = str(el.ravel()[0]) if el.size else ""
            elif isinstance(el, np.ndarray) and el.dtype == object:
                arr[idx] = np.array(
                    [_char_to_str(e) for e in el.ravel()], dtype=object
                )
        return arr
    return arr


def load_mat_table(path, varname):
    """Load a trial table as a DataFrame from a v5 struct (see
    :func:`save_mat`) or a v7.3 MATLAB table (MCOS, via
    ``load_task_table``)."""
    path = Path(path)
    if is_v73(path):
        from spirals_py.task.preprocessing.load_task_table import load_task_table

        return load_task_table(path, varname)
    m = loadmat(path)
    if varname not in m:
        raise KeyError(f"{varname} not found in {path}")
    return struct_to_df(m[varname])


def nanmean(a, axis=None, keepdims=False):
    """MATLAB mean(..., 'omitnan'): all-NaN slices give NaN (numpy
    nanmean warns; the warning is suppressed)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis, keepdims=keepdims)


def nanstd(a, axis=None):
    """MATLAB std(..., 'omitnan') (N-1 normalization)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis, ddof=1)
