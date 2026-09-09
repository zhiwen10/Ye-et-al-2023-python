from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _decode_char(ds):
    return "".join(chr(v) for v in ds[()].ravel())


def load_task_table(path, varname="T_all"):
    """Adaptation: decode a MATLAB table saved in a v7.3 (HDF5) .mat file
    (scipy.io.loadmat cannot read MATLAB tables/MCOS objects).

    Returns a pandas DataFrame. Numeric columns become float64 arrays;
    cell-of-char columns (e.g. ``label``) become object arrays of str.

    Pairing heuristic: the variable names are stored as a cell of char
    arrays and the columns as a same-length cell of arrays elsewhere in
    the file's ``#refs#`` group; the two cells are matched by length and
    element order (verified against all ``task_outcome``/``trial_trace``
    files of this dataset).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if varname not in f:
            raise KeyError(f"{varname} not found in {path}")
        cells = []

        def _visit(name, obj):
            if (
                isinstance(obj, h5py.Dataset)
                and obj.attrs.get("MATLAB_class", b"") == b"cell"
                and not obj.attrs.get("MATLAB_empty", 0)
            ):
                cells.append(obj)

        f.visititems(_visit)

        def _is_char_cell(els):
            return all(
                isinstance(e, h5py.Dataset)
                and e.attrs.get("MATLAB_class", b"") == b"char"
                for e in els
            )

        def _is_data_cell(els):
            # numeric datasets or cells of char (string columns), no structs/groups
            for e in els:
                if not isinstance(e, h5py.Dataset):
                    return False
                cls = e.attrs.get("MATLAB_class", b"")
                if cls == b"cell":
                    sub = [f[r] for r in e[()].flat]
                    if not sub or not _is_char_cell(sub):
                        return False
                elif cls not in (b"double", b"single", b"int32", b"uint32", b"logical"):
                    return False
            return True

        pairs = []
        for c in cells:
            els = [f[r] for r in c[()].flat]
            if not els or not _is_char_cell(els):
                continue
            for c2 in cells:
                if c2 is c:
                    continue
                els2 = [f[r] for r in c2[()].flat]
                if len(els2) == len(els) and _is_data_cell(els2):
                    pairs.append((els, els2))
        if not pairs:
            raise ValueError(f"Could not decode table {varname} in {path}")
        names_els, data_els = max(pairs, key=lambda p: len(p[0]))
        names = [_decode_char(e) for e in names_els]
        out = {}
        for name, e in zip(names, data_els):
            if isinstance(e, h5py.Dataset) and e.attrs.get("MATLAB_class", b"") == b"cell":
                out[name] = np.array(
                    [_decode_char(f[r]) for r in e[()].flat], dtype=object
                )
            else:
                out[name] = np.asarray(e[()]).ravel()
    return pd.DataFrame(out)
