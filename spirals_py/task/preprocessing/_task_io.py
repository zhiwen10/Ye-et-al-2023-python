"""Backwards-compatible alias: the shared MATLAB .mat IO helpers moved to
:mod:`spirals_py.utils.matio` (they are used by all preprocessing
pipelines, not only pipeline6)."""

from spirals_py.utils.matio import *  # noqa: F401,F403
from spirals_py.utils.matio import (  # noqa: F401
    _char_to_str,
    _load_cell_v73,
    _matlab_class,
    _read_ref,
    _write_cell,
    _write_char,
    _write_element,
    _write_numeric,
    df_to_struct,
    is_v73,
    load_mat_cell,
    load_mat_table,
    load_mat_var,
    load_mat_vars,
    nanmean,
    nanstd,
    save_mat,
    save_mat73,
    struct_to_df,
)
