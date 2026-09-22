"""Data-root resolution for the python translation.

All notebooks and tools read their inputs from the MATLAB data release
root and write regenerated files into a separate python output tree, so
the release is never overwritten.  Both roots are resolved per call so
they can be configured through environment variables on any platform:

===================================  ==================  ==================
root                                 environment var     default (Windows)
===================================  ==================  ==================
release inputs (read-only)           ``SPIRALS_DATA_ROOT``   ``D:\\data``
python output tree                   ``SPIRALS_OUT_ROOT``    ``D:\\data_python``
===================================  ==================  ==================

On macOS/Linux the defaults fall back to ``~/data`` and
``~/data_python`` (export the variables to point elsewhere; forward
slashes on any platform, backslashes only on Windows).

:func:`release_twin` bridges the two trees: when a chained intermediate
(e.g. the spiral-detection output consumed by the grouping step, when
detection is skipped because it is slow) has not been regenerated under
the output root yet, it resolves to the corresponding release file
instead.
"""

import os
from pathlib import Path

_WINDOWS_DEFAULTS = {
    "data": r"D:\data",
    "out": r"D:\data_python",
}
_POSIX_DEFAULTS = {
    "data": "~/data",
    "out": "~/data_python",
}


def data_root():
    """MATLAB data-release root (inputs, read-only; ``SPIRALS_DATA_ROOT``
    overrides; ``D:\\data`` on Windows, ``~/data`` elsewhere)."""
    default = _WINDOWS_DEFAULTS["data"] if os.name == "nt" else _POSIX_DEFAULTS["data"]
    return Path(os.environ.get("SPIRALS_DATA_ROOT", default)).expanduser()


def out_root():
    """Python output tree root (``SPIRALS_OUT_ROOT`` overrides;
    ``D:\\data_python`` on Windows, ``~/data_python`` elsewhere)."""
    default = _WINDOWS_DEFAULTS["out"] if os.name == "nt" else _POSIX_DEFAULTS["out"]
    return Path(os.environ.get("SPIRALS_OUT_ROOT", default)).expanduser()


def release_twin(path, data_root_dir):
    """Resolve *path* for reading.

    Returns *path* itself when it exists.  Otherwise, if *path* lies
    under :func:`out_root`, the same relative path under the release
    root *data_root_dir* is returned when it exists — so steps whose
    upstream python output was skipped fall back to the release file.
    """
    path = Path(path)
    if path.exists():
        return path
    try:
        rel = path.relative_to(out_root())
    except ValueError:
        return path
    twin = Path(data_root_dir) / rel
    return twin if twin.exists() else path


def copy_release_twin(path, data_root_dir):
    """Copy the release twin of *path* into the output tree and return it.

    Used for cached interactive ROI files: the polygon is only drawn
    when neither the output-tree copy nor the release original exists.
    Returns the resolved path, or *path* when no twin was found.
    """
    path = Path(path)
    twin = release_twin(path, data_root_dir)
    if twin != path and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(twin, path)
        return path
    return twin
