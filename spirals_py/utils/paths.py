"""Output-root redirection for the python preprocessing pipelines.

The pipeline notebooks read their inputs from the MATLAB data release
(``D:\\data``) but write every regenerated file into a separate python
output tree (``D:\\data_python`` by default, overridable through the
``SPIRALS_OUT_ROOT`` environment variable), so the release files are
never overwritten.  :func:`release_twin` bridges the two trees: when a
chained intermediate (e.g. the spiral-detection output consumed by the
grouping step, when detection is skipped because it is slow) has not
been regenerated under the output root yet, it resolves to the
corresponding release file instead.
"""

import os
from pathlib import Path

_DEFAULT_OUT_ROOT = r"D:\data_python"


def out_root():
    """Python output tree root (``D:\\data_python`` unless overridden)."""
    return Path(os.environ.get("SPIRALS_OUT_ROOT", _DEFAULT_OUT_ROOT))


def release_twin(path, data_folder):
    """Resolve *path* for reading.

    Returns *path* itself when it exists.  Otherwise, if *path* lies
    under :func:`out_root`, the same relative path under *data_folder*
    (the release root) is returned when it exists — so steps whose
    upstream python output was skipped fall back to the release file.
    """
    path = Path(path)
    if path.exists():
        return path
    try:
        rel = path.relative_to(out_root())
    except ValueError:
        return path
    twin = Path(data_folder) / rel
    return twin if twin.exists() else path


def copy_release_twin(path, data_folder):
    """Copy the release twin of *path* into the output tree and return it.

    Used for cached interactive ROI files: the polygon is only drawn
    when neither the output-tree copy nor the release original exists.
    Returns the resolved path, or *path* when no twin was found.
    """
    path = Path(path)
    twin = release_twin(path, data_folder)
    if twin != path and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(twin, path)
        return path
    return twin
