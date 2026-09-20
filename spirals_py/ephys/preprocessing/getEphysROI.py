from pathlib import Path

import numpy as np

from spirals_py.ephys.utils import get_session_info2
from spirals_py.spirals.plots._fig1_helpers_s1 import _padZeros


def _save_roi_mat73(path, verts):
    """Save the (N, 2) ROI vertices as a (1, 1) object cell 'roi' in the
    v7.3 layout.  save_mat73 writes cell elements at the HDF5 root
    instead of under #refs# (where MATLAB stores them), so the vertex
    dataset is written under #refs# directly to match the release files
    that _load_roi_vertices scans."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        d = refs.create_dataset("r0", data=np.asarray(verts, dtype=np.float64).T)
        d.attrs["MATLAB_class"] = np.bytes_("double")
        cell = f.create_dataset("roi", shape=(1, 1), dtype=h5py.ref_dtype)
        cell[0, 0] = d.ref
        cell.attrs["MATLAB_class"] = np.bytes_("cell")


def getEphysROI(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getEphysROI.m

    Interactive ROI drawing: sessions with an existing
    <fname>_roi.mat are skipped (MATLAB roi_exist); otherwise the mean
    image padded by 120 zero pixels is displayed and a polygon ROI is
    drawn with matplotlib's PolygonSelector (click vertices; close with
    Enter / by clicking the first vertex; the window is closed to
    accept).  The vertices (N, 2) [x, y] in the padded 120-frame are
    saved as a (1, 1) object cell 'roi' (v7.3, see _save_roi_mat73),
    laid out so that _load_roi_vertices recovers them exactly.
    (Adaptation: MATLAB drawpolygon/polyshape replaced by
    PolygonSelector + raw vertices.)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    params_downscale = 1
    halfpadding = 120
    roi_exist = np.zeros(len(T), dtype=int)
    for kk in range(len(T)):
        ops = get_session_info2(T, kk, data_folder)
        fname = ops.fname
        mimg = np.load(Path(ops.session_root) / "meanImage.npy")
        # apply mask, this helps speed up spiral detection later
        mimg1 = mimg[::params_downscale, ::params_downscale]
        mimg2 = _padZeros(mimg1, halfpadding)
        filename = save_folder / f"{fname}_roi.mat"
        if filename.exists():
            roi_exist[kk] = 1
        else:
            roi_exist[kk] = 0
            fig, ax = plt.subplots()
            ax.imshow(mimg2, cmap="gray")
            selector = PolygonSelector(ax, lambda verts: None)
            plt.show()
            roi = np.asarray(selector.verts, dtype=float)
            _save_roi_mat73(filename, roi)
            plt.close(fig)
    return roi_exist
