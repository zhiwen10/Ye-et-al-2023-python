"""Translated from revision/axons/getMOroi.m

Interactive polygon ROI over the isocortex outline plot with the MO
bias segments (colored by bias_angle through the colorcet C06 LUT,
scale 15 * pc_ratio) and the MOp reference star at (377, 428).  The
selected vertices are saved as a (1, 1) object cell named roi whose
element is the (N, 2) [x, y] vertex array (stored transposed as a
2 x N dataset under #refs#, exactly what
spirals_py.spirals.plots._fig1_helpers_s1._load_roi_vertices expects;
MATLAB saves the polyshape MCOS object, which Python cannot write).

MATLAB's drawpolygon is replaced by matplotlib.widgets.PolygonSelector:
draw the polygon in the figure window, then close the window to save.
If <save_folder>/MO_roi.mat already exists the drawing is skipped and
a note printed (MATLAB would overwrite the ROI).

Skipped dead code of the MATLAB source: the unused color1 LUT
(colorcet('C06','N',12)).
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import PolygonSelector

from spirals_py.utils.atlas import overlayOutlines
from spirals_py.utils.io import load_outline_coords_h5
from spirals_py.utils.paths import copy_release_twin, out_root, release_twin

CENTER_MOP = (377, 428)  # MOp

_C06_CSV = Path(__file__).resolve().parents[2] / "task" / "plots" / "colorcet_C06.csv"


def _save_roi(path, verts):
    """Save the (1, 1) roi cell in the save_mat73 v7.3 layout, with the
    (N, 2) vertex array stored transposed under #refs# (where genuine
    MATLAB v7.3 cell elements live and where _load_roi_vertices looks;
    matio.save_mat73 currently writes cell elements at the HDF5 root,
    which that reader cannot see)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        el = refs.create_dataset("r0", data=np.asarray(verts, dtype=np.float64).T)
        el.attrs["MATLAB_class"] = np.bytes_("double")
        dset = f.create_dataset("roi", data=np.array([[el.ref]], dtype=h5py.ref_dtype))
        dset.attrs["MATLAB_class"] = np.bytes_("cell")


def _colorcet_C06(N=256):
    """colorcet('C06','N',N) from the exported C06 LUT
    (spirals_py/task/plots/colorcet_C06.csv)."""
    base = np.loadtxt(_C06_CSV, delimiter=" ")
    if N >= base.shape[0]:
        return base
    xi = np.arange(N) * (base.shape[0] - 1) / (N - 1)
    return np.stack(
        [np.interp(xi, np.arange(base.shape[0]), base[:, j]) for j in range(3)], axis=1
    )


def _interp_colors(x1, cmap_arr, v):
    """interp1(x1, cmap_arr, v) per color channel."""
    return np.stack(
        [np.interp(v, x1, cmap_arr[:, j]) for j in range(3)], axis=-1
    )


def getMOroi(data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    out = copy_release_twin(save_folder / "MO_roi.mat", data_folder)
    if out.exists():
        print(f"getMOroi: {out} already exists, skipping")
        return

    coords = load_outline_coords_h5(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )
    T1 = pd.read_csv(
        release_twin(
            out_root() / "revision" / "axons" / "Axon_bias_all_cells_MO.csv",
            data_folder,
        )
    )

    color2 = _colorcet_C06(180)
    scale1 = 15
    x1 = np.linspace(-90, 90, 180)
    colora3 = np.nan_to_num(_interp_colors(x1, color2, T1["bias_angle"].to_numpy()))
    soma_center1 = T1[["soma_center_1", "soma_center_2"]].to_numpy()
    axon_bias = T1[["axon_bias_1", "axon_bias_2"]].to_numpy()
    pc_ratio = T1["pc_ratio"].to_numpy()

    _, ax1 = plt.subplots(figsize=(4, 4))
    overlayOutlines(coords, 1, (0.8, 0.8, 0.8), ax=ax1)
    for i in range(len(T1)):
        ax1.plot(
            [
                soma_center1[i, 0] - axon_bias[i, 0] * scale1 * pc_ratio[i],
                soma_center1[i, 0] + axon_bias[i, 0] * scale1 * pc_ratio[i],
            ],
            [
                soma_center1[i, 1] - axon_bias[i, 1] * scale1 * pc_ratio[i],
                soma_center1[i, 1] + axon_bias[i, 1] * scale1 * pc_ratio[i],
            ],
            color=colora3[i],
            linewidth=1,
        )
    ax1.scatter(*CENTER_MOP, s=36, marker="*", c="k")
    ax1.invert_yaxis()  # MATLAB YDir reverse
    ax1.set_aspect("equal")  # axis image
    ax1.set_axis_off()

    print("getMOroi: draw the MO polygon ROI in the figure window, then close it to save")

    roi_holder = {}

    def _onselect(verts):
        roi_holder["verts"] = np.asarray(verts, dtype=float)

    PolygonSelector(ax1, _onselect)
    plt.show()

    verts = roi_holder.get("verts")
    if verts is None:
        print("getMOroi: no ROI selected, nothing saved")
        return
    _save_roi(out, verts)
    return verts
