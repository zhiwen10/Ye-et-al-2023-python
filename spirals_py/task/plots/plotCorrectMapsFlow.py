from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

from spirals_py.task.preprocessing.trace_filt_nan import trace_filt_nan
from spirals_py.utils.atlas import plotOutline
from spirals_py.utils.io import loadStructureTree
from spirals_py.utils.optical_flow import HS_flowfield


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m"""
    st = loadStructureTree(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    maskPath = [
        "/997/8/567/688/695/315/500/985/",  # MOp
        "/997/8/567/688/695/315/500/993/",  # MOs
        "/997/8/567/688/695/315/31/",  # ACA
        "/997/8/567/688/695/315/453/378/",  # SS2
        "/997/8/567/688/695/315/453/322/",  # SSp
        "/997/8/567/688/695/315/247/",  # AUD
        "/997/8/567/688/695/315/669/",  # VIS
        "/997/8/567/688/695/315/254/",  # RSP
        "/997/8/567/688/695/315/22",  # VISa
        "/997/8/567/688/695/315/541/",  # TEa
        "/997/8/567/688/695/315/677/",  # VISC
    ]
    return maskPath, st


def _load_atlas1(data_folder):
    with h5py.File(Path(data_folder) / "tables" / "horizontal_cortex_atlas_50um.mat", "r") as f:
        atlas1 = np.asarray(f["atlas1"]).T  # HDF5 stores MATLAB arrays transposed
    return atlas1


def _load_task_mask(data_folder, downscale):
    with h5py.File(Path(data_folder) / "tables" / "task_mask_all_mice.mat", "r") as f:
        BW2 = np.asarray(f["BW2"]).T
    return BW2[::downscale, ::downscale].astype(bool)


def _subplottight(fig, n, m, i):
    """Translated from utils/subplottight.m (i is 1-based, row-major over n rows x m cols)."""
    r = (i - 1) // m
    c = (i - 1) % m
    return fig.add_axes([c / m, 1 - (r + 1) / n, 1 / m, 1 / n])


def _parula():
    """MATLAB parula(64), exported from MATLAB R2026a (see parula64.csv)."""
    data = np.loadtxt(Path(__file__).with_name("parula64.csv"), delimiter=" ")
    return LinearSegmentedColormap.from_list("parula", data)


def _colorcet_c06():
    """colorcet('C06'), exported from dependencies/colorcet/colorcet.m (see colorcet_C06.csv)."""
    data = np.loadtxt(Path(__file__).with_name("colorcet_C06.csv"), delimiter=" ")
    return LinearSegmentedColormap.from_list("C06", data)


def plotCorrectMapsFlow(data_folder, save_folder):
    """Translated from task/plots/plotCorrectMapsFlow.m

    Plot mean widefield maps in correct trials: raw frames, 2-8 Hz phase
    frames and phase frames with Horn-Schunck flow field overlaid.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    atlas1 = _load_atlas1(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    BW2 = _load_task_mask(data_folder, 16)
    with h5py.File(
        data_folder / "task" / "task_mean_maps" / "task_mean_maps_all_mice.mat", "r"
    ) as f:
        # MATLAB (83, 72, 141, 3) -> x, y, t, condition
        trace_mean_all = np.asarray(f["trace_mean_all"]).transpose(3, 2, 1, 0)

    hemi = None
    scale3 = 5 / 16
    lineColor = "k"
    v = np.array([[37, 38], [58, 38], [58, 59], [37, 59]]) - 1  # MATLAB 1-based -> 0-based
    parula = _parula()
    c06 = _colorcet_c06()

    h5b = plt.figure(figsize=(9.5, 6))
    for kk in range(1):
        # -2:2 seconds, 141 samples
        trace_mean_current = trace_mean_all[:, :, :, kk]
        traceFilt2, tracePhase2 = trace_filt_nan(trace_mean_current)
        tracePhase3 = tracePhase2.transpose(2, 0, 1)[:, 34:59, 9:60]
        vxRaw_all, vyRaw_all = HS_flowfield(tracePhase3, False)
        # only use -1:1 seconds, 35 samples
        trace_mean4 = trace_mean_current[:, :, 53:88]
        tracePhase4 = tracePhase2[:, :, 53:88]
        vxRaw_all1 = vxRaw_all[53:88]
        vyRaw_all1 = vyRaw_all[53:88]
        if kk == 0:
            # MATLAB prctile ignores NaNs; plain np.percentile would
            # return NaN for the NaN-masked mean maps (all-dark rendering)
            cmax = np.nanpercentile(trace_mean4, 99.98)
            cmin = np.nanpercentile(trace_mean4, 0.02)
        for i in range(14):
            iframe = i + 15  # 0-based; MATLAB iframe = i+15 (1-based)
            frame_temp1 = trace_mean4[:, :, iframe]
            frame_temp2 = tracePhase4[:, :, iframe]

            ax1 = _subplottight(h5b, 3, 15, kk * 45 + i + 1)
            ax1.imshow(frame_temp1, cmap=parula, vmin=cmin, vmax=cmax, alpha=BW2)
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax1)
            ax1.set_aspect("equal")
            ax1.axis("off")

            ax2 = _subplottight(h5b, 3, 15, kk * 45 + i + 16)
            ax2.imshow(frame_temp2, cmap=c06, vmin=-np.pi, vmax=np.pi, alpha=BW2)
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax2)
            ax2.set_aspect("equal")
            ax2.axis("off")
            ax2.add_patch(
                Polygon(v, closed=True, edgecolor="k", facecolor="none", linewidth=0.8)
            )

            # flow field subsampled by skip=2 and magnified x2, padded back
            skip = 2
            zoom_scale = 2
            vxRaw1 = vxRaw_all1[iframe][::skip, ::skip] * zoom_scale
            vyRaw1 = vyRaw_all1[iframe][::skip, ::skip] * zoom_scale
            rows = np.arange(34, 59, skip)  # MATLAB rows 35:2:59 (1-based)
            cols = np.arange(9, 60, skip)  # MATLAB cols 10:2:60 (1-based)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            valid = BW2[rr, cc]
            ax3 = _subplottight(h5b, 3, 15, kk * 45 + i + 31)
            pos = ax3.get_position()
            ax3.set_position([pos.x0, pos.y0, pos.width - 0.005, pos.height - 0.005])
            ax3.imshow(frame_temp2, cmap=c06, vmin=-np.pi, vmax=np.pi, alpha=BW2)
            # MATLAB quiver(...,'autoScale','off'): arrows in data units
            ax3.quiver(
                cc[valid],
                rr[valid],
                vxRaw1[valid],
                vyRaw1[valid],
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
                headwidth=3,
                headlength=4,
            )
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax3)
            ax3.set_aspect("equal")
            ax3.axis("off")
            ax3.set_xlim(36, 57)  # MATLAB xlim([37,58])
            ax3.set_ylim(58, 37)  # MATLAB ylim([38,59]) with imagesc y-axis
        cb4 = _subplottight(h5b, 3, 15, 15)
        im_cb = cb4.imshow(
            trace_mean4[:, :, 17], cmap=parula, vmin=cmin, vmax=cmax, visible=False
        )
        cb4.axis("off")
        cb = h5b.colorbar(im_cb, ax=cb4)
        a = cb.ax.get_position()
        cb.ax.set_position([a.x0 - 0.02, a.y0, a.width, a.height / 8])
    h5b.savefig(save_folder / "Fig5b_correct_mean_maps2.pdf", bbox_inches="tight")
    return h5b
