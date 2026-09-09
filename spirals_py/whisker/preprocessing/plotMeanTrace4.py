"""Translated from whisker/preprocessing/plotMeanTrace4.m"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.signal import find_peaks

from spirals_py.utils.atlas import plotOutline

from ._whisker_utils import _get_cortex_atlas_path, _imresize, _load_outline

_MASK_PATHS4 = [
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
    "/997/8/567/688/695/315/453/322/329/",  # SSp-bfd
    "/997/8/567/688/695/315/453/322/353/",  # SSp-n
    "/997/8/567/688/695/315/453/322/337/",  # SSp-ll
    "/997/8/567/688/695/315/453/322/345/",  # SSp-m
    "/997/8/567/688/695/315/453/322/369/",  # SSp-ul
    "/997/8/567/688/695/315/453/322/361/",  # SSp-tr
    "/997/8/567/688/695/315/453/322/182305689/",  # SSp-un
]

_SSP_C = [
    "/997/8/567/688/695/315/453/322/329/614454341/",
    "/997/8/567/688/695/315/453/322/329/614454348/",
    "/997/8/567/688/695/315/453/322/329/614454355/",
    "/997/8/567/688/695/315/453/322/329/614454362/",
    "/997/8/567/688/695/315/453/322/329/614454369/",
    "/997/8/567/688/695/315/453/322/329/614454376/",
]


def _draw_outlines(ax, st2, atlas2, ssp_bfd_path, scale3, lineColor, lineColor1, lw1, lw2):
    plotOutline(_MASK_PATHS4[0:11], st2, atlas2, None, scale3, lineColor, lw1, ax=ax)
    plotOutline([_MASK_PATHS4[4]], st2, atlas2, None, scale3, lineColor1, lw2, ax=ax)
    plotOutline(_MASK_PATHS4[12:18], st2, atlas2, None, scale3, lineColor1, lw2, ax=ax)
    plotOutline(list(ssp_bfd_path), st2, atlas2, None, scale3, lineColor1, lw1, ax=ax)
    plotOutline(_SSP_C, st2, atlas2, None, scale3, (0.5, 0.5, 0.5), lw1, ax=ax)


def plotMeanTrace4(mimgtransformed, wf_mean2, data_folder, code_folder):
    """Translated from whisker/preprocessing/plotMeanTrace4.m

    mimgtransformed: [rows, cols] reference image (165x143).
    wf_mean2: [rows, cols, frames] mean evoked map (165x143x141).
    code_folder: repo root containing data_plus/horizontal_cortex_atlas_25um_ssp_bfd.mat
    and data_plus/structures_ssp_bfd.csv.
    The bandpass/Hilbert computation in the MATLAB original is unused for
    display and is omitted. cbrewer2('seq','YlOrRd',10) is approximated by
    matplotlib's YlOrRd colormap (same ColorBrewer anchors).
    """
    code_folder = Path(code_folder)
    with h5py.File(
        code_folder / "data_plus" / "horizontal_cortex_atlas_25um_ssp_bfd.mat", "r"
    ) as f:
        atlas2 = f["atlas1"][:].T  # v7.3 -> MATLAB [row, col]
    projectedAtlas1, _, _ = _load_outline(data_folder)
    st2 = pd.read_csv(code_folder / "data_plus" / "structures_ssp_bfd.csv")
    ssp_bfd_path = st2.loc[st2["parent_structure_id"] == 329, "structure_id_path"].tolist()
    BW = projectedAtlas1 != 0

    # cbrewer2('seq','YlOrRd',10) approximation
    color2 = plt.get_cmap("YlOrRd")(np.linspace(0, 1, 10))[:, :3]
    lineColor = "k"
    lineColor1 = "w"
    lineWidth1 = 1
    lineWidth2 = 1.5
    scale3 = 5 / 8
    center = [77, 104]  # [row, col] in the 165x143 grid (MATLAB 1-based)
    a, b = 50 * 2, 100 * 2  # height
    c, d = 80 * 2, 130 * 2  # width
    v1 = np.array([[c, a], [d, a], [d, b], [c, b]])

    fig = plt.figure(figsize=(6, 3))

    mimg = mimgtransformed / mimgtransformed.max()
    mimg = _imresize(mimg, (330, 286))
    low, high = np.percentile(mimg, [2, 98])
    mimgRGB = np.clip((mimg - low) / (high - low), 0, 1)
    BW3 = _imresize(BW.astype(float), (330, 286))

    th2 = np.arange(1, 361, 45)
    px1, py1 = center[1], center[0]
    r = 12
    cx2 = np.round(r * np.cos(np.deg2rad(th2)) + px1)
    cy2 = np.round(r * np.sin(np.deg2rad(th2)) + py1)
    pixel = np.column_stack([cy2, cx2])  # [row, col], MATLAB 1-based

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(mimgRGB, cmap="gray", alpha=BW3)
    _draw_outlines(ax1, st2, atlas2, ssp_bfd_path, scale3, lineColor, lineColor1, lineWidth1, lineWidth2)
    ax1.scatter(pixel[:, 1] * 2, pixel[:, 0] * 2, s=16, c=color2[2:10])
    ax1.scatter([center[1] * 2], [center[0] * 2], s=16, c="w")
    ax1.add_patch(Polygon(v1, closed=True, fill=False, edgecolor="k", linewidth=2))
    ax1.set_aspect("equal")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(mimgRGB, cmap="gray", alpha=BW3)
    _draw_outlines(ax2, st2, atlas2, ssp_bfd_path, scale3, lineColor, lineColor1, lineWidth1, lineWidth2)
    ax2.scatter(pixel[:, 1] * 2, pixel[:, 0] * 2, s=16, c=color2[2:10])
    ax2.scatter([center[1] * 2], [center[0] * 2], s=16, c="w")
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_ylim(b, a)
    ax2.set_xlim(c, d)

    ax3 = fig.add_subplot(1, 3, 3)
    scalea = 1.5
    t1 = np.linspace(-2, 2, 141)
    for i in range(pixel.shape[0]):
        row, col = int(pixel[i, 0]) - 1, int(pixel[i, 1]) - 1  # -> 0-based
        trace_raw = wf_mean2[row, col, :]
        ax3.plot(
            t1[63:85],
            trace_raw[63:85] * scalea * 100 + scalea * i,
            color=color2[i + 2],
            linewidth=1,
        )
        locs, _ = find_peaks(trace_raw, prominence=0.001)
        ax3.scatter(
            t1[locs],
            trace_raw[locs] * scalea * 100 + scalea * i,
            s=6,
            c="k",
        )
    ax3.tick_params(labelsize=8)
    ax3.set_yticklabels([])
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.axvline(t1[70], linestyle="--", color="k")

    return fig
