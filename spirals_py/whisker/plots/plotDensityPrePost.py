"""Translated from whisker/plots/plotDensityPrePost.m (Fig. 5d and 5f)"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from spirals_py.utils.atlas import overlayOutlines, plotOutline

from spirals_py.whisker.preprocessing._whisker_utils import (
    _get_cortex_atlas_path,
    _load_atlas_50um,
    _load_outline,
    _select_area,
)

# In the MATLAB original areaPath(8) is never assigned, so the string array
# contains "" at that position; startsWith(spath, "") matches every entry and
# the "sensory area" selection reduces to the whole hemisphere. The empty
# string is kept here to replicate that behavior exactly.
_SENSORY_AREA = [
    "/997/8/567/688/695/315/453/",  # SS
    "/997/8/567/688/695/315/247/",  # AUD
    "/997/8/567/688/695/315/669/",  # VIS
    "/997/8/567/688/695/315/254/",  # RSP
    "/997/8/567/688/695/315/22/312782546/",  # VISa
    "/997/8/567/688/695/315/22/417/",  # VISrl
    "/997/8/567/688/695/315/541/",  # TEa
    "",
    "/997/8/567/688/695/315/677/",  # VISC
    "/997/8/567/688/695/1089/",  # HPF
    "/997/8/567/688/695/315/677/",  # CTXsp
]


def _density_color_plot2(pwAllRaw, histbin):
    """Translated from task/utils/density_color_plot2.m"""
    half = histbin / 2
    xy = pwAllRaw[:, :2]
    unique_xy = np.unique(xy, axis=0)  # sorted, like MATLAB unique(...,'rows')
    out = np.zeros((unique_xy.shape[0], 3))
    out[:, :2] = unique_xy
    for k, (x, y) in enumerate(unique_xy):
        out[k, 2] = np.sum(
            (xy[:, 0] >= x - half)
            & (xy[:, 0] <= x + half)
            & (xy[:, 1] >= y - half)
            & (xy[:, 1] <= y + half)
        )
    return out


def _ismember_rows(A, B):
    """MATLAB ismember(A, B, 'rows') logical output."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    dt = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    return np.isin(A.view(dt).ravel(), B.view(dt).ravel())


def _ratio_ccw(spirals_cell, index_xy):
    ratios = np.zeros(len(spirals_cell))
    for kk, sp in enumerate(spirals_cell):
        lia = _ismember_rows(sp[:, :2], index_xy)
        sp2 = sp[lia, :]
        ratios[kk] = np.sum(sp2[:, 3] == -1) / sp2.shape[0]
    return ratios


def plotDensityPrePost(data_folder, save_folder):
    """Plot spiral density maps pre/post stimulus (Fig. 5d) and pre/post
    rotation-direction ratios per mouse (Fig. 5f).

    The MOs/MOp select_area calls in the MATLAB original are unused by the
    plot and are omitted. Returns (h5d, h5f) figure handles.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1 = _load_atlas_50um(data_folder)
    projectedAtlas1, _, coords = _load_outline(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    with h5py.File(
        data_folder / "whisker" / "spirals_peri_stim" / "Whisker_spirals_pre_post.mat", "r"
    ) as f:
        spirals_pre_all = f["spirals_pre_all"][:].T
        spirals_post_all = f["spirals_post_all"][:].T
        pre_refs = f["spirals_pre_cell"][:].ravel()
        post_refs = f["spirals_post_cell"][:].ravel()
        spirals_pre_cell = [f[r][:].T for r in pre_refs]
        spirals_post_cell = [f[r][:].T for r in post_refs]

    indexSSp_right = _select_area(_SENSORY_AREA, st, projectedAtlas1, "right")
    indexSSp_left = _select_area(_SENSORY_AREA, st, projectedAtlas1, "left")

    ratio_post = _ratio_ccw(spirals_post_cell, indexSSp_right)
    ratio_pre = _ratio_ccw(spirals_pre_cell, indexSSp_right)
    ratio_post2 = _ratio_ccw(spirals_post_cell, indexSSp_left)
    ratio_pre2 = _ratio_ccw(spirals_pre_cell, indexSSp_left)

    mouseN = 5
    h5f = plt.figure(figsize=(4, 4))
    ax = h5f.add_subplot(1, 2, 1)
    ax.scatter(np.ones(mouseN), ratio_pre2, s=4, c="k")
    ax.scatter(np.ones(mouseN) * 2, ratio_post2, s=4, c="k")
    for kk in range(mouseN):
        ax.plot([1, 2], [ratio_pre2[kk], ratio_post2[kk]], "k")
    ax.axhline(0.5, color="k", linestyle="--")
    ax.set_ylim(0.4, 0.8)
    ax.set_xticks([1, 2], ["Pre", "Post"])
    ax.set_ylabel("CCW Ratio")
    p3 = stats.ttest_rel(ratio_pre2, ratio_post2).pvalue
    ax.text(1.2, 0.7, f"p = {p3:.4g}")

    ax = h5f.add_subplot(1, 2, 2)
    ax.scatter(np.ones(mouseN), ratio_pre, s=4, c="k")
    ax.scatter(np.ones(mouseN) * 2, ratio_post, s=4, c="k")
    for kk in range(mouseN):
        ax.plot([1, 2], [ratio_pre[kk], ratio_post[kk]], "k")
    ax.axhline(0.5, color="k", linestyle="--")
    ax.set_ylim(0.4, 0.8)
    ax.set_xticks([1, 2], ["Pre", "Post"])
    ax.set_ylabel("CW Ratio")
    p3 = stats.ttest_rel(ratio_pre, ratio_post).pvalue
    ax.text(1.2, 0.7, f"p = {p3:.4g}")
    h5f.tight_layout()
    h5f.savefig(save_folder / "Fig5f_SpiralsRatio.pdf")

    pixSize = 0.01  # mm/pix after registration
    pixArea = pixSize**2
    hist_bin = 40
    trialN = 4000
    hemi = None
    scale3 = 5 / 1
    h5d = plt.figure(figsize=(8, 6))
    for j, spirals_all in enumerate([spirals_pre_all, spirals_post_all]):
        ax = h5d.add_subplot(1, 2, j + 1)
        unique_spirals = _density_color_plot2(spirals_all, hist_bin)
        unique_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)  # counts/mm^2
        unique_unit = unique_unit / trialN / 5 * 35  # spirals/(mm^2*s)
        ax.scatter(
            unique_spirals[:, 0],
            unique_spirals[:, 1],
            s=3,
            c=unique_unit,
            cmap="hot",
            vmin=0,
            vmax=3,
        )
        overlayOutlines(coords, 1, ax=ax)
        plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, "k", ax=ax)
        ax.set_xlim(0, 1140)
        ax.set_ylim(1320, 0)  # MATLAB set(gca,'Ydir','reverse')
        ax.set_aspect("equal")
        ax.axis("off")
        h5d.colorbar(ax.collections[0], ax=ax)
    h5d.savefig(save_folder / "Fig5d_SpiralsPrePost.pdf")

    return h5d, h5f
