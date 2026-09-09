"""Translated from ephys/plots/plotVarMap.m
(Extended Data Fig.13a: variance explained maps for all sessions)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ._prediction_example_utils import (
    _get_cortex_atlas_path,
    _load_atlas50,
    _load_outline_mat,
    _load_tform,
    _imwarp2d,
    _subplottight,
)
from spirals_py.ephys.utils import get_session_info2
from spirals_py.utils.atlas import plotOutline
from spirals_py.utils.colormaps import inferno


def plotVarMap(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotVarMap.m

    Returns hs13a. (Deviation: the roi .mat file MATLAB loads contains an
    unused graphics object and is skipped.)
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    projectedAtlas1, projectedTemplate1 = _load_outline_mat(data_folder)
    atlas1, _template1 = _load_atlas50(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    BW = (projectedAtlas1 > 0).astype(float)

    area = ["THAL", "STR", "CORTEX", "MB"]

    prediction_folder = data_folder / "ephys" / "dv_prediction"
    regist_folder = data_folder / "ephys" / "rf_tform_4x"

    scale = 1
    scale3 = 5 / scale
    hemi = None
    hs13a = plt.figure(figsize=(10, 3))
    count1 = 1
    mean_var_t = None
    for areai in [1, 2, 4]:
        current_T = T[T["Area"].str.contains(area[areai - 1], na=False).astype(bool)]
        for kk in range(len(current_T)):
            pos = kk + (count1 - 1) * 12
            ax2 = _subplottight(hs13a, 3, 12, pos)
            ops = get_session_info2(current_T, kk, data_folder)
            fname = f"{ops.mn}_{ops.tdb}_{ops.en}"
            with h5py.File(
                prediction_folder / f"{fname}_dv_predict.mat", "r"
            ) as f:
                explained_var_all = np.asarray(f["explained_var_all"][()])
            explained_var_all = explained_var_all.transpose(
                *range(explained_var_all.ndim)[::-1]
            )
            explained_var_all = np.where(explained_var_all < 0, 0, explained_var_all)
            var_mean = explained_var_all.mean(axis=2)
            fname1 = f"{ops.mn}_{ops.tdb}_{ops.en}_tform_4x"
            try:
                T4 = _load_tform(regist_folder / f"{fname1}.mat")
            except (OSError, KeyError, ValueError) as e:
                # some 2024 sessions have empty registration artifacts
                print(f"WARNING: skipping {fname1} (no tform: {e})")
                ax2.remove()
                continue
            mean_var_t = _imwarp2d(var_mean, T4, projectedTemplate1.shape)
            ax2.imshow(mean_var_t, cmap=inferno, alpha=BW, vmin=0, vmax=0.8)
            ax2.set_aspect("equal")
            ax2.axis("off")
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, "k", ax=ax2)
            ax2.text(0, 50, ops.mn)
        count1 += 1

    ax3 = _subplottight(hs13a, 3, 12, 35)
    im2 = ax3.imshow(mean_var_t, cmap=inferno, alpha=BW, vmin=0, vmax=0.8)
    ax3.set_aspect("equal")
    ax3.axis("off")
    plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, "k", ax=ax3)
    cb = plt.colorbar(im2, ax=ax3, fraction=0.046)
    cb.set_ticks([0, 0.4, 0.8])
    cb.set_ticklabels(["0", "0.4", "0.8"])

    hs13a.savefig(save_folder / "FigS13a_all_variance_maps.pdf", bbox_inches="tight")
    return hs13a
