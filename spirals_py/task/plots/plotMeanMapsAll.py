from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.task.plots.plotCorrectMapsFlow import (
    _colorcet_c06,
    _get_cortex_atlas_path,
    _load_atlas1,
    _load_task_mask,
    _parula,
    _subplottight,
)
from spirals_py.utils.atlas import plotOutline


def plotMeanMapsAll(data_folder, save_folder):
    """Translated from task/plots/plotMeanMapsAll.m

    Mean widefield maps (raw and 2-8 Hz phase) in correct / incorrect /
    miss trials, from the precomputed task_mean_maps_all_mice.mat.

    Adaptations: the unused loads in the MATLAB source
    (horizontal_cortex_template_50um.mat, isocortex outline) are skipped;
    output is .png instead of MATLAB's -dpdf.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    atlas1 = _load_atlas1(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    t2 = np.arange(-2, 2 + 1e-9, 1 / 35)
    downscale = 16
    scale3 = 5 / 16
    lineColor = "k"
    labels = ["correct", "incorrect", "miss"]

    BW2 = _load_task_mask(data_folder, downscale)
    with h5py.File(
        data_folder / "task" / "task_mean_maps" / "task_mean_maps_all_mice.mat", "r"
    ) as f:
        # MATLAB (83, 72, 141, 3) -> x, y, t, condition
        trace_mean_all = np.asarray(f["trace_mean_all"]).transpose(3, 2, 1, 0)
        tracePhase_all = np.asarray(f["tracePhase_all"]).transpose(3, 2, 1, 0)
    # MATLAB prctile ignores NaNs; plain np.percentile would return NaN
    # for the NaN-masked mean maps (all-dark rendering)
    cmax = np.nanpercentile(trace_mean_all, 99.98)
    cmin = np.nanpercentile(trace_mean_all, 0.02)

    hemi = None
    parula = _parula()
    c06 = _colorcet_c06()
    hs15bdf = plt.figure(figsize=(9.5, 8.5))
    for kk in range(3):
        trace_mean3 = trace_mean_all[:, :, :, kk]
        tracePhase3 = tracePhase_all[:, :, :, kk]
        for i in range(18):
            iframe = 70 + i  # 0-based; MATLAB iframe = 70+i (1-based)

            ax3 = _subplottight(hs15bdf, 6, 19, i + 1 + 2 * 19 * kk)
            ax3.imshow(
                trace_mean3[:, :, iframe], cmap=parula, vmin=cmin, vmax=cmax, alpha=BW2
            )
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax3)
            ax3.set_aspect("equal")
            ax3.axis("off")
            if kk == 0:
                ax3.set_title("%gms" % matlab_round(t2[iframe] * 1000))

            ax4 = _subplottight(hs15bdf, 6, 19, i + 20 + 2 * 19 * kk)
            ax4.imshow(
                tracePhase3[:, :, iframe], cmap=c06, vmin=-np.pi, vmax=np.pi, alpha=BW2
            )
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax4)
            ax4.set_aspect("equal")
            ax4.axis("off")

        if kk == 0:
            cb4 = _subplottight(hs15bdf, 6, 19, 19)
            im_cb = cb4.imshow(
                trace_mean3[:, :, 18], cmap=parula, vmin=cmin, vmax=cmax, visible=False
            )
            cb4.axis("off")
            cb = hs15bdf.colorbar(im_cb, ax=cb4)
            a = cb.ax.get_position()
            cb.ax.set_position([a.x0 - 0.02, a.y0, a.width, a.height / 8])

        hs15bdf.text(
            0.5, 0.9 - kk * 0.34, labels[kk], ha="center", va="bottom"
        )
    hs15bdf.savefig(save_folder / "FigS15bdf_mean_maps_all.png", bbox_inches="tight")
    return hs15bdf
