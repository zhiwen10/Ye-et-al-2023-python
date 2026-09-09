from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spirals_py.task.plots._task_helpers import (
    imwarp,
    loadUVt2_h5,
    load_tform,
)
from spirals_py.task.plots._task_helpers_s15 import (
    getPhotodiodeTime,
    load_block,
    load_projectedAtlas1,
)
from spirals_py.task.plots.plotCorrectMapsFlow import (
    _get_cortex_atlas_path,
    _load_atlas1,
    _parula,
    _subplottight,
)
from spirals_py.utils.atlas import plotOutline


def plotSpiralTrialExample(data_folder, save_folder):
    """Translated from task/plots/plotSpiralTrialExample.m

    Single-trial widefield maps for 3 trials with spirals and 3 trials
    without (example session ZYE_0085, task session 6).

    Adaptations:
    - the MATLAB source loads ZYE_0085_half_spirals4.mat,
      ZYE_0085_spirals_task_sort.mat (260 MB), the task_outcome and
      task_trial_ID files, but never uses them downstream; those loads
      are skipped.
    - only the first 50 SVD components are warped to atlas space (only
      those are used; saves ~1 GB of transient memory).
    - the colorbar is drawn once instead of once per trial (identical
      position and content in the MATLAB source).
    - output is .png instead of MATLAB's -dpdf.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    atlas1 = _load_atlas1(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    mn = "ZYE_0085"
    session = 6  # 1-based row into the filtered session table
    T_session = pd.read_excel(data_folder / "task" / "sessions" / f"{mn}.xlsx")
    T1 = T_session[T_session.label == "task"]
    T1 = T1[(T1.hit_left > 0.7) & (T1.hit_right > 0.7)]
    row = T1.iloc[session - 1]
    fname = "{}_{}_{}".format(
        row.MouseID, pd.Timestamp(row.date).strftime("%Y%m%d"), int(row.folder)
    )
    session_root = data_folder / "task" / "task_svd" / fname

    U, V, t, mimg = loadUVt2_h5(session_root)
    ncomp = 50
    dV = np.hstack(
        [np.zeros((V.shape[0], 1)), np.diff(V, axis=1)]
    )  # derivative of V
    dV1 = dV[:ncomp, :].astype(float)

    block = load_block(session_root)
    win = [0, 3000]
    allPD2 = getPhotodiodeTime(session_root, win)

    ntrial = np.size(block.events.endTrialValues)
    # sometimes, last trial doesn't have a response
    norepeatValues = np.ravel(block.events.repeatNumValues)[:ntrial]
    norepeat_indx = norepeatValues == 1
    allPD2 = np.ravel(allPD2)[:ntrial][norepeat_indx]

    # time index of each trial start
    indx2 = np.empty(allPD2.size, dtype=int)
    for i in range(allPD2.size):
        ta = t - allPD2[i]
        indx2[i] = np.argmax(ta > 0)  # MATLAB find(ta>0,1,'first')

    # atlas registration
    downscale = 16
    tform = load_tform(data_folder / "task" / "rfmap" / f"{fname}.mat")
    sizeTemplate = (1320, 1140)
    Ut = imwarp(U[:, :, :ncomp], tform, sizeTemplate)
    mimgt = imwarp(mimg, tform, sizeTemplate)
    mimgt = mimgt[::downscale, ::downscale]
    Ut1 = Ut[::downscale, ::downscale, :]

    BW3 = load_projectedAtlas1(data_folder)[::downscale, ::downscale].astype(bool)
    hemi = None
    scale3 = 5 / 16
    lineColor = "k"

    indx = [576, 623, 608, 54, 65, 136]  # 3 spiral examples, 3 no-spiral examples
    cmax = 0.03
    cmin = -0.03
    hs15gh = plt.figure(figsize=(9.5, 4))
    parula = _parula()
    ny, nx, _ = Ut1.shape
    Ut1a = Ut1.reshape(ny * nx, Ut1.shape[2], order="F").astype(float)
    for kk in range(len(indx)):
        current_trial = indx[kk] - 1  # 1-based -> 0-based
        it = indx2[current_trial]
        # MATLAB dV1(:, indx_t-4:indx_t+32) with 1-based indx_t = it+1
        trace = Ut1a @ dV1[:, it - 4 : it + 33]
        # divide by mean image; 0/0 -> NaN outside the FOV, masked by BW3
        with np.errstate(divide="ignore", invalid="ignore"):
            trace2d = trace.reshape(ny, nx, -1, order="F") / mimgt[:, :, None]

        if kk < 3:
            row_i = kk + 1
        else:
            row_i = kk + 2  # 1-based row in the 10 x 19 grid
        for i in range(18):
            ax4 = _subplottight(hs15gh, 10, 19, (row_i - 1) * 19 + i + 1)
            ax4.imshow(
                trace2d[:, :, i], cmap=parula, vmin=cmin, vmax=cmax, alpha=BW3
            )
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax4)
            ax4.set_aspect("equal")
            ax4.axis("off")

    cb4 = _subplottight(hs15gh, 10, 19, 19)
    im_cb = cb4.imshow(
        trace2d[:, :, 18], cmap=parula, vmin=cmin, vmax=cmax, visible=False
    )
    cb4.axis("off")
    cb = hs15gh.colorbar(im_cb, ax=cb4)
    a = cb.ax.get_position()
    cb.ax.set_position([a.x0 - 0.02, a.y0, a.width, a.height / 8])

    hs15gh.text(0.5, 0.9, "Spiral", ha="center", va="top")
    hs15gh.text(0.5, 0.5, "No spiral", ha="center", va="top")
    hs15gh.savefig(
        save_folder / "FigS15gh_example_spiral_nospiral.png", bbox_inches="tight"
    )
    return hs15gh
