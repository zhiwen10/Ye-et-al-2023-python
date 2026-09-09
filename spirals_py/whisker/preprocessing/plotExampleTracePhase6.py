"""Translated from whisker/preprocessing/plotExampleTracePhase6.m"""

import numpy as np
import matplotlib.pyplot as plt

from spirals_py.utils.atlas import plotOutline

from ._whisker_utils import _bandpass_phase, _colorcet_c06, _subplottight


def plotExampleTracePhase6(wf_mean2, BW2, maskPath, st, atlas1):
    """Translated from whisker/preprocessing/plotExampleTracePhase6.m

    wf_mean2: [rows, cols, frames, trials] single-trial maps (660x570x141x3).
    matplotlib's viridis stands in for MATLAB's parula colormap.
    Returns (fig, meanTrace2, tracePhase) — the latter two are from the last
    trial, as in MATLAB.
    """
    lineColor = "k"
    hemi = None
    scale3 = 5 / 2
    alpha = BW2.astype(float)
    cmap_c06 = _colorcet_c06()

    fig = plt.figure(figsize=(9.5, 4.0))
    meanTrace2 = tracePhase = None
    for kk in range(3):
        wf_mean = wf_mean2[:, :, :, kk]
        meanTrace2, tracePhase = _bandpass_phase(wf_mean)
        cmax, cmin = 0.03, -0.03
        for i in range(1, 20):
            idx = 66 + i - 1  # MATLAB frame 66+i (1-based) -> 0-based

            ax1 = _subplottight(fig, 10, 19, kk * 19 + i)
            im_raw = ax1.imshow(
                wf_mean[:, :, idx], cmap="viridis", vmin=cmin, vmax=cmax, alpha=alpha
            )
            ax1.axis("off")
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax1)
            if i == 19:
                fig.colorbar(im_raw, ax=ax1)

            ax2 = _subplottight(fig, 10, 19, (kk + 3) * 19 + i)
            ax2.imshow(
                tracePhase[:, :, idx], cmap=cmap_c06, vmin=-np.pi, vmax=np.pi, alpha=alpha
            )
            ax2.axis("off")
            plotOutline(maskPath[0:11], st, atlas1, hemi, scale3, lineColor, ax=ax2)

    return fig, meanTrace2, tracePhase
