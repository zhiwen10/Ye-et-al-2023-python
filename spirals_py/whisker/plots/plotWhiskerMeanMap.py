"""Translated from whisker/plots/plotWhiskerMeanMap.m (Fig. 5b)"""

from pathlib import Path

import h5py
import numpy as np

from spirals_py.whisker.preprocessing._whisker_utils import (
    _get_cortex_atlas_path,
    _imresize,
    _load_atlas_50um,
    _load_outline,
)
from spirals_py.whisker.preprocessing.plotMeanTracePhase7 import plotMeanTracePhase7


def plotWhiskerMeanMap(data_folder, save_folder):
    """Plot mean whisker-evoked maps across 5 mice (raw frames, phase, and
    zoomed maps with detected spirals overlaid).

    The redundant bandpass/Hilbert pre-computation in the MATLAB original
    (its results are overwritten inside plotMeanTracePhase7) is omitted.
    Returns the figure handle.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1 = _load_atlas_50um(data_folder)
    projectedAtlas1, _, _ = _load_outline(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    BW = projectedAtlas1 != 0
    BW2 = BW[::2, ::2]

    with h5py.File(
        data_folder / "whisker" / "whisker_mean_maps" / "whisker_spirals_mean_all.mat", "r"
    ) as f:
        wf_mean2 = f["wf_mean2"][:].transpose(2, 1, 0)  # v7.3 -> [row, col, frame]
    wf_mean3 = _imresize(wf_mean2, (660, 570))

    with h5py.File(
        data_folder / "whisker" / "whisker_mean_maps" / "whisker_spirals_group_fftn.mat", "r"
    ) as f:
        cells = f["archiveCell"][:].ravel()
        spirals = np.vstack([f[r][:].T for r in cells])  # MATLAB cell2mat(archiveCell)

    fig, meanTrace2, tracePhase = plotMeanTracePhase7(
        wf_mean3, BW2, maskPath, st, atlas1, spirals
    )
    fig.savefig(save_folder / "Fig5b_WhiskerEvokedMeanMaps.pdf")
    return fig
