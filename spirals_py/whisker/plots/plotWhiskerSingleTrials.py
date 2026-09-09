"""Translated from whisker/plots/plotWhiskerSingleTrials.m (Fig. S15a)"""

from pathlib import Path

import h5py

from spirals_py.whisker.preprocessing._whisker_utils import (
    _get_cortex_atlas_path,
    _imresize,
    _load_atlas_50um,
    _load_outline,
)
from spirals_py.whisker.preprocessing.plotExampleTracePhase6 import plotExampleTracePhase6


def plotWhiskerSingleTrials(data_folder, save_folder):
    """Plot example single trials (raw frames and phase) with spirals.
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    atlas1 = _load_atlas_50um(data_folder)
    projectedAtlas1, _, _ = _load_outline(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    BW = projectedAtlas1 != 0
    BW2 = BW[::2, ::2]

    with h5py.File(data_folder / "whisker" / "single_trials" / "single_trial_maps2.mat", "r") as f:
        # v7.3 [trial, frame, col, row] -> MATLAB [row, col, frame, trial]
        wf1 = f["wf1"][:].transpose(3, 2, 1, 0)
    wf2 = _imresize(wf1, (660, 570))

    fig, _, _ = plotExampleTracePhase6(wf2, BW2, maskPath, st, atlas1)
    fig.savefig(save_folder / "figS15a_ZYE_0092_example_trials2.pdf")
    return fig
