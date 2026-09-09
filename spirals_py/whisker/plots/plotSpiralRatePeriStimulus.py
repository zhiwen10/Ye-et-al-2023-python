"""Translated from whisker/plots/plotSpiralRatePeriStimulus.m (Fig. 5e)"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from spirals_py.utils.plotting import shadedErrorBar


def plotSpiralRatePeriStimulus(data_folder, save_folder):
    """Plot rotating-wave rate over time around whisker stimulation,
    separately for left and right cortex. Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(
        data_folder / "whisker" / "spirals_peri_stim" / "Whisker_spirals_peri_stimulus.mat",
        "r",
    ) as f:
        # v7.3 [frames, bins, mice] -> MATLAB [mice, bins, frames]
        spiral_count_sum_left = f["spiral_count_sum_left"][:].transpose(2, 1, 0)
        spiral_count_sum_right = f["spiral_count_sum_right"][:].transpose(2, 1, 0)

    ylim2 = 12
    mouseN = 5
    t1 = np.linspace(-1, 1, 71)  # MATLAB -1:1/35:1
    h5e = plt.figure(figsize=(4, 2.5))

    ax = h5e.add_subplot(1, 2, 1)
    counts = spiral_count_sum_left.sum(axis=1)  # squeeze(sum(...,2))
    mean_left = counts.mean(axis=0)
    sem_left = counts.std(axis=0) / np.sqrt(mouseN)
    shadedErrorBar(t1, mean_left, sem_left, lineProps="g", ax=ax)
    for t in (t1[35], t1[38], t1[45]):  # MATLAB t1(36), t1(39), t1(46)
        ax.axvline(t, color="k", linestyle="--")
    ax.set_ylim(0, ylim2)
    ax.set_ylabel("Rotating waves/s")

    ax = h5e.add_subplot(1, 2, 2)
    counts = spiral_count_sum_right.sum(axis=1)
    mean_right = counts.mean(axis=0)
    sem_right = counts.std(axis=0) / np.sqrt(mouseN)
    shadedErrorBar(t1, mean_right, sem_right, lineProps="g", ax=ax)
    for t in (t1[35], t1[38], t1[45]):
        ax.axvline(t, color="k", linestyle="--")
    ax.set_ylim(0, ylim2)
    ax.set_ylabel("Rotating waves/s")

    h5e.tight_layout()
    h5e.savefig(save_folder / "Fig5e_WhiskerSpiralsPeriStimulus.pdf")
    return h5e
