"""Translated from revision2/prediction_motion_energy/sort_sprials_by_arousal.m
(Extended Data Fig.13g: spiral density sorted by arousal phase)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._ephys_helpers import (
    filter_facevideo_table,
    phaseSpiralHistogram2,
    spiralDensityBins2,
)
from ._prediction_example_utils import _load_outline_mat
from spirals_py.utils.plotting import shadedErrorBar


def sort_sprials_by_arousal(T, data_folder, save_folder):
    """Original MATLAB file:
    revision2/prediction_motion_energy/sort_sprials_by_arousal.m

    Returns hs13g. Note: the data_folder argument takes precedence over
    the hard-coded path in the MATLAB source; the facevideo table filter
    is skipped when the session table lacks that column.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    projectedAtlas1, _projectedTemplate1 = _load_outline_mat(data_folder)
    BW = projectedAtlas1 > 0  # atlas brain boundary binary mask
    rows, cols = np.nonzero(BW)
    brain_index = np.column_stack([cols + 1, rows + 1])  # [col, row], 1-based

    area = ["THAL", "STR", "CORTEX", "MB"]

    T = filter_facevideo_table(T)

    n_bins = 18
    low_freq_band = [0.05, 0.5]  # phase-providing frequency (slow oscillation)
    spirals_sort = phaseSpiralHistogram2(T, data_folder, n_bins, brain_index, low_freq_band)
    count_sample = spiralDensityBins2(T, data_folder, spirals_sort)

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

    color2 = ["g", "r", "c", "m"]
    hs13g = plt.figure(figsize=(6, 3))
    count1 = 0
    for kk in [1, 2, 4]:  # 'THAL', 'STR'; 'MB'
        count1 += 1
        ax = hs13g.add_subplot(1, 3, count1)
        indx = T["Area"].str.contains(area[kk - 1], na=False).to_numpy()
        count_sample_temp = count_sample[indx, :]
        for session in range(count_sample_temp.shape[0]):
            spiral = count_sample_temp[session, :]
            ax.plot(phase_centers, spiral, color=[0.5, 0.5, 0.5])
        ax.set_xticks(np.arange(-np.pi, np.pi + 1e-9, np.pi / 2))
        ax.set_xticklabels(["-pi", "-1/2*pi", "0", "1/2*pi", "pi"])
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, 6)
        ax.set_yticks(np.arange(0, 6 + 1e-9, 2))
        count_mean = count_sample_temp.mean(axis=0)
        count_sem = count_sample_temp.std(axis=0, ddof=1) / np.sqrt(
            count_sample_temp.shape[0]
        )
        shadedErrorBar(
            phase_centers, count_mean, count_sem,
            lineProps=f"-{color2[kk - 1]}", ax=ax,
        )
        ax.set_xlabel("Phase of face motion")
        ax.set_ylabel("Peak density (Centers/mm2*s)")

    hs13g.savefig(save_folder / "FigS13g_spirals_arousal.pdf", bbox_inches="tight")
    return hs13g
