"""Translated from
revision/frequency_band/preprocessing/getSpiralsScrambled_freq.m
(frequency-band revision of Extended Data Fig.3)."""

from pathlib import Path

import numpy as np
from tqdm import tqdm


def getSpiralsScrambled_freq(T, data_folder, save_folder):
    """Translated from
    revision/frequency_band/preprocessing/getSpiralsScrambled_freq.m

    For the three frequency bands [0.1 0.2], [0.5 2], [2 8] Hz: runs
    getSpiralGroupingScrambled_freq (frame-scrambled grouping, 10
    permutations per session) into
    <save_folder>/<freq_folder>/<fname>/<fname>_scramble_group_<ii>.mat.

    MATLAB quirk kept: this band list starts at 0.1_0.2Hz while
    getSpiralsGroup_freq groups 0.2_0.5Hz as its slowest band, so the
    scrambled control of the 0.2_0.5Hz band is never produced.
    getSpiralGroupingScrambled_freq is imported lazily (translated
    separately).
    """
    freq_all = np.array([[0.1, 0.2], [0.5, 2], [2, 8]])
    save_folder = Path(save_folder)
    from spirals_py.spirals.preprocessing.getSpiralGroupingScrambled_freq import (
        getSpiralGroupingScrambled_freq,
    )

    for ifreq in tqdm(range(freq_all.shape[0]), desc="getSpiralsScrambled_freq"):
        freq = freq_all[ifreq]
        freq_folder = f"{freq[0]:g}_{freq[1]:g}Hz"
        save_folder1 = save_folder / freq_folder
        getSpiralGroupingScrambled_freq(T, freq, data_folder, save_folder1)
