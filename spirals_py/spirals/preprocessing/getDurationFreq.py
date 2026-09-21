"""Translated from
revision/frequency_band/preprocessing/getDurationFreq.m (frequency-band
revision of Extended Data Fig.3)."""

import numpy as np
from tqdm import tqdm


def getDurationFreq(T, data_folder, save_folder):
    """Translated from
    revision/frequency_band/preprocessing/getDurationFreq.m

    Runs getSpiralDurationRatio_freq (spiral-duration ratios of the
    grouped vs frame-scrambled spirals) for freq = [0.1 0.2] Hz only:
    MATLAB iterates `for ifreq = 1` over freq_all = [0.1 0.2; 0.5 2; 2 8],
    so bands 2-3 are dead code (kept in the array for parity).
    getSpiralDurationRatio_freq is imported lazily (translated
    separately).
    """
    freq_all = np.array([[0.1, 0.2], [0.5, 2], [2, 8]])
    from spirals_py.spirals.preprocessing.getSpiralDurationRatio_freq import (
        getSpiralDurationRatio_freq,
    )

    for ifreq in tqdm(range(1), desc="getDurationFreq"):
        freq = freq_all[ifreq]
        getSpiralDurationRatio_freq(T, freq, data_folder, save_folder)
