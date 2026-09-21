"""Translated from
revision/frequency_band/preprocessing/getSpiralDensityLine_all.m
(frequency-band revision of Extended Data Fig.3)."""

import numpy as np
from tqdm import tqdm


def getSpiralDensityLine_all(T, data_folder, save_folder):
    """Translated from
    revision/frequency_band/preprocessing/getSpiralDensityLine_all.m

    Runs getSpiralsDensityLine_freq (spiral-density profile along the
    fixed atlas line x = 105:520, y = 780:225) for the frequency bands
    [0.1, 0.2] and [0.5, 2] Hz.

    Bug fixed: the MATLAB iterates `for ifreq = 1` over freq_all =
    [0.05 0.5; 0.5 2; 2 8], i.e. only the 0.05_0.5Hz band - but no
    0.05_0.5Hz grouping exists anywhere (getSpiralsGroup_freq's band
    list is [0.2 0.5; 0.5 2; 2 8], and the release's spirals_freq tree
    only contains 0.1_0.2Hz and 0.5_2Hz).  The released outputs are
    spiralDensityLine_0.1_0.2Hz.mat and spiralDensityLine_0.5_2Hz.mat,
    so those two bands are computed here.
    getSpiralsDensityLine_freq is imported lazily (translated separately).
    """
    freq_all = np.array([[0.1, 0.2], [0.5, 2]])
    from spirals_py.spirals.preprocessing.getSpiralsDensityLine_freq import (
        getSpiralsDensityLine_freq,
    )

    for ifreq in tqdm(range(freq_all.shape[0]), desc="getSpiralDensityLine_all"):
        freq = freq_all[ifreq]
        getSpiralsDensityLine_freq(T, freq, data_folder, save_folder)
