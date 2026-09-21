"""Translated from
revision/frequency_band/preprocessing/getSpiralsFFTnFreq.m (frequency-band
revision of Extended Data Fig.3)."""

import numpy as np
from tqdm import tqdm


def getSpiralsFFTnFreq(T, data_folder, save_folder):
    """Translated from
    revision/frequency_band/preprocessing/getSpiralsFFTnFreq.m

    Full FFTN-control pipeline for freq = [0.05 0.5] Hz only: MATLAB
    iterates `for i = 1` over freq_all = [0.05 0.5; 0.5 2; 2 8], so bands
    2-3 are dead code (kept in the array for parity).  For that band,
    runs the spiral detection on the raw (label 'control') and the
    3-D-FFT phase-scrambled (label 'fftn') data, then the per-label
    spiral maps (getFFTNSpiralsMap) and stats (getFFTNSpiralsStats).

    getSpiralDetectionFftnRaw / getSpiralDetectionFftnPermute are
    imported lazily (translated separately).
    """
    freq_all = np.array([[0.05, 0.5], [0.5, 2], [2, 8]])
    label1 = "control"
    label2 = "fftn"
    from spirals_py.spirals.preprocessing.getFFTNSpiralsMap import getFFTNSpiralsMap
    from spirals_py.spirals.preprocessing.getFFTNSpiralsStats import (
        getFFTNSpiralsStats,
    )
    from spirals_py.spirals.preprocessing.getSpiralDetectionFftnPermute import (
        getSpiralDetectionFftnPermute,
    )
    from spirals_py.spirals.preprocessing.getSpiralDetectionFftnRaw import (
        getSpiralDetectionFftnRaw,
    )

    for i in tqdm(range(1), desc="getSpiralsFFTnFreq"):
        freq = freq_all[i]
        getSpiralDetectionFftnRaw(T, freq, data_folder, save_folder)
        getSpiralDetectionFftnPermute(T, freq, data_folder, save_folder)
        getFFTNSpiralsMap(T, freq, label1, data_folder, save_folder)
        getFFTNSpiralsStats(T, freq, label1, data_folder, save_folder)
        getFFTNSpiralsMap(T, freq, label2, data_folder, save_folder)
        getFFTNSpiralsStats(T, freq, label2, data_folder, save_folder)
