"""Translated from revision/power_spectrum/getPowerBandRatio.m
(pipeline1 revision, Extended Data Fig.1e-f input).

Percentage of 0.3-8 Hz power within the 2-8 Hz band for the 8 hardcoded
area-pixel example traces of each session (same extraction as
getExamplePixelTrace_005_8Hz, on V rather than dV): MATLAB
bandpower(rawTrace', 35, [2 8]) / bandpower(rawTrace', 35, [0.3 8]) *
100, averaged over the 5 SSp pixels.  Saves power_ratio_all_sessions.mat
(power_ratio (4, nSessions), areaNames) for plotPowerRatio3 /
plotPowerRatioRegression3.

MATLAB bandpower(x, fs, freqrange) is approximated by _bandpower: a
one-sided periodogram with a rectangular window of the full data length
(mean removed) integrated with the trapezoid rule over the
endpoint-inclusive band, which is the MATLAB implementation up to the
periodogram detrend/window conventions (the band ratio that is actually
saved is insensitive to them).

Dead code of the MATLAB source skipped: dV, the
horizontal_cortex_atlas_50um.mat / outline / get_cortex_atlas_path /
root1 / ctx / color1 / nameList definitions.
"""

from pathlib import Path

import numpy as np
from scipy.signal import periodogram
from tqdm import tqdm

from spirals_py.spirals.preprocessing.getExamplePixelTrace_005_8Hz import (
    _FS,
    _example_pixel_traces,
)
from spirals_py.utils.matio import save_mat73

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _bandpower(x, fs, freqrange):
    """MATLAB bandpower(x, fs, [f1 f2]) for x (time x channels), per
    column."""
    nfft = x.shape[0]
    f, pxx = periodogram(
        x, fs=fs, window="boxcar", nfft=nfft, detrend="constant", axis=0
    )
    idx = (f >= freqrange[0]) & (f <= freqrange[1])
    return _trapz(pxx[idx], f[idx], axis=0)


def getPowerBandRatio(T, data_folder, save_folder):
    """Translated from revision/power_spectrum/getPowerBandRatio.m"""
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    per_power = np.zeros((8, len(T)))
    for kk in tqdm(range(len(T)), desc="getPowerBandRatio"):
        _, _, rawTrace = _example_pixel_traces(T, data_folder, kk)  # V-based traces

        pband = _bandpower(rawTrace.T, _FS, [2, 8])
        ptot = _bandpower(rawTrace.T, _FS, [0.3, 8])
        per_power[:, kk] = 100 * (pband / ptot)

    power_ratio = np.vstack(
        [
            per_power[0:2, :],
            per_power[2:7, :].mean(axis=0)[None, :],  # average across SSp pixels
            per_power[7:8, :],
        ]
    )
    areaNames = np.array([["VISp", "RSP", "SSp", "MOs"]], dtype=object)
    save_mat73(
        save_folder / "power_ratio_all_sessions.mat",
        {"power_ratio": power_ratio, "areaNames": areaNames},
    )
