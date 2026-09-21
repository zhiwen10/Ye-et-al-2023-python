"""Translated from revision/power_spectrum/setAlphaThreshold.m
(pipeline1 revision, Extended Data Fig.1f input).

Per-session alpha (2-8 Hz) epoch statistics at the SSp-ul example
pixel: the dV-based trace is band-passed (butter(2), filtfilt) and its
Hilbert amplitude smoothed with a 17-frame (0.5 s) movmean; epochs above
threshold = 0.006 lasting > 17 frames are marked by getAlphaEpoch.
Saves alpha_threshold/<fname>_alpha_threshold.mat (threshold,
alpha_epoch2, alpha_binary, alpha_ratio) for the alpha-ratio panel of
plotPowerRatioRegression3.

The MATLAB source is a script run with T / data_folder taken from the
base workspace (the pipeline calls it with no arguments); they are
explicit parameters here.  Adaptations:
- the file is written to <save_folder>/alpha_threshold (the consumer
  reads spirals/spirals_power_spectrum2/alpha_threshold/...), whereas
  the MATLAB script saves into its current folder;
- the two interactive threshold-checking figures are skipped;
- alpha_binary / alpha_epoch2 are stored as double (save_mat73 cannot
  write logicals).

getAlphaEpoch.m (revision/power_spectrum/getAlphaEpoch.m) is translated
here as _get_alpha_epoch; its MATLAB source errors when the trace never
crosses the threshold (epoch_start(1) of an empty array) - that case
returns no epochs instead.  Dead code skipped: rawTraceV (figure only),
the horizontal_cortex_atlas_50um.mat / outline /
get_cortex_atlas_path / root1 / ctx / color1 / nameList definitions.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spirals_py.spirals.preprocessing.getExamplePixelTrace_005_8Hz import (
    _bandpass_hilbert,
    _example_pixel_traces,
)
from spirals_py.utils.matio import save_mat73

_THRESHOLD = 0.006
_MIN_EPOCH = 17  # frames


def _movmean17(x):
    """MATLAB movmean(x, 17): centered window, shrinking at the edges."""
    n = x.size
    h = 17 // 2
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    idx = np.arange(n)
    lo = np.maximum(idx - h, 0)
    hi = np.minimum(idx + h + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def _get_alpha_epoch(traceAmp2, threshold):
    """Translated from revision/power_spectrum/getAlphaEpoch.m.

    Returns (alpha_epoch2 (n, 2) 1-based inclusive epoch bounds kept at
    length > _MIN_EPOCH, alpha_binary (1, n) logical row)."""
    alpha1 = traceAmp2 > threshold
    alpha1_diff = np.concatenate([[0], np.diff(alpha1.astype(int).ravel())])
    epoch_start = np.flatnonzero(alpha1_diff == 1) + 1
    epoch_end = np.flatnonzero(alpha1_diff == -1) + 1

    if epoch_start.size == 0 and epoch_end.size == 0:
        # MATLAB errors here (epoch_start(1) of an empty array); a trace
        # entirely above/below the threshold is one epoch / no epoch
        if alpha1.all():
            return (
                np.array([[1, alpha1.size]], dtype=int),
                np.ones(alpha1.shape, dtype=bool),
            )
        return np.zeros((0, 2), dtype=int), np.zeros(alpha1.shape, dtype=bool)
    if epoch_start.size and (epoch_end.size == 0 or epoch_start[0] < epoch_end[0]):
        pass
    else:
        epoch_start = np.concatenate([[1], epoch_start])
    epoch_n = min(epoch_start.size, epoch_end.size)
    alpha_epoch = np.column_stack([epoch_start[:epoch_n], epoch_end[:epoch_n]])
    epoch_length = alpha_epoch[:, 1] - alpha_epoch[:, 0]
    alpha_epoch2 = alpha_epoch[epoch_length > _MIN_EPOCH]

    alpha_binary = np.zeros(alpha1.shape, dtype=bool)
    for s, e in alpha_epoch2:
        alpha_binary[0, s - 1 : e] = True
    return alpha_epoch2, alpha_binary


def setAlphaThreshold(T, data_folder, save_folder):
    """Translated from revision/power_spectrum/setAlphaThreshold.m"""
    save_folder1 = Path(save_folder) / "alpha_threshold"
    save_folder1.mkdir(parents=True, exist_ok=True)

    for kk in tqdm(range(len(T)), desc="setAlphaThreshold"):
        fname, rawTrace, _ = _example_pixel_traces(T, data_folder, kk)

        rawTrace = rawTrace - rawTrace.mean(axis=1, keepdims=True)
        _, _, traceAmp = _bandpass_hilbert(rawTrace)
        traceAmp2 = _movmean17(traceAmp[2])  # smooth mean of 0.5s (SSp-ul pixel)
        traceAmp3 = traceAmp2[None, :]

        alpha_epoch2, alpha_binary = _get_alpha_epoch(traceAmp3, _THRESHOLD)
        alpha_ratio = alpha_binary.sum() / alpha_binary.size

        save_mat73(
            save_folder1 / f"{fname}_alpha_threshold.mat",
            {
                "threshold": _THRESHOLD,
                "alpha_epoch2": alpha_epoch2,
                "alpha_binary": alpha_binary,
                "alpha_ratio": alpha_ratio,
            },
        )
