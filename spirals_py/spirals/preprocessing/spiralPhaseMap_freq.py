"""Band-passed spiral phase-map reconstruction from SVD components.

Translations of three near-identical MATLAB helpers:
- spirals/utils/spirals_detection/spiralPhaseMap_freq.m  (freq band argument)
- spirals/plots/spiralPhaseMap_freq2.m                   (trace2d returns raw trace)
- ephys/utils/spiralPhaseMap4.m                          (fixed 2-8 Hz band)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert, medfilt2d


def _spiral_phase_map(U, dV, t, params, freq, rate, raw_trace2d=False):
    lowpass = params.get("lowpass", 0)
    gsmooth = params.get("gsmooth", 0)
    nSV = U.shape[2]
    Fs = (1 / np.median(np.diff(t))) * (1 / rate)
    x, y = U.shape[0], U.shape[1]

    # MATLAB reshape(U, x*y, nSV): column-major pixel order
    Ur = U.reshape(-1, nSV, order="F")
    meanTrace = np.asarray(Ur @ dV, dtype=float)  # pixels x time
    tsize = meanTrace.shape[1]

    if gsmooth:
        mt = meanTrace.reshape(x, y, tsize, order="F")
        for kk in range(tsize):
            mt[:, :, kk] = medfilt2d(mt[:, :, kk], 20)
        meanTrace = mt.reshape(-1, tsize, order="F")

    if rate != 1:
        n = int(np.floor((tsize - 1) / rate)) + 1
        tq = 1 + np.arange(n) * rate  # MATLAB 1:rate:tsize
        meanTrace = interp1d(
            np.arange(1, tsize + 1), meanTrace, axis=1, kind="linear"
        )(tq)
        tsize = n

    meanTrace = meanTrace - meanTrace.mean(axis=1, keepdims=True)
    meanRaw = meanTrace

    # MATLAB filters along the time dimension of meanTrace'
    if lowpass:
        f1, f2 = butter(2, 1 / (Fs / 2), btype="low")
    else:
        f1, f2 = butter(2, np.asarray(freq, dtype=float) / (Fs / 2), btype="bandpass")
    # MATLAB filtfilt pads with 3*(max(len(a),len(b))-1) odd samples
    # (scipy's default is 3*max(len(a),len(b)) and also errors when the
    # segment is exactly that long)
    padlen = 3 * (max(len(f1), len(f2)) - 1)
    meanTrace = filtfilt(f1, f2, meanTrace, axis=1, padlen=padlen)

    traceHilbert = hilbert(meanTrace, axis=1)
    tracePhase = np.angle(traceHilbert)
    traceAmp = np.abs(traceHilbert)

    trace2d = meanRaw if raw_trace2d else meanTrace
    # MATLAB reshape(tsize, x, y) then permute([2,3,1]) == F-order reshape here
    tracePhase = tracePhase.reshape(x, y, tsize, order="F")
    traceAmp = traceAmp.reshape(x, y, tsize, order="F")
    trace2d = trace2d.reshape(x, y, tsize, order="F")
    return trace2d, traceAmp, tracePhase


def spiralPhaseMap_freq(U, dV, t, params, freq, rate, mimg=None):
    """Translated from spirals/utils/spirals_detection/spiralPhaseMap_freq.m"""
    if mimg is not None:
        U = U / mimg
    return _spiral_phase_map(U, dV, t, params, freq, rate, raw_trace2d=False)


def spiralPhaseMap_freq2(U, dV, t, params, freq, rate, mimg=None):
    """Translated from spirals/plots/spiralPhaseMap_freq2.m

    Identical to spiralPhaseMap_freq except trace2d is the raw (unfiltered)
    trace.
    """
    if mimg is not None:
        U = U / mimg
    return _spiral_phase_map(U, dV, t, params, freq, rate, raw_trace2d=True)


def spiralPhaseMap4(U, dV, t, params, rate, mimg=None):
    """Translated from ephys/utils/spiralPhaseMap4.m (fixed 2-8 Hz band)."""
    if mimg is not None:
        U = U / mimg
    return _spiral_phase_map(U, dV, t, params, [2, 8], rate, raw_trace2d=False)
