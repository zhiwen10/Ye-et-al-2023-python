"""Phase-scrambled spiral phase-map reconstruction from SVD components.

Translated from spirals/utils/spirals_detection/spiralPhaseMap_fftn_freq.m
(3-D FFT phase randomization control used by Extended Data Fig.3).
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert


def spiralPhaseMap_fftn_freq(U1, dV1, t, params, freq, rate, rng=None):
    """Translated from spirals/utils/spirals_detection/spiralPhaseMap_fftn_freq.m

    Returns (trace2d, traceAmp, tracePhase), each (x, y, tsize). The pixel
    phase of the 3-D FFT is randomly permuted (MATLAB randperm) before the
    inverse transform; pass a numpy Generator as rng for reproducibility.
    """
    lowpass = params.get("lowpass", 0)
    nSV = U1.shape[2]
    Fs = (1 / np.median(np.diff(t))) * (1 / rate)
    x, y = U1.shape[0], U1.shape[1]

    Ur = U1.reshape(x * y, nSV, order="F")
    meanTrace = np.asarray(Ur @ dV1, dtype=float)
    tsize = meanTrace.shape[1]

    meanTrace = meanTrace.reshape(x, y, tsize, order="F")
    data_fft = np.fft.fftn(meanTrace)
    phase = np.angle(data_fft)
    if rng is None:
        rng = np.random.default_rng()
    indx = rng.permutation(phase.size)
    phase1 = phase.ravel(order="F")[indx].reshape(phase.shape, order="F")
    mag = np.abs(data_fft)
    meanTrace1 = np.real(np.fft.ifftn(mag * np.exp(1j * phase1)))
    del data_fft, phase, mag
    meanTrace1 = meanTrace1.reshape(x * y, tsize, order="F")

    if rate != 1:
        tq = np.arange(1, tsize + 1, rate)
        meanTrace1 = interp1d(
            np.arange(1, tsize + 1), meanTrace1, axis=1, kind="linear"
        )(tq)
        tsize = tq.size

    meanTrace1 = meanTrace1 - meanTrace1.mean(axis=1, keepdims=True)
    meanTrace1 = meanTrace1.T  # tsize x pixels; filters act along columns

    if lowpass:
        b1, a1 = butter(2, 0.2 / (Fs / 2), btype="low")
    else:
        b1, a1 = butter(
            2, np.asarray(freq, dtype=float) / (Fs / 2), btype="bandpass"
        )
    meanTrace1 = filtfilt(b1, a1, meanTrace1, axis=0)

    traceHilbert = hilbert(meanTrace1, axis=0)
    tracePhase = np.angle(traceHilbert)
    traceAmp = np.abs(traceHilbert)
    trace2d = meanTrace1

    def _to_xyz(m):
        # MATLAB reshape(tsize,x,y) + permute([2,3,1])
        return m.T.reshape(x, y, tsize, order="F")

    return _to_xyz(trace2d), _to_xyz(traceAmp), _to_xyz(tracePhase)
