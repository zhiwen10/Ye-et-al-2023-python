import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def trace_filt_nan(trace_mean):
    """Translated from task/preprocessing/trace_filt_nan.m

    trace_mean shape: x * y * t. Bandpass 2-8 Hz, then Hilbert phase.
    NaN pixels (after registration) are excluded from filtering and stay NaN.
    """
    x, y, t = trace_mean.shape
    trace_mean2 = trace_mean.reshape(x * y, t, order="F")
    Fs = 35
    freq = [2, 8]
    indx1 = ~np.isnan(trace_mean2[:, 0])
    trace_mean4 = trace_mean2[indx1, :]
    meanTrace = (trace_mean4 - trace_mean4.mean(axis=1, keepdims=True)).T
    b, a = butter(2, np.asarray(freq) / (Fs / 2), btype="bandpass")
    traceFilt = filtfilt(b, a, meanTrace, axis=0)
    tracePhase = np.angle(hilbert(traceFilt, axis=0))
    tracePhase2 = np.full(trace_mean2.shape, np.nan)
    traceFilt2 = np.full(trace_mean2.shape, np.nan)
    tracePhase2[indx1, :] = tracePhase.T
    traceFilt2[indx1, :] = traceFilt.T
    traceFilt2 = traceFilt2.reshape(x, y, t, order="F")
    tracePhase2 = tracePhase2.reshape(x, y, t, order="F")
    return traceFilt2, tracePhase2
