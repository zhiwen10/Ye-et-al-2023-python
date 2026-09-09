import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert


def taskTrace_upsample(trace_mean_correct, freq, rate):
    """Translated from task/preprocessing/taskTrace_upsample.m

    Upsample traces in time by 1/rate, bandpass filter at freq (Hz) and
    compute the Hilbert phase. trace_mean_correct shape: x * y * t.
    """
    x, y, tsize = trace_mean_correct.shape
    meanTrace = trace_mean_correct.reshape(x * y, tsize, order="F")
    Fs = 35 / rate
    tq = 1 + np.arange(int(round((tsize - 1) / rate)) + 1) * rate  # MATLAB 1:rate:tsize
    meanTrace = interp1d(
        np.arange(1, tsize + 1), meanTrace.T, axis=0, kind="linear"
    )(tq).T
    tsize2 = tq.size
    indx1 = ~np.isnan(meanTrace[:, 0])
    trace_mean4 = meanTrace[indx1, :]
    meanTrace2 = (trace_mean4 - trace_mean4.mean(axis=1, keepdims=True)).T
    b, a = butter(2, np.asarray(freq) / (Fs / 2), btype="bandpass")
    traceFilt = filtfilt(b, a, meanTrace2, axis=0)
    tracePhase = np.angle(hilbert(traceFilt, axis=0))
    tracePhase2 = np.full(meanTrace.shape, np.nan)
    traceFilt2 = np.full(meanTrace.shape, np.nan)
    tracePhase2[indx1, :] = tracePhase.T
    traceFilt2[indx1, :] = traceFilt.T
    meanTrace = meanTrace - meanTrace.mean(axis=1, keepdims=True)
    trace_mean3 = meanTrace.reshape(x, y, tsize2, order="F")
    traceFilt3 = traceFilt2.reshape(x, y, tsize2, order="F")
    tracePhase3 = tracePhase2.reshape(x, y, tsize2, order="F")
    return trace_mean3, traceFilt3, tracePhase3
