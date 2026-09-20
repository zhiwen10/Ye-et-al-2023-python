from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from spirals_py.task.preprocessing._task_io import (
    load_mat_var,
    nanmean,
    save_mat73,
)


def getMeanMapsAll(data_folder, save_folder):
    """Translated from task/preprocessing/getMeanMapsAll.m

    Average the per-mouse mean maps (getMeanMapSession) of each trial
    type across mice, then across the first 3 contrasts (100 / 50 / 25%
    left), and band-pass filter 2-8 Hz with Hilbert phase.  Saved as
    task_mean_maps_all_mice.mat (trace_mean_all / traceFilt_all /
    tracePhase_all, (83, 72, 141, 3)).
    """
    fnames = ["ZYE_0085", "ZYE_0088", "ZYE_0090", "ZYE_0091"]
    labels = ["correct", "incorrect", "miss"]
    freq = [2, 8]
    Fs = 35
    individual = Path(data_folder) / "task" / "task_mean_maps" / "individual"

    trace_mean_all = np.zeros((83, 72, 141, 3))
    traceFilt_all = np.zeros((83, 72, 141, 3))
    tracePhase_all = np.zeros((83, 72, 141, 3))

    for kk, label in enumerate(labels):
        # cat(5, ...) across mice, then mean over mice
        per_mouse = [
            load_mat_var(
                individual / f"{mn}_mean_map_{label}.mat", "trace_correct_mean"
            )
            for mn in fnames
        ]
        trace_correct_mean_all = np.stack(per_mouse, axis=4)  # (83,72,141,11,4)
        trace_correct_mean_all1 = nanmean(trace_correct_mean_all, axis=4)

        # mean across the first 3 contrasts (MATLAB cts = 1:3)
        trace_mean = nanmean(trace_correct_mean_all1[:, :, :, :3], axis=3)
        # MATLAB reshape column-major: (83*72, 141)
        trace_mean2 = trace_mean.reshape(83 * 72, 141, order="F")

        # take care of NaN pixels after registration before filtering
        indx1 = ~np.isnan(trace_mean2[:, 0])
        trace_mean4 = trace_mean2[indx1, :]
        meanTrace = (trace_mean4 - trace_mean4.mean(axis=1, keepdims=True)).T
        b, a = butter(2, np.asarray(freq, dtype=float) / (Fs / 2), btype="bandpass")
        traceFilt = filtfilt(b, a, meanTrace, axis=0)
        traceHilbert = hilbert(traceFilt, axis=0)
        tracePhase = np.angle(traceHilbert)
        tracePhase2 = np.full(trace_mean2.shape, np.nan)
        traceFilt2 = np.full(trace_mean2.shape, np.nan)
        tracePhase2[indx1, :] = tracePhase.T
        traceFilt2[indx1, :] = traceFilt.T

        trace_mean2 = trace_mean2 - trace_mean2.mean(axis=1, keepdims=True)
        trace_mean3 = trace_mean2.reshape(83, 72, 141, order="F")
        traceFilt3 = traceFilt2.reshape(83, 72, 141, order="F")
        tracePhase3 = tracePhase2.reshape(83, 72, 141, order="F")

        trace_mean_all[:, :, :, kk] = trace_mean3
        traceFilt_all[:, :, :, kk] = traceFilt3
        tracePhase_all[:, :, :, kk] = tracePhase3
        print(f"getMeanMapsAll: {label}")

    save_mat73(
        Path(save_folder) / "task_mean_maps_all_mice.mat",
        {
            "trace_mean_all": trace_mean_all,
            "traceFilt_all": traceFilt_all,
            "tracePhase_all": tracePhase_all,
        },
    )
