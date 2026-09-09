from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from spirals_py.task.plots._task_helpers2 import load_task_freq_arrays, load_task_outcome


def plotTraceExample2_8Hz(data_folder, save_folder):
    """Translated from task/plots/plotTraceExample2_8Hz.m

    Example session (ZYE_0091): 2-8 Hz filtered traces at pixel 1 for the
    low-contrast trials (|left-right| <= 0.125) of correct/miss trials,
    plus 10 single-trial examples per outcome.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    mn = "ZYE_0091"
    trialWin = [-4, 4]
    freq = [2, 8]

    T_all = load_task_outcome(data_folder, mn)
    arrs = load_task_freq_arrays(
        data_folder / "task" / "task_outcome" / f"{mn}_task_freq_to{freq[1]}Hz.mat",
        ["wf_all", "amp_all", "contrast_all"],
    )
    wf_all = arrs["wf_all"] * 100  # (281, comp, trial)
    amp_all = arrs["amp_all"]
    contrast_all = arrs["contrast_all"]  # (trial, 2)

    t1 = np.arange(trialWin[0], trialWin[1] + 1 / 35, 1 / 35)
    stimOn = int((t1.size - 1) / 2)  # 0-based; MATLAB (numel(t1)-1)/2+1
    amp1 = amp_all[:, 0, :]  # MATLAB squeeze(amp_all(:,1,:))
    amp2 = amp1[122:141, :].sum(axis=0)  # MATLAB 123:141
    p1 = np.percentile(amp2, 90)
    indx = amp2 > p1
    contrast_all1 = contrast_all[indx, :]
    T_all1 = T_all[indx]
    wf_all1 = wf_all[:, 0, indx]  # MATLAB squeeze(wf_all(:,1,indx)) -> (281, nSel)

    trace1_demean = wf_all1 - wf_all1.mean(axis=0)
    trace1_demean = trace1_demean.astype(float)
    Fs = 35
    freq = [2, 8]
    b, a = butter(2, np.asarray(freq) / (Fs / 2), btype="bandpass")
    traceFilt = filtfilt(b, a, trace1_demean, axis=0)
    traceHilbert = hilbert(traceFilt, axis=0)
    tracePhase = np.angle(traceHilbert)
    traceAmp = np.abs(traceHilbert)

    cdiff = np.abs(contrast_all1[:, 0] - contrast_all1[:, 1])
    index = (cdiff > 0) & (cdiff <= 0.125)
    traceRaw = wf_all1[:, index]
    taskR = T_all1[index]
    label_all = taskR["label"].to_numpy()
    scale = 1
    label = ["correct", "miss"]
    color1 = ["k", "r"]
    h5i = plt.figure(figsize=(4.5, 4.5))
    for m in range(2):
        index1 = label_all == label[m]
        traceRaw1 = traceRaw[:, index1]
        N = traceRaw1.shape[1]
        traceRaw2 = traceRaw1[:, :20]
        traceRaw1_mean = traceRaw1.mean(axis=1)

        ax = h5i.add_subplot(3, 2, m + 1)
        ax.plot(t1, traceRaw1_mean, color1[m])
        ax.axvline(t1[stimOn], linestyle="--", color="k")
        ax.set_ylim(-4 * scale, 4 * scale)

        # let's only use 15-32 trials as examples, since they look good
        if m == 0:
            sel = [2, 5, 6, 7, 9, 11, 12, 14, 19, 20]
        else:
            sel = [1, 4, 5, 8, 11, 13, 14, 18, 19, 20]
        traceRaw2 = traceRaw2[:, np.array(sel) - 1]
        ax = h5i.add_subplot(3, 2, (3 + m, 5 + m))
        for i in range(10):
            ax.plot(t1, traceRaw2[:, i] + 10 * i * scale, color1[m])
        ax.plot([-3.5, -3.5], [0, 8], "r")
        ax.axvline(t1[stimOn], linestyle="--", color="k")
        ax.set_ylim(-10, 110)

    h5i.savefig(save_folder / "Fig5i_example_2_8Hz.png", bbox_inches="tight")
    return h5i
