from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.task.plots._task_helpers2 import (
    load_high_perf_task_sessions,
    load_task_outcome,
)


def plotMeanTraceExampleSlow(data_folder, save_folder):
    """Translated from task/plots/plotMeanTraceExampleSlow.m

    Example session (ZYE_0091): mean 0.05-2 Hz (raw) widefield traces in
    correct (black) and miss (red) trials, per contrast, from the
    trial_trace file (pixel 1, left hemisphere).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    h5f = plt.figure(figsize=(9.5, 2.5))
    mn = "ZYE_0091"
    load_high_perf_task_sessions(data_folder, mn)  # MATLAB reads T1 (unused)

    trialWin = [-4, 4]
    T_all = load_task_outcome(data_folder, mn, varname="T_all")
    with h5py.File(
        data_folder / "task" / "trial_trace" / f"{mn}_task.mat", "r"
    ) as f:
        # stored (nPix, 281, nTrials) -> MATLAB (6, 281, nTrials)
        wf_all = np.asarray(f["wf_all"]).transpose(2, 1, 0)
        contrast_all = np.asarray(f["contrast_all"]).T  # (nTrials, 2)
    wf_all = wf_all * 100
    t1 = np.arange(trialWin[0], trialWin[1] + 1 / 35, 1 / 35)
    stimOn = int((t1.size - 1) / 2)  # 0-based; MATLAB (numel(t1)-1)/2+1
    contrasts = np.array(
        [
            [0.06, 0.125, 0.25, 0.5, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    color1 = ["k", "r"]
    for ii in range(5):
        index = (
            (contrast_all[:, 0] == contrasts[0, ii])
            & (contrast_all[:, 1] == contrasts[1, ii])
        )  # 1 is left, 2 is right
        trace1 = wf_all[:, :, index]  # (6, 281, nSel)
        traceRaw = trace1.transpose(1, 0, 2)  # MATLAB permute [2,1,3]
        taskR = T_all[index]
        scale = 1
        label_all = taskR["label"].to_numpy()
        for m in range(2):
            ax = h5f.add_subplot(1, 10, ii * 2 + m + 1)
            if m == 0:
                label = "correct"
            else:
                label = "miss"
            index1 = label_all == label
            traceRaw1 = traceRaw[:, :, index1]
            traceRaw1_mean = traceRaw1[:, 0, :].mean(axis=1)

            ax.plot(t1, traceRaw1_mean, color1[m])
            ax.axvline(t1[stimOn], linestyle="--", color="k")
            ax.set_ylim(-4 * scale, 4 * scale)
            if m == 0:
                ax.set_title(f"C{int(np.floor(contrasts[0, ii] * 100))}")

    h5f.savefig(save_folder / "Fig5f_mean_trace_example_slow.png", bbox_inches="tight")
    return h5f
