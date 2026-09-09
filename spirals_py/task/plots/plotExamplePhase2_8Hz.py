from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.task.plots._task_helpers2 import (
    load_task_freq_arrays,
    load_task_outcome,
)


def plotExamplePhase2_8Hz(data_folder, save_folder):
    """Translated from task/plots/plotExamplePhase2_8Hz.m

    Example session (ZYE_0091): mean 2-8 Hz responses per contrast (left
    column = left hemisphere, right column = right hemisphere), 20
    single-trial examples of high 2-8 Hz amplitude correct trials,
    left-vs-right hemisphere onset-phase scatter and phase histograms.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    freq_high = [2, 8]
    freq_low = [0.05, 2]
    mn = "ZYE_0091"
    trialWin = [-4, 4]
    T_all = load_task_outcome(data_folder, mn)
    arrs = load_task_freq_arrays(
        data_folder / "task" / "task_outcome" / f"{mn}_task_freq_to{freq_high[1]}Hz.mat",
        ["wf_all", "phase_all", "amp_all"],
    )
    amp_high = arrs["amp_all"]
    phase_all = arrs["phase_all"]  # (281, comp, trial)
    wf_all = arrs["wf_all"] * 100
    amp_high1 = amp_high[122:141, 0, :].mean(axis=0)  # MATLAB 123:141
    p_high1 = np.percentile(amp_high1, 75)
    contrasts = np.array(
        [
            [0.06, 0.125, 0.25, 0.5, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.06, 0.125, 0.25, 0.5, 1],
        ]
    )
    t1 = np.arange(trialWin[0], trialWin[1] + 1 / 35, 1 / 35)
    stimOn = int((t1.size - 1) / 2)  # 0-based; MATLAB (numel(t1)-1)/2+1

    xlim1 = 2
    hs14efg = plt.figure(figsize=(8.5, 5.5))
    lab = T_all["label"].to_numpy()
    left = T_all["left_contrast"].to_numpy()
    right = T_all["right_contrast"].to_numpy()
    for i in range(5):
        indx = (
            (left == contrasts[0, i])
            & (right == contrasts[1, i])
            & (lab == "correct")
        )
        wf_temp = wf_all[:, 0:2, indx]
        wf_mean = wf_temp.mean(axis=2)  # (281, 2)

        indx1 = (
            (left == contrasts[0, i])
            & (right == contrasts[1, i])
            & (lab == "correct")
            & (amp_high1 > p_high1)
        )
        wf_temp1 = wf_all[:, 0:2, indx1]

        ax = hs14efg.add_subplot(5, 10, (i) * 2 + 1)
        ax.plot(t1, wf_mean[:, 0], "k")  # left hemisphere
        ax.axvline(t1[stimOn], linestyle="--", color="k")
        ax.set_xlim(-xlim1, xlim1)
        ax.set_ylim(-3, 3)

        ax = hs14efg.add_subplot(5, 10, (i) * 2 + 2)
        ax.plot(t1, wf_mean[:, 1], "r")  # right hemisphere
        ax.axvline(t1[stimOn], linestyle="--", color="k")
        ax.set_xlim(-xlim1, xlim1)
        ax.set_ylim(-3, 3)

        ax = hs14efg.add_subplot(5, 10, ((i) * 2 + 11, (i) * 2 + 21))
        for k in range(20):
            ax.plot(t1, wf_temp1[:, 0, k] + k * 6, "k")  # left hemisphere
        ax.axvline(t1[stimOn], linestyle="--", color="k")
        ax.set_xlim(-xlim1, xlim1)
        ax.set_ylim(-20, 140)
        ax.set_yticklabels([])
        if i == 0:
            ax.plot([0, 0], [0, 8], "r")

        ax = hs14efg.add_subplot(5, 10, ((i) * 2 + 12, (i) * 2 + 22))
        for k in range(20):
            ax.plot(t1, wf_temp1[:, 1, k] + k * 6, "r")  # right hemisphere
        ax.axvline(t1[stimOn], linestyle="--", color="k")
        ax.set_xlim(-xlim1, xlim1)
        ax.set_ylim(-20, 140)
        ax.set_yticklabels([])

        # note: phase_all[140, 0:2][:, indx] -- numpy puts the boolean-indexed
        # axis first when advanced indices are separated by a slice
        phase_temp = phase_all[140, 0:2][:, indx]  # MATLAB phase_all(141,1:2,indx)
        ax = hs14efg.add_subplot(5, 10, ((i) * 2 + 31, (i) * 2 + 32))
        ax.scatter(
            phase_temp[0, :], phase_temp[1, :], s=6, c="k"  # left hemisphere
        )
        ax.set_aspect("equal")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
        ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])

        ax = hs14efg.add_subplot(5, 10, ((i) * 2 + 41, (i) * 2 + 42))
        edges = np.arange(-np.pi, np.pi + np.pi / 4, np.pi / 4)
        ax.hist(
            phase_temp[0, :],
            bins=edges,
            edgecolor="k",
            facecolor=[0.5, 0.5, 0.5],
        )
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, 150)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])

    hs14efg.savefig(
        save_folder / "FigS14efg_phase_histogram_2_8Hz.png", bbox_inches="tight"
    )
    return hs14efg
