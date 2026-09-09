from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from spirals_py.task.plots._task_helpers import cbrewer2_seq
from spirals_py.task.plots._task_helpers2 import (
    load_task_freq_arrays,
    load_task_outcome,
)


def plotMeanTraceAcrossContrasts(data_folder, save_folder):
    """Translated from task/plots/plotMeanTraceAcrossContrasts.m

    Mean 2-8 Hz responses across contrasts in passive-viewing and task
    sessions: contralateral (Reds) and ipsilateral (Greys) hemisphere
    traces at pixel 124:159, peak dF/F per contrast and contrast legends.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    fname = ["ZYE_0085", "ZYE_0088", "ZYE_0090", "ZYE_0091"]
    freq = [2, 8]
    cs = [0.06, 0.125, 0.25, 0.5, 1]
    t1 = np.arange(-0.5, 0.5 + 1 / 35, 1 / 35)
    pixel_index = np.arange(124, 160) - 1  # MATLAB 124:159 (1-based)

    nT = 281  # trial window is -4:4 s at 35 Hz
    trace_task_contra = np.full((nT, 5, 4), np.nan)
    trace_task_ipsi = np.full((nT, 5, 4), np.nan)
    for m in range(4):
        mn = fname[m]
        T_all = load_task_outcome(data_folder, mn)
        arrs = load_task_freq_arrays(
            data_folder / "task" / "task_outcome" / f"{mn}_task_freq_to{freq[1]}Hz.mat",
            ["wf_all", "contrast_all"],
        )
        wf_all = arrs["wf_all"] * 100  # (281, comp, trial)

        lab = T_all["label"].to_numpy()
        left = T_all["left_contrast"].to_numpy()
        right = T_all["right_contrast"].to_numpy()
        for kk in range(5):
            index1 = (lab == "correct") & ((left - right) == cs[kk])
            trace_all1_conrta = wf_all[:, 1, index1]  # MATLAB wf_all(:,2,index1)
            index2 = (lab == "correct") & ((right - left) == cs[kk])
            trace_all2_conrta = wf_all[:, 0, index2]  # MATLAB wf_all(:,1,index2)
            trace_all3_conrta = np.concatenate(
                [trace_all1_conrta, trace_all2_conrta], axis=1
            )
            trace_task_contra[:, kk, m] = trace_all3_conrta.mean(axis=1)

            trace_all1_ipsi = wf_all[:, 0, index1]
            trace_all2_ipsi = wf_all[:, 1, index2]
            trace_all3_ipsi = np.concatenate(
                [trace_all1_ipsi, trace_all2_ipsi], axis=1
            )
            trace_task_ipsi[:, kk, m] = trace_all3_ipsi.mean(axis=1)

    trace_passive_contra = np.full((nT, 5, 4), np.nan)
    trace_passive_ipsi = np.full((nT, 5, 4), np.nan)
    for m in range(4):
        mn = fname[m]
        arrs = load_task_freq_arrays(
            data_folder
            / "task"
            / "task_outcome"
            / f"{mn}_passive_freq_to{freq[1]}Hz.mat",
            ["wf_all", "contrast_all"],
        )
        wf_all = arrs["wf_all"] * 100
        contrast_all = arrs["contrast_all"]

        cdiff = contrast_all[:, 0] - contrast_all[:, 1]
        for kk in range(5):
            index = cdiff == cs[kk]
            trace_all_contra = wf_all[:, 1, index]
            trace_passive_contra[:, kk, m] = trace_all_contra.mean(axis=1)

            trace_all_ipsi = wf_all[:, 0, index]
            trace_passive_ipsi[:, kk, m] = trace_all_ipsi.mean(axis=1)

    color1 = cbrewer2_seq("Reds", 5)
    color2 = cbrewer2_seq("Greys", 5)
    cmap1 = LinearSegmentedColormap.from_list("Reds5", color1)
    cmap2 = LinearSegmentedColormap.from_list("Greys5", color2)

    def _mean_sem(trace):
        # trace: (nT, 4 mice); MATLAB squeeze(trace(:,i,:)) -> mean/sem over mice
        mean1 = trace.mean(axis=1)
        sem = trace.std(axis=1) / np.sqrt(4)
        return mean1, sem

    hs14cd = plt.figure(figsize=(9, 4))
    ax = hs14cd.add_subplot(2, 4, 1)
    passive_contra_mean = np.empty(5)
    passive_contra_sem = np.empty(5)
    for i in range(5):
        mean1, sem = _mean_sem(trace_passive_contra[:, i, :])
        passive_contra_mean[i] = mean1[145] - mean1[140]  # MATLAB 146 - 141
        passive_contra_sem[i] = sem[145]
        ax.plot(
            t1, mean1[pixel_index], color=color1[i], linewidth=2
        )
    ax.axvline(0.14, color="k")
    ax.set_ylim(-2, 4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F (%)")

    ax = hs14cd.add_subplot(2, 4, 2)
    passive_ipsi_mean = np.empty(5)
    passive_ipsi_sem = np.empty(5)
    for i in range(5):
        mean1, sem = _mean_sem(trace_passive_ipsi[:, i, :])
        passive_ipsi_mean[i] = mean1[145] - mean1[140]
        passive_ipsi_sem[i] = sem[145]
        ax.plot(
            t1, mean1[pixel_index], color=color2[i], linewidth=2
        )
    ax.axvline(0.14, color="k")
    ax.set_ylim(-2, 4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F (%)")

    ax = hs14cd.add_subplot(2, 4, 3)
    ax.errorbar(np.arange(1, 6), passive_contra_mean, yerr=passive_contra_sem, fmt="r")
    ax.errorbar(np.arange(1, 6), passive_ipsi_mean, yerr=passive_ipsi_sem, fmt="k")
    ax.set_xlim(1, 5)
    ax.set_ylim(-1, 3)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xticklabels(["6", "12", "25", "50", "100"])
    ax.set_xlabel("Contrasts(%)")
    ax.set_ylabel("dF/F (%)")

    ax = hs14cd.add_subplot(2, 4, 4)
    im = ax.imshow(np.arange(1, 6).reshape(1, 5), cmap=cmap2, aspect="auto", vmin=1, vmax=5)
    ax.axis("off")
    cb = hs14cd.colorbar(im, ax=ax, ticks=np.arange(1, 6))
    cb.ax.set_yticklabels(["6", "12", "25", "50", "100"])

    ax = hs14cd.add_subplot(2, 4, 5)
    task_contra_mean = np.empty(5)
    task_contra_sem = np.empty(5)
    for i in range(5):
        mean1, sem = _mean_sem(trace_task_contra[:, i, :])
        task_contra_mean[i] = mean1[145] - mean1[140]
        task_contra_sem[i] = sem[145]
        ax.plot(
            t1, mean1[pixel_index], color=color1[i], linewidth=2
        )
    ax.axvline(0.14, color="k")
    ax.set_ylim(-2, 4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F (%)")

    ax = hs14cd.add_subplot(2, 4, 6)
    task_ipsi_mean = np.empty(5)
    task_ipsi_sem = np.empty(5)
    for i in range(5):
        mean1, sem = _mean_sem(trace_task_ipsi[:, i, :])
        task_ipsi_mean[i] = mean1[145] - mean1[140]
        task_ipsi_sem[i] = sem[145]
        ax.plot(
            t1, mean1[pixel_index], color=color2[i], linewidth=2
        )
    ax.axvline(0.14, color="k")
    ax.set_ylim(-2, 4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F (%)")

    ax = hs14cd.add_subplot(2, 4, 7)
    ax.errorbar(np.arange(1, 6), task_contra_mean, yerr=task_contra_sem, fmt="r")
    ax.errorbar(np.arange(1, 6), task_ipsi_mean, yerr=task_ipsi_sem, fmt="k")
    ax.set_xlim(1, 5)
    ax.set_ylim(-1, 3)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xticklabels(["6", "12", "25", "50", "100"])
    ax.set_xlabel("Contrasts(%)")
    ax.set_ylabel("dF/F (%)")

    ax = hs14cd.add_subplot(2, 4, 8)
    im = ax.imshow(np.arange(1, 6).reshape(1, 5), cmap=cmap1, aspect="auto", vmin=1, vmax=5)
    ax.axis("off")
    cb1 = hs14cd.colorbar(im, ax=ax, ticks=np.arange(1, 6))
    cb1.ax.set_yticklabels(["6", "12", "25", "50", "100"])

    hs14cd.savefig(save_folder / "FigS14cd_mean_trace.png", bbox_inches="tight")
    return hs14cd
