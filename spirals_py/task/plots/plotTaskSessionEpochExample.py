from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from spirals_py.task.plots._task_helpers import (
    computeVelocity2,
    correctCounterDiscont,
    imwarp,
    load_tform,
    load_h5_var,
    loadUVt2_h5,
    tsToT,
)
from spirals_py.task.plots._task_helpers2 import (
    getPhotodiodeTime2,
    load_high_perf_task_sessions,
    load_task_outcome,
)
from spirals_py.task.preprocessing.load_task_table import load_task_table


def plotTaskSessionEpochExample(data_folder, save_folder):
    """Translated from task/plots/plotTaskSessionEpochExample.m

    Example task session epoch (ZYE_0091, first high-performance session,
    1920-2020 s): widefield traces at V1 left/right and SSp-ul pixels,
    wheel velocity, trial events (stimulus onset, response/reward,
    contrast) and outcome rasters with low-arousal (miss/repeat) states.

    Adaptations:
    - The *_Block.mat file is a pre-v7.3 .mat and is read with
      scipy.io.loadmat (the MATLAB dir('*_Block.mat') match is
      case-insensitive on Windows).
    - dV, contrasts_unique and trace1_demean of the MATLAB code are
      unused downstream and skipped.
    - getPhotodiodeTime is taken from _task_helpers2 (corrected
      schmittTimes translation).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    win = [0, 5000]
    downscale = 16
    Visp = np.array(
        [
            [810, 280],  # left hemisphere
            [810, 880],  # right hemisphere
            [900, 280],  # left hemisphere
            [900, 880],  # right hemisphere
            [512, 850],  # right SSp-ul
            [768, 640],  # right RSP
        ]
    )
    Visp_ds = np.round(Visp / downscale).astype(int)

    mn = "ZYE_0091"
    T1 = load_high_perf_task_sessions(data_folder, mn)
    kk = 0  # MATLAB kk = 1
    T_all = load_task_outcome(data_folder, mn)
    trial_all = load_task_table(
        data_folder / "task" / "task_outcome" / f"{mn}_task_trial_ID.mat", "trial_all"
    )
    indexa = trial_all["session"].to_numpy() == (kk + 1)
    T_all = T_all[indexa].reset_index(drop=True)
    mn = str(T1.MouseID.iloc[kk])
    tda = pd.Timestamp(T1.date.iloc[kk])
    en = int(T1.folder.iloc[kk])
    td = tda.strftime("%Y-%m-%d")
    tdb = tda.strftime("%Y%m%d")

    # load data and block
    fname = f"{mn}_{tdb}_{en}"
    session_root = data_folder / "task" / "task_svd" / fname
    U, V, t, mimg = loadUVt2_h5(session_root)  # load U, V, t
    block_files = [
        p for p in session_root.glob("*.mat") if p.name.lower().endswith("_block.mat")
    ]
    block = loadmat(block_files[0], squeeze_me=True, struct_as_record=False)["block"]
    # get photodiode time
    allPD2 = getPhotodiodeTime2(session_root, win)

    # rotaryEncoder
    sigName = "rotaryEncoder"
    pd_sig = load_h5_var(session_root / f"{sigName}_raw.mat", "pd").ravel()
    tlTimes = load_h5_var(
        session_root / f"{sigName}_timestamps_Timeline.mat", "tlTimes"
    )
    tt = tsToT(tlTimes, pd_sig.size)
    fs1 = 1 / np.mean(np.diff(tt))
    wh = correctCounterDiscont(pd_sig)
    vel = computeVelocity2(wh, 0.01, fs1)
    vela = np.interp(t, tt, vel)

    ntrial = np.size(block.events.endTrialValues)
    norepeatValues = np.asarray(block.events.repeatNumValues).ravel()[:ntrial]
    response = np.asarray(block.events.responseValues).ravel()[:ntrial]
    norepeat_indx = norepeatValues == 1
    response_time = (
        np.asarray(block.events.responseTimes).ravel()[:ntrial]
        - np.asarray(block.events.stimulusOnTimes).ravel()[:ntrial]
    )

    allPD2 = allPD2[:ntrial]
    left_contrast = np.asarray(block.events.contrastLeftValues).ravel()[:ntrial]
    right_contrast = np.asarray(block.events.contrastRightValues).ravel()[:ntrial]
    contrast_all = np.column_stack([left_contrast, right_contrast])
    # load atlas registration
    tform = load_tform(data_folder / "task" / "rfmap" / f"{fname}.mat")
    sizeTemplate = (1320, 1140)
    Ut = imwarp(U, tform, sizeTemplate)
    mimgt = imwarp(mimg, tform, sizeTemplate)
    mimgt = mimgt[::downscale, ::downscale]
    Ut1 = Ut[::downscale, ::downscale, :50]
    dV1 = V[:50, :].astype(float)

    # get all traces for 4+2 pixels
    wf1 = np.empty((6, t.size))
    for j in range(6):
        wfa = Ut1[Visp_ds[j, 0] - 1, Visp_ds[j, 1] - 1, :] @ dV1
        wfa = wfa / mimgt[Visp_ds[j, 0] - 1, Visp_ds[j, 1] - 1]
        wf1[j, :] = wfa
    wf1 = wf1 * 100

    # get time index for all events
    indx = np.empty(allPD2.size, dtype=int)
    for i in range(allPD2.size):
        ta = t - allPD2[i]
        indx[i] = int(np.argmax(ta > 0))  # MATLAB find(ta>0, 1,'first')

    contrast_all1 = contrast_all.copy()
    contrast_all1[:, 0] = contrast_all1[:, 0] * -1
    contrast_all1 = contrast_all1.sum(axis=1)
    contrast_ttl1 = np.column_stack([indx, contrast_all1])
    contrast_ttl = np.zeros(t.size)
    trial_start = np.zeros(t.size)
    response1 = np.zeros(t.size)
    for i in range(allPD2.size):
        idx = int(contrast_ttl1[i, 0])  # 1-based index into t
        contrast_ttl[idx : idx + 5] = contrast_ttl1[i, 1] * 4
        trial_start[idx : idx + 5] = 4
        if response[i] != 0:
            delay = int(round(response_time[i] * 35))
            response1[idx + delay : idx + 5 + delay] = response[i] * 4

    task_label1 = T_all["label"].to_numpy()
    task_label = np.full(indx.size, "repeat", dtype=object)
    task_label[norepeat_indx] = task_label1

    trial_onset2 = np.full(task_label.size, np.nan)
    trial_onset2[norepeat_indx] = T_all["wheel_onset"].to_numpy(dtype=float)
    trial_onset2 = trial_onset2 + allPD2

    trial_correct = np.full(t.size, np.nan)
    trial_incorrect = np.full(t.size, np.nan)
    trial_miss = np.full(t.size, np.nan)
    trial_falarm = np.full(t.size, np.nan)
    trial_reject = np.full(t.size, np.nan)
    trial_repeat = np.full(t.size, np.nan)
    for i in range(task_label.size):
        current_label = task_label[i]
        idx = int(contrast_ttl1[i, 0])  # 1-based index into t
        target = None
        if current_label == "correct":
            target = trial_correct
        elif current_label == "incorrect":
            target = trial_incorrect
        elif current_label == "miss":
            target = trial_miss
        elif current_label in ("falarmL", "falarmR"):
            target = trial_falarm
        elif current_label == "reject":
            target = trial_reject
        elif current_label == "repeat":
            target = trial_repeat
        if target is not None:
            target[idx - 1 : idx + 5] = 4
            target[idx - 3 : idx - 1] = 0
            target[idx + 5 : idx + 7] = 0

    hs14b = plt.figure(figsize=(10, 4))
    ax = hs14b.add_subplot(1, 1, 1)
    ax.plot(t, wf1[0, :] / 2, "k")
    ax.plot(t, wf1[1, :] / 2 + 5, "k")
    ax.plot(t, wf1[4, :] / 2 + 10, "k")

    ax.plot(t, vela / 500 + 25, "b")
    ax.scatter(trial_onset2, np.full(trial_onset2.size, 25), s=6, c="r", zorder=3)
    ax.plot(t, contrast_ttl + 45, "k")
    ax.plot(t, response1 + 35, "k")

    label_all = ["correct", "incorrect", "miss", "reject", "falarm", "repeat"]
    color_all = ["k", "y", "r", "g", "b", "m"]
    ax.plot(t, trial_start + 55, "k")
    ax.plot(t, trial_correct + 55, "k")
    ax.plot(t, trial_incorrect + 55, "y")
    ax.plot(t, trial_miss + 55, "r")
    ax.plot(t, trial_reject + 55, "g")
    ax.plot(t, trial_falarm + 55, "b")
    ax.plot(t, trial_repeat + 55, "m")

    for x in t[indx - 1]:
        ax.axvline(x, linestyle="--", color="k")

    ax.set_yticks([0, 5, 10, 25, 35, 45, 55])
    ax.set_yticklabels(
        ["V1-L", "V1-R", "SSp-ul", "Wheel", "Reward", "Contrast", "Trial-start"]
    )
    for k, lab in enumerate(label_all):
        hs14b.text(0.0, 0.9 - 0.05 * k, lab, color=color_all[k], transform=hs14b.transFigure)
    tt1 = 1920
    tt2 = 2020
    ax.set_xlim(tt1, tt2)
    hs14b.savefig(
        save_folder / f"FigS14b_{mn}_session{kk + 1}_example_{tt1}to{tt2}s.png",
        bbox_inches="tight",
    )
    return hs14b
