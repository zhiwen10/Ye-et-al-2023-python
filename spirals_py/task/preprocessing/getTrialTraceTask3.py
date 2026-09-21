from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm

from spirals_py.task.plots._task_helpers import imwarp, loadUVt2_h5, load_tform
from spirals_py.task.plots._task_helpers_s15 import (
    getPhotodiodeTime,
    load_block,
    matlab_round,
)
from spirals_py.task.preprocessing._task_utils import first_index_after, session_dirs


def getTrialTraceTask3(data_folder, T1, win, trialWin, freq):
    """Translated from task/preprocessing/getTrialTraceTask3.m

    Widefield traces at 6 cortical pixels (left/right VISp, left/right
    VISp edge, right SSp-ul, right ALM) around the photodiode stimulus
    onset of every non-repeat trial of the sessions in T1, band-pass
    filtered at *freq* with Hilbert phase and amplitude.

    Returns (wf_all, filt_all, phase_all, amp_all, contrast_all):
    (281, 6, nTrials) arrays (trialWin = [-4, 4] at 35 Hz) and the
    (nTrials, 2) left/right contrasts.

    The dV derivative of the MATLAB source is unused downstream and is
    skipped.
    """
    downscale = 16
    # left/right hemisphere V1, V1 edge, right SSp-ul, right ALM
    Visp = np.array(
        [
            [810, 280],
            [810, 880],
            [900, 280],
            [900, 880],
            [512, 850],
            [340, 705],
        ]
    )
    Visp_ds = matlab_round(Visp / downscale).astype(int)

    nStart = int(-trialWin[0] * 35)
    nEnd = int(trialWin[1] * 35)

    wf_all, filt_all, phase_all, amp_all, contrast_all = [], [], [], [], []

    for kk in tqdm(range(len(T1)), desc="getTrialTraceTask3"):
        fname, session_root = session_dirs(data_folder, T1, kk)
        U, V, t, mimg = loadUVt2_h5(session_root)  # load U, V, t
        block = load_block(session_root)
        # get photodiode time
        allPD2 = getPhotodiodeTime(session_root, win)

        # load atlas registration
        tform = load_tform(Path(data_folder) / "task" / "rfmap" / f"{fname}.mat")
        sizeTemplate = (1320, 1140)
        Ut = imwarp(U, tform, sizeTemplate)
        mimgt = imwarp(mimg, tform, sizeTemplate)
        mimgt = mimgt[::downscale, ::downscale]
        Ut1 = Ut[::downscale, ::downscale, :50]
        dV1 = V[:50, :].astype(float)

        # get all traces for the 6 pixels: wf1 is (time, 6)
        wf1 = np.empty((t.size, 6))
        for j in range(6):
            wfa = Ut1[Visp_ds[j, 0] - 1, Visp_ds[j, 1] - 1, :] @ dV1
            wfa = wfa / mimgt[Visp_ds[j, 0] - 1, Visp_ds[j, 1] - 1]
            wf1[:, j] = wfa

        trace1_demean = wf1 - wf1.mean(axis=0, keepdims=True)
        Fs = 35
        b, a = butter(2, np.asarray(freq, dtype=float) / (Fs / 2), btype="bandpass")
        traceFilt = filtfilt(b, a, trace1_demean, axis=0)
        traceHilbert = hilbert(traceFilt, axis=0)
        tracePhase = np.angle(traceHilbert)
        traceAmp = np.abs(traceHilbert)

        # snippets around every photodiode time
        nPD = allPD2.size
        wf1b = np.empty((nStart + nEnd + 1, 6, nPD))
        filt1b = np.empty_like(wf1b)
        phase1b = np.empty_like(wf1b)
        amp1b = np.empty_like(wf1b)
        indx = first_index_after(t, allPD2)  # 1-based
        for i in range(nPD):
            lo = indx[i] - nStart - 1  # MATLAB indx-nStart (1-based) -> 0-based
            hi = indx[i] + nEnd  # MATLAB indx+nEnd inclusive -> 0-based exclusive
            if lo < 0 or hi > t.size:
                raise IndexError(
                    f"trial window [{lo}, {hi}) outside the session {fname}"
                )
            wf1b[:, :, i] = wf1[lo:hi, :]
            filt1b[:, :, i] = traceFilt[lo:hi, :]
            phase1b[:, :, i] = tracePhase[lo:hi, :]
            amp1b[:, :, i] = traceAmp[lo:hi, :]

        # only non-repeat trials
        ntrial = np.size(block.events.endTrialValues)
        norepeatValues = np.asarray(block.events.repeatNumValues).ravel()[:ntrial]
        norepeat_indx = norepeatValues == 1
        if nPD < ntrial:
            raise IndexError(f"fewer photodiode events than trials in {fname}")

        sel = np.arange(nPD)[:ntrial][norepeat_indx]
        wf_all.append(wf1b[:, :, :ntrial][:, :, norepeat_indx])
        filt_all.append(filt1b[:, :, :ntrial][:, :, norepeat_indx])
        phase_all.append(phase1b[:, :, :ntrial][:, :, norepeat_indx])
        amp_all.append(amp1b[:, :, :ntrial][:, :, norepeat_indx])

        left_contrast = np.asarray(block.events.contrastLeftValues).ravel()[:ntrial]
        right_contrast = np.asarray(block.events.contrastRightValues).ravel()[:ntrial]
        contrast_all.append(
            np.column_stack(
                [left_contrast[norepeat_indx], right_contrast[norepeat_indx]]
            )
        )
        print(f"getTrialTraceTask3: {fname} ({kk + 1}/{len(T1)})")

    return (
        np.concatenate(wf_all, axis=2),
        np.concatenate(filt_all, axis=2),
        np.concatenate(phase_all, axis=2),
        np.concatenate(amp_all, axis=2),
        np.vstack(contrast_all),
    )
