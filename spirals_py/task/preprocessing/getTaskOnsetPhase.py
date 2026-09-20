from pathlib import Path

from spirals_py.task.preprocessing._task_io import save_mat73
from spirals_py.task.preprocessing._task_utils import TASK_MICE, load_task_sessions
from spirals_py.task.preprocessing.getTrialTraceTask3 import getTrialTraceTask3


def getTaskOnsetPhase(data_folder, save_folder, freq):
    """Translated from task/preprocessing/getTaskOnsetPhase.m

    Phase and amplitude at 6 cortical pixels around stimulus onset
    ([-4, 4] s at 35 Hz) for every non-repeat trial, band-pass filtered
    at *freq*, saved as [mouse]_task_freq_to[freq[1]]Hz.mat.

    The MATLAB source also loads task/task_outcome/[mouse]_task_outcome
    but never uses it; that load is skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for mn in TASK_MICE:
        T1 = load_task_sessions(data_folder, mn)
        win = [0, 5000]
        trialWin = [-4, 4]
        wf_all, filt_all, phase_all, amp_all, contrast_all = getTrialTraceTask3(
            data_folder, T1, win, trialWin, freq
        )
        save_mat73(
            save_folder / f"{mn}_task_freq_to{freq[1]:g}Hz.mat",
            {
                "wf_all": wf_all,
                "filt_all": filt_all,
                "phase_all": phase_all,
                "amp_all": amp_all,
                "contrast_all": contrast_all,
            },
        )
