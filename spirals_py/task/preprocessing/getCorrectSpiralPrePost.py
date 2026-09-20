from pathlib import Path

import numpy as np

from spirals_py.task.preprocessing._task_io import (
    load_mat_cell,
    load_mat_var,
    save_mat73,
)
from spirals_py.task.preprocessing._task_utils import (
    TASK_MICE,
    density_color_plot2,
)
from spirals_py.task.preprocessing.combineSpiralsLR import combineSpiralsLR


def getCorrectSpiralPrePost(data_folder, save_folder):
    """Translated from task/preprocessing/getCorrectSpiralPrePost.m

    Concatenate the correct-trial task spirals of all mice over five
    frames before (MATLAB frames 63:67) and after (77:81) stimulus
    onset, count unique spiral centers (density_color_plot2) and save
    them as CorrectSpiralsPrePost.mat (unique_spirals_all as a 1x2
    cell + trialN).

    The atlas / pixel-size loads of the MATLAB source are unused
    downstream and are skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    frames_pre = np.arange(62, 67)  # MATLAB 63:67 (1-based)
    frames_post = np.arange(76, 81)  # MATLAB 77:81 (1-based)
    correct_trialN_all = 0
    spirals_correct_pre_all = []
    spirals_correct_post_all = []

    for mn in TASK_MICE:
        sort_file = (
            data_folder / "task" / "spirals" / f"{mn}_spirals_task_sort.mat"
        )
        spiral_correct_L = load_mat_cell(sort_file, "spiral_correct_L").ravel()
        spiral_correct_R = load_mat_cell(sort_file, "spiral_correct_R").ravel()
        correct_trialN = int(
            load_mat_var(sort_file, "correct_L_trialN").ravel()[0]
        ) + int(load_mat_var(sort_file, "correct_R_trialN").ravel()[0])

        spiral_correct_all = combineSpiralsLR(spiral_correct_L, spiral_correct_R)

        spirals_correct_pre = np.vstack([spiral_correct_all[k] for k in frames_pre])
        spirals_correct_post = np.vstack([spiral_correct_all[k] for k in frames_post])
        spirals_correct_pre_all.append(spirals_correct_pre)
        spirals_correct_post_all.append(spirals_correct_post)
        correct_trialN_all += correct_trialN
        print(f"getCorrectSpiralPrePost: {mn}")

    hist_bin = 40
    unique_spirals_correct_pre = density_color_plot2(
        np.vstack(spirals_correct_pre_all), hist_bin
    )
    unique_spirals_correct_post = density_color_plot2(
        np.vstack(spirals_correct_post_all), hist_bin
    )
    unique_spirals_all = np.empty((1, 2), dtype=object)
    unique_spirals_all[0, 0] = unique_spirals_correct_pre
    unique_spirals_all[0, 1] = unique_spirals_correct_post
    trialN = np.array([[correct_trialN_all, correct_trialN_all]], dtype=float)

    save_mat73(
        save_folder / "CorrectSpiralsPrePost.mat",
        {"unique_spirals_all": unique_spirals_all, "trialN": trialN},
    )
