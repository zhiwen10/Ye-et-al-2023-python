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
from spirals_py.task.preprocessing.getConcatTrials import getConcatTrials
from spirals_py.utils.paths import out_root, release_twin


def getPassiveSpiralPrePost(data_folder, save_folder):
    """Translated from task/preprocessing/getPassiveSpiralPrePost.m

    Concatenate the high-contrast passive-viewing spirals of all mice
    over ten frames before (MATLAB frames 59:68) and after (75:84)
    stimulus onset, count unique spiral centers (density_color_plot2)
    and save them as PassiveSpiralsPrePost.mat (unique_spirals_all as
    a 1x2 cell + trialN).

    The atlas / pixel-size loads of the MATLAB source are unused
    downstream and are skipped.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    frames_pre = np.arange(58, 68)  # MATLAB 59:68 (1-based)
    frames_post = np.arange(74, 84)  # MATLAB 75:84 (1-based)
    high_trialN_all = 0
    spirals_high_pre_all = []
    spirals_high_post_all = []

    for mn in TASK_MICE:
        sort_file = release_twin(
            out_root() / "task" / "spirals" / f"{mn}_spirals_passive_sort.mat",
            data_folder,
        )
        spiral_high_stimL = load_mat_cell(sort_file, "spiral_high_stimL")
        spiral_high_stimR = load_mat_cell(sort_file, "spiral_high_stimR")
        high_trialN = int(
            load_mat_var(sort_file, "spiral_high_stimL_trialN").ravel()[0]
        ) + int(load_mat_var(sort_file, "spiral_high_stimR_trialN").ravel()[0])

        spiral_high_L = getConcatTrials(spiral_high_stimL)
        spiral_high_R = getConcatTrials(spiral_high_stimR)
        spiral_high_all = combineSpiralsLR(spiral_high_L, spiral_high_R)

        spirals_high_pre = np.vstack([spiral_high_all[k] for k in frames_pre])
        spirals_high_post = np.vstack([spiral_high_all[k] for k in frames_post])
        spirals_high_pre_all.append(spirals_high_pre)
        spirals_high_post_all.append(spirals_high_post)
        high_trialN_all += high_trialN
        print(f"getPassiveSpiralPrePost: {mn}")

    hist_bin = 40
    unique_spirals_high_pre = density_color_plot2(
        np.vstack(spirals_high_pre_all), hist_bin
    )
    unique_spirals_high_post = density_color_plot2(
        np.vstack(spirals_high_post_all), hist_bin
    )
    unique_spirals_all = np.empty((1, 2), dtype=object)
    unique_spirals_all[0, 0] = unique_spirals_high_pre
    unique_spirals_all[0, 1] = unique_spirals_high_post
    trialN = np.array([[high_trialN_all, high_trialN_all]], dtype=float)

    save_mat73(
        save_folder / "PassiveSpiralsPrePost.mat",
        {"unique_spirals_all": unique_spirals_all, "trialN": trialN},
    )
