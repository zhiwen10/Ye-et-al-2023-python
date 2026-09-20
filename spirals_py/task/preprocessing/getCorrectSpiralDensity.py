from pathlib import Path

import numpy as np

from spirals_py.task.preprocessing._task_io import (
    load_mat_cell,
    load_mat_table,
    save_mat73,
)
from spirals_py.task.preprocessing._task_utils import (
    TASK_MICE,
    density_color_plot2,
)
from spirals_py.task.preprocessing.getConcatTrials import getConcatTrials


def getCorrectSpiralDensity(data_folder, save_folder):
    """Translated from task/preprocessing/getCorrectSpiralDensity.m

    Spiral density maps before (frames 62-68) and after (frames 76-82)
    stimulus onset for correct / incorrect / miss trials, from the
    *_spirals_task_sort.mat files of getTaskSpirals (which must have run
    before, as in the MATLAB pipeline), pooled across mice.

    The atlas loads of the MATLAB source are unused downstream and are
    skipped.  Saved as spiral_density_map_trial_types.mat with
    unique_spirals_pre / unique_spirals_post (3x1 cells of [x y count]
    arrays) and trialN.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    spiral_folder = data_folder / "task" / "spirals"
    labels = ["correct", "incorrect", "miss"]
    frames_pre = np.arange(62, 69)  # MATLAB 62:68
    frames_post = np.arange(76, 83)  # MATLAB 76:82

    trialN = np.zeros(3)
    spiral_temp_pre_all = [np.zeros((0, 5))] * 3
    spiral_temp_post_all = [np.zeros((0, 5))] * 3

    for mn in TASK_MICE:
        sort_file = spiral_folder / f"{mn}_spirals_task_sort.mat"
        spiral_all = load_mat_cell(sort_file, "spiral_all")
        T_all = load_mat_table(sort_file, "T_all")
        left = T_all.left_contrast.to_numpy()
        right = T_all.right_contrast.to_numpy()
        lab = T_all.label.to_numpy()

        for j, label in enumerate(labels):
            if j == 0:
                indx = (lab == label) & (left - right) > 0
            else:
                indx = (lab == label) & np.abs(left - right) > 0
            spiral_temp = spiral_all[indx, :]
            spiral_temp1 = getConcatTrials(spiral_temp)
            spiral_temp_pre = np.vstack([spiral_temp1[f - 1] for f in frames_pre])
            spiral_temp_post = np.vstack([spiral_temp1[f - 1] for f in frames_post])

            spiral_temp_pre_all[j] = np.vstack(
                [spiral_temp_pre_all[j], spiral_temp_pre]
            )
            spiral_temp_post_all[j] = np.vstack(
                [spiral_temp_post_all[j], spiral_temp_post]
            )
            trialN[j] += int(indx.sum())

    # keep radius-100 spirals with direction >= -2
    spiral_temp_pre_all2 = []
    spiral_temp_post_all2 = []
    for j in range(3):
        pre = spiral_temp_pre_all[j]
        post = spiral_temp_post_all[j]
        spiral_temp_pre_all2.append(pre[(pre[:, 2] == 100) & (pre[:, 3] >= -2), :])
        spiral_temp_post_all2.append(post[(post[:, 2] == 100) & (post[:, 3] >= -2), :])

    hist_bin = 40
    unique_spirals_pre = np.empty((3, 1), dtype=object)
    unique_spirals_post = np.empty((3, 1), dtype=object)
    for j in range(3):
        unique_spirals_pre[j, 0] = density_color_plot2(
            spiral_temp_pre_all2[j], hist_bin
        )
        unique_spirals_post[j, 0] = density_color_plot2(
            spiral_temp_post_all2[j], hist_bin
        )

    save_mat73(
        save_folder / "spiral_density_map_trial_types.mat",
        {
            "unique_spirals_pre": unique_spirals_pre,
            "unique_spirals_post": unique_spirals_post,
            "trialN": trialN,
        },
    )
