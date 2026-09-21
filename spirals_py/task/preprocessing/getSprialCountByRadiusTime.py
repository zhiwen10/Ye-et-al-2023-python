from pathlib import Path

import numpy as np

from spirals_py.task.preprocessing._task_io import (
    load_mat_cell,
    load_mat_table,
    save_mat73,
)
from spirals_py.task.preprocessing._task_utils import TASK_MICE
from spirals_py.utils.paths import out_root, release_twin


def getSprialCountByRadiusTime(data_folder, save_folder):
    """Translated from task/preprocessing/getSprialCountByRadiusTime.m

    Fraction of trials with a spiral of each radius (40:10:100 pixels)
    at each frame around stimulus onset ([-2, 2] s), per trial type and
    mouse, x 35 (maps to Hz).  Saved as
    task_spiral_count_by_radius_time.mat (spiral_count_sum_all
    (3, 4, 7, 141) and labels).

    The atlas loads of the MATLAB source only feed the unused
    area_cortex and are skipped; the task_outcome table is already
    contained in the *_spirals_task_sort.mat file.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    labels = ["correct", "incorrect", "miss"]
    radius = np.arange(40, 101, 10)  # MATLAB 40:10:100
    spiral_count_sum_all = np.zeros((3, 4, 7, 141))

    for label_id, label in enumerate(labels):
        for i, mn in enumerate(TASK_MICE):
            sort_file = release_twin(
                out_root() / "task" / "spirals" / f"{mn}_spirals_task_sort.mat",
                data_folder,
            )
            spiral_all = load_mat_cell(sort_file, "spiral_all")
            T_all = load_mat_table(sort_file, "T_all")
            left = T_all.left_contrast.to_numpy()
            right = T_all.right_contrast.to_numpy()
            lab = T_all.label.to_numpy()

            index = (lab == label) & (np.abs(left - right) > 0)
            spiral_label = spiral_all[index, :]
            trialN = int(index.sum())

            for kk in range(7):
                spiral_count = np.zeros((spiral_label.shape[0], 141))
                for m in range(spiral_label.shape[0]):
                    for n in range(141):
                        spiral_temp = spiral_label[m, n]
                        if spiral_temp is not None and getattr(spiral_temp, "size", 0):
                            indx = spiral_temp[:, 2] == radius[kk]
                            if indx.any():
                                spiral_count[m, n] = 1
                if trialN:
                    spiral_count_sum_all[label_id, i, kk, :] = (
                        spiral_count.sum(axis=0) / trialN * 35
                    )
                else:
                    spiral_count_sum_all[label_id, i, kk, :] = np.nan
            print(f"getSprialCountByRadiusTime: {mn} {label}")

    save_mat73(
        save_folder / "task_spiral_count_by_radius_time.mat",
        {
            "spiral_count_sum_all": spiral_count_sum_all,
            "labels": np.array(labels, dtype=object).reshape(-1, 1),
        },
    )
