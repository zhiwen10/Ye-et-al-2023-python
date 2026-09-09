from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers_s15 import plot_spiral_pre_post


def plotCorrectSpiralPrePost(data_folder, save_folder):
    """Translated from task/plots/plotCorrectSpiralPrePost.m

    Spiral density maps (spirals/(mm^2*s)) before and after stimulus
    onset in correct task trials, from the precomputed
    CorrectSpiralsPrePost.mat. Output is .png instead of MATLAB's -dpdf.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    return plot_spiral_pre_post(
        data_folder,
        save_folder,
        mat_file="CorrectSpiralsPrePost.mat",
        frames_pre=np.arange(63, 68),  # MATLAB 63:67
        labels=["Correct pre-stim", "Correct post-stim"],
        out_name="FigS15j_task_spirals_pre_post.png",
    )
