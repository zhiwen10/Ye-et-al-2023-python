from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers_s15 import plot_spiral_pre_post


def plotPassiveSpiralPrePost(data_folder, save_folder):
    """Translated from task/plots/plotPassiveSpiralPrePost.m

    Spiral density maps (spirals/(mm^2*s)) before and after stimulus
    onset in passive viewing (high-contrast stimuli), from the
    precomputed PassiveSpiralsPrePost.mat. Output is .png instead of
    MATLAB's -dpdf.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    return plot_spiral_pre_post(
        data_folder,
        save_folder,
        mat_file="PassiveSpiralsPrePost.mat",
        frames_pre=np.arange(59, 69),  # MATLAB 59:68
        labels=["High pre-stim", "High post-stim"],
        out_name="FigS15i_passive_spirals_pre_post.png",
    )
