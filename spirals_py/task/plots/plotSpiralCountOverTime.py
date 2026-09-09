from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.task.plots._task_helpers_s15 import load_projectedAtlas1
from spirals_py.utils.plotting import shadedErrorBar


def plotSpiralCountOverTime(data_folder, save_folder):
    """Translated from task/plots/plotSpiralCountOverTime.m

    Spiral counts per cortical area (7 atlas regions) across time
    (-2 to 2 s around stimulus onset), for 3 trial types x 4 mice, from
    the precomputed task_spiral_count_by_radius_time.mat.

    Adaptations: the unused atlas loads of the MATLAB source
    (horizontal_cortex_atlas_50um.mat, get_cortex_atlas_path) are
    skipped; only projectedAtlas1 is read to compute the cortical area.
    Output is .png instead of MATLAB's -dpdf.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    BW = load_projectedAtlas1(data_folder).astype(bool)
    pixSize = 0.01  # mm/pix after registration
    pixArea = pixSize**2
    pix_sum = BW.sum()
    area_cortex = pix_sum * pixArea

    with h5py.File(
        data_folder / "task" / "spirals" / "task_spiral_count_by_radius_time.mat", "r"
    ) as f:
        # MATLAB (3, 4, 7, 141) -> trial type, mouse, area, time
        spiral_count_sum_all = np.asarray(f["spiral_count_sum_all"]).transpose(
            3, 2, 1, 0
        )
    spiral_count_sum_all = spiral_count_sum_all / area_cortex

    color1 = ["k", "g", "r"]
    hs15k = plt.figure(figsize=(9, 6))
    for i in range(3):
        spiral_count_sum_all1 = spiral_count_sum_all[i]
        spiral_count_sum_mean = spiral_count_sum_all1.mean(axis=0)
        spiral_count_sum_sem = spiral_count_sum_all1.std(axis=0, ddof=1) / np.sqrt(4)
        for j in range(7):
            ax = hs15k.add_subplot(3, 7, j + i * 7 + 1)
            shadedErrorBar(
                np.arange(1, 142),
                spiral_count_sum_mean[j],
                spiral_count_sum_sem[j],
                lineProps=f"-{color1[i]}",
                ax=ax,
            )
            ax.axvline(71, linestyle="--", color="k")
            ax.set_xticks([1, 35, 70, 105, 141])
            ax.set_xticklabels(["-2", "-1", "0", "1", "2"])
            ax.set_ylim(0, 0.08)
    hs15k.savefig(
        save_folder / "FigS15k_task_spiral_count_over_time.png", bbox_inches="tight"
    )
    return hs15k
