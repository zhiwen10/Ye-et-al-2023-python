"""Translated from spirals_mirror/plots/plotMatchingIndexHEMI.m (Figure 3i)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals_mirror.plots._helpers import (
    _imresize,
    _load_h5_var,
    _ttest2_scalar,
)


def plotMatchingIndexHEMI(data_folder, save_folder, rng=None):
    """Translated from spirals_mirror/plots/plotMatchingIndexHEMI.m

    rng: numpy Generator (default seeded, unlike MATLAB's unseeded randperm).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    mi = data_folder / "spirals_mirror" / "matching_index"
    TheColorImage_all = _load_h5_var(
        mi / "hemi_weights_8points_allsessions.mat", "TheColorImage_all"
    )
    intensity_all = _load_h5_var(mi / "axon_intensity_all_hemi.mat", "intensity_all")
    gcamp_mean = TheColorImage_all.mean(axis=3)

    target_hw = TheColorImage_all.shape[:2]
    intensity_all1 = np.stack(
        [_imresize(intensity_all[:, :, i], target_hw) for i in range(8)], axis=2
    )

    dot_real = 0.0
    for kkk in range(8):
        gcamp = gcamp_mean[:, :, kkk]
        axon = intensity_all1[:, :, kkk]
        dot_real = dot_real + np.dot(gcamp.ravel(), axon.ravel())
    dot_real_all2 = dot_real

    dot_perm_all2 = np.zeros(1000)
    for i in range(1000):
        interation = rng.permutation(8)
        intensity_all2 = intensity_all1[:, :, interation]
        dot_perm = 0.0
        for kkk in range(8):
            gcamp = gcamp_mean[:, :, kkk]
            axon = intensity_all2[:, :, kkk]
            dot_perm = dot_perm + np.dot(gcamp.ravel(), axon.ravel())
        dot_perm_all2[i] = dot_perm
    _ttest2_scalar(dot_real_all2, dot_perm_all2)

    # normalize between 0 and 1
    dot_all2 = np.concatenate([[dot_real_all2], dot_perm_all2])
    dot_all2 = dot_all2 - dot_all2.min()
    dot_all2 = dot_all2 / dot_all2.max()

    # significance test (MATLAB computes it; the result is not plotted)
    mean_indx = dot_all2[1:1001].mean()
    std_indx = dot_all2[1:1001].std(ddof=1)
    sem_indx = std_indx / np.sqrt(1000)
    _ttest2_scalar(dot_all2[0], dot_all2[1:1001])

    h3i = plt.figure(figsize=(3, 3))
    ax = h3i.add_subplot(1, 1, 1)
    edges = np.arange(0, 1 + 1e-9, 0.05)
    ax.hist(dot_all2[1:], bins=edges)
    ax.axvline(dot_all2[0], color="r")
    ax.set_xlim(0, 1.2)

    h3i.savefig(save_folder / "Fig3i_matching_index_hemi.png", bbox_inches="tight")
    h3i.savefig(save_folder / "Fig3i_matching_index_hemi.pdf", bbox_inches="tight")
    return h3i
