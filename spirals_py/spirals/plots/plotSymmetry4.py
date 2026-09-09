from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_h5var


def plotSymmetry4(data_folder, save_folder):
    """Translated from revision/plane_wave/plots/plotSymmetry4.m

    Scatter of mean flow angles between areas (right SSp vs right MO and vs
    left MO) for frames where both exceed the plane-wave threshold 0.6.
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_folder / "revision" / "plane_wave" / "flow_mirror_all.mat", "r") as f:
        vxy_MO_left = _load_h5var(f, "vxy_MO_left")
        vxy_MO_right = _load_h5var(f, "vxy_MO_right")
        vxy_SSp_left = _load_h5var(f, "vxy_SSp_left")
        vxy_SSp_right = _load_h5var(f, "vxy_SSp_right")
    vxy_all = np.stack([vxy_MO_left, vxy_MO_right, vxy_SSp_left, vxy_SSp_right], axis=2)
    angle_all = np.angle(vxy_all)
    amp_all = np.abs(vxy_all)

    threthold = 0.6
    # plane-wave frame ratios per area and across area pairs (computed in
    # MATLAB; used for reference, only the scatters are plotted)
    ratio = np.empty((15, 4))
    for k in range(4):
        for i in range(15):
            amp_temp = amp_all[i, :, k]
            ratio[i, k] = np.sum(amp_temp >= threthold) / amp_temp.size
    ratioM = np.empty((15, 2))
    for k in range(2):
        for i in range(15):
            amp_temp = amp_all[i, :, k + 1]
            amp_temp2 = amp_all[i, :, 3]
            ratioM[i, k] = np.sum((amp_temp > threthold) & (amp_temp2 > threthold)) / amp_temp.size

    pi_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    pi_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    pairs = [(3, 2), (3, 1)]  # MATLAB (:,:,4) vs (:,:,3) and vs (:,:,2)
    hs8o, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, (k1, k2) in zip(axes, pairs):
        angle_all1 = angle_all[:, :, k1].ravel()
        angle_all2 = angle_all[:, :, k2].ravel()
        amp_all1 = amp_all[:, :, k1].ravel()
        amp_all2 = amp_all[:, :, k2].ravel()
        index = (amp_all1 >= threthold) & (amp_all2 >= threthold)
        ax.scatter(angle_all1[index], angle_all2[index], 4, "k")
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels)
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticks(pi_ticks)
        ax.set_yticklabels(pi_labels)

    hs8o.tight_layout()
    hs8o.savefig(save_folder / "FigS8o_wave_symmetry.pdf", bbox_inches="tight")
    plt.show()
    return hs8o
