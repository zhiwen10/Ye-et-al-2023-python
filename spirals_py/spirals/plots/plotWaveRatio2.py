from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, ttest_rel

from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_h5var


def plotWaveRatio2(data_folder, save_folder):
    """Translated from revision/plane_wave/plots/plotWaveRatio2.m

    Plane-wave ratio (fraction of frames with plane-wave index > 0.4) vs
    spiral-wave ratio (peak spiral density / 35) per session; 11 gcamp7
    sessions black, 4 gcamp8 sessions red, group mean +/- SEM error bars.
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # plane wave index (15 sessions)
    with h5py.File(data_folder / "revision" / "plane_wave" / "flow_mirror_all2.mat", "r") as f:
        vxy_all = np.stack([_load_h5var(f, n) for n in
                            ["vxy_MO_left", "vxy_MO_right", "vxy_SSp_left", "vxy_SSp_right"]], axis=2)
    vxy_all2 = vxy_all.sum(axis=2) / 4
    amp_all = np.abs(vxy_all2)
    ratio = np.array([np.sum(amp_all[i] > 0.4) / amp_all.shape[1] for i in range(15)])

    # spiral wave peak density (15 sessions)
    with h5py.File(data_folder / "spirals" / "spirals_density" / "spiralDensityLinePerSession.mat", "r") as f:
        count_sample = f["count_sample"][()].T  # MATLAB (416, 15)
    count_sample[count_sample < 0] = 0
    max_density = count_sample.max(axis=0)
    spiral_peak_ratio = max_density / 35

    # plane wave index, new sessions (not plotted in MATLAB, kept for parity)
    with h5py.File(data_folder / "revision" / "plane_wave" / "flow_mirror_all_newsession2.mat", "r") as f:
        vxy_all_new = np.stack([_load_h5var(f, n) for n in
                                ["vxy_MO_left", "vxy_MO_right", "vxy_SSp_left", "vxy_SSp_right"]], axis=2)
    amp_all_new = np.abs(vxy_all_new.sum(axis=2) / 4)
    ratio_new = np.array([np.sum(amp_all_new[i] > 0.4) / amp_all_new.shape[1] for i in range(4)])
    with h5py.File(data_folder / "revision" / "plane_wave" / "spiralDensityLinePerSession_new.mat", "r") as f:
        count_sample_new = f["count_sample"][()].T
    count_sample_new[count_sample_new < 0] = 0
    spiral_peak_ratio_new = count_sample_new.max(axis=0) / 35

    def stats1(x):
        return x.mean(), x.std(ddof=1) / np.sqrt(x.size)

    mean_g7_spiral, sem_g7_spiral = stats1(spiral_peak_ratio[:11])
    mean_g8_spiral, sem_g8_spiral = stats1(spiral_peak_ratio[11:15])
    mean_g7_plane, sem_g7_plane = stats1(ratio[:11])
    mean_g8_plane, sem_g8_plane = stats1(ratio[11:15])

    hs8m, ax3 = plt.subplots(figsize=(2.5, 2.5))
    ax3.scatter(ratio[:11], spiral_peak_ratio[:11], c="k")
    ax3.scatter(ratio[11:15], spiral_peak_ratio[11:15], c="r")
    ax3.errorbar(mean_g7_plane, mean_g7_spiral, yerr=sem_g7_spiral, xerr=sem_g7_plane,
                 fmt="none", linewidth=1.5, color="k")
    ax3.errorbar(mean_g8_plane, mean_g8_spiral, yerr=sem_g8_spiral, xerr=sem_g8_plane,
                 fmt="none", linewidth=1.5, color="r")
    ax3.plot([0, 0.4], [0, 0.4], "k--")
    ax3.set_xlim([0, 0.4])
    ax3.set_ylim([0, 0.4])
    ax3.set_xlabel("Plane wave ratio")
    ax3.set_ylabel("Spiral wave ratio")
    ticks = np.arange(0, 0.41, 0.1)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(["0", "10%", "20%", "30%", "40%"])
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(["0", "10%", "20%", "30%", "40%"])

    hh, pp = ttest_rel(ratio[11:15], spiral_peak_ratio[11:15])
    hh1, plane_p1 = ttest_ind(ratio[:11], ratio[11:15])
    hh2, spiral_p1 = ttest_ind(spiral_peak_ratio[:11], spiral_peak_ratio[11:15])
    print(f"paired t-test g8 plane vs spiral: p = {pp:.4g}")
    print(f"ttest2 plane g7 vs g8: p = {plane_p1:.4g}; spiral g7 vs g8: p = {spiral_p1:.4g}")

    hs8m.tight_layout()
    hs8m.savefig(save_folder / "FigS8m_wave_ratio.pdf", bbox_inches="tight")
    plt.show()
    return hs8m
