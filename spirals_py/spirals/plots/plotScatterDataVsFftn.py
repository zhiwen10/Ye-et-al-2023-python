"""Translated from spirals/plots/plotScatterDataVsFftn.m (Extended Data Fig.3d-e).

Peak spiral density (within SSp) per session for the data vs the 3-D-FFT
phase-scrambled control, per spiral radius (10:10:100 pixels), plus the
percentage change across sessions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel

from spirals_py.spirals.plots._fig1_helpers_s3 import (
    _ismember_rows_int,
    _load_density_file,
    _session_strings,
    get_ssp_index,
)


def plotScatterDataVsFftn(T, data_folder, save_folder, freq):
    """Translated from spirals/plots/plotScatterDataVsFftn.m

    The paired/one-sample t-tests collected in the MATLAB original
    (ha/pa, h3/p3) are not returned or plotted there; they are computed
    here as ha/pa and h3/p3 for parity."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    freq_name = f"{freq[0]:g}_{freq[1]:g}Hz"
    local_data_folder = data_folder / "spirals" / "spirals_fftn"
    control_data_folder = local_data_folder / freq_name / "control_stats"
    fftn_data_folder = local_data_folder / freq_name / "fftn_stats"

    color1 = "k"
    ssp_index = get_ssp_index(data_folder)

    pixSize = 0.01  # mm/pix
    pixArea = pixSize**2
    hist_bin = 40
    n_sessions = len(T)
    count_sample_control = np.zeros((n_sessions, 10))
    count_sample_permute = np.zeros((n_sessions, 10))

    for kk in range(n_sessions):
        mn, td, tdb, en = _session_strings(T, kk)
        fname = f"{mn}_{tdb}_{en}"
        control_cells, control_frames = _load_density_file(
            control_data_folder / f"{fname}_density.mat"
        )
        permute_cells, permute_frames = _load_density_file(
            fftn_data_folder / f"{fname}_density.mat"
        )
        for count in range(10):
            spirals_control = control_cells[count]
            spirals_permute = permute_cells[count]
            spirals_control = spirals_control[
                _ismember_rows_int(spirals_control[:, :2], ssp_index)
            ]
            spirals_permute = spirals_permute[
                _ismember_rows_int(spirals_permute[:, :2], ssp_index)
            ]
            unit_control = spirals_control[:, 2] / (hist_bin * hist_bin * pixArea)
            unit_control = unit_control / control_frames[0] * 35
            unit_permute = spirals_permute[:, 2] / (hist_bin * hist_bin * pixArea)
            unit_permute = unit_permute / permute_frames[0] * 35
            count_sample_control[kk, count] = unit_control.max() if unit_control.size else 0.0
            count_sample_permute[kk, count] = unit_permute.max() if unit_permute.size else 0.0

    hs3d = plt.figure()
    for radius in range(10):
        ax = hs3d.add_subplot(1, 10, radius + 1)
        control_all = count_sample_control[:, radius]
        permute_all = count_sample_permute[:, radius]
        ax.scatter(np.ones(n_sessions), control_all, s=4, c=color1)
        ax.scatter(np.ones(n_sessions) * 2, permute_all, s=4, c=color1)
        for kk in range(n_sessions):
            ax.plot([1, 2], [control_all[kk], permute_all[kk]], color=color1)
        ax.set_ylim(0, 2.5)
    hs3d.savefig(save_folder / f"FigS3d_spirals_across_session_{freq_name}.png")
    plt.show()

    ha = np.empty(10)
    pa = np.empty(10)
    for radius in range(10):
        ha[radius], pa[radius] = ttest_rel(
            count_sample_control[:, radius], count_sample_permute[:, radius]
        )

    percentage_change = (count_sample_control - count_sample_permute) / count_sample_permute

    h3 = np.empty(10)
    p3 = np.empty(10)
    for i in range(10):
        h3[i], p3[i] = ttest_1samp(percentage_change[:, i], 0)

    mean_percentage_change = percentage_change.mean(axis=0)
    std_percentage_change = percentage_change.std(axis=0, ddof=1)
    sem_percentage_change = std_percentage_change / np.sqrt(n_sessions)

    x_point = np.random.normal(0, 0.1, n_sessions)
    scatter_x = np.tile(np.arange(1, 11), (n_sessions, 1)) + x_point[:, None]

    hs3e = plt.figure()
    ax = hs3e.add_subplot(1, 1, 1)
    for radius in range(10):
        ax.scatter(
            scatter_x[:, radius], percentage_change[:, radius] * 100, s=6, c=color1
        )
    x = np.arange(1, 11)
    ax.bar(x, mean_percentage_change * 100, facecolor="none", edgecolor="k")
    ax.errorbar(x, mean_percentage_change * 100, yerr=sem_percentage_change * 100,
                fmt="none", ecolor="k")
    ax.set_ylim(-100, 500)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Percentage change (%)")
    hs3e.savefig(save_folder / f"FigS3e_spirals_change_across_session_{freq_name}.png")
    plt.show()
    return hs3d, hs3e
