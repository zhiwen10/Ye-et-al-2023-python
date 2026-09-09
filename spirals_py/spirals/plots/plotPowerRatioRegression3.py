"""Translated from spirals/plots/plotPowerRatioRegression3.m (Extended Data Fig.1f)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _load_h5(path, varname):
    """Load one variable from a MATLAB v7.3 file, restoring MATLAB axis order."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[varname]).T


def _cbrewer2(ctype, cname, n):
    """Translated from dependencies/cbrewer2/cbrewer2.m (exact palette sizes only)."""
    palettes = {
        ("seq", "OrRd", 6): [
            [254, 240, 217], [253, 212, 158], [253, 187, 132], [252, 141, 89],
            [227, 74, 51], [179, 0, 0],
        ],
    }
    return np.array(palettes[(ctype, cname, n)], dtype=float) / 255.0


def plotPowerRatioRegression3(T, data_folder, save_folder):
    """Translated from spirals/plots/plotPowerRatioRegression3.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    freqN = 350
    freq2 = 201  # index of 10Hz (MATLAB 1-based)

    # load power ratio
    ratio_file = data_folder / "spirals" / "spirals_power_spectrum2" / "power_ratio_all_sessions.mat"
    power_ratio = _load_h5(ratio_file, "power_ratio")  # (4, 15)
    # power ratio vs spiral density regression
    density_file = data_folder / "spirals" / "spirals_density" / "spiralDensityLinePerSession.mat"
    count_sample = _load_h5(density_file, "count_sample")  # (416, 15)
    count_sample = np.where(count_sample < 0, 0, count_sample)
    max_density = count_sample.max(axis=0)

    data_folder1 = data_folder / "spirals" / "spirals_power_spectrum2" / "example_traces_005_8Hz"
    n_sessions = len(T)
    alpha_ratio_all = np.zeros(n_sessions)
    y_ratio_all = np.zeros((n_sessions, freq2 - 11 + 1))
    slope = np.zeros(n_sessions)
    offset = np.zeros(n_sessions)
    for kk in range(n_sessions):
        # session info
        mn = T.MouseID.iloc[kk]
        tdb = pd.Timestamp(T.date.iloc[kk]).strftime("%Y%m%d")
        en = T.folder.iloc[kk]
        fname = f"{mn}_{tdb}_{en}"
        alpha_file = data_folder / "spirals" / "spirals_power_spectrum2" / "alpha_threshold" / f"{fname}_alpha_threshold.mat"
        alpha_ratio_all[kk] = _load_h5(alpha_file, "alpha_ratio").ravel()[0]

        fft_file = data_folder1 / f"{fname}_fft.mat"
        psdx_mean = _load_h5(fft_file, "psdx_mean")  # (351, 8)
        freq1 = _load_h5(fft_file, "freq1").ravel()
        psdx_SSp = psdx_mean[:, 2:7].mean(axis=1, keepdims=True)
        psdx_mean2 = np.hstack([psdx_mean[:, 0:2], psdx_SSp, psdx_mean[:, 7:8]])
        # freq = 0.5 is at index 11 (MATLAB 1-based) -> 10 (0-based)
        x = np.log10(freq1[10:freq2])
        y = np.log10(psdx_mean2[10:freq2, 2])
        y1 = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
        y_ratio_all[kk] = y - y1
        slope[kk] = (y[-1] - y[0]) / (x[-1] - x[0])
        offset[kk] = y[0]

    max_density1 = max_density[:11]
    max_density2 = max_density[11:15]
    mean_density1 = max_density1.mean()
    mean_density2 = max_density2.mean()
    sem_density1 = max_density1.std(ddof=1) / np.sqrt(11)
    sem_density2 = max_density2.std(ddof=1) / np.sqrt(4)

    power_ratio_ssp = power_ratio[2]
    mean_ratio1 = power_ratio_ssp[:11].mean()
    mean_ratio2 = power_ratio_ssp[11:15].mean()
    sem_ratio1 = power_ratio_ssp[:11].std(ddof=1) / np.sqrt(11)
    sem_ratio2 = power_ratio_ssp[11:15].std(ddof=1) / np.sqrt(4)

    alpha_ratio_all = alpha_ratio_all * 100
    mean_alpha_ratio1 = alpha_ratio_all[:11].mean()
    mean_alpha_ratio2 = alpha_ratio_all[11:15].mean()
    sem_alpha_ratio1 = alpha_ratio_all[:11].std(ddof=1) / np.sqrt(11)
    sem_alpha_ratio2 = alpha_ratio_all[11:15].std(ddof=1) / np.sqrt(4)

    p0 = stats.ttest_ind(max_density[:11], max_density[11:15]).pvalue
    p1 = stats.ttest_ind(power_ratio_ssp[:11], power_ratio_ssp[11:15]).pvalue
    p2 = stats.ttest_ind(alpha_ratio_all[:11], alpha_ratio_all[11:15]).pvalue
    print(f"ttest2 density p = {p0:.4g}, power ratio p = {p1:.4g}, alpha ratio p = {p2:.4g}")

    hs1f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 3.75))
    ax1.scatter(max_density[:11], power_ratio_ssp[:11], color="k")
    ax1.errorbar(mean_density1, mean_ratio1, yerr=sem_ratio1, xerr=sem_density1,
                 marker="o", color="k", linestyle="None")
    ax1.scatter(max_density[11:15], power_ratio_ssp[11:15], color="r")
    ax1.errorbar(mean_density2, mean_ratio2, yerr=sem_ratio2, xerr=sem_density2,
                 marker="o", color="r", linestyle="None")
    ax1.set_xlim([0, 5])
    ax1.set_ylim([0, 50])
    ax1.set_xticks(np.arange(0, 6))
    ax1.set_xlabel("Spiral density (spirals/mm2*s)")
    ax1.set_ylabel("2-8 Hz power ratio (%)")

    ax2.scatter(max_density[:11], alpha_ratio_all[:11], color="k")
    ax2.errorbar(mean_density1, mean_alpha_ratio1, yerr=sem_alpha_ratio1, xerr=sem_density1,
                 marker="o", color="k", linestyle="None")
    ax2.scatter(max_density[11:15], alpha_ratio_all[11:15], color="r")
    ax2.errorbar(mean_density2, mean_alpha_ratio2, yerr=sem_alpha_ratio2, xerr=sem_density2,
                 marker="o", color="r", linestyle="None")
    ax2.set_xlim([0, 5])
    ax2.set_ylim([0, 50])
    ax2.set_ylabel("2-8 Hz epoch ratio (%)")
    ax2.set_xlabel("Spiral density (spirals/mm2*s)")
    ax2.set_xticks(np.arange(0, 6))

    # example session (kk = 12, MATLAB 1-based)
    kk = 11
    mn = T.MouseID.iloc[kk]
    tdb = pd.Timestamp(T.date.iloc[kk]).strftime("%Y%m%d")
    en = T.folder.iloc[kk]
    fname = f"{mn}_{tdb}_{en}"
    fft_file = data_folder1 / f"{fname}_fft.mat"
    psdx_mean = _load_h5(fft_file, "psdx_mean")
    freq1 = _load_h5(fft_file, "freq1").ravel()
    psdx_SSp = psdx_mean[:, 2:7].mean(axis=1, keepdims=True)
    psdx_mean2 = np.hstack([psdx_mean[:, 0:2], psdx_SSp, psdx_mean[:, 7:8]])
    x = np.log10(freq1[10:freq2])
    y = np.log10(psdx_mean2[10:freq2, 2])
    y1 = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])

    ax3.plot(x, y, color="k")
    ax3.plot(x, y1, color=[0.2, 0.2, 0.2])
    freq_value3 = np.array([0.5, 2, 4, 6, 8, 10])
    log_freq_value3 = np.log10(freq_value3)
    ax3.set_xticks(log_freq_value3)
    ax3.set_xticklabels(["0.5", "2", "4", "6", "8", "10"])
    ax3.axvline(log_freq_value3[2], color="k", linestyle="--")
    ax3.set_xlim([np.log10(0.5), np.log10(10)])
    ax3.set_xlabel("log10(frequency)")
    ax3.set_ylabel("log10(power) (dF/F2)")

    color1 = plt.get_cmap("gray")(np.linspace(1, 0, 15))[:, :3]  # flipud(gray(15))
    color3 = np.vstack([color1[:11], _cbrewer2("seq", "OrRd", 6)[2:6]])
    for kk in range(15):
        ax4.plot(x, y_ratio_all[kk], color=color3[kk])
    ax4.set_xticks(log_freq_value3)
    ax4.set_xticklabels(["0.5", "2", "4", "6", "8", "10"])
    ax4.set_ylim([-0.2, 1.2])
    ax4.axvline(log_freq_value3[2], color="k", linestyle="--")
    ax4.set_xlim([np.log10(0.5), np.log10(10)])
    ax4.set_xlabel("log10(frequency)")
    ax4.set_ylabel("log10(power) (dF/F2)")

    hs1f.savefig(save_folder / "FigS1f_spiral_density_vs_power2.pdf", bbox_inches="tight")
    plt.show()
    return hs1f
