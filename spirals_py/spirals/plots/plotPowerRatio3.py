"""Translated from spirals/plots/plotPowerRatio3.m (Extended Data Fig.1e)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _load_h5(path, varname):
    """Load one variable from a MATLAB v7.3 file, restoring MATLAB axis order."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[varname]).T


def _load_h5_strings(path, varname):
    """Load a MATLAB cell array of char vectors from a v7.3 file as a list of str."""
    out = []
    with h5py.File(path, "r") as f:
        refs = f[varname][()]
        for ref in refs.flat:
            out.append("".join(chr(c) for c in np.asarray(f[ref]).ravel()))
    return out


def _cbrewer2(ctype, cname, n):
    """Translated from dependencies/cbrewer2/cbrewer2.m (exact palette sizes only)."""
    palettes = {
        ("qual", "Set1", 9): [
            [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
            [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
            [153, 153, 153],
        ],
    }
    return np.array(palettes[(ctype, cname, n)], dtype=float) / 255.0


def plotPowerRatio3(data_folder, save_folder):
    """Translated from spirals/plots/plotPowerRatio3.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    ratio_file = data_folder / "spirals" / "spirals_power_spectrum2" / "power_ratio_all_sessions.mat"
    power_ratio = _load_h5(ratio_file, "power_ratio")  # (4, 15)
    areaNames = _load_h5_strings(ratio_file, "areaNames")
    ratio_mean = power_ratio.mean(axis=1)
    ratio_sem = power_ratio.std(axis=1, ddof=1) / np.sqrt(15)

    color1 = _cbrewer2("qual", "Set1", 9)
    # swap color for VISp and RSP, to match with example trace
    color2 = color1.copy()
    color2[1] = color1[2]
    color2[2] = color1[1]

    hs1e, ax = plt.subplots(figsize=(3.75, 3.75))
    index = np.ones(15)
    for i in range(4):
        ax.scatter((i + 1) * index, power_ratio[i], color="k", s=8)
        ax.errorbar(i + 1, ratio_mean[i], yerr=ratio_sem[i], color=color2[i + 1],
                    linewidth=2, linestyle="None", marker="_")
    for i in range(3):
        ax.plot([(i + 1) * index, (i + 2) * index], power_ratio[i : i + 2], color=[0.8, 0.8, 0.8])

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(areaNames)
    ax.set_yticks(np.arange(0, 70, 10))
    ax.set_ylabel("2-8 Hz Power Ratio (%)")

    hs1e.savefig(save_folder / "FigS1e_2-8hz_power_ratio.pdf", bbox_inches="tight")
    plt.show()

    # compare RSP to VISp, SSP, MOs
    p1 = [stats.ttest_rel(power_ratio[1], power_ratio[i]).pvalue for i in [0, 2, 3]]
    # compare SSp to VISp, RSP, MOs
    p2 = [stats.ttest_rel(power_ratio[2], power_ratio[i]).pvalue for i in [0, 1, 3]]
    print("paired t-test RSP vs [VISp, SSp, MOs] p =", p1)
    print("paired t-test SSp vs [VISp, RSP, MOs] p =", p2)

    # power ratio vs spiral density regression
    density_file = data_folder / "spirals" / "spirals_density" / "spiralDensityLinePerSession.mat"
    count_sample = _load_h5(density_file, "count_sample")  # (416, 15)
    count_sample = np.where(count_sample < 0, 0, count_sample)
    max_density = count_sample.max(axis=0)

    r1 = np.corrcoef(max_density, power_ratio[2])[0, 1]
    fit = stats.linregress(max_density, power_ratio[2])
    # adjusted R^2 like fitlm.Rsquared.Adjusted (n = 15, 1 predictor)
    n = max_density.size
    r2a = 1 - (1 - fit.rvalue**2) * (n - 1) / (n - 2)
    print(f"corr(max_density, SSp power ratio): r = {r1:.4f}, adjusted R^2 = {r2a:.4f}")
    return hs1e
