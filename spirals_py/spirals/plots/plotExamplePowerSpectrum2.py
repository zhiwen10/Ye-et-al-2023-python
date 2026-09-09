"""Translated from spirals/plots/plotExamplePowerSpectrum2.m (Extended Data Fig.1d)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plotExamplePowerSpectrum2(T, data_folder, save_folder):
    """Translated from spirals/plots/plotExamplePowerSpectrum2.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # plot all power spectrum
    data_folder1 = data_folder / "spirals" / "spirals_power_spectrum2" / "example_traces_005_8Hz"
    nameList2 = ["VISp", "RSP", "SSp", "MOs"]
    freq_value = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10])
    log_freq_value = np.log10(freq_value)
    freqN = 350

    color1 = plt.get_cmap("gray")(np.linspace(1, 0, 15))[:, :3]  # flipud(gray(15))
    color2 = np.vstack([np.zeros((9, 3)), _cbrewer2("seq", "OrRd", 6)])

    hs1d, axes = plt.subplots(1, 4, figsize=(8.75, 3.75))
    for kk in range(len(T)):
        # session info
        mn = T.MouseID.iloc[kk]
        tdb = pd.Timestamp(T.date.iloc[kk]).strftime("%Y%m%d")
        en = T.folder.iloc[kk]
        fname = f"{mn}_{tdb}_{en}"
        fft_file = data_folder1 / f"{fname}_fft.mat"
        psdx_mean = _load_h5(fft_file, "psdx_mean")  # (351, 8)
        freq1 = _load_h5(fft_file, "freq1").ravel()

        psdx_SSp = psdx_mean[:, 2:7].mean(axis=1, keepdims=True)
        psdx_mean2 = np.hstack([psdx_mean[:, 0:2], psdx_SSp, psdx_mean[:, 7:8]])

        colora = color1 if kk < 11 else color2
        for i in range(4):
            ax = axes[i]
            # MATLAB 2:freqN (1-based) -> 1:freqN (0-based)
            ax.plot(np.log10(freq1[1:freqN]), np.log10(psdx_mean2[1:freqN, i]), color=colora[kk])
            ax.set_xlim([log_freq_value[0], log_freq_value[-1]])
            ax.set_xticks(log_freq_value)
            ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4", "6", "8", "10"])
            ax.set_xlabel("log10(Frequency)")
            ax.set_ylabel("log10(Power) (df/f^2)")
            ax.set_ylim([-9, -2])
            ax.set_title(nameList2[i])

    hs1d.savefig(save_folder / "FigS1d_example_fft_all.pdf", bbox_inches="tight")
    plt.show()
    return hs1d
