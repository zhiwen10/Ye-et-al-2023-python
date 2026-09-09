"""Translated from spirals/plots/plotSpiralDuration.m

Plots the spiral duration ratio (red) against the scrambled distribution
(black), mean +/- STD across the 15 sessions of spiralSessions3.xlsx.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotSpiralDuration(T, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralDuration.m

    T is the session table (pandas DataFrame read from spiralSessions3.xlsx).
    Duration .mat files are v7.3; N_ratio loads as (50,), N_ratio_scramble
    transposes back to MATLAB orientation (50, 10).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    N_ratio_all = []
    N_ratio_scramble_all = []
    for kk in range(T.shape[0]):
        mn = T["MouseID"].iloc[kk]
        tda = pd.to_datetime(T["date"].iloc[kk])
        en = T["folder"].iloc[kk]
        tdb = tda.strftime("%Y%m%d")
        fname = f"{mn}_{tdb}_{en}.mat"
        filename = data_folder / "spirals" / "spirals_duration" / fname
        with h5py.File(filename, "r") as f:
            N_ratio = np.asarray(f["N_ratio"]).ravel()
            N_ratio_scramble = np.asarray(f["N_ratio_scramble"]).T  # (50, 10)
        N_ratio_scramble_all.append(N_ratio_scramble.mean(axis=1))
        N_ratio_all.append(N_ratio)

    N_ratio_all = np.stack(N_ratio_all, axis=1)
    N_ratio_scramble_all = np.stack(N_ratio_scramble_all, axis=1)
    mean_N = N_ratio_all.mean(axis=1)
    std_N = N_ratio_all.std(axis=1, ddof=1)  # MATLAB std default
    mean_N_scramble = N_ratio_scramble_all.mean(axis=1)
    std_N_scramble = N_ratio_scramble_all.std(axis=1, ddof=1)

    h1d = plt.figure(figsize=(2, 3))
    x = np.arange(1, 51) / 35
    plt.errorbar(x, mean_N, yerr=std_N, color="r", capsize=15)
    plt.errorbar(x, mean_N_scramble, yerr=std_N_scramble, color="k", capsize=15)
    plt.xlim(0, 15 / 35)
    plt.ylim(0, 1)
    t1 = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(t1, [str(v) for v in t1])
    plt.xlabel("Spiral duration (ms)")
    plt.ylabel("Spiral ratio")

    h1d.tight_layout()
    h1d.savefig(save_folder / "Fig1d_spiral_duration_scramble.pdf")
    plt.show()
    return h1d
