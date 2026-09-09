"""Translated from spirals_mirror/plots/plotVarianceExplained.m (Figure 3c)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spirals_py.spirals_mirror.plots._helpers import _load_h5_var
from spirals_py.utils.plotting import shadedErrorBar


def plotVarianceExplained(T, data_folder, save_folder):
    """Translated from spirals_mirror/plots/plotVarianceExplained.m

    T is the pandas session table; plots the mean +/- SEM across sessions of
    the variance explained by the reduced-rank regression ranks.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    ffolder1 = data_folder / "spirals_mirror" / "regression_ap"
    ffolder2 = data_folder / "spirals_mirror" / "regression_hemi"

    def load_var_all(ffolder, suffix):
        var_all = []
        for kk in range(len(T)):
            mn = T["MouseID"].iloc[kk]
            tda = T["date"].iloc[kk]
            en = T["folder"].iloc[kk]
            tdb = pd.Timestamp(tda).strftime("%Y%m%d")
            fname = f"{mn}_{tdb}_{int(en)}{suffix}"
            var_all.append(_load_h5_var(ffolder / fname, "explained_var5").ravel())
        return np.column_stack(var_all)

    var_all_ap = load_var_all(ffolder1, "-AP.mat")
    mean_var_ap_all = var_all_ap.mean(axis=1)
    std_var_ap_all = var_all_ap.std(axis=1, ddof=1)
    sem_var_ap_all = std_var_ap_all / var_all_ap.shape[1]

    var_all_hemi = load_var_all(ffolder2, "-hemi.mat")
    mean_var_hemi_all = var_all_hemi.mean(axis=1)
    std_var_hemi_all = var_all_hemi.std(axis=1, ddof=1)
    sem_var_hemi_all = std_var_hemi_all / var_all_hemi.shape[1]

    h3c, axs = plt.subplots(1, 2, figsize=(7, 3))
    shadedErrorBar(np.arange(1, 51), mean_var_ap_all, sem_var_ap_all, "-r", ax=axs[0])
    axs[0].set_ylim(0.7, 1)
    axs[0].axvline(16, linestyle="--", color="k")
    axs[0].set_xlabel("Components")
    axs[0].set_ylabel("Variance Explained")
    axs[0].set_title("AP prediction")

    shadedErrorBar(
        np.arange(1, 51), mean_var_hemi_all, sem_var_hemi_all, "-g", ax=axs[1]
    )
    axs[1].set_ylim(0.7, 1)
    axs[1].axvline(16, linestyle="--", color="k")
    axs[1].set_xlabel("Components")
    axs[1].set_ylabel("Variance Explained")
    axs[1].set_title("hemi prediction")

    h3c.savefig(save_folder / "Fig3c_prediction_accuracy_rank.png", bbox_inches="tight")
    h3c.savefig(save_folder / "Fig3c_prediction_accuracy_rank.pdf", bbox_inches="tight")
    return h3c
