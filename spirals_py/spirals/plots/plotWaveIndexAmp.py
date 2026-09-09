from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots.plotSpiralSyncIndex import _load_h5var


def plotWaveIndexAmp(data_folder, save_folder):
    """Translated from revision/plane_wave/plots/plotWaveIndexAmp.m

    Plane-wave index (mean flow-field synchrony in right SSp) binned by
    2-8Hz amplitude, mean +/- SEM across 15 sessions.
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_folder / "revision" / "plane_wave" / "flow_mirror_all.mat", "r") as f:
        vxy_MO_left = _load_h5var(f, "vxy_MO_left")
        vxy_MO_right = _load_h5var(f, "vxy_MO_right")
        vxy_SSp_left = _load_h5var(f, "vxy_SSp_left")
        vxy_SSp_right = _load_h5var(f, "vxy_SSp_right")
    with h5py.File(data_folder / "revision" / "plane_wave" / "flow_mirror_amp_all.mat", "r") as f:
        traceAmpt_MO_left = f["traceAmpt_MO_left"][()].T
        traceAmpt_MO_right = f["traceAmpt_MO_right"][()].T
        traceAmpt_SSp_left = f["traceAmpt_SSp_left"][()].T
        traceAmpt_SSp_right = f["traceAmpt_SSp_right"][()].T

    vxy_all = np.stack([vxy_MO_left, vxy_MO_right, vxy_SSp_left, vxy_SSp_right], axis=2)
    sync_amp_all = np.abs(vxy_all)
    osci_amp_all = np.stack(
        [traceAmpt_MO_left, traceAmpt_MO_right, traceAmpt_SSp_left, traceAmpt_SSp_right], axis=2)

    edges = np.arange(0, 0.025 + 1e-12, 0.00125)
    sync_amp_bins = np.full((edges.size - 1, 15), np.nan)
    for kk in range(15):
        sync_amp_temp = sync_amp_all[kk, :, 3]  # MATLAB (:,:,4) = SSp_right
        osci_amp_temp = osci_amp_all[kk, :-1, 3]
        for i in range(edges.size - 1):
            a = np.flatnonzero((osci_amp_temp >= edges[i]) & (osci_amp_temp < edges[i + 1]))
            if a.size:
                sync_amp_bins[i, kk] = sync_amp_temp[a].mean()

    sync_amp_bins_mean = np.nanmean(sync_amp_bins, axis=1)
    sync_amp_bins_sem = np.nanstd(sync_amp_bins, axis=1, ddof=1) / np.sqrt(15)

    hs8l, ax = plt.subplots(figsize=(2.5, 3.5))
    ax.errorbar(edges[0:8] * 100, sync_amp_bins_mean[0:8], sync_amp_bins_sem[0:8], fmt="k")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("2-8Hz amp(dF/F, %)")
    ax.set_ylabel("Plane wave index")
    ax.set_xlim([0, 1])
    ax.set_ylim([0.2, 0.6])
    ax.set_yticks(np.arange(0.2, 0.61, 0.1))

    hs8l.tight_layout()
    hs8l.savefig(save_folder / "FigS8l_planewave_index_vs_amp.pdf", bbox_inches="tight")
    plt.show()
    return hs8l
