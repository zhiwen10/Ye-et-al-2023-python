from pathlib import Path

import h5py
import matplotlib.pyplot as plt


def plotMotionEnergyIndex(data_folder, save_folder):
    """Translated from spirals/plots/plotMotionEnergyIndex.m

    Sync/spirality/order index vs motion energy, one line per session
    (sessions 1-9 black, 10-13 red). Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    path = data_folder / "spirals" / "spirals_index" / "energy_sync_bin_0_0.01_all.mat"
    with h5py.File(path, "r") as f:
        edges = f["edges"][()].ravel()
        sync_sort = f["sync_sort"][()].T       # MATLAB (bins, sessions)
        spirality_sort = f["spirality_sort"][()].T
        index_sort = f["index_sort"][()].T

    hs8j, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [("sync index", sync_sort), ("spirality index", spirality_sort), ("order index", index_sort)]
    for i in range(9):
        for ax, (_, arr) in zip(axes, labels):
            ax.plot(edges[:-1], arr[:, i], "k")
    for i in range(9, 13):
        for ax, (_, arr) in zip(axes, labels):
            ax.plot(edges[:-1], arr[:, i], "r")
    for ax, (ylab, _) in zip(axes, labels):
        ax.set_xlabel("Motion energy")
        ax.set_ylabel(ylab)
        ax.set_ylim([0, 1])

    hs8j.tight_layout()
    hs8j.savefig(save_folder / "Figs8j_motion_vs_orderness.pdf", bbox_inches="tight")
    plt.show()
    return hs8j
