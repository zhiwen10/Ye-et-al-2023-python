"""Translated from whisker/plots/plotWhiskerMeanTrace.m (Fig. 5c)"""

from pathlib import Path

import h5py

from spirals_py.whisker.preprocessing.plotMeanTrace4 import plotMeanTrace4


def plotWhiskerMeanTrace(data_folder, save_folder, code_folder):
    """Plot mean traces around the SSp center on the reference image.

    code_folder: repository root containing data_plus/
    (horizontal_cortex_atlas_25um_ssp_bfd.mat, structures_ssp_bfd.csv).
    The atlas/outline loading and select_area calls in the MATLAB original
    are unused by the plot and are omitted.
    Returns the figure handle.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(
        data_folder / "whisker" / "whisker_mean_maps" / "whisker_spirals_mean_all.mat", "r"
    ) as f:
        wf_mean2 = f["wf_mean2"][:].transpose(2, 1, 0)
    with h5py.File(
        data_folder / "whisker" / "whisker_mean_maps" / "ZYE94_mimg.mat", "r"
    ) as f:
        mimgtransformed = f["mimgtransformed"][:].T

    fig = plotMeanTrace4(mimgtransformed, wf_mean2, data_folder, code_folder)
    fig.savefig(save_folder / "Fig5c_WhiskerMeanTrace.pdf")
    return fig
