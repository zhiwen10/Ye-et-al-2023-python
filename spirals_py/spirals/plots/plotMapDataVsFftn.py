"""Translated from spirals/plots/plotMapDataVsFftn.m (Extended Data Fig.3c).

Spiral density maps (all sessions combined) for the data vs the 3-D-FFT
phase-scrambled control, per spiral radius (10:10:100 pixels).
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots.plotSpiralTimeSeries3d import (
    _get_cortex_atlas_path,
    _load_atlas_data,
)
from spirals_py.utils.atlas import plotOutline


def _matlab_round10(x):
    return np.floor(x * 10 + 0.5) / 10  # MATLAB round(x*10)/10 (x > 0)


def plotMapDataVsFftn(data_folder, save_folder, freq):
    """Translated from spirals/plots/plotMapDataVsFftn.m

    Adaptation: the MATLAB colorbar position tweak (shift right, halve
    height) is approximated with colorbar(..., shrink=0.5)."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    freq_name = f"{freq[0]:g}_{freq[1]:g}Hz"
    local_data_folder = data_folder / "spirals" / "spirals_fftn"
    control_data_folder = local_data_folder / freq_name / "control_map"
    fftn_data_folder = local_data_folder / freq_name / "fftn_map"

    atlas1, coords, projectedAtlas1 = _load_atlas_data(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    scale3 = 5
    pixSize = 0.01  # mm/pix
    pixArea = pixSize**2
    hist_bin = 40
    with h5py.File(control_data_folder / "total_frames.mat", "r") as f:
        frame_all = float(np.asarray(f["frame_all"]).ravel()[0])

    hs3c = plt.figure(figsize=(9, 9))
    count = 1
    for radius in range(10, 101, 10):
        with h5py.File(control_data_folder / f"histogram_radius_{radius}.mat", "r") as f:
            control = np.asarray(f["unique_spirals"]).T
        with h5py.File(fftn_data_folder / f"histogram_radius_{radius}.mat", "r") as f:
            permute = np.asarray(f["unique_spirals"]).T

        unique_spirals_unit_control = control[:, 2] / (hist_bin * hist_bin * pixArea)
        unique_spirals_unit_control = unique_spirals_unit_control / frame_all * 35
        unique_spirals_unit_permute = permute[:, 2] / (hist_bin * hist_bin * pixArea)
        unique_spirals_unit_permute = unique_spirals_unit_permute / frame_all * 35
        cmax = max(unique_spirals_unit_control.max(), unique_spirals_unit_permute.max())

        count1 = count + 5 if radius >= 60 else count
        for row, (sp, unit) in enumerate(
            [(control, unique_spirals_unit_control),
             (permute, unique_spirals_unit_permute)]
        ):
            ax = hs3c.add_subplot(4, 5, row * 5 + count1)
            ax.scatter(sp[:, 0], sp[:, 1], s=3, c=unit, cmap="hot",
                       vmin=0, vmax=cmax, rasterized=True)
            for sl in (slice(0, 3), slice(3, 4), slice(4, 5), slice(5, 11), slice(0, 11)):
                plotOutline(maskPath[sl], st, atlas1, [], scale3, ax=ax)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_xlim(0, 1140)
            ax.set_ylim(1320, 0)  # MATLAB Ydir reverse
            cb = hs3c.colorbar(
                ax.collections[0], ax=ax, fraction=0.046, pad=0.04, shrink=0.5
            )
            cb.set_ticks([0, cmax])
            cb.set_ticklabels(["0", f"{_matlab_round10(cmax):g}"])
        count += 1

    hs3c.savefig(save_folder / f"FigS3c_spirals_radius_{freq_name}.png")
    plt.show()
    return hs3c
