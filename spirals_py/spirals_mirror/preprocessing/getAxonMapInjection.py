"""Translated from spirals_mirror/preprocessing/getAxonMapInjection.m
(pipeline3_spirals_mirror.m, Figure 3j)."""
from pathlib import Path

import numpy as np

from spirals_py.ephys.plots._ephys_helpers import nrrdread
from spirals_py.spirals_mirror.plots._helpers import (
    _get_cortex_atlas_path,
    _load_atlas_tables,
)
from spirals_py.spirals_mirror.preprocessing._regression_utils import (
    create3dMask,
)
from spirals_py.utils.matio import save_mat73

NAME_LIST = [
    "SSp_bfd",
    "SSp_n",
    "SSp_m",
    "SSp_ll",
    "SSp_ul",
    "SSp_tr",
    "RSP",
    "VISp",
]


def getAxonMapInjection(data_folder, save_folder):
    """Translated from spirals_mirror/preprocessing/getAxonMapInjection.m

    Sum of each coronal sensory axon-projection volume over its first axis
    (isocortex mask only, no area mask), normalized per area by its max;
    saves injection_intensity (264, 228, 8) to
    axon_intensity_all_injection.mat.

    Bug fixed: the MATLAB original builds dataFolder with a Windows
    backslash ('spirals_mirror\\AxonProjectionVolumn'), which finds no
    files on other platforms; the forward-slash path is used here. The
    dead root1 / areaName / maskPath definitions are skipped and the
    unused atlas-table loads with it.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    _, _, projectedAtlas1, _ = _load_atlas_tables(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)

    # create cortex mask
    atlas, metaAVGT = nrrdread(data_folder / "tables" / "annotation_50.nrrd")
    ctx = "/997/8/567/688/"
    cortexMask = create3dMask(ctx, st, atlas)

    dataFolder = data_folder / "spirals_mirror" / "AxonProjectionVolumn"

    injection_intensity = []
    for k in range(8):
        matches = sorted(dataFolder.glob(f"{NAME_LIST[k]}_coronal*.nrrd"))
        if not matches:
            raise FileNotFoundError(
                f"no {NAME_LIST[k]}_coronal*.nrrd volume in {dataFolder}"
            )
        data, meta = nrrdread(matches[0])
        data[~cortexMask] = 0
        TheColorImage = np.sum(data, axis=0)  # squeeze(sum(data, 1))
        colorIntensity = TheColorImage / TheColorImage.max()
        injection_intensity.append(colorIntensity)

    injection_intensity = np.stack(injection_intensity, axis=2)
    save_mat73(
        save_folder / "axon_intensity_all_injection.mat",
        {"injection_intensity": injection_intensity},
    )
