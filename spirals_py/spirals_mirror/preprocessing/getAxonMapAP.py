"""Translated from spirals_mirror/preprocessing/getAxonMapAP.m
(pipeline3_spirals_mirror.m, Figure 3j-k)."""
from pathlib import Path

import numpy as np

from spirals_py.ephys.plots._ephys_helpers import nrrdread
from spirals_py.spirals_mirror.plots._helpers import (
    _bw_mask,
    _get_cortex_atlas_path,
    _load_atlas_tables,
)
from spirals_py.spirals_mirror.plots.plotCortexDivision import (
    FRONTAL_AREA_PATHS,
)
from spirals_py.spirals_mirror.preprocessing._regression_utils import (
    create3dMask,
    select_area_indices,
)
from spirals_py.utils.matio import save_mat73

SCALE = 5
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


def _read_axon_volume(dataFolder, name):
    matches = sorted(Path(dataFolder).glob(f"{name}*.nrrd"))
    if not matches:
        raise FileNotFoundError(f"no {name}*.nrrd volume in {dataFolder}")
    return nrrdread(matches[0])


def _projection_intensity(dataFolder, cortexMask, name, BW):
    data, meta = _read_axon_volume(dataFolder, name)
    data[~cortexMask] = 0
    TheColorImage = np.sum(data, axis=0)  # squeeze(sum(data, 1))
    TheColorImage[~BW.astype(bool)] = 0
    return TheColorImage / TheColorImage.max()


def getAxonMapAP(data_folder, save_folder):
    """Translated from spirals_mirror/preprocessing/getAxonMapAP.m

    Sums each coronal sensory axon-projection volume over its first axis,
    restricts it to the right MO (frontal) mask on the scale=5 (264 x 228)
    grid, normalizes by the max and saves intensity_all (264, 228, 8) to
    axon_intensity_all_ap.mat.

    The MATLAB original passes Utransformed = zeros(size(projectedAtlas1))
    (a 2-D array, so Uselected is meaningless and unused); select_area_indices
    is called with Utransformed=None instead. The unused metaAVGT / meta
    outputs and the dead outline loads are skipped.
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
    spath = st["structure_id_path"].astype(str)

    indexMO, UselectedMO = select_area_indices(
        FRONTAL_AREA_PATHS, spath, None, projectedAtlas1, "right", SCALE
    )
    BW_MO = _bw_mask(indexMO, projectedAtlas1[::SCALE, ::SCALE].shape)

    intensity_all = np.stack(
        [
            _projection_intensity(
                dataFolder,
                cortexMask,
                f"{NAME_LIST[k]}_coronal",
                BW_MO,
            )
            for k in range(8)
        ],
        axis=2,
    )
    save_mat73(
        save_folder / "axon_intensity_all_ap.mat",
        {"intensity_all": intensity_all},
    )
