"""Translated from spirals_mirror/preprocessing/getAxonMapHEMI.m
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
    SENSORY_AREA_PATHS,
)
from spirals_py.spirals_mirror.preprocessing._regression_utils import (
    create3dMask,
    select_area_indices,
)
from spirals_py.spirals_mirror.preprocessing.getAxonMapAP import (
    _projection_intensity,
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


def getAxonMapHEMI(data_folder, save_folder):
    """Translated from spirals_mirror/preprocessing/getAxonMapHEMI.m

    Sum of each coronal sensory axon-projection volume over its first
    axis, restricted to the LEFT-hemisphere sensory union mask on the
    scale=5 (264 x 228) grid, normalized by the max; saves intensity_all
    (264, 228, 8) to axon_intensity_all_hemi.mat.

    The MATLAB original passes Utransformed = zeros(size(projectedAtlas1))
    (unused; select_area_indices is called with Utransformed=None instead)
    and its dead comment block / outline loads are skipped.
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

    # mask and Kernel regression map for right
    scale = SCALE
    Utransformed = None
    hemi = "left"
    indexleft, UselectedLeft = select_area_indices(
        SENSORY_AREA_PATHS, spath, Utransformed, projectedAtlas1, hemi, scale
    )
    BW_left = _bw_mask(indexleft, projectedAtlas1[::scale, ::scale].shape)

    intensity_all = np.stack(
        [
            _projection_intensity(
                dataFolder,
                cortexMask,
                f"{NAME_LIST[k]}_coronal",
                BW_left,
            )
            for k in range(8)
        ],
        axis=2,
    )
    save_mat73(
        save_folder / "axon_intensity_all_hemi.mat",
        {"intensity_all": intensity_all},
    )
