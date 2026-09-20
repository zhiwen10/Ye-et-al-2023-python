"""Translated from revision/axons/getAxonBiasTableMO.m

Same axon arbor bias table as getAxonBiasTable.m, extended to the 6
motor / frontal regions (MOp, MOs, ACA, PL, ILA, ORB) whose axon
search region is the full MO block, and referenced to the MOp center
[377, 428].  Written as Axon_bias_all_cells_MO.csv (no layer columns).

Skipped dead code of the MATLAB source: the MOp / MOs / ACA index
groups (only used by the commented-out per-region axon restriction),
the horizontal-cortex-atlas / outline loads and BW / BW1, and the
commented-out rotated-angle / histogram block.  The MATLAB `lables{k}
= []` typo in the empty-region branch is not ported (labels are simply
empty there, so the alignment of labels_all with the concatenated
cells is preserved).
"""

from pathlib import Path

from spirals_py.axons.preprocessing._axon_utils import getRegionIndex
from spirals_py.axons.preprocessing.getAxonBiasTable import (
    SENSORY_LABEL,
    _get_axon_bias_table,
    sensory_axon_indx,
)
from spirals_py.utils.io import loadStructureTree

MO_LABEL = ["MOp", "MOs", "ACA", "PL", "ILA", "ORB"]

ALL_LABEL_MO = SENSORY_LABEL + MO_LABEL


def getAxonBiasTableMO(data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    st = loadStructureTree(data_folder / "tables" / "structure_tree_safe_2017.csv")
    axon_indx = sensory_axon_indx(st)
    MO_indx = getRegionIndex(MO_LABEL, st)
    for k in range(16, 22):
        axon_indx[k] = MO_indx

    center = [377, 428]  # MOp
    T = _get_axon_bias_table(data_folder, ALL_LABEL_MO, axon_indx, center)
    T.to_csv(save_folder / "Axon_bias_all_cells_MO.csv", index=False)
    return T
