"""Translated from revision/axons/getAxonBiasTableMO2.m

Axon arbor bias table of the 6 motor / frontal regions only (MOp, MOs,
ACA, PL, ILA, ORB), all with the full MO block as axon search region,
referenced to the MOp center [377, 428].  Written as
Axon_bias_all_cells_MO2.csv (no layer columns).

Skipped dead code of the MATLAB source: the horizontal-cortex-atlas /
outline loads and BW / BW1.  The MATLAB `lables{k} = []` typo in the
empty-region branch is not ported (labels are simply empty there).
"""

from pathlib import Path

from spirals_py.axons.preprocessing._axon_utils import getRegionIndex
from spirals_py.axons.preprocessing.getAxonBiasTable import _get_axon_bias_table
from spirals_py.utils.io import loadStructureTree

MO_LABEL2 = ["MOp", "MOs", "ACA", "PL", "ILA", "ORB"]


def getAxonBiasTableMO2(data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    st = loadStructureTree(data_folder / "tables" / "structure_tree_safe_2017.csv")
    MO_indx = getRegionIndex(MO_LABEL2, st)
    axon_indx = {k: MO_indx for k in range(1, 7)}

    center = [377, 428]  # MOp
    T = _get_axon_bias_table(data_folder, MO_LABEL2, axon_indx, center)
    T.to_csv(save_folder / "Axon_bias_all_cells_MO2.csv", index=False)
    return T
