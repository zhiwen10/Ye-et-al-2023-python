"""Translated from revision/axons/getAxonBiasTableSSp2.m

Axon arbor bias table over the 15 split sensory regions (VISp, VISam,
VISpm, VISli, VISl, VISpor, RSP, AUD, TEa, SSp-ll, SSp-ul, SSp-m,
SSp-n, SSp-un, SSs), referenced to the PPC center [350, 700].  Axon
search regions follow the split list: AUD / TEa -> AUD areas, the
visual / RSP entries -> the full sensory block, SSp subregions -> the
large SSp region, SSs -> SSs.  Written as
Axon_bias_all_cells_SSp2.csv (no layer columns).

The MATLAB consumers of this table read it from the data folder's
axons/ directory (revision2/axon_center/getAxonBiasTableSSp.m), not
from save_folder; save_folder is kept exactly as passed, as in MATLAB.

Skipped dead code of the MATLAB source: the horizontal-cortex-atlas /
outline loads and BW / BW1, and the commented-out rotated-angle /
histogram block.  The MATLAB `lables{k} = []` typo in the empty-region
branch is not ported (labels are simply empty there).
"""

from pathlib import Path

from spirals_py.axons.preprocessing._axon_utils import getRegionIndex
from spirals_py.axons.preprocessing.getAxonBiasTable import (
    AUD_LABEL,
    SENSORY_LABEL,
    SSs_LABEL,
    SSp_LABEL,
    _get_axon_bias_table,
)
from spirals_py.utils.io import loadStructureTree

SSP2_LABEL = [
    "VISp", "VISam", "VISpm", "VISli", "VISl", "VISpor", "RSP", "AUD", "TEa",
    "SSp-ll", "SSp-ul", "SSp-m", "SSp-n", "SSp-un", "SSs",
]


def getAxonBiasTableSSp2(data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    st = loadStructureTree(data_folder / "tables" / "structure_tree_safe_2017.csv")
    sensory_indx = getRegionIndex(SENSORY_LABEL, st)
    AUD_indx = getRegionIndex(AUD_LABEL, st)
    SSs_indx = getRegionIndex(SSs_LABEL, st)
    SSp_indx = getRegionIndex(SSp_LABEL, st)
    axon_indx = {8: AUD_indx, 9: AUD_indx}
    for kk in range(1, 8):
        axon_indx[kk] = sensory_indx
    for kk in range(10, 15):
        axon_indx[kk] = SSp_indx
    axon_indx[15] = SSs_indx

    center = [350, 700]  # PPC
    T = _get_axon_bias_table(data_folder, SSP2_LABEL, axon_indx, center)
    T.to_csv(save_folder / "Axon_bias_all_cells_SSp2.csv", index=False)
    return T
