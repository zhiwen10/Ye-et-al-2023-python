"""Translated from axons/preprocessing/getAxonBiasTable.m

Axon arbor bias of each neuron in the sensory regions, via SVD of the
axon-terminal cloud (first PC) and its angle to the soma-center vector
referenced to the SSp-un center [244, 542].  Axon search regions are
restricted by soma location so that long-range projections are
excluded (AUD/TEa cells -> AUD areas, SSp cells -> the large SSp
region, SSs cells -> SSs).  The result is written as
Axon_bias_all_cells.csv with columns soma_center_1/2, soma_angle,
axon_bias_1/2, bias_angle, pc_ratio, center_bias_angle, labels_all,
layers_1..layers_5 (the cortical layer flags TF of the MATLAB table,
expanded by writetable into one column per layer pattern
"1", "2/3", "4", "5", "6").

Naming: the MATLAB source writes Axon_bias_all_cells3.csv, but every
consumer (plotAxonOrientation.py, plotBiasHitogram.py,
plotAxonFlowMatch2.py) reads axons/Axon_bias_all_cells.csv, so the
output is saved under that name.

Skipped dead code of the MATLAB source: the horizontal-cortex-atlas /
outline loads and BW / BW1 (no plotting here), and layer_counts = 
sum(T{:, 9}) which is never used.  The complex z = exp(1i *
soma_angle) table column (renamed soma_polar_angle) is dropped: no
consumer reads it and writetable stores it as text.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirals_py.axons.preprocessing._axon_utils import (
    getAxonSVD,
    getAxonTerminal,
    getCellPos,
    getRegionIndex,
    load_allCoords,
    load_annotation_volume,
)
from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.utils.io import loadStructureTree

SENSORY_LABEL = [
    "VIS", "RSP", "AUD", "TEa", "VISa", "VISC", "VISrl",
    "SSp-tr", "SSp-ll", "SSp-ul", "SSp-m", "SSp-n", "SSp-bfd", "SSp-un", "SSs",
]
AUD_LABEL = ["AUD", "TEa", "VISC"]
SSs_LABEL = ["SSs"]
SSp_LABEL = [
    "VISrl", "SSp-tr", "SSp-ll", "SSp-ul", "SSp-m",
    "SSp-n", "SSp-bfd", "SSp-un",
]
LAYER_PAT = ["1", "2/3", "4", "5", "6"]


def sensory_axon_indx(st):
    """Axon search regions (1-based label positions) of the 15 sensory
    labels, shared by getAxonBiasTable.m and getAxonBiasTableMO.m."""
    sensory_indx = getRegionIndex(SENSORY_LABEL, st)
    AUD_indx = getRegionIndex(AUD_LABEL, st)
    SSs_indx = getRegionIndex(SSs_LABEL, st)
    SSp_indx = getRegionIndex(SSp_LABEL, st)
    axon_indx = {}
    for kk in [1, 2, 5, 6, 7]:
        axon_indx[kk] = sensory_indx
    axon_indx[3] = AUD_indx
    axon_indx[4] = AUD_indx
    for kk in range(8, 15):
        axon_indx[kk] = SSp_indx
    axon_indx[15] = SSs_indx
    return axon_indx


def _get_axon_bias_table(data_folder, all_label, axon_indx, center, layers=False):
    """Shared body of getAxonBiasTable{,MO,MO2,SSp2}.m: collect cells per
    region, keep those with surviving axon terminals, run the terminal
    SVD and compute the bias angles referenced to *center*."""
    data_folder = Path(data_folder)
    allCoords = load_allCoords(data_folder / "axons" / "all_cell_with_parents.mat")
    st = loadStructureTree(data_folder / "tables" / "structure_tree_safe_2017.csv")
    av = load_annotation_volume(data_folder)

    hemi = 1  # only look at left hemisphere
    soma_all2 = []
    axon_terminal_all2 = []
    labels_all = []
    for k, iregion in enumerate(tqdm(all_label, desc="getAxonBiasTable")):
        st_region_indx = getRegionIndex([iregion], st)
        cell_id = getCellPos(allCoords, av, st_region_indx)
        soma_all, axon_terminal_all = getAxonTerminal(
            cell_id, allCoords, av, axon_indx[k + 1], hemi
        )
        soma_all2.extend(soma_all)
        axon_terminal_all2.extend(axon_terminal_all)
        labels_all.extend([iregion] * len(soma_all))

    keep = [a.shape[0] > 0 for a in axon_terminal_all2]
    labels_all = [lab for lab, a in zip(labels_all, keep) if a]
    soma_all3 = [s for s, a in zip(soma_all2, keep) if a]
    axon_terminal_all3 = [a for a, a2 in zip(axon_terminal_all2, keep) if a2]

    soma_center, axon_vector, angle1, polarity = getAxonSVD(
        axon_terminal_all3, soma_all3
    )

    # bias angle for each cell, reference to the region center
    orthog_vector = soma_center - np.asarray(center, dtype=float)
    cross_z = np.abs(
        axon_vector[:, 0] * orthog_vector[:, 1]
        - axon_vector[:, 1] * orthog_vector[:, 0]
    )
    dot = (
        axon_vector[:, 0] * orthog_vector[:, 0]
        + axon_vector[:, 1] * orthog_vector[:, 1]
    )
    ang_diff2 = np.rad2deg(np.arctan2(cross_z, dot))
    soma_angle = np.arctan2(orthog_vector[:, 1], orthog_vector[:, 0])

    T = pd.DataFrame(
        {
            "soma_center_1": soma_center[:, 0],
            "soma_center_2": soma_center[:, 1],
            "soma_angle": soma_angle,
            "axon_bias_1": axon_vector[:, 0],
            "axon_bias_2": axon_vector[:, 1],
            "bias_angle": angle1,
            "pc_ratio": polarity,
            "center_bias_angle": ang_diff2,
            "labels_all": labels_all,
        }
    )

    if layers:
        soma_all4 = np.array([s[0] for s in soma_all3])
        ind = matlab_round(soma_all4).astype(int)
        indxa = av[ind[:, 0] - 1, ind[:, 1] - 1, ind[:, 2] - 1]
        names = st["name"].to_numpy()[indxa.astype(int) - 1]
        for i, pat1 in enumerate(LAYER_PAT):
            T[f"layers_{i + 1}"] = [pat1 in n for n in names]
    return T


def getAxonBiasTable(data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    st = loadStructureTree(data_folder / "tables" / "structure_tree_safe_2017.csv")
    axon_indx = sensory_axon_indx(st)

    center = [244, 542]  # SSp-un
    T = _get_axon_bias_table(data_folder, SENSORY_LABEL, axon_indx, center, layers=True)
    T.to_csv(save_folder / "Axon_bias_all_cells.csv", index=False)
    return T
