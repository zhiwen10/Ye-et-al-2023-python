"""Morphology helpers shared by the axon preprocessing pipeline.

Translations of axons/utils/{getRegionIndex,getCellPos,getAxonTerminal,
filter_axon_position,getAxonSVD}.m plus the shared loaders for the
single-cell morphology dataset, the 10 um annotation volume and the
structure tree.
"""

from pathlib import Path

import numpy as np

from spirals_py.task.plots._task_helpers_s15 import matlab_round
from spirals_py.utils.io import loadStructureTree, load_cell_array_h5


def load_allCoords(path):
    """allCoords cell array from axons/all_cell_with_parents.mat (v7.3):
    one (n, 6) matrix per cell
    [pointType, x, y, z, unique_point_id, point_parent_id] (1 um atlas
    coordinates)."""
    return load_cell_array_h5(path, "allCoords")


def load_annotation_volume(data_folder):
    """annotation_volume_10um_by_index.npy: voxel values are 1-based rows
    of the structure tree (0-based st.index + 1), [x y z] orientation."""
    return np.load(Path(data_folder) / "tables" / "annotation_volume_10um_by_index.npy")


def getRegionIndex(region_label, st):
    """Translated from axons/utils/getRegionIndex.m

    Returns the 1-based structure-tree rows whose structure_id_path
    contains '/<id>/' for the id of every acronym in *region_label*
    (i.e. the region and all its descendants), concatenated in label
    order; these values live in the same space as the annotation-volume
    voxel values.
    """
    spath = st["structure_id_path"].astype(str)
    st_region_indx = []
    for label in region_label:
        region_id = st.loc[st["acronym"] == label, "id"].to_numpy()
        for rid in np.atleast_1d(region_id):
            st_region_indx.extend(
                np.flatnonzero(spath.str.contains(f"/{int(rid)}/", na=False)) + 1
            )
    return np.asarray(st_region_indx, dtype=float)


def getCellPos(allCoords, av, filter_region):
    """Translated from axons/utils/getCellPos.m

    Returns 1-based indices into allCoords of the cells whose soma
    (pointType 1) maps, after round(xyz / 10), into a voxel of
    *filter_region* in the 10 um annotation volume.

    MATLAB round (half away from zero) is used.  MATLAB errors when a
    rounded soma coordinate is 0 (sub2ind); here such soma points are
    dropped before the lookup instead, and coordinates beyond the
    volume bounds raise IndexError like MATLAB.
    """
    cell_id = []
    filter_region = np.asarray(filter_region, dtype=float).ravel()
    for nIdx in range(len(allCoords)):
        noi = np.asarray(allCoords[nIdx], dtype=float)
        somaCoords = noi[:, 0] == 1
        coordsIdx = matlab_round(noi[:, 1:4] / 10).astype(int)
        coordsIdx = coordsIdx[somaCoords]
        coordsIdx = coordsIdx[np.all(coordsIdx >= 1, axis=1)]
        if coordsIdx.shape[0] == 0:
            continue
        stIdxSoma = av[coordsIdx[:, 0] - 1, coordsIdx[:, 1] - 1, coordsIdx[:, 2] - 1]
        if np.isin(stIdxSoma, filter_region).any():
            cell_id.append(nIdx + 1)
    return cell_id


def filter_axon_position(axonCoords, av, st_region_indx, hemi):
    """Translated from axons/utils/filter_axon_position.m

    Rounds axon coordinates to 10 um, clamps them into the annotation
    volume (MATLAB's 'fixing rounding errors' block) and keeps those in
    *st_region_indx* and, for hemi == 1 / 2, in the left / right
    hemisphere (3rd coordinate vs size(av, 3) / 2).  Returns the (n, 3)
    rounded 10 um [x y z] coordinates.
    """
    axonCoords2 = matlab_round(np.asarray(axonCoords, dtype=float)[:, 1:4] / 10)
    axonCoords2 = axonCoords2.astype(int)
    # fixing rounding errors:
    axonCoords2[axonCoords2 < 1] = 1
    for j in range(3):
        axonCoords2[axonCoords2[:, j] > av.shape[j], j] = av.shape[j]
    axon_area = av[axonCoords2[:, 0] - 1, axonCoords2[:, 1] - 1, axonCoords2[:, 2] - 1]
    in_region = np.isin(axon_area, np.asarray(st_region_indx, dtype=float))
    if hemi is None:
        keep = in_region
    elif hemi == 1:
        keep = in_region & (axonCoords2[:, 2] < av.shape[2] / 2)
    elif hemi == 2:
        keep = in_region & (axonCoords2[:, 2] > av.shape[2] / 2)
    return axonCoords2[keep]


def getAxonTerminal(cell_id, allCoords, av, filter_region, hemi):
    """Translated from axons/utils/getAxonTerminal.m

    Terminals of a cell are the points whose unique_point_id (column 5)
    never appears in a point_parent_id (column 6) of that cell; axon
    terminals are the type-2 ones, filtered through
    filter_axon_position.  Soma coordinates (type 1) are returned
    unrounded at 10 um scale (xyz / 10)."""
    soma_all = []
    axon_all = []
    type1 = 1
    type2 = 2
    for i in range(len(cell_id)):
        noi = np.asarray(allCoords[cell_id[i] - 1], dtype=float)
        parentsa = np.unique(noi[:, 5])
        x1 = np.isin(noi[:, 4], parentsa)
        noi1 = noi[~x1]
        axonCoords = noi1[noi1[:, 0] == type2]
        axon_all.append(filter_axon_position(axonCoords, av, filter_region, hemi))
        somaCoords = noi[noi[:, 0] == type1]
        soma_all.append(somaCoords[:, 1:4] / 10)
    return soma_all, axon_all


def getAxonSVD(axon_terminal_all, soma_all):
    """Translated from axons/utils/getAxonSVD.m

    First PC of each cell's axon-terminal cloud: the (z, x) columns of
    the 10 um coordinates (axon_terminal_all(:, [3, 1]) / soma (:, [3,
    1])), zero-meaned, econ-SVD of the transposed cloud.  Returns
    (soma_center, axon_vector, angle1, polarity), where angle1 is
    round(atand(U21 / U11)) + 1 clamped to <= 90 and polarity is
    S1 / S2 - 1."""
    cell_n = len(axon_terminal_all)
    soma_center = np.zeros((cell_n, 2))
    axon_vector = np.zeros((cell_n, 2))
    angle1 = np.zeros(cell_n)
    S_diag = np.zeros((2, cell_n))
    for icell in range(cell_n):
        axon_current = np.asarray(axon_terminal_all[icell], dtype=float)
        axon_current_2d = axon_current[:, [2, 0]]

        soma_current_2d = np.asarray(soma_all[icell], dtype=float)
        soma_current_2d = soma_current_2d[0]
        soma_center[icell] = soma_current_2d[[2, 0]]

        # zero mean before svd
        axon_center = axon_current_2d.mean(axis=0)
        axon_current_2d = axon_current_2d - axon_center

        # axon svd
        U, S, _ = np.linalg.svd(axon_current_2d.T, full_matrices=False)
        S_diag[: S.size, icell] = S
        axon_vector[icell] = [U[0, 0], U[1, 0]]
        angle1[icell] = matlab_round(np.rad2deg(np.arctan(U[1, 0] / U[0, 0]))) + 1
    angle1[angle1 > 90] = 90
    polarity = S_diag[0, :] / S_diag[1, :] - 1
    return soma_center, axon_vector, angle1, polarity
