"""Spiral detection batch driver and spatiotemporal grouping.

Translations of
- spirals/utils/spirals_detection/spiralDetectionAlgorithm.m
- spirals/utils/spirals_detection/getGroupingAlgorithm.m
- spirals/utils/spirals_detection/groupSpirals.m

The per-frame detection chain (spiralAlgorithm / checkClusterXY /
doubleCheckSpiralsAlgorithm / spatialRefine / spiralRadiusCheck2) is
reused from ``spirals_py.spirals.plots._fig1_helpers_s1``.
"""

import time

import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _checkClusterXY,
    _doubleCheckSpiralsAlgorithm,
    _padZeros,
    _spatialRefine,
    _spiralAlgorithm,
    _spiralRadiusCheck2,
)
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq


def detect_padded_frames(tracePhase, params, frame_start):
    """The per-frame detection chain of spiralDetectionAlgorithm.m (lines
    17-29) / getSpiralsWhiskerMeanMaps.m (lines 104-116), run over a
    zero-padded phase stack.  *frame_start* is the 1-based movie frame
    number of the first frame in *tracePhase*."""
    pwAll = np.zeros((0, 5))
    nframe = tracePhase.shape[2]
    for frame in range(1, nframe + 1):
        A = tracePhase[:, :, frame - 1]
        pwAll1 = _spiralAlgorithm(A, params)  # coarse search
        frameN = frame + frame_start - 1
        pwAll2 = _checkClusterXY(pwAll1, params["dThreshold"])
        pwAll3 = _doubleCheckSpiralsAlgorithm(A, pwAll2, params)
        pwAll4 = _spatialRefine(A, pwAll3, params)
        pwAll5 = _spiralRadiusCheck2(A, pwAll4, params)
        if pwAll5.size:
            # attach frame ID label
            pwAll5 = np.column_stack([pwAll5, np.full((pwAll5.shape[0], 1), frameN)])
        else:
            pwAll5 = np.zeros((0, 5))  # MATLAB [] keeps vertcat valid
        pwAll = np.vstack([pwAll, pwAll5])
    return pwAll


def spiralDetectionAlgorithm(U1, dV, t, params, freq, rate):
    """Translated from
    spirals/utils/spirals_detection/spiralDetectionAlgorithm.m

    Batched spiral detection: epochs of params['epochL'] frames (plus
    +-35 filter-margin frames) are band-pass filtered with
    spiralPhaseMap_freq, and the per-frame detection chain is run on the
    padded phase stack.  Returns pwAll (N, 5)
    [x y radius direction frame] with unpadded 1-based frame numbers.
    """
    if "xxRoi" not in params:
        params["xxRoi"] = params["xx"].ravel()
        params["yyRoi"] = params["yy"].ravel()

    pwAll = np.zeros((0, 5))
    frameRange = params["frameRange"]
    for kkk in range(len(frameRange) - 1):  # MATLAB 1:numel(frameRange)-1
        tic = time.time()
        frameStart = frameRange[kkk]
        frameEnd = frameStart + params["epochL"] - 1
        # extra 2*35 frames before/after the filter margin
        frameTemp = np.arange(frameStart - 35, frameEnd + 35 + 1)
        dV1 = dV[:50, frameTemp - 1]
        t1 = t[frameTemp - 1]
        # filter data at a pre-defined frequency range
        _, _, tracePhase1 = spiralPhaseMap_freq(U1, dV1, t1, params, freq, rate)
        # reduce the 2*35 filter-margin frames
        tracePhase1 = tracePhase1[:, :, 35 : tracePhase1.shape[2] - 35]
        # pad with edge zeros and run the per-frame chain
        tracePhase = _padZeros(tracePhase1, params["halfpadding"])
        pwAll = np.vstack([pwAll, detect_padded_frames(tracePhase, params, frameStart)])
        print(
            f"spiralDetectionAlgorithm: frames {frameStart}/"
            f"{params['frameN1']}; {time.time() - tic:.1f} s"
        )
    if pwAll.size:
        # recalculate spiral 2d coordinates without padding
        pwAll[:, :2] = pwAll[:, :2] - params["halfpadding"]
    return pwAll


def groupSpirals(archiveCell, cell1, nextSpirals, newRound, newAdd):
    """Translated from spirals/utils/spirals_detection/groupSpirals.m

    Attach next-frame spirals within 30 px of the last spiral of an
    existing cluster; archive clusters that went >= 3 frames without a
    new spiral.  Returns (archiveCell, cell1, remain, newRound, newAdd).
    """
    grouped = []
    # last spiral of each cluster in the intermediate assembly cell1
    lastSpiralsG = np.vstack([c[-1, :] for c in cell1])
    # only consider last spirals within 3 frames of the next frame
    indx1 = lastSpiralsG[:, -1] < nextSpirals[0, -1] - 2
    lastSpiralsG[indx1, :] = np.nan
    if not np.all(np.isnan(lastSpiralsG)):
        for j in range(nextSpirals.shape[0]):
            tSpirals = nextSpirals[j, :]
            if not np.isnan(tSpirals).any():  # MATLAB if not(isnan(...))
                distance = np.linalg.norm(lastSpiralsG[:, :2] - tSpirals[:2], axis=1)
                if np.isnan(distance).all():
                    continue
                indx = int(np.nanargmin(distance))
                minV = distance[indx]
                if minV < 30:
                    # attach that spiral to the existing cluster
                    cell1[indx] = np.vstack([cell1[indx], tSpirals])
                    newAdd[indx] = newAdd[indx] + 1
                    grouped.append(j)
    # spirals that were not grouped remain
    keep = np.setdiff1d(np.arange(nextSpirals.shape[0]), np.array(grouped, dtype=int))
    remain = nextSpirals[keep, :]
    # archive clusters iterated >= 3 frames with >= 3 frames without adds
    newMiss = newRound - newAdd
    indxG = (newRound >= 3) & (newMiss >= 3)
    if indxG.any():
        archiveCell = archiveCell + [cell1[k] for k in np.flatnonzero(indxG)]
        cell1 = [cell1[k] for k in np.flatnonzero(~indxG)]
        newRound = newRound[~indxG]
        newAdd = newAdd[~indxG]
    return archiveCell, cell1, remain, newRound, newAdd


def getGroupingAlgorithm(filteredSpirals):
    """Translated from
    spirals/utils/spirals_detection/getGroupingAlgorithm.m

    Spatiotemporal grouping of detected spirals into clusters (see the
    extensive MATLAB header comment): spirals in nearby frames with
    distance < 30 pixels are grouped in a cluster; a cluster is archived
    after >= 3 frames without a nearby new spiral.  Returns
    (archiveCell, test_stats) where archiveCell is a list of (m, 5)
    arrays and test_stats flags duplicates/missed spirals.
    """
    filteredSpirals = np.atleast_2d(np.asarray(filteredSpirals, dtype=float))
    allFrames = np.unique(filteredSpirals[:, -1])  # frames with spirals
    firstFrame = allFrames[0]
    lastFrame = allFrames[-1]
    frameIterator = int(lastFrame - firstFrame + 1)

    archiveCell = []
    cell1 = []
    newRound = np.zeros(0)
    newAdd = np.zeros(0)
    currentSpirals = np.zeros((0, filteredSpirals.shape[1]))
    cFrame = allFrames[0]
    for i in range(frameIterator):
        if not cell1:
            # cell1 empty: (re)collect the current frame's not-yet-archived
            # spirals as new singleton clusters
            if len(archiveCell) > 100:
                # only check the last 100 archived clusters
                tempSpirals = np.vstack(archiveCell[-101:])
            elif archiveCell:
                tempSpirals = np.vstack(archiveCell)
            else:
                tempSpirals = np.zeros((0, filteredSpirals.shape[1]))
            currentSpirals = filteredSpirals[filteredSpirals[:, -1] == cFrame, :]
            if currentSpirals.size and tempSpirals.size:
                a = _ismember_rows_pairs(currentSpirals, tempSpirals)
                if a.any():
                    currentSpirals = currentSpirals[~a, :]
            if currentSpirals.size:
                cell1 = [row[None, :] for row in currentSpirals]
                newRound = np.zeros(len(cell1))
                newAdd = np.zeros(len(cell1))
        elif cell1 and currentSpirals.size:
            # spirals of the previous frame not attached to any cluster
            # become new clusters
            cell1 = cell1 + [row[None, :] for row in currentSpirals]
            newRound = np.concatenate([newRound, np.zeros(currentSpirals.shape[0])])
            newAdd = np.concatenate([newAdd, np.zeros(currentSpirals.shape[0])])
            currentSpirals = np.zeros((0, filteredSpirals.shape[1]))
        newRound = newRound + 1
        nextFrame = cFrame + 1
        nextSpirals = filteredSpirals[filteredSpirals[:, -1] == nextFrame, :]
        if nextSpirals.size and cell1:
            archiveCell, cell1, currentSpirals, newRound, newAdd = groupSpirals(
                archiveCell, cell1, nextSpirals, newRound, newAdd
            )
        else:
            currentSpirals = np.zeros((0, filteredSpirals.shape[1]))
        cFrame = cFrame + 1

    # check all spirals were archived, and there were no duplication
    test_stats = np.ones(2)
    groupedSpirals = (
        np.vstack(archiveCell + cell1) if (archiveCell or cell1) else np.zeros((0, 5))
    )
    if currentSpirals.size:
        groupedSpirals = np.vstack([groupedSpirals, currentSpirals])
    if groupedSpirals.size:
        groupedSpirals = groupedSpirals[np.argsort(groupedSpirals[:, 3], kind="stable")]
        uniq = np.unique(groupedSpirals, axis=0)
        if uniq.shape[0] != groupedSpirals.shape[0]:
            test_stats[0] = 0  # duplicate rows
        missed = filteredSpirals[~_ismember_rows_pairs(filteredSpirals, groupedSpirals)]
        if missed.size:
            test_stats[1] = 0  # missed spirals
    return archiveCell, test_stats


def _ismember_rows_pairs(a, b):
    """MATLAB ismember(a, b, 'rows') logical output for 2-D a, b."""
    b_view = np.ascontiguousarray(b).view([("", b.dtype)] * b.shape[1])
    a_view = np.ascontiguousarray(a).view([("", a.dtype)] * a.shape[1])
    return np.isin(a_view.ravel(), b_view.ravel())
