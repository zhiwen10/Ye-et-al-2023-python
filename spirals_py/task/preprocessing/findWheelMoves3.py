import warnings

import numpy as np

from spirals_py.task.plots._task_helpers import _gausswin


def findWheelMoves3(
    pos,
    t,
    Fs,
    posThresh=8.0,
    tThresh=0.2,
    minGap=0.1,
    posThreshOnset=1.5,
    minDur=0.05,
    batchSize=10000,
):
    """Translated from task/preprocessing/findWheelMoves3.m

    IBL wheel-movement detector: a sample is 'moving' when the position
    changes by more than posThresh within the next tThresh seconds;
    small gaps are merged, then precise onsets are found by looking back
    from the end of each window until the position is within
    posThreshOnset of the window start.

    Returns (moveOnsets, moveOffsets, moveAmps, peakVelTimes), times in
    seconds.  The makePlots option of the MATLAB source is not
    implemented (it is never used by the pipeline).

    Faithful details kept from the MATLAB implementation:
    - windows are computed in overlapping batches and truncated at each
      batch end (MATLAB hankel(pos(batch), nan(1, tThreshSamps)));
    - in the precise-onset pass only onsets inside each batch's first
      batchSize - tThreshSamps - 1 samples are processed (an onset
      landing exactly on a batch boundary keeps its approximate sample,
      because its deviation row stays zero-initialized);
    - MATLAB max/min skip NaN, so NaN-padded truncated windows shrink.
    """
    if batchSize <= 0:
        raise ValueError("batchSize must be positive")
    Fs = float(Fs)
    rawT = np.asarray(t, dtype=float).ravel()
    rawPos = np.asarray(pos, dtype=float).ravel()

    # evenly-sampled position
    n = int(np.floor((rawT[-1] - rawT[0]) * Fs)) + 1
    t = rawT[0] + np.arange(n) / Fs
    pos = np.interp(t, rawT, rawPos)
    L = t.size
    T = int(round(tThresh * Fs))
    if T < 1:
        raise ValueError("tThresh * Fs must round to at least 1 sample")

    # total deviation of each tThresh-long window (batched)
    totalDev = np.full(L, np.nan)
    c = 0
    while True:
        i2proc = np.arange(1, batchSize + 1) + c  # 1-based sample indices
        i2proc = i2proc[i2proc <= L]
        if i2proc.size == 0:
            break
        seg = pos[i2proc - 1]
        m = seg.size
        idx = np.arange(m)[:, None] + np.arange(T)[None, :]
        w = np.full((m, T), np.nan)
        ok = idx < m
        w[ok] = seg[idx[ok]]  # past the batch end: NaN, as in hankel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            totalDev[i2proc - 1] = np.nanmax(w, axis=1) - np.nanmin(w, axis=1)
        if i2proc[-1] == L:
            break
        c += batchSize - T
        if batchSize - T <= 0:
            break

    isMoving = np.concatenate([[False], totalDev > posThresh])
    isMoving[-1] = False  # make sure we end on an offset

    def _onset_samples():
        return np.flatnonzero(~isMoving[:-1] & isMoving[1:]) + 1  # 1-based

    def _offset_samples():
        return np.flatnonzero(isMoving[:-1] & ~isMoving[1:]) + 1  # 1-based

    # fill in small gaps
    moveOnsetSamps = _onset_samples()
    moveOffsetSamps = _offset_samples()
    if moveOnsetSamps.size >= 2 and moveOffsetSamps.size >= 1:
        tooShort = np.flatnonzero(
            (moveOnsetSamps[1:] - moveOffsetSamps[:-1]) / Fs < minGap
        )
        for q in tooShort:
            # MATLAB isMoving(offset : onset) = true (1-based, inclusive)
            isMoving[moveOffsetSamps[q] - 1 : moveOnsetSamps[q + 1]] = True

    # precise movement onset samples
    moveOnsetSamps = _onset_samples()
    nOn = moveOnsetSamps.size
    if nOn == 0:
        empty = np.empty(0)
        return empty, empty, empty, np.empty(0)

    wheelToepOnsetsDev = np.zeros((nOn, T))
    c = 0
    while True:
        i2proc = np.arange(1, batchSize + 1) + c  # 1-based
        # MATLAB: intersect(i2proc(1:end-tThreshSamps-1), moveOnsetSamps)
        if i2proc.size > T + 1:
            prefix = i2proc[: i2proc.size - T - 1]
        else:
            prefix = i2proc[:0]
        for row in np.flatnonzero(np.isin(moveOnsetSamps, prefix)):
            m = moveOnsetSamps[row]
            w = pos[m - 1 : m - 1 + T]
            wheelToepOnsetsDev[row] = np.abs(w - w[0])
        i2proc = i2proc[i2proc <= L]
        if i2proc.size == 0 or i2proc[-1] >= moveOnsetSamps[-1]:
            break
        c += batchSize - T
        if batchSize - T <= 0:
            break

    # look back from the window end for the last sample close to the onset
    hasOnset = wheelToepOnsetsDev > posThreshOnset
    onsetLags = np.zeros(nOn, dtype=int)
    for row in range(nOn):
        quiet = np.flatnonzero(~hasOnset[row])
        if quiet.size:
            onsetLags[row] = quiet[-1]  # MATLAB tThreshSamps - min(b) = col - 1
    moveOnsetSamps = moveOnsetSamps + onsetLags
    moveOnsets = t[moveOnsetSamps - 1]

    # offsets: actual end of isMoving
    moveOffsetSamps = _offset_samples()
    moveOffsets = t[moveOffsetSamps - 1]

    # drop movements that are too brief
    moveDurs = moveOffsets - moveOnsets
    keep = ~(moveDurs < minDur)
    moveOnsetSamps = moveOnsetSamps[keep]
    moveOnsets = moveOnsets[keep]
    moveOffsetSamps = moveOffsetSamps[keep]
    moveOffsets = moveOffsets[keep]

    # join movements separated by less than minGap
    if moveOnsets.size:
        moveGaps = moveOnsets[1:] - moveOffsets[:-1]
        gapTooSmall = moveGaps < minGap
        moveOnsets = moveOnsets[np.concatenate([[True], ~gapTooSmall])]
        moveOnsetSamps = moveOnsetSamps[np.concatenate([[True], ~gapTooSmall])]
        moveOffsets = moveOffsets[np.concatenate([[~gapTooSmall], [True]])]
        moveOffsetSamps = moveOffsetSamps[np.concatenate([[~gapTooSmall], [True]])]

    moveAmps = pos[moveOffsetSamps - 1] - pos[moveOnsetSamps - 1]

    vel = np.convolve(np.diff(np.concatenate([[0.0], pos])), _gausswin(10), "same")
    peakVelTimes = np.full(moveOnsets.size, np.nan)
    for m in range(moveOnsets.size):
        thisV = np.abs(vel[moveOnsetSamps[m] - 1 : moveOffsetSamps[m]])
        if thisV.size:
            peakVelTimes[m] = moveOnsets[m] + (np.argmax(thisV) + 1) / Fs

    return moveOnsets, moveOffsets, moveAmps, peakVelTimes
