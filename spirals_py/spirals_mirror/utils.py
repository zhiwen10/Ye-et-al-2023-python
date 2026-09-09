"""Translated from spirals_mirror/utils/redoSVD.m

(other spirals_mirror/utils functions, CanonCor2.m and sseExplainedCal.m, are
only called by the preprocessing functions and are not translated here)
"""
import numpy as np


def _normc(X):
    """MATLAB normc: scale each column to unit 2-norm."""
    X = np.asarray(X)
    n = np.sqrt((X.astype(np.float64) ** 2).sum(axis=0))
    n[n == 0] = 1.0
    return (X / n.astype(X.dtype)).astype(X.dtype)


def redoSVD(U, V, nSVD=2000):
    """Translated from spirals_mirror/utils/redoSVD.m

    Get raw data from a set of U (usually a selected subspace from the
    original U), V and compute a new set of Unew and Vnew. Batches the
    U*V products like the MATLAB original.

    Returns Unew (uSize x nSVD), Vnew (nSVD x ntotframes) and Sv
    (nSVD,) as float32, matching the MATLAB single-precision pipeline.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    NavgFramesSVD = 5000
    ntotframes = V.shape[1]
    nt0 = int(np.ceil(ntotframes / NavgFramesSVD))
    NavgFramesSVD = int(np.floor(ntotframes / nt0))
    imgbatchSize = nt0 * int(np.floor(1000 / nt0))
    nbatch = int(np.floor(ntotframes / imgbatchSize))
    # read U, V and average nearby frames
    uSize = U.shape[0]
    ix = 0
    mov = np.zeros((uSize, NavgFramesSVD), dtype=np.float32)
    for ibatch in range(1, nbatch + 1):
        batchStart = (ibatch - 1) * imgbatchSize
        batchEnd = ibatch * imgbatchSize
        data = U @ V[:, batchStart:batchEnd]
        data = data.reshape(uSize, nt0, -1)
        davg = data.mean(axis=1)  # MATLAB mean(data, 2)
        mov[:, ix : ix + davg.shape[1]] = davg
        ix = ix + davg.shape[1]
    mov = mov[:, :ix]  # MATLAB: mov(:, :, (ix+1):end) = []
    # get COV matrix for mov
    mov = mov - mov.mean(axis=1, keepdims=True)
    COV = (mov.T @ mov) / mov.shape[0]
    useGPU = 0
    # svd for COV matrix to get new Ua
    if nSVD < 1000 or COV.shape[0] > 1e4:
        from scipy.sparse.linalg import eigsh

        vals, vecs = eigsh(COV.astype(np.float64), k=min(nSVD, COV.shape[0] - 2))
        order = np.argsort(vals)[::-1]
        Va = vecs[:, order]
        Sv = vals[order]
    else:
        if useGPU:
            raise NotImplementedError("GPU branch not translated")
        Usvd, Ssvd, _ = np.linalg.svd(COV, full_matrices=False)
        Va = Usvd  # COV is symmetric PSD: singular vectors = eigenvectors
        Sv = Ssvd
    Va = Va[:, :nSVD]
    Sv = Sv[:nSVD].astype(np.float32)
    Ua = _normc(mov @ Va).astype(np.float32)
    # get new V (Fs) from new U and data, batch by batch
    ix = 0
    Fs = np.zeros((nSVD, ntotframes), dtype=np.float32)
    for ibatch in range(1, nbatch + 2):
        batchStart = (ibatch - 1) * imgbatchSize
        if ibatch == nbatch + 1 and ntotframes % imgbatchSize:
            batchEnd = ntotframes
        else:
            batchEnd = ibatch * imgbatchSize
        batchEnd = min(batchEnd, ntotframes)
        if batchEnd <= batchStart:
            continue
        data = U @ V[:, batchStart:batchEnd]
        # subtract mean as we did before
        data = data - data.mean(axis=1, keepdims=True)
        Fs[:, ix : ix + data.shape[1]] = Ua.T @ data
        ix = ix + data.shape[1]
    return Ua, Fs, Sv
