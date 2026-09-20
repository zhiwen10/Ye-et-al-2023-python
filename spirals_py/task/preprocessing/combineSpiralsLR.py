import numpy as np


def combineSpiralsLR(spiral_L, spiral_R):
    """Translated from task/preprocessing/combineSpiralsLR.m

    Combine left- and right-stimulus spiral lists (each a list of 141
    per-frame arrays) frame by frame."""
    spiral_all = []
    for i in range(141):
        L = spiral_L[i] if spiral_L[i] is not None else np.zeros((0, 5))
        R = spiral_R[i] if spiral_R[i] is not None else np.zeros((0, 5))
        spiral_all.append(np.vstack([L, R]))
    return spiral_all
