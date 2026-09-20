import numpy as np


def getConcatTrials(spiral_all):
    """Translated from task/preprocessing/getConcatTrials.m

    Concatenate the spiral arrays of all trials (rows of the
    (nTrials, 141) cell array *spiral_all*) frame by frame.
    Returns a list of 141 (n, 5) arrays.
    """
    spiral_all = np.asarray(spiral_all, dtype=object)
    n_trials = spiral_all.shape[0]
    out = []
    for i in range(141):
        current = [
            spiral_all[m, i] if spiral_all[m, i] is not None else np.zeros((0, 5))
            for m in range(n_trials)
        ]
        if current:
            out.append(np.vstack(current))
        else:
            out.append(np.zeros((0, 5)))
    return out
