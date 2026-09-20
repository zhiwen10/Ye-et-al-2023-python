"""Translated from whisker/preprocessing/SpiralCountBins.m"""

import numpy as np


def SpiralCountBins(spiral_cell):
    """Translated from whisker/preprocessing/SpiralCountBins.m

    For every radius bin (40:10:100) and every trial/frame cell, flag
    whether the cell holds at least one spiral with exactly that radius
    (MATLAB spiral_temp(:,3) == radius(j), exact equality); the flags are
    summed over trials and divided by the trial count.  Returns
    (7, nCols).
    """
    radius = np.arange(40, 101, 10)
    trialN = spiral_cell.shape[0]
    spiral_count_sum_all = np.zeros((radius.size, spiral_cell.shape[1]))
    for j in range(radius.size):
        spiral_count = np.zeros(spiral_cell.shape)
        for m in range(spiral_cell.shape[0]):
            for n in range(spiral_cell.shape[1]):
                spiral_temp = spiral_cell[m, n]
                if spiral_temp.size:
                    indx = spiral_temp[:, 2] == radius[j]
                    if spiral_temp[indx].size:
                        spiral_count[m, n] = 1
        spiral_count_sum_all[j] = spiral_count.sum(axis=0) / trialN
    return spiral_count_sum_all
