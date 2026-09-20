"""Translated from whisker/preprocessing/sortSpirals.m"""

import numpy as np

from spirals_py.task.preprocessing._task_utils import ismember_rows

from ._whisker_session import bin_spirals_by_frame
from .SpiralCountBins import SpiralCountBins


def sortSpirals(pwAll, indexSSp, frames_stimOn):
    """Translated from whisker/preprocessing/sortSpirals.m

    Restrict pwAll to the [x y] pixels of indexSSp, bin the spirals into
    the frames_stimOn (nTrials, nCols) trial window and return the
    per-radius/per-frame flag-count (7, nCols) of SpiralCountBins.
    """
    lia2 = ismember_rows(pwAll[:, :2], indexSSp)
    pwAll2 = pwAll[lia2]
    spiral_cell2 = bin_spirals_by_frame(pwAll2, frames_stimOn)
    spiral_count_sum = SpiralCountBins(spiral_cell2)
    return spiral_count_sum
