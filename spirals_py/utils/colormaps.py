import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

# matplotlib ships the same inferno colormap; re-export it
inferno = colormaps["inferno"]


def colormap_RedWhiteBlue(n=100, gamma=0.6):
    """Translated from utils/colormap_RedWhiteBlue.m

    Returns a matplotlib LinearSegmentedColormap matching the MATLAB values.
    """
    nn = np.arange(n + 1)
    cm = (
        np.stack(
            [
                np.concatenate([np.full(n, n), np.arange(n, -1, -1)]),
                np.concatenate([nn, np.arange(n - 1, -1, -1)]),
                np.concatenate([nn, np.full(n, n)]),
            ],
            axis=1,
        )
        / n
    ) ** gamma
    return LinearSegmentedColormap.from_list("RedWhiteBlue", cm)
