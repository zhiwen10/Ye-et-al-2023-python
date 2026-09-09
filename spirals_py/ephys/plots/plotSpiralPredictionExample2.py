"""Translated from ephys/plots/plotSpiralPredictionExample2.m
(Extended Data Fig.12c,d: example spiral prediction from midbrain spiking
data, session kk = 20, ZYE_0067_20221026_3)."""

from ._prediction_example_utils import _plot_prediction_example


def plotSpiralPredictionExample2(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotSpiralPredictionExample2.m

    Returns (hs12c, hs12d): summary figure and frame montage of raw vs
    predicted phase maps.
    """
    return _plot_prediction_example(
        T,
        data_folder,
        save_folder,
        kk=20,
        epochs=(250, 350),
        frame=range(29, 40),
        abcd=(50, 100, 80, 130),
        fig_tag=("FigS12c", "FigS12d"),
        wave=False,
    )
