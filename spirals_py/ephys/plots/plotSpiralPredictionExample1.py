"""Translated from ephys/plots/plotSpiralPredictionExample1.m
(Extended Data Fig.12a,b: example spiral prediction from striatal spiking
data, session kk = 10, ZYE_0041_20210903_2)."""

from ._prediction_example_utils import _plot_prediction_example


def plotSpiralPredictionExample1(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotSpiralPredictionExample1.m

    Returns (hs12a, hs12b): summary figure (anatomy, variance map, df/f
    traces, spike raster, circular variance of flow-angle difference) and
    frame montage of raw vs predicted phase maps.
    """
    return _plot_prediction_example(
        T,
        data_folder,
        save_folder,
        kk=10,
        epochs=(110950, 111050),
        frame=range(40, 51),
        abcd=(40, 90, 20, 70),
        fig_tag=("FigS12a", "FigS12b"),
        wave=False,
    )
