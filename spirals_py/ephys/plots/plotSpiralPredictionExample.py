"""Translated from ephys/plots/plotSpiralPredictionExample.m
(Figure 4b-e: example spiral prediction from thalamic spiking data,
session kk = 6, ZYE_0060_20220326_1)."""

from ._prediction_example_utils import _plot_prediction_example


def plotSpiralPredictionExample(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotSpiralPredictionExample.m

    Returns (h4bc, h4de): summary figure (anatomy, variance map, df/f
    traces, spike raster, circular variance of flow-angle difference) and
    frame montage of raw vs predicted phase maps with optical-flow quiver
    overlay.
    """
    return _plot_prediction_example(
        T,
        data_folder,
        save_folder,
        kk=6,
        epochs=(13700, 13800),
        frame=range(45, 56),
        abcd=(30, 80, 20, 70),
        fig_tag=("Fig4bc", "Fig4de"),
        wave=True,
    )
