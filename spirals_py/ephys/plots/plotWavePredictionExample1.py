"""Translated from ephys/plots/plotWavePredictionExample1.m
(Extended Data Fig.12e,f: example wave prediction from striatal spiking
data, session kk = 10, ZYE_0041_20210903_2)."""

from ._prediction_example_utils import _plot_prediction_example


def plotWavePredictionExample1(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotWavePredictionExample1.m

    Returns (hs12e, hs12f): summary figure and frame montage of raw vs
    predicted phase maps with optical-flow quiver overlay.
    """
    return _plot_prediction_example(
        T,
        data_folder,
        save_folder,
        kk=10,
        epochs=(62950, 63050),
        frame=range(25, 36),
        abcd=(40, 90, 20, 70),
        fig_tag=("FigS12e", "FigS12f"),
        wave=True,
    )
