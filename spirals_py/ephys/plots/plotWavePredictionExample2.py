"""Translated from ephys/plots/plotWavePredictionExample2.m
(Extended Data Fig.12g,h: example wave prediction from midbrain spiking
data, session kk = 20, ZYE_0067_20221026_3)."""

from ._prediction_example_utils import _plot_prediction_example


def plotWavePredictionExample2(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotWavePredictionExample2.m

    Returns (hs12g, hs12h): summary figure and frame montage of raw vs
    predicted phase maps with optical-flow quiver overlay.
    """
    return _plot_prediction_example(
        T,
        data_folder,
        save_folder,
        kk=20,
        epochs=(9050, 9150),
        frame=range(44, 55),
        abcd=(50, 100, 80, 130),
        fig_tag=("FigS12g", "FigS12h"),
        wave=True,
    )
