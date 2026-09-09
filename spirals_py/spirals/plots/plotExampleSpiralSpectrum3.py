"""Translated from spirals/plots/plotExampleSpiralSpectrum3.m (Extended Data
Fig.1i2): power spectrum of the ZYE_0067 raw cortex traces (averaged over
left-hemisphere sensory-area pixels) around the example spiral epoch."""

from spirals_py.spirals.plots._fig1_helpers_s1 import _exampleSpiralSpectrumCore


def plotExampleSpiralSpectrum3(T, data_folder, save_folder):
    """Translated from spirals/plots/plotExampleSpiralSpectrum3.m; returns hs1i2."""
    return _exampleSpiralSpectrumCore(
        T, data_folder, save_folder,
        kk=12,            # ZYE_0067
        first_frame=35,
        frame_anchor=89053,
        out_name="FigS1i2_example_time_series_spectrum.png",
    )
