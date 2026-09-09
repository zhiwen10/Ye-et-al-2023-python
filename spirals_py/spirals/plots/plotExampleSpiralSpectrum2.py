"""Translated from spirals/plots/plotExampleSpiralSpectrum2.m (Extended Data
Fig.1g2): power spectrum of the LK_0003 raw cortex traces (averaged over
left-hemisphere sensory-area pixels) around the example spiral epoch."""

from spirals_py.spirals.plots._fig1_helpers_s1 import _exampleSpiralSpectrumCore


def plotExampleSpiralSpectrum2(T, data_folder, save_folder):
    """Translated from spirals/plots/plotExampleSpiralSpectrum2.m; returns hs1g2."""
    return _exampleSpiralSpectrumCore(
        T, data_folder, save_folder,
        kk=11,            # LK_0003
        first_frame=34,
        frame_anchor=70885,
        out_name="FigS1g2_example_time_series_spectrum.png",
    )
