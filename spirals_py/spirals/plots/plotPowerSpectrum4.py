"""Translated from spirals/plots/plotPowerSpectrum4.m (Extended Data Fig.1c)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.utils import load_outline_coords_h5, overlayOutlines


def _load_h5(path, varname):
    """Load one variable from a MATLAB v7.3 file, restoring MATLAB axis order."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[varname]).T


def _bandpower_psd(pxx, freq, f1, f2):
    """MATLAB bandpower(pxx, freq, [f1 f2], 'psd'): rectangle-rule integration of
    an already-computed PSD over the band (endpoints included). pxx: (..., nfreq)."""
    mask = (freq >= f1) & (freq <= f2)
    return pxx[..., mask].sum(axis=-1) * (freq[1] - freq[0])


def plotPowerSpectrum4(T, data_folder, save_folder):
    """Translated from spirals/plots/plotPowerSpectrum4.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # load atlas brain horizontal projection and outline (10um resolution)
    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    projectedAtlas1 = _load_h5(outline_file, "projectedAtlas1")
    coords = load_outline_coords_h5(outline_file, "coords")

    pixel = np.array([
        [845, 835],  # VISp
        [775, 650],  # RSP
        [520, 850],  # SSp-ll
        [290, 700],  # MOs
    ])
    scale = 8
    pixel = np.round(pixel / scale).astype(int)

    # load tapered power spectrum for all sessions
    spec_file = data_folder / "spirals" / "spirals_power_spectrum2" / "fftSpectrumAllNew2.mat"
    psdxAllNorm = _load_h5(spec_file, "psdxAllNorm")  # (165, 143, 29, 15)
    freq = _load_h5(spec_file, "freq").ravel()
    BW = projectedAtlas1[::scale, ::scale] != 0

    powerRatio1 = np.full(BW.shape + (len(T),), np.nan)
    for kk in range(len(T)):
        psdxAllNormMean = psdxAllNorm[:, :, :, kk]
        psdx2 = psdxAllNormMean.reshape(-1, psdxAllNormMean.shape[2])  # (npix, nfreq)
        powerAlpha = _bandpower_psd(psdx2, freq, 2, 7.9)
        powerTotal = _bandpower_psd(psdx2, freq, 0.3, 7.9)
        powerRatio = powerAlpha / powerTotal
        powerRatio[powerRatio > 1] = np.nan
        powerRatio[powerRatio < 0] = np.nan
        powerRatio1[:, :, kk] = powerRatio.reshape(BW.shape)
    powerRatio2 = np.nanmean(powerRatio1, axis=2)

    indx1 = int(np.argmin(np.abs(freq - 2)))
    indx2 = int(np.argmin(np.abs(freq - 8)))
    psdxAllNormMean = psdxAllNorm.mean(axis=3)
    # sum power between 2-8Hz (MATLAB indx1:indx2 inclusive)
    powerAlpha = psdxAllNormMean[:, :, indx1 : indx2 + 1].sum(axis=2)

    hs1c, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.25, 3.75))
    im1 = ax1.imshow(powerAlpha, cmap="hot", alpha=BW.astype(float))
    overlayOutlines(coords, scale, "w", ax=ax1)
    ax1.axis("off")
    ax1.set_aspect("equal")
    hs1c.colorbar(im1, ax=ax1)
    for i in range(4):
        ax1.scatter(pixel[i, 1], pixel[i, 0], s=12, color="k")

    im2 = ax2.imshow(powerRatio2, cmap="hot", alpha=BW.astype(float), vmin=0, vmax=0.4)
    overlayOutlines(coords, scale, "w", ax=ax2)
    for i in range(4):
        ax2.scatter(pixel[i, 1], pixel[i, 0], s=12, color="k")
    ax2.axis("off")
    ax2.set_aspect("equal")
    cb2 = hs1c.colorbar(im2, ax=ax2)
    cb2.set_ticks(np.arange(0, 0.5, 0.1))
    cb2.set_ticklabels(["0", "0.1", "0.2", "0.3", "0.4"])

    hs1c.savefig(save_folder / "FigS1c_2-8hz_power_map_15mice.pdf", bbox_inches="tight")
    plt.show()
    return hs1c
