"""Translated from spirals/preprocessing/getTaperPowerMap3.m (pipeline1,
Extended Data Fig.1c input).

Tapered power spectrum per pixel per session: V is cut into 2 s epochs
(70 frames at 35 Hz) and mean-subtracted per epoch and SVD component,
the chronux computePowerMap2 chain computes the multi-taper FFT
(params.tapers = [1 1] DPSS, pad = 0 -> nfft = 2^nextpow2(70) = 128,
fpass [0.1 8] -> 29 frequencies), |J|^2 is projected to pixels through
the 8x-downsampled U ./ mimg, averaged over epochs, upsampled 8x
(bicubic imresize), warped onto the atlas template
(projectedTemplate1) and downsampled to the 165 x 143 grid.  Saves
fftSpectrumAllNew2.mat (psdxAllNorm, freq) for plotPowerSpectrum4.

computePowerMap2.m + its chronux helpers (spirals/utils/
{getparams,change_row_to_column,getfgrid,dpsschk,mtfftc}.m) are folded
into _compute_power_map2 (not assigned modules).  Taper details: MATLAB
dpss returns unit-energy tapers, dpsschk scales them by sqrt(Fs) and
mtfftc divides the padded FFT by Fs; scipy.signal.windows.dpss is
unit-energy as well, so the scaling transfers exactly.  Only |J|^2 is
used, so DPSS sign conventions are irrelevant.  nfft = max(2^
(nextpow2(N)+pad), N); the frequency grid is 0:Fs/nfft:Fs cut to nfft
points and restricted to fpass (endpoint-inclusive), as getfgrid.m.

Deviations / dead code of the MATLAB source:
- psdxAllNorm is preallocated zeros(165,143,29,15) in MATLAB; here the
  session count comes from len(T) and the frequency count from the
  computed grid (identical for the 15 released sessions);
- the channel FFT is chunked (identical results, bounded memory);
- dead code skipped: BW, dV, the mask_ZYE12.mat load, trialT,
  SpixelMean (:,:,1:end) (no-op), SpixelMean2 (pixel mean, unused) and
  the commented-out per-session save; projectedTemplate1 is the
  variable of that name in tables/
  isocortex_horizontal_projection_outline.mat.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.signal.windows import dpss
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _imwarp_row,
    _load_session_row,
    _load_tform,
)
from spirals_py.spirals_mirror.plots._helpers import _imresize
from spirals_py.utils.matio import save_mat73

_PARAMS = {"tapers": [1, 1], "Fs": 35.0, "fpass": [0.1, 8], "pad": 0}
_EPOCH = 35 * 2  # number of timestamps for 2 seconds


def _compute_power_map2(data, params, chunk=65536):
    """computePowerMap2.m (chronux getparams/getfgrid/dpsschk/mtfftc):
    tapered FFT of data (N samples x C channels); returns J
    (findx, K, C) scaled 1/Fs and the fpass-restricted frequency
    vector f."""
    tapers = params["tapers"]
    pad = params.get("pad", 0)
    Fs = params.get("Fs", 1.0)
    fpass = params.get("fpass", [0.0, Fs / 2])
    N, C = data.shape
    nfft = max(2 ** (int(np.ceil(np.log2(N))) + pad), N)
    f = np.arange(nfft) * (Fs / nfft)  # getfgrid: 0:df:Fs cut to nfft
    findx = (f >= fpass[0]) & (f <= fpass[-1])
    TW, K = int(tapers[0]), int(tapers[1])
    w = dpss(N, TW, K)
    w = w / np.sqrt((w**2).sum(axis=1, keepdims=True)) * np.sqrt(Fs)

    nf = int(findx.sum())
    J = np.empty((nf, K, C), dtype=complex)
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        for k in range(K):
            Jk = np.fft.fft(data[:, lo:hi] * w[k][:, None], n=nfft, axis=0) / Fs
            J[:, k, lo:hi] = Jk[findx]
    return J, f[findx]


def getTaperPowerMap3(T, data_folder, save_folder):
    """Translated from spirals/preprocessing/getTaperPowerMap3.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    with h5py.File(outline_file, "r") as f:
        template_shape = np.asarray(f["projectedTemplate1"]).T.shape

    n_sessions = len(T)
    psdxAllNorm = None
    freq = None
    for kk in tqdm(range(n_sessions), desc="getTaperPowerMap3"):
        # session info
        _, _, _, fname, U, V, _, mimg, _ = _load_session_row(data_folder, T, kk + 1)
        # registration
        tform = _load_tform(
            data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat"
        )

        epoch = _EPOCH
        epochN = V.shape[1] // epoch
        # trialV(:, k, :) = V(:, 1+(k-1)*epoch : k*epoch)
        trialV = V[:, : epochN * epoch].reshape(V.shape[0], epochN, epoch)
        nV = trialV.shape[0]

        thisV = trialV.reshape(nV * epochN, epoch, order="F")
        thisV = thisV - thisV.mean(axis=1, keepdims=True)
        J1, f1 = _compute_power_map2(thisV.T, _PARAMS)
        nTaper = _PARAMS["tapers"][1]
        nf = f1.size

        # J1 (nf, K, C) -> (nV, nf*K*epochN): freq fastest, taper, epoch
        J1 = J1.reshape(nf, nTaper, nV, epochN, order="F")
        J1 = J1.transpose(2, 0, 1, 3)  # permute [3 1 2 4]
        J1 = J1.reshape(nV, nf * nTaper * epochN, order="F")

        # example trace SVD (MATLAB reshape column-major pixel order)
        U1 = U[::8, ::8, :] / mimg[::8, ::8][:, :, None]
        Ur = U1.reshape(-1, U1.shape[2], order="F")
        Jpixel = Ur @ J1
        Spixel = (Jpixel.conj() * Jpixel).real
        Spixel = Spixel.reshape(Spixel.shape[0], nf, nTaper * epochN, order="F")
        SpixelMean = Spixel.mean(axis=2)  # squeeze(mean(Spixel, 3))
        SpixelMean = SpixelMean.reshape(U1.shape[0], U1.shape[1], nf, order="F")

        SpixelMean1 = _imresize(SpixelMean, (U1.shape[0] * 8, U1.shape[1] * 8))
        # imwarp onto projectedTemplate1, then 1:8:end
        psdxMeanTransformed = _imwarp_row(SpixelMean1, tform, template_shape, stride=8)

        if psdxAllNorm is None:
            psdxAllNorm = np.zeros(
                psdxMeanTransformed.shape[:2] + (nf, n_sessions)
            )
        psdxAllNorm[:, :, :, kk] = psdxMeanTransformed
        freq = f1

    save_mat73(
        save_folder / "fftSpectrumAllNew2.mat",
        {"psdxAllNorm": psdxAllNorm, "freq": freq.reshape(1, -1)},
    )
