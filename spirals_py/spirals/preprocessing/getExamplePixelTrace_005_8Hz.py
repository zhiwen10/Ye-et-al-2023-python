"""Translated from revision/power_spectrum/getExamplePixelTrace_005_8Hz.m
(pipeline1 revision, Extended Data Fig.1d).

Per-session FFT power spectra of the 8 hardcoded area-pixel example
traces (VISp, RSP, 5x SSp, MOs; session 3 uses shifted SSp-m / SSp-n
pixels): U is warped to the atlas frame and 8x downsampled, each pixel
trace is U_pixel * V(1:50,:) / warped mimg, cut into 20 s epochs (700
frames at 35 Hz) and mean-subtracted, and fft_spectrum (one-sided
periodogram, 1/(Fs*N) scaling with doubled interior bins) is averaged
over epochs.  Saves example_traces_005_8Hz/<fname>_fft.mat (freq1,
psdx_mean (351, 8)) for plotExamplePowerSpectrum2 /
plotPowerRatioRegression3.

fft_spectrum.m (revision/power_spectrum/fft_spectrum.m) is translated
here as _fft_spectrum (not an assigned module).

Dead code of the MATLAB source skipped: the dV computation, mimgt, the
butter/filtfilt/hilbert block (traceFilt..traceAmp feed nothing), the
horizontal_cortex_atlas_50um.mat / get_cortex_atlas_path / root1 / ctx /
color1 / nameList definitions and the clearvars list.  The pixel trace
extraction shared with getPowerBandRatio / setAlphaThreshold is
_example_pixel_traces.
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm

from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _imwarp_row,
    _load_session_row,
    _load_tform,
)
from spirals_py.task.plots._task_helpers_s15 import load_projectedAtlas1, matlab_round
from spirals_py.utils.matio import save_mat73

_FS = 35.0
_PARAMS = {"downscale": 8, "lowpass": 0, "gsmooth": 0}
_PIXEL = np.array(
    [
        [845, 835],  # VISp
        [775, 650],  # RSP
        [590, 750],  # SSp-ul
        [520, 850],  # SSp-ll
        [480, 950],  # SSp-m
        [550, 950],  # SSp-n
        [682, 905],  # SSp-bfd
        [290, 700],  # MOs
    ],
    dtype=float,
)


def _example_pixel_traces(T, data_folder, kk):
    """Session block shared by getExamplePixelTrace_005_8Hz /
    getPowerBandRatio / setAlphaThreshold (kk 0-based): returns
    (fname, rawTrace, rawTraceV) where rawTrace uses dV(1:50,:) and
    rawTraceV V(1:50,:), both divided by the warped mimg."""
    fname, U, V, dV, mimg = _session_warp(T, data_folder, kk)
    rows, cols, mimg_pix = _pixel_rc(data_folder, U, mimg, kk, fname)
    Utransformed_pix = U[rows, cols, :]  # (8, 50)
    rawTrace = (Utransformed_pix @ dV[:50, :]) / mimg_pix[:, None]
    rawTraceV = (Utransformed_pix @ V[:50, :]) / mimg_pix[:, None]
    return fname, rawTrace, rawTraceV


def _session_warp(T, data_folder, kk):
    """Load session kk (0-based), warp U (50 components) and mimg into the
    atlas frame at 1/8 resolution (_imwarp_row with stride equals the
    MATLAB imwarp followed by 1:downscale:end)."""
    _, _, _, fname, U, V, _, mimg, dV = _load_session_row(data_folder, T, kk + 1)
    tform = _load_tform(
        Path(data_folder) / "spirals" / "rf_tform" / f"{fname}_tform.mat"
    )
    downscale = _PARAMS["downscale"]
    atlas_shape = load_projectedAtlas1(data_folder).shape
    U = _imwarp_row(U[:, :, :50].astype(np.float64), tform, atlas_shape, stride=downscale)
    mimg = _imwarp_row(mimg.astype(np.float64), tform, atlas_shape, stride=downscale)
    return fname, U, V, dV, mimg


def _pixel_rc(data_folder, U, mimg, kk, fname=None):
    """0-based (rows, cols) of the 8 area pixels on the downsampled
    warped grid, and the warped mimg at those pixels (MATLAB
    round(pixel/downscale), half away from zero)."""
    pixel_copy = _PIXEL.copy()
    if kk == 2:  # MATLAB kk == 3
        pixel_copy[4] = [560, 920]
        pixel_copy[5] = [655, 890]
    pixel_copy = matlab_round(pixel_copy / _PARAMS["downscale"]).astype(int)
    rows = pixel_copy[:, 0] - 1
    cols = pixel_copy[:, 1] - 1
    if not (
        rows.min() >= 0
        and cols.min() >= 0
        and rows.max() < U.shape[0]
        and cols.max() < U.shape[1]
    ):
        raise IndexError(f"area pixels outside warped grid for {fname}")
    return rows, cols, mimg[rows, cols]


def _fft_spectrum(trace):
    """Translated from revision/power_spectrum/fft_spectrum.m (Fs = 35).

    trace is ntrace x timestamps; returns (freq1, psdx (nfreq x ntrace),
    psdx_mean).  One-sided periodogram: psdx = |fft(:, 1:N/2+1)|^2 /
    (Fs*N) with the interior bins doubled (MATLAB fft scaling)."""
    x = trace - trace.mean(axis=1, keepdims=True)
    N = x.shape[1]
    Fs = _FS
    # MATLAB fft of the transposed traces: (N x ntrace) -> one-sided
    xdft1 = np.fft.fft(x, axis=1)[:, : N // 2 + 1].T
    psdx = (1.0 / (Fs * N)) * np.abs(xdft1) ** 2
    psdx[1:-1, :] = 2 * psdx[1:-1, :]
    freq1 = np.arange(N // 2 + 1) * (Fs / N)
    psdx_mean = psdx.mean(axis=1)
    return freq1, psdx, psdx_mean


def _bandpass_hilbert(rawTrace):
    """2-8 Hz butter(2) filtfilt + hilbert of the row traces; returns
    (traceFilt, tracePhase, traceAmp).  MATLAB filtfilt pads with
    3*(max(len(a),len(b))-1) odd samples (scipy default differs)."""
    freq = np.array([2.0, 8.0])
    f1, f2 = butter(2, freq / (_FS / 2), btype="bandpass")
    padlen = 3 * (max(len(f1), len(f2)) - 1)
    traceFilt = filtfilt(f1, f2, rawTrace, axis=1, padlen=padlen)
    traceHilbert = hilbert(traceFilt, axis=1)
    return traceFilt, np.angle(traceHilbert), np.abs(traceHilbert)


def getExamplePixelTrace_005_8Hz(T, data_folder, save_folder):
    """Translated from
    revision/power_spectrum/getExamplePixelTrace_005_8Hz.m"""
    save_folder1 = Path(save_folder) / "example_traces_005_8Hz"
    save_folder1.mkdir(parents=True, exist_ok=True)

    for kk in tqdm(range(len(T)), desc="getExamplePixelTrace_005_8Hz"):
        fname, _, rawTrace = _example_pixel_traces(T, data_folder, kk)  # V-based traces

        rawTrace = rawTrace - rawTrace.mean(axis=1, keepdims=True)
        epochLength = int(20 * _FS)
        epochs = rawTrace.shape[1] // epochLength
        # MATLAB traceEpoch(i, epoch, :) = rawTrace over the epoch windows
        traceEpoch = rawTrace[:, : epochs * epochLength].reshape(
            8, epochs, epochLength
        )
        traceEpoch1 = traceEpoch.reshape(8 * epochs, epochLength, order="F")

        freq1, psdx, _ = _fft_spectrum(traceEpoch1)
        psdx = psdx.reshape(freq1.size, 8, epochs, order="F")
        psdx_mean = psdx.mean(axis=2)

        save_mat73(
            save_folder1 / f"{fname}_fft.mat",
            {"freq1": freq1.reshape(1, -1), "psdx_mean": psdx_mean},
        )
