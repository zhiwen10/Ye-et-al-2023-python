import numpy as np
from scipy.signal import convolve2d


def _conv2_same(A, kernel):
    """MATLAB conv2(A, kernel, 'same').

    For even-sized kernel dimensions MATLAB keeps the central part starting
    at offset floor(k/2), shifted by one element relative to scipy's
    mode='same' (e.g. MATLAB conv([1 2 3],[1 1],'same') == [3 5 3]).
    """
    A = np.asarray(A, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    full = convolve2d(A, kernel, mode="full")
    r0 = kernel.shape[0] // 2
    c0 = kernel.shape[1] // 2
    return full[r0 : r0 + A.shape[0], c0 : c0 + A.shape[1]]


def computeDerivatives_mod(im1, im2):
    """Translated from utils/horn_schunck/computeDerivatives_mod.m

    (the MATLAB file's internal function is named computeDerivatives)
    """
    im1 = np.asarray(im1, dtype=float)
    if im2 is None or im2.size == 0:
        im2 = np.zeros_like(im1)

    # by Nick Steinmetz
    fx = np.angle(np.exp(1j * _conv2_same(im1, [[-1, 1]]))) / 2 + np.angle(
        np.exp(1j * _conv2_same(im2, [[-1, 1]]))
    ) / 2
    fy = np.angle(np.exp(1j * _conv2_same(im1, [[-1], [1]]))) / 2 + np.angle(
        np.exp(1j * _conv2_same(im2, [[-1], [1]]))
    ) / 2
    ft = np.angle(np.exp(1j * (im1 - im2)))

    fx = np.angle(np.exp(1j * fx))
    fy = np.angle(np.exp(1j * fy))
    return fx, fy, ft


def HS_phase_mod(im1, im2):
    """Translated from utils/horn_schunck/HS_phase_mod.m

    Note: the MATLAB function accepts extra arguments but unconditionally
    overwrites them with the hardcoded values used here.
    """
    alpha = 1
    ite = 100
    u = np.zeros_like(im1, dtype=float)
    v = np.zeros_like(im2, dtype=float)

    fx, fy, ft = computeDerivatives_mod(im1, im2)
    kernel_1 = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]])
    for _ in range(ite):
        uAvg = _conv2_same(u, kernel_1)
        vAvg = _conv2_same(v, kernel_1)
        u = uAvg - (fx * ((fx * uAvg) + (fy * vAvg) + ft)) / (alpha**2 + fx**2 + fy**2)
        v = vAvg - (fy * ((fx * uAvg) + (fy * vAvg) + ft)) / (alpha**2 + fx**2 + fy**2)

    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0
    return u, v


def HS_flowfield(tracePhase, useGPU=False):
    """Translated from utils/horn_schunck/HS_flowfield.m

    frameN is the first dimension. useGPU is accepted for signature
    compatibility but ignored (no GPU support).
    """
    frameN = tracePhase.shape[0]
    vxRaw = np.zeros((frameN - 1, tracePhase.shape[1], tracePhase.shape[2]))
    vyRaw = np.zeros((frameN - 1, tracePhase.shape[1], tracePhase.shape[2]))

    for k in range(frameN - 1):
        A1 = tracePhase[k]
        A2 = tracePhase[k + 1]
        vxRaw[k], vyRaw[k] = HS_phase_mod(A1, A2)
        if (k + 1) % 100 == 0:
            print(f"frame {k + 1}/{frameN}")

    return vxRaw, vyRaw
