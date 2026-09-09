"""Shared private helpers for the spirals_mirror plot modules
(adapted from the equivalents in spirals_py/spirals/plots/plotSpiralTimeSeries3d.py
and spirals_py/whisker/preprocessing/_whisker_utils.py)."""
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from skimage.transform import resize

from spirals_py.spirals_mirror.plots.plotCortexDivision import (
    FRONTAL_AREA_PATHS,
    MASK_PATHS,
    SENSORY_AREA_PATHS,
    _colorcet,
    _load_atlas_tables,
    _select_area,
    _bw_mask,
    _get_cortex_atlas_path,
)


def _load_h5_var(path, varname):
    """Load a numeric variable from a v7.3 .mat file, transposed back to
    MATLAB orientation (MATLAB stores arrays with reversed dimensions)."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[varname]).T


def _load_tform(path):
    """Load a v7.3 affine2d object; returns its 3x3 TransformationMatrix T in
    MATLAB orientation, with [u v 1] = [x y 1] * T (transformPointsForward)."""
    with h5py.File(path, "r") as f:
        T = None
        for name in f["#refs#"]:
            g = f["#refs#"][name]
            if isinstance(g, h5py.Group) and "TransformationMatrix" in g:
                T = np.asarray(g["TransformationMatrix"]).T
                break
    if T is None:
        raise ValueError(f"No TransformationMatrix found in {path}")
    return T


def _imwarp(img, T, out_shape, stride=1):
    """MATLAB imwarp(img, affine2d(T), 'OutputView', imref2d(out_shape)),
    bilinear interpolation, FillValues=0. T is the row-convention
    TransformationMatrix from _load_tform ([u v 1] = [x y 1] * T, translation
    in the bottom row), so the output->input map is [x y 1] = [u v 1] * T^-1.
    stride>1 samples the output grid at 1:stride:end, matching imwarp
    followed by (1:stride:end, 1:stride:end) slicing. img may be 2-D or 3-D
    (each 2-D plane is warped, as imwarp does for N-D input). Preserves the
    input dtype (float32 stays float32)."""
    Tinv = np.linalg.inv(T)
    rows = np.arange(0, out_shape[0], stride) + 1  # 1-based output y centers
    cols = np.arange(0, out_shape[1], stride) + 1  # 1-based output x centers
    X, Y = np.meshgrid(cols, rows)
    ones = np.ones_like(X)
    uv = np.stack([X, Y, ones], axis=-1) @ Tinv  # (R, C, 3)
    xin = uv[..., 0] / uv[..., 2] - 1  # 0-based input column
    yin = uv[..., 1] / uv[..., 2] - 1  # 0-based input row
    coords = [yin, xin]

    def warp_plane(plane):
        return ndimage.map_coordinates(
            np.asarray(plane), coords, order=1, mode="constant", cval=0
        )

    if img.ndim == 2:
        return warp_plane(img)
    planes = [warp_plane(img[:, :, k]) for k in range(img.shape[2])]
    return np.stack(planes, axis=2)


def _zscore_rows(X):
    """MATLAB zscore(X, [], 2): z-score each row (std with N-1)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (X - mu) / sd


def _matlab_round(a):
    """MATLAB round(): half away from zero (np.round is half-to-even)."""
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _imresize(img, out_hw):
    """MATLAB imresize(img, [h, w]): bicubic (order=3), antialiasing only
    when shrinking."""
    out_hw = (int(out_hw[0]), int(out_hw[1]))
    return resize(
        np.asarray(img, dtype=float),
        out_hw,
        order=3,
        anti_aliasing=(out_hw[0] < img.shape[0]) or (out_hw[1] < img.shape[1]),
        preserve_range=True,
    )


def _imshow_color_overlay(ax, intensity, color, square_normalize=True):
    """MATLAB pattern: imshow(thisColorImage) with set(...,'AlphaData',
    colorIntensity), where thisColorImage is a solid-color RGB image.

    square_normalize=True reproduces the kernel overlays
    (colorIntensity = (TheColorImage/maxI).^2); False uses the intensity
    directly as alpha (already in [0,1]; NaN -> 0, matching MATLAB's
    transparent AlphaData=NaN)."""
    intensity = np.nan_to_num(np.asarray(intensity, dtype=float), nan=0.0)
    if square_normalize:
        alpha = np.clip((intensity / intensity.max()) ** 2, 0, 1)
    else:
        alpha = np.clip(intensity, 0, 1)
    rgb = np.ones(intensity.shape + (3,)) * np.asarray(color, dtype=float)
    return ax.imshow(rgb, alpha=alpha)


def _kernel_slice(k1_real, index_from, index_to, shape, pr, pc):
    """squeeze(kernel_full2(:,:,pr,pc)) from the MATLAB kernel construction:

        kernel_temp2 = zeros(numel,1);
        kernel_temp2(index_from) = k1_real(:, j);
        kernel_full(:, index_to(j)) = kernel_temp2;

    computed lazily for a single target pixel (the 4-D dense tensor is never
    needed; this is exactly the corresponding (pr,pc) slice). pr, pc are
    1-based (row, col) like MATLAB. Returns None if (pr,pc) is outside
    index_to (MATLAB would give an all-zero slice)."""
    img = np.zeros(shape)
    lin0 = (pc - 1) * shape[0] + (pr - 1)  # 0-based column-major linear index
    j = int(np.searchsorted(index_to, lin0))
    if j >= index_to.size or index_to[j] != lin0:
        return img
    flat = img.ravel(order="F")
    flat[index_from] = k1_real[:, j]
    img = flat.reshape(shape, order="F")
    return img


def _build_regression(UA, VA, UB, VB, UselectedA, UnewA, Vtest):
    """Shared reduced-rank regression of plotExampleKernelHEMI/AP.

    MATLAB (dimensionally-inconsistent) line ``kk1 = regressor'\\signal1``
    is computed as its least-squares equivalent
    ``kk1 = (R*R')^-1 * R*signalB'`` (see the plot module docstrings), where
    R = zscore(VA) and signalB = UB*VB.

    UA/VA, UB/VB: (Unew(:,1:50), Vnew(1:50,:)) from redoSVD for the two
    areas; UselectedA: raw selected pixels of area A; UnewA: all redoSVD
    components of area A; Vtest: V(:, 78000:end)."""
    regressor1 = _zscore_rows(VA)
    regressor = regressor1[:, :58000]
    kk1 = np.linalg.solve(
        regressor @ regressor.T,
        (regressor @ np.asarray(VB, dtype=np.float64).T) @ np.asarray(UB, dtype=np.float64).T,
    )
    k1_real = np.asarray(UA, dtype=np.float64) @ kk1
    # prepare test regressor
    data = np.asarray(UselectedA, dtype=np.float64) @ Vtest
    # subtract mean as we did before
    data = data - data.mean(axis=1, keepdims=True)
    regressor_test = np.asarray(UnewA, dtype=np.float64).T @ data
    regressor_test_z = _zscore_rows(regressor_test)
    return kk1, k1_real, regressor_test_z


def _ttest2_scalar(x, y):
    """MATLAB ttest2(x, y) for scalar x (pooled two-sample t-test);
    returns (t, p). The MATLAB plots compute it but do not plot it."""
    from scipy import stats

    y = np.asarray(y, dtype=float)
    n1, n2 = 1, y.size
    sp = np.sqrt((n2 - 1) * y.var(ddof=1) / (n1 + n2 - 2))
    t = (float(x) - y.mean()) / (sp * np.sqrt(1 / n1 + 1 / n2))
    p = 2 * stats.t.cdf(-abs(t), n1 + n2 - 2)
    return t, p


# 8 example points (1-based rows/cols on the 165 x 143 downscaled grid),
# identical in plotKernelMaps*/plotMapsSession
POINT8 = np.array(
    [
        [84, 117],  # SSp-bfd
        [69, 122],  # SSp-n
        [55, 118],  # SSp-m
        [65, 105],  # SSp-ll
        [73, 96],  # SSp-ul
        [83, 93],  # SSp-tr
        [97, 79],  # RSP
        [115, 105],  # VISp
    ]
)

NAME8 = ["SSp-bfd", "SSp-n", "SSp-m", "SSp-ll", "SSp-ul", "SSp-tr", "RSP", "VISp"]
