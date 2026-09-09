"""Translated from spirals/plots/plotExampleOscillation.m (Extended Data Fig.1a-b)."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import butter, filtfilt

from spirals_py.spirals import loadUVt1
from spirals_py.utils import load_outline_coords_h5, overlayOutlines


def _load_h5(path, varname):
    """Load one variable from a MATLAB v7.3 file, restoring MATLAB axis order."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[varname]).T


def _load_tform(path):
    """Load an affine2d tform saved in a v7.3 .mat file; returns MATLAB T (3x3).

    MATLAB affine2d: [x_out y_out 1] = [x_in y_in 1] @ T."""
    with h5py.File(path, "r") as f:
        for key in f["#refs#"]:
            grp = f["#refs#"][key]
            if isinstance(grp, h5py.Group) and "TransformationMatrix" in grp:
                return np.asarray(grp["TransformationMatrix"]).T
    raise ValueError(f"No TransformationMatrix found in {path}")


def _imwarp(img, T, out_shape, step=1):
    """MATLAB imwarp(img, tform, 'OutputView', imref2d(out_shape)) with bilinear
    interpolation and fill value 0. `step` evaluates only every `step`-th output
    pixel in both dimensions (equivalent to imwarp followed by 1:step:end)."""
    invT = np.linalg.inv(T)
    rows = np.arange(0, out_shape[0], step)
    cols = np.arange(0, out_shape[1], step)
    cc, rr = np.meshgrid(cols, rows)  # 0-based pixel indices
    # intrinsic coordinates of output pixel centers are 1-based (x = col, y = row)
    x = cc + 1.0
    y = rr + 1.0
    xin = invT[0, 0] * x + invT[0, 1] * y + invT[0, 2]
    yin = invT[1, 0] * x + invT[1, 1] * y + invT[1, 2]
    coords = np.array([yin - 1.0, xin - 1.0])  # 0-based (row, col)
    if img.ndim == 2:
        return map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
    out = np.empty(coords.shape[1:] + (img.shape[2],), dtype=np.float64)
    for k in range(img.shape[2]):
        out[..., k] = map_coordinates(img[..., k], coords, order=1, mode="constant", cval=0.0)
    return out


def _cbrewer2(ctype, cname, n):
    """Translated from dependencies/cbrewer2/cbrewer2.m (exact palette sizes only)."""
    palettes = {
        ("qual", "Set1", 9): [
            [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
            [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
            [153, 153, 153],
        ],
        ("seq", "YlOrRd", 7): [
            [255, 255, 178], [254, 217, 118], [254, 178, 76], [253, 141, 60],
            [252, 78, 42], [227, 26, 28], [177, 0, 38],
        ],
        ("seq", "YlOrRd", 9): [
            [255, 255, 204], [255, 237, 160], [254, 217, 118], [254, 178, 76],
            [253, 141, 60], [252, 78, 42], [227, 26, 28], [189, 0, 38],
            [128, 0, 38],
        ],
        ("seq", "OrRd", 6): [
            [254, 240, 217], [253, 212, 158], [253, 187, 132], [252, 141, 89],
            [227, 74, 51], [179, 0, 0],
        ],
        ("seq", "Greys", 9): [
            [255, 255, 255], [240, 240, 240], [217, 217, 217], [189, 189, 189],
            [150, 150, 150], [115, 115, 115], [82, 82, 82], [37, 37, 37],
            [0, 0, 0],
        ],
    }
    return np.array(palettes[(ctype, cname, n)], dtype=float) / 255.0


def plotExampleOscillation(data_folder, save_folder):
    """Translated from spirals/plots/plotExampleOscillation.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # load atlas brain horizontal projection and outline (10um resolution)
    outline_file = data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    projectedAtlas1 = _load_h5(outline_file, "projectedAtlas1")
    coords = load_outline_coords_h5(outline_file, "coords")

    mn = "ZYE_0052"
    tdb = "20211218"
    en = 2
    fname = f"{mn}_{tdb}_{en}"
    session_root = data_folder / "spirals" / "svd" / fname
    U, V, t, mimg = loadUVt1(session_root)  # load U, V, t
    dV = np.concatenate([np.zeros((V.shape[0], 1)), np.diff(V, axis=1)], axis=1)

    lick_wheel = data_folder / "spirals" / "spirals_example" / f"{fname}_lick_wheel.mat"
    wheelE = _load_h5(lick_wheel, "wheelE").ravel()
    licking = _load_h5(lick_wheel, "licking").ravel()
    lick_t = _load_h5(lick_wheel, "lick_t").ravel()

    # registration: load atlas transformation matrix tform
    tform = _load_tform(data_folder / "spirals" / "rf_tform" / f"{fname}_tform.mat")

    sizeTemplate = [1320, 1140]
    mimgt = _imwarp(mimg.astype(np.float64), tform, sizeTemplate)

    BW = projectedAtlas1 != 0

    # ZYE52 points [row, col]
    points = np.array([
        [764, 640],  # RSP
        [848, 860],  # V1
        [534, 834],  # S1
        [328, 748],  # MO
    ])
    # sample U and mimg at the inverse-mapped locations of the 4 points;
    # equivalent to Ut1 = imwarp(U)/imwarp(mimg) indexed at those pixels
    invT = np.linalg.inv(tform)
    xy = points[:, ::-1].astype(float)  # (x = col, y = row), 1-based intrinsic
    xin = invT[0, 0] * xy[:, 0] + invT[0, 1] * xy[:, 1] + invT[0, 2]
    yin = invT[1, 0] * xy[:, 0] + invT[1, 1] * xy[:, 1] + invT[1, 2]
    n_sv = 50
    coords3 = np.vstack([
        np.repeat(yin - 1.0, n_sv),
        np.repeat(xin - 1.0, n_sv),
        np.tile(np.arange(n_sv), 4),
    ])
    u_pts = map_coordinates(U.astype(np.float64), coords3, order=1, mode="constant", cval=0.0).reshape(4, n_sv)
    mimg_pts = map_coordinates(mimg.astype(np.float64), np.vstack([yin - 1.0, xin - 1.0]), order=1, mode="constant", cval=0.0)
    Ut1 = u_pts / mimg_pts[:, None]

    f1, f2 = butter(2, np.array([2.0, 8.0]) / (35.0 / 2.0), btype="bandpass")
    trace = np.zeros((4, dV.shape[1]))
    for i in range(4):
        trace[i] = Ut1[i].astype(float) @ dV[:50, :]
        trace[i] = filtfilt(f1, f2, trace[i])

    color1 = _cbrewer2("qual", "Set1", 9)
    hs1ab, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.75))
    ax1.imshow(mimgt, cmap="gray", alpha=BW.astype(float))
    overlayOutlines(coords, 1, "w", ax=ax1)
    ax1.axis("off")
    ax1.set_aspect("equal")
    for i in range(4):
        ax1.scatter(points[i, 1], points[i, 0], s=24, color=color1[i + 1], marker="o")

    ax2.plot(t, wheelE / 10000 + 5000, linewidth=1, color="k")
    ax2.plot(t, licking * 20 + 4000, linewidth=1, color=color1[0])
    ax2.vlines(lick_t, 4000, 4400, color="k")  # MATLAB: plot([lick_t,lick_t],[0,1]*400+4000)
    for i in range(4):
        ax2.plot(t, trace[i] * 90000 + 1000 * i, linewidth=1, color=color1[i + 1])
    ax2.plot([484, 484], [1000, 1900], "r")
    ax2.set_yticks(np.arange(0, 6000, 1000))
    ax2.set_yticklabels(["RSP", "VISp", "SSp", "MO", "licking", "wheel"])
    ax2.set_xlim([476, 486])
    ax2.set_xticks(np.arange(476, 487, 2))
    ax2.set_xlabel("Time (s)")

    hs1ab.savefig(save_folder / "FigS1ab_example_trace.pdf", bbox_inches="tight")
    plt.show()
    return hs1ab
