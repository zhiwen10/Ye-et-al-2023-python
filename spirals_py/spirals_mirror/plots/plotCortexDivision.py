"""Translated from spirals_mirror/plots/plotCortexDivision.m (Figure 3b)."""
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spirals_py.utils.atlas import plotOutline

# colorcet C06 map (256 x 3), extracted from dependencies/colorcet/colorcet.m
_C06 = np.fromstring(
    """
    0.967258 0.214042 0.102845 0.966570 0.219832 0.091900 0.966331 0.228338 0.081637 0.966519 0.239172 0.071996
    0.967102 0.251979 0.063086 0.968036 0.266329 0.054632 0.969272 0.281871 0.046867 0.970767 0.298273 0.039573
    0.972473 0.315307 0.032902 0.974330 0.332735 0.027416 0.976304 0.350364 0.022767 0.978352 0.368072 0.018802
    0.980444 0.385766 0.015391 0.982543 0.403389 0.012442 0.984627 0.420846 0.009625 0.986675 0.438147 0.007288
    0.988672 0.455241 0.005171 0.990611 0.472130 0.003222 0.992482 0.488803 0.001403 0.994272 0.505275 0.000000
    0.995981 0.521558 0.000000 0.997608 0.537666 0.000000 0.999139 0.553589 0.000000 1.000000 0.569344 0.000000
    1.000000 0.584942 0.000000 1.000000 0.600370 0.000000 1.000000 0.615646 0.000000 1.000000 0.630773 0.000000
    1.000000 0.645716 0.000000 1.000000 0.660484 0.000000 1.000000 0.675041 0.000000 1.000000 0.689353 0.000000
    1.000000 0.703360 0.000000 1.000000 0.717014 0.000000 1.000000 0.730228 0.000000 1.000000 0.742928 0.000000
    0.997013 0.755011 0.000000 0.993130 0.766371 0.000000 0.988329 0.776911 0.000000 0.982556 0.786513 0.000000
    0.975736 0.795073 0.000000 0.967848 0.802510 0.000000 0.958855 0.808751 0.000000 0.948767 0.813729 0.000000
    0.937612 0.817431 0.000000 0.925428 0.819845 0.000000 0.912286 0.821003 0.000000 0.898259 0.820945 0.000000
    0.883442 0.819745 0.000000 0.867926 0.817497 0.000000 0.851803 0.814297 0.000000 0.835174 0.810255 0.000000
    0.818120 0.805486 0.000000 0.800730 0.800102 0.000000 0.783061 0.794197 0.000000 0.765184 0.787863 0.000000
    0.747133 0.781199 0.000000 0.728950 0.774261 0.000000 0.710670 0.767120 0.000000 0.692310 0.759819 0.000000
    0.673870 0.752399 0.000000 0.655374 0.744889 0.000000 0.636825 0.737311 0.000000 0.618208 0.729683 0.000000
    0.599517 0.722003 0.000000 0.580768 0.714300 0.000088 0.561966 0.706602 0.001364 0.543085 0.698894 0.002744
    0.524134 0.691206 0.004275 0.505112 0.683549 0.006019 0.486026 0.675934 0.008055 0.466891 0.668390 0.010459
    0.447696 0.660936 0.013625 0.428506 0.653626 0.017232 0.409328 0.646501 0.021673 0.390210 0.639615 0.027147
    0.371224 0.633033 0.033854 0.352444 0.626828 0.042288 0.333947 0.621087 0.051310 0.315826 0.615887 0.061030
    0.298170 0.611326 0.071492 0.281144 0.607485 0.082491 0.264861 0.604438 0.094141 0.249454 0.602260 0.106460
    0.235087 0.601018 0.119351 0.221824 0.600750 0.132930 0.209884 0.601479 0.147119 0.199340 0.603211 0.161961
    0.190342 0.605917 0.177370 0.182908 0.609585 0.193351 0.177115 0.614136 0.209879 0.172868 0.619533 0.226924
    0.170148 0.625665 0.244390 0.168788 0.632475 0.262299 0.168606 0.639871 0.280529 0.169448 0.647759 0.299096
    0.171027 0.656083 0.317888 0.173211 0.664739 0.336896 0.175777 0.673694 0.356074 0.178617 0.682872 0.375370
    0.181467 0.692238 0.394792 0.184329 0.701733 0.414298 0.187046 0.711358 0.433869 0.189539 0.721058 0.453513
    0.191760 0.730837 0.473200 0.193686 0.740671 0.492943 0.195221 0.750557 0.512737 0.196441 0.760481 0.532591
    0.197199 0.770419 0.552475 0.197513 0.780361 0.572398 0.197387 0.790299 0.592363 0.196812 0.800219 0.612347
    0.195731 0.810088 0.632372 0.194229 0.819890 0.652406 0.192201 0.829594 0.672443 0.189741 0.839135 0.692449
    0.186833 0.848469 0.712380 0.183498 0.857524 0.732221 0.179800 0.866222 0.751891 0.175772 0.874463 0.771333
    0.171544 0.882150 0.790461 0.167203 0.889172 0.809197 0.162846 0.895409 0.827424 0.158662 0.900747 0.845049
    0.154786 0.905081 0.861958 0.151447 0.908303 0.878038 0.148796 0.910333 0.893193 0.146996 0.911104 0.907341
    0.146145 0.910582 0.920396 0.146272 0.908746 0.932321 0.147403 0.905621 0.943092 0.149439 0.901233 0.952696
    0.152216 0.895660 0.961167 0.155674 0.888981 0.968552 0.159518 0.881296 0.974916 0.163693 0.872718 0.980351
    0.167887 0.863362 0.984945 0.172057 0.853333 0.988796 0.176027 0.842762 0.992012 0.179752 0.831735 0.994684
    0.183085 0.820339 0.996910 0.186032 0.808673 0.998776 0.188478 0.796786 1.000000 0.190407 0.784752 1.000000
    0.191837 0.772596 1.000000 0.192739 0.760370 1.000000 0.193096 0.748094 1.000000 0.192898 0.735780 1.000000
    0.192140 0.723457 1.000000 0.190862 0.711130 1.000000 0.189087 0.698802 1.000000 0.186788 0.686516 1.000000
    0.184010 0.674267 1.000000 0.180788 0.662075 1.000000 0.177226 0.649939 1.000000 0.173330 0.637891 1.000000
    0.169319 0.625957 1.000000 0.165226 0.614171 1.000000 0.161462 0.602586 1.000000 0.158143 0.591244 1.000000
    0.155704 0.580216 1.000000 0.154501 0.569561 1.000000 0.155005 0.559385 1.000000 0.157568 0.549762 1.000000
    0.162547 0.540788 1.000000 0.170054 0.532570 1.000000 0.180129 0.525186 1.000000 0.192687 0.518745 1.000000
    0.207468 0.513305 1.000000 0.224143 0.508952 1.000000 0.242446 0.505707 1.000000 0.262021 0.503635 1.000000
    0.282580 0.502716 1.000000 0.303812 0.502940 1.000000 0.325533 0.504288 1.000000 0.347546 0.506662 1.000000
    0.369627 0.510027 1.000000 0.391686 0.514292 1.000000 0.413584 0.519347 1.000000 0.435251 0.525109 1.000000
    0.456595 0.531482 1.000000 0.477598 0.538379 1.000000 0.498229 0.545699 1.000000 0.518449 0.553369 1.000000
    0.538274 0.561341 1.000000 0.557697 0.569539 1.000000 0.576746 0.577926 1.000000 0.595415 0.586468 1.000000
    0.613739 0.595124 1.000000 0.631747 0.603884 1.000000 0.649445 0.612708 1.000000 0.666873 0.621602 1.000000
    0.684044 0.630551 1.000000 0.700981 0.639500 1.000000 0.717697 0.648450 1.000000 0.734210 0.657392 1.000000
    0.750547 0.666275 1.000000 0.766714 0.675093 1.000000 0.782732 0.683789 1.000000 0.798604 0.692327 1.000000
    0.814326 0.700619 1.000000 0.829902 0.708608 1.000000 0.845304 0.716198 1.000000 0.860517 0.723283 1.000000
    0.875501 0.729774 1.000000 0.890207 0.735534 0.999657 0.904582 0.740457 0.993875 0.918551 0.744419 0.987002
    0.932045 0.747317 0.978941 0.944985 0.749039 0.969640 0.957277 0.749516 0.959055 0.968856 0.748694 0.947169
    0.979649 0.746529 0.933992 0.989595 0.743021 0.919579 0.998650 0.738178 0.903989 1.000000 0.732068 0.887313
    1.000000 0.724744 0.869658 1.000000 0.716301 0.851139 1.000000 0.706836 0.831891 1.000000 0.696460 0.812014
    1.000000 0.685290 0.791640 1.000000 0.673431 0.770879 1.000000 0.660986 0.749815 1.000000 0.648057 0.728551
    1.000000 0.634732 0.707149 1.000000 0.621067 0.685666 1.000000 0.607133 0.664149 1.000000 0.592967 0.642637
    1.000000 0.578604 0.621156 1.000000 0.564087 0.599728 1.000000 0.549411 0.578359 1.000000 0.534593 0.557081
    1.000000 0.519636 0.535864 1.000000 0.504548 0.514748 1.000000 0.489292 0.493723 1.000000 0.473908 0.472799
    1.000000 0.458378 0.451997 1.000000 0.442700 0.431298 1.000000 0.426857 0.410724 1.000000 0.410864 0.390277
    1.000000 0.394716 0.370000 1.000000 0.378449 0.349916 1.000000 0.362094 0.330040 1.000000 0.345681 0.310420
    1.000000 0.329330 0.291088 0.998725 0.313084 0.272096 0.994867 0.297137 0.253559 0.991053 0.281634 0.235487
    0.987329 0.266827 0.217935 0.983746 0.253015 0.200971 0.980371 0.240489 0.184719 0.977255 0.229689 0.169173
    0.974453 0.221039 0.154320 0.972022 0.214834 0.140283 0.969990 0.211521 0.127002 0.968399 0.211229 0.114510
    """,
    sep=" ",
).reshape(256, 3)


def _colorcet(N=9):
    """Translated from dependencies/colorcet/colorcet.m (C06 map only).

    colorcet('C06', 'N', N): linear subsampling of the 256-entry map.
    """
    if N >= _C06.shape[0]:
        return _C06.copy()
    xi = np.arange(N) * (_C06.shape[0] - 1) / (N - 1)
    return np.stack([np.interp(xi, np.arange(_C06.shape[0]), _C06[:, j]) for j in range(3)], axis=1)


# structure tree paths (areaPath lists from the MATLAB plot functions)
SENSORY_AREA_PATHS = [
    "/997/8/567/688/695/315/453/",  # SS
    "/997/8/567/688/695/315/247/",  # AUD
    "/997/8/567/688/695/315/669/",  # VIS
    "/997/8/567/688/695/315/254/",  # RSP
    "/997/8/567/688/695/315/22/312782546/",  # VISa
    "/997/8/567/688/695/315/22/417/",  # VISrl
    "/997/8/567/688/695/315/541/",  # TEa
    "/997/8/567/688/695/315/677/",  # VISC (areaPath(9) in MATLAB)
    "/997/8/567/688/695/1089/",  # HPF
    "/997/8/567/688/695/315/677/",  # CTXsp
]
FRONTAL_AREA_PATHS = [
    "/997/8/567/688/695/315/500/",  # MO
    "/997/8/567/688/695/315/31/",  # ACA
    "/997/8/567/688/695/315/972/",  # PL
    "/997/8/567/688/695/315/44/",  # ILA
]
MASK_PATHS = [
    "/997/8/567/688/695/315/500/985/",  # MOp
    "/997/8/567/688/695/315/500/993/",  # MOs
    "/997/8/567/688/695/315/31/",  # ACA
    "/997/8/567/688/695/315/453/378/",  # SS2
    "/997/8/567/688/695/315/453/322/",  # SSp
    "/997/8/567/688/695/315/247/",  # AUD
    "/997/8/567/688/695/315/669/",  # VIS
    "/997/8/567/688/695/315/254/",  # RSP
    "/997/8/567/688/695/315/22",  # VISa
    "/997/8/567/688/695/315/541/",  # TEa
    "/997/8/567/688/695/315/677/",  # VISC
]


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m"""
    st = pd.read_csv(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    return MASK_PATHS, st


def _load_atlas_tables(data_folder):
    """Load the v7.3 atlas/template tables in MATLAB (row, col) orientation."""
    tables = Path(data_folder) / "tables"
    with h5py.File(tables / "horizontal_cortex_atlas_50um.mat", "r") as h:
        atlas1 = np.array(h["atlas1"]).T
    with h5py.File(tables / "horizontal_cortex_template_50um.mat", "r") as h:
        template1 = np.array(h["template1"]).T
    with h5py.File(tables / "isocortex_horizontal_projection_outline.mat", "r") as h:
        projectedAtlas1 = np.array(h["projectedAtlas1"]).T
        projectedTemplate1 = np.array(h["projectedTemplate1"]).T
    return atlas1, template1, projectedAtlas1, projectedTemplate1


def _select_area(area_paths, spath, Utransformed, projectedAtlas1, hemi, scale):
    """Translated from spirals/utils/select_area.m

    spath: pandas Series of structure_id_path strings; MATLAB st.index is the
    0-based row number (loadStructureTree.m). index is a MATLAB-style linear
    (column-major) index into the downsampled atlas grid.
    """
    mask = np.zeros(len(spath), dtype=bool)
    for p in area_paths:
        mask |= spath.str.startswith(p).to_numpy()
    idFilt = np.flatnonzero(mask)
    pa = projectedAtlas1.copy()
    pa[~np.isin(pa, idFilt)] = 0
    if hemi == "right":
        pa[:, : pa.shape[1] // 2] = 0
    elif hemi == "left":
        pa[:, pa.shape[1] // 2 :] = 0
    pa2 = pa[::scale, ::scale]
    index = np.flatnonzero(pa2.ravel(order="F"))
    if Utransformed is not None and Utransformed.ndim == 3:
        UregDown = Utransformed[::scale, ::scale, :]
        UregDown1d = UregDown.reshape(-1, UregDown.shape[2], order="F")
        Uselected = UregDown1d[index, :]
    else:
        Uselected = np.zeros((index.size, 1), dtype=np.float32)
    return index, Uselected


def _bw_mask(index, shape=(165, 143)):
    # (fix: BW.ravel(order="F") returns a copy, so the assignment has to be
    # done on a flat vector that is then reshaped back column-major)
    flat = np.zeros(shape[0] * shape[1])
    flat[index] = 1
    return flat.reshape(shape, order="F")


def plotCortexDivision(data_folder, save_folder):
    """Translated from spirals_mirror/plots/plotCortexDivision.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    atlas1, template1, projectedAtlas1, projectedTemplate1 = _load_atlas_tables(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    spath = st["structure_id_path"].astype(str)

    # only select cortex in the atlas
    spath2 = spath.str.startswith("/997/8/567/")
    idFilt = np.flatnonzero(spath2.to_numpy())
    Lia = np.isin(projectedAtlas1, idFilt)
    projectedAtlas1 = projectedAtlas1.copy()
    projectedAtlas1[~Lia] = 0

    scale = 8
    template2 = projectedTemplate1[::scale, ::scale]

    point = np.array(
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

    # mask for right and left sensory cortex
    indexright, _ = _select_area(SENSORY_AREA_PATHS, spath, None, projectedAtlas1, "right", scale)
    indexleft, _ = _select_area(SENSORY_AREA_PATHS, spath, None, projectedAtlas1, "left", scale)
    BW_left = _bw_mask(indexleft)
    BW_right = _bw_mask(indexright)

    # right SSp and MO index
    indexSSp, _ = _select_area(SENSORY_AREA_PATHS, spath, None, projectedAtlas1, "right", scale)
    indexMO, _ = _select_area(FRONTAL_AREA_PATHS, spath, None, projectedAtlas1, "right", scale)
    BW_MO = _bw_mask(indexMO)
    BW_SSp = _bw_mask(indexSSp)

    color1 = _colorcet(9)
    scale3 = 5 / 8
    lineColor = "k"
    lineColor1 = "w"

    h3b, axs = plt.subplots(2, 2, figsize=(7, 7))

    # ap division: MO
    ax = axs[0, 0]
    ax.imshow(template2, cmap="gray", alpha=BW_MO)
    plotOutline(maskPath[0:3], st, atlas1, "right", scale3, lineColor, ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    # ap division: SSp + points
    ax = axs[0, 1]
    ax.imshow(template2, cmap="gray", alpha=BW_SSp)
    for kkk in range(8):
        ax.scatter(
            point[kkk, 1] - 1,
            point[kkk, 0] - 1,
            s=24,
            color=color1[kkk],
            edgecolors="none",
        )
    plotOutline(maskPath[3], st, atlas1, "right", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[4], st, atlas1, "right", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[5:11], st, atlas1, "right", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[3:11], st, atlas1, "right", scale3, lineColor, ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    # hemi division: left
    ax = axs[1, 0]
    ax.imshow(template2, cmap="gray", alpha=BW_left)
    plotOutline(maskPath[3], st, atlas1, "left", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[5:11], st, atlas1, "left", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[3:11], st, atlas1, "left", scale3, lineColor, ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    # SSp + points (right)
    ax = axs[1, 1]
    ax.imshow(template2, cmap="gray", alpha=BW_SSp)
    for kkk in range(8):
        ax.scatter(
            point[kkk, 1] - 1,
            point[kkk, 0] - 1,
            s=24,
            color=color1[kkk],
            edgecolors="none",
        )
    plotOutline(maskPath[3], st, atlas1, "right", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[4], st, atlas1, "right", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[5:11], st, atlas1, "right", scale3, lineColor1, ax=ax)
    plotOutline(maskPath[3:11], st, atlas1, "right", scale3, lineColor, ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")

    h3b.savefig(save_folder / "Fig3b_cortex_division.pdf", bbox_inches="tight")
    plt.show()
    return h3b
