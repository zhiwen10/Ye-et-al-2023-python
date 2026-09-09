from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import gaussian_kde, pearsonr

from spirals_py.spirals.plots.plotSpiralSyncIndex import (
    _load_projected_atlas,
    _session_info,
)


def _scatter_kde(ax, x, y, size=3):
    """Translated from utils/scatter_kde.m (2-D kernel density colored
    scatter; gaussian_kde approximates MATLAB ksdensity)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    c = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    return ax.scatter(x, y, s=size, c=c)


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m"""
    st = pd.read_csv(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    maskPath = [
        "/997/8/567/688/695/315/500/985/",  # MOp
        "/997/8/567/688/695/315/500/993/",  # MOs
        "/997/8/567/688/695/315/31/",       # ACA
        "/997/8/567/688/695/315/453/378/",  # SS2
        "/997/8/567/688/695/315/453/322/",  # SSp
        "/997/8/567/688/695/315/247/",      # AUD
        "/997/8/567/688/695/315/669/",      # VIS
        "/997/8/567/688/695/315/254/",      # RSP
        "/997/8/567/688/695/315/22",        # VISa
        "/997/8/567/688/695/315/541/",      # TEa
        "/997/8/567/688/695/315/677/",      # VISC
    ]
    return maskPath, st


def _select_area(sensoryArea, st, projectedAtlas1, hemi):
    """Translated from spirals/utils/select_area.m (simplified: scale=1,
    U/template handling dropped since the result is unused downstream).
    MATLAB st.index is the 0-based row number in the structure tree, which is
    what projectedAtlas1 stores."""
    spath = st["structure_id_path"].astype(str)
    mask = np.zeros(len(st), dtype=bool)
    for p in sensoryArea:
        mask |= spath.str.startswith(p).to_numpy()
    idFilt1 = np.flatnonzero(mask)  # == st.index(spath3) in MATLAB
    atlas = projectedAtlas1.copy()
    Lia1 = np.isin(atlas, idFilt1)
    atlas[~Lia1] = 0
    if hemi == "right":
        atlas[:, : atlas.shape[1] // 2] = 0
    elif hemi == "left":
        atlas[:, atlas.shape[1] // 2:] = 0
    index = np.flatnonzero(atlas)
    return index


def plotMotionEnergyAmpX(T, data_folder, save_folder):
    """Translated from spirals/plots/plotMotionEnergyAmpX.m

    Correlation between 2-8Hz amplitude and motion energy across 13 sessions.
    Returns the figure handle."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()

    projectedAtlas1, _ = _load_projected_atlas(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    areaPath = ["/997/8/567/688/695/315/453/322/"]  # SSp
    indexright = _select_area(areaPath, st, projectedAtlas1, "right")
    BW_right = np.zeros(projectedAtlas1.shape, dtype=int)
    BW_right[np.unravel_index(indexright, BW_right.shape)] = 1

    R1 = np.empty(13)
    p1 = np.empty(13)
    count1 = 0
    data1 = None
    for kk in list(range(0, 6)) + list(range(8, 15)):  # MATLAB [1:6, 9:15]
        mn, tdb, en, fname = _session_info(T, kk)
        amp_file = data_folder / "spirals" / "spirals_index" / f"{fname}_amp.mat"
        me_file = data_folder / "spirals" / "spirals_index" / f"{fname}_motion_energy.mat"
        if not (amp_file.exists() and me_file.exists()):
            raise FileNotFoundError(f"missing amp/motion_energy for {fname}")
        with h5py.File(amp_file, "r") as f:
            traceAmp_mean = f["traceAmp_mean"][()].ravel()
        with h5py.File(me_file, "r") as f:
            image_energy2 = f["image_energy2"][()].ravel()
        tsize = min(traceAmp_mean.size, image_energy2.size)
        data1 = np.column_stack([traceAmp_mean[:tsize], image_energy2[:tsize]])
        data1 = data1[~np.isnan(data1[:, 1])]
        R, p = pearsonr(data1[:, 0], data1[:, 1])  # corrcoef in MATLAB
        R1[count1] = R
        p1[count1] = p
        count1 += 1

    indx = rng.permutation(data1.shape[0])[:10000]
    data2 = data1[indx, :]
    X = sm.add_constant(data2[:, 0])
    mdl = sm.OLS(data2[:, 1], X).fit()
    x1 = np.arange(0.002, 0.015 + 1e-12, 0.001)
    y1 = mdl.params[0] + mdl.params[1] * x1

    hs8gh, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _scatter_kde(ax1, data2[:, 0], data2[:, 1], size=3)
    ax1.collections[0].set_cmap(plt.get_cmap("hot"))
    ax1.plot(x1, y1, "k")
    ax1.set_xlabel("2-8Hz amplitude")
    ax1.set_ylabel("Motion energy")
    ax1.text(0.005, 1400000, f"r = {np.round(R1[-1] * 10) / 10}")

    mean_r = R1.mean()
    sem_r = R1.std(ddof=1) / np.sqrt(13)
    rnd1 = rng.normal(0, 0.1, 13)
    xa = np.zeros(13) + rnd1
    ax2.bar([0], [mean_r])
    ax2.scatter(xa, R1)
    ax2.errorbar([0], [mean_r], yerr=[sem_r], fmt="k")
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 0.2])

    hs8gh.tight_layout()
    hs8gh.savefig(save_folder / "Figs8gh_motion_amp_correlation.pdf", bbox_inches="tight")
    plt.show()
    return hs8gh
