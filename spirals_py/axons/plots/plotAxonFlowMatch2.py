"""Translated from axons/plots/plotAxonFlowMatch2.m

Plots the mean spiral optical flow field sampled at axon locations and the
axon-flow matching index against a permutation null (Fig. 2e-f).
"""

from pathlib import Path

import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy import stats

from spirals_py.utils.atlas import plotOutline
from spirals_py.utils.optical_flow import HS_flowfield

# maskPath cell array from plotAxonFlowMatch2.m
MASK_PATHS = [
    "/997/8/567/688/695/315/669/",  # VIS
    "/997/8/567/688/695/315/254/",  # RSP
    "/997/8/567/688/695/315/247/",  # AUD
    "/997/8/567/688/695/315/22",  # VISa
    "/997/8/567/688/695/315/677/",  # VISC
    "/997/8/567/688/695/315/22/417/",  # VISrl
    "/997/8/567/688/695/315/453/322/361/",  # SSp-tr
    "/997/8/567/688/695/315/453/322/337/",  # SSp-ll
    "/997/8/567/688/695/315/453/322/369/",  # SSp-ul
    "/997/8/567/688/695/315/453/322/345/",  # SSp-m
    "/997/8/567/688/695/315/453/322/353/",  # SSp-n
    "/997/8/567/688/695/315/453/322/329/",  # SSp-bfd
    "/997/8/567/688/695/315/453/322/182305689/",  # SSp-un
    "/997/8/567/688/695/315/453/378/",  # SSs
    "/997/8/567/688/695/315/453/322/",  # SSp
    "/997/8/567/688/695/315/500/985/",  # MOp
    "/997/8/567/688/695/315/500/993/",  # MOs
    "/997/8/567/688/695/315/31/",  # ACA
    "/997/8/567/688/695/315/541/",  # TEa
    "/997/8/567/688/695/315/669/385/",  # VISp
    "/997/8/567/688/695/315/669/312782628/",
    "/997/8/567/688/695/315/669/409/",
]


def _colorcet_C06(N=256):
    """colorcet('C06','N',N) via the Python colorcet package (CET_C6)."""
    base = np.array([to_rgb(c) for c in colorcet.palette["CET_C6"]])
    if N >= base.shape[0]:
        return base
    xi = np.arange(N) * (base.shape[0] - 1) / (N - 1)
    return np.stack(
        [np.interp(xi, np.arange(base.shape[0]), base[:, j]) for j in range(3)], axis=1
    )


def _get_cortex_atlas_path(data_folder):
    """Translated from spirals/utils/get_cortex_atlas_path.m.

    Only the structure tree is used downstream; maskPath is returned for
    signature parity with MATLAB.
    """
    st = pd.read_csv(Path(data_folder) / "tables" / "structure_tree_safe_2017.csv")
    maskPath = [
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
    return maskPath, st


def _interp_colors(x1, cmap_arr, v):
    """interp1(x1, cmap_arr, v) per color channel."""
    return np.stack(
        [np.interp(v, x1, cmap_arr[:, j]) for j in range(3)], axis=-1
    )


def _ttest2_scalar(x, y):
    """MATLAB ttest2(x, y) for scalar x (pooled two-sample t-test)."""
    y = np.asarray(y, dtype=float)
    n1, n2 = 1, y.size
    sp = np.sqrt((n2 - 1) * y.var(ddof=1) / (n1 + n2 - 2))
    t = (float(x) - y.mean()) / (sp * np.sqrt(1 / n1 + 1 / n2))
    p = 2 * stats.t.cdf(-abs(t), n1 + n2 - 2)
    return t, p


def _load_mean_spiral_phase(T, data_folder):
    """Concatenate spiral_phase_all_norm over sessions and average (circ_mean).

    MATLAB cat(4, ...) + circ_mean(..., [], 4); here the complex sum is
    accumulated per session to avoid holding all frames in memory, which is
    mathematically identical for unit weights.
    """
    acc = None
    for _, row in T.iterrows():
        mn = row["MouseID"]
        tdb = pd.Timestamp(row["date"]).strftime("%Y%m%d")
        en = row["folder"]
        fname = f"{mn}_{tdb}_{en}"
        fp = (
            Path(data_folder)
            / "axons"
            / "spirals_70pixels_mean_flow"
            / f"{fname}_mean_flow2.mat"
        )
        with h5py.File(fp) as f:
            # stored transposed: (frames, 2, 143, 165) -> MATLAB (165, 143, 2, frames)
            arr = np.array(f["spiral_phase_all_norm"]).transpose(3, 2, 1, 0)
        s = np.exp(1j * arr).sum(axis=3)
        acc = s if acc is None else acc + s
        print(f"loaded {fname}")
    return np.angle(acc)  # (165, 143, 2), same as circ_mean(cat4(...), dim=3)


def plotAxonFlowMatch2(T, data_folder, save_folder, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    with h5py.File(data_folder / "tables" / "horizontal_cortex_atlas_50um.mat") as f:
        # v7.3, stored transposed: MATLAB array is (264, 228)
        atlas1 = np.array(f["atlas1"]).T
    with h5py.File(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    ) as f:
        # stored transposed: MATLAB array is (1320, 1140)
        projectedAtlas1 = np.array(f["projectedAtlas1"]).T
    maskPath, st = _get_cortex_atlas_path(data_folder)
    maskPath = MASK_PATHS

    colora2 = _colorcet_C06(180)

    BW = projectedAtlas1.astype(bool)
    scale = 8
    BW1 = BW[::scale, ::scale]  # (165, 143)
    BW1_left = BW1.copy()
    BW1_left[:, 71:] = False  # MATLAB (:,72:end) = 0

    # --- axon bias vectors for all cells with morphology ---
    T1 = pd.read_csv(data_folder / "axons" / "Axon_bias_all_cells.csv")
    soma_center = T1[["soma_center_1", "soma_center_2"]].to_numpy() / scale
    axon_vector = T1[["axon_bias_1", "axon_bias_2"]].to_numpy()
    pc_ratio = T1["pc_ratio"].to_numpy()
    soma_center1 = np.round(soma_center).astype(int)
    cell_n = soma_center.shape[0]
    axon_vector1 = axon_vector * pc_ratio[:, None]

    # --- mean spiral phase map and mean optical flow field ---
    spiral_phase_mean = _load_mean_spiral_phase(T, data_folder)
    spiral_phase_mean1 = np.transpose(spiral_phase_mean, (2, 0, 1))  # (2, 165, 143)
    vxRaw, vyRaw = HS_flowfield(spiral_phase_mean1, False)
    vxRaw = np.squeeze(vxRaw)
    vyRaw = np.squeeze(vyRaw)

    # --- sample flow field at cell locations ---
    vxRaw1 = vxRaw.copy()
    vyRaw1 = vyRaw.copy()
    # MATLAB: vxRaw1(:,144/2:-1:1) = -vxRaw(:,142/2+1:end) (mirror to left hemi)
    vxRaw1[:, 71::-1] = -vxRaw[:, 71:]
    vyRaw1[:, 71::-1] = vyRaw[:, 71:]
    flow_vxy = np.zeros((cell_n, 2))
    for i in range(cell_n):
        flow_vxy[i, 0] = vxRaw1[soma_center1[i, 1] - 1, soma_center1[i, 0] - 1]
        flow_vxy[i, 1] = vyRaw1[soma_center1[i, 1] - 1, soma_center1[i, 0] - 1]

    # --- dot products, real and 1000x permuted ---
    sum_dot = np.abs(np.sum(flow_vxy * axon_vector1, axis=1)).sum()
    N_flow = np.linalg.norm(flow_vxy, axis=1)
    N_axon = np.linalg.norm(axon_vector1, axis=1)
    N_abs = (N_flow * N_axon).sum()
    sum_dot_perm_all = np.zeros(1000)
    for k in range(1000):
        index = rng.permutation(cell_n)
        axon_vector_perm = axon_vector1[index]
        sum_dot_perm_all[k] = np.abs(np.sum(flow_vxy * axon_vector_perm, axis=1)).sum()
    _ttest2_scalar(sum_dot, sum_dot_perm_all)

    # --- plot optical flow field and matching index ---
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_angle = np.round(np.rad2deg(np.arctan(vyRaw1 / vxRaw1)))
    x1 = np.linspace(-90, 90, 180)
    colora3 = np.nan_to_num(_interp_colors(x1, colora2, flow_angle))

    vxRaw2 = np.full(vxRaw1.shape, np.nan)
    vyRaw2 = np.full(vyRaw1.shape, np.nan)
    skip = 3
    zoom_scale = 2
    vxRaw2[::skip, ::skip] = vxRaw1[::skip, ::skip] * zoom_scale
    vyRaw2[::skip, ::skip] = vyRaw1[::skip, ::skip] * zoom_scale
    vxRaw2[~BW1] = np.nan
    vyRaw2[~BW1] = np.nan

    lineColor = "k"
    lineColor1 = "w"
    hemi = "left"
    scale3 = 5 / 8

    h2df, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    plt.sca(ax1)
    for i in list(range(0, 15)) + list(range(18, 22)):  # MATLAB [1:15,19:22]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor1, ax=ax1)
    for i in list(range(0, 6)) + [13, 14, 18]:  # MATLAB [1:6,14,15,19]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor, ax=ax1)
    for i in range(15, 18):  # MATLAB [16:18]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor, ax=ax1)
    for i in [13, 14]:  # MATLAB [14:15]
        plotOutline([maskPath[i]], st, atlas1, hemi, scale3, lineColor, ax=ax1)
    scale1 = 1
    for i in range(cell_n):
        ax1.plot(
            [
                soma_center1[i, 0] - flow_vxy[i, 0] * scale1,
                soma_center1[i, 0] + flow_vxy[i, 0] * scale1,
            ],
            [
                soma_center1[i, 1] - flow_vxy[i, 1] * scale1,
                soma_center1[i, 1] + flow_vxy[i, 1] * scale1,
            ],
            color=colora3[soma_center1[i, 1] - 1, soma_center1[i, 0] - 1],
            linewidth=1,
        )
    ax1.invert_yaxis()  # MATLAB YDir reverse
    ax1.set_aspect("equal")
    ax1.set_axis_off()

    sum_dot_perm_all1 = sum_dot_perm_all / N_abs
    sum_dot1 = sum_dot / N_abs
    _, p2 = _ttest2_scalar(sum_dot1, sum_dot_perm_all1)

    ax2.hist(sum_dot_perm_all1)
    ax2.axvline(sum_dot1)
    ax2.text(1, 120, str(p2))
    ax2.text(1, 130, str(sum_dot1))
    ax2.set_ylim(0, 200)
    ax2.set_xlim(0.6, 0.75)
    ax2.set_xticks(np.arange(0.6, 0.751, 0.05))
    ax2.set_xticklabels(["0.6", "0.65", "0.7", "0.75"])
    ax2.set_yticks(np.arange(0, 201, 50))
    ax2.set_yticklabels(["0", "50", "100", "150", "200"])

    h2df.savefig(save_folder / "Fig2df_axon_flow_match.pdf", bbox_inches="tight")
    return h2df
