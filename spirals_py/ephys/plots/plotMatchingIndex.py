"""Translated from ephys/plots/plotMatchingIndex.m
(Figure 4h,i: phase (h) and wave/flow (i) matching index vs shuffle
condition, by area)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._ephys_helpers import _load_mat_var, anovan2, ttest2_right_1vN, ttest_paired_h
from spirals_py.ephys.utils import get_session_info2


def _load_flow_var(rfolder, fname):
    """Load flow_var/phase_var/edges (v7 or v7.3 .mat)."""
    return (
        np.atleast_2d(_load_mat_var(rfolder / (fname + ".mat"), "flow_var").astype(float)),
        np.atleast_2d(_load_mat_var(rfolder / (fname + ".mat"), "phase_var").astype(float)),
        np.atleast_2d(_load_mat_var(rfolder / (fname + ".mat"), "edges").astype(float)),
    )


def _sig_ratios(T, data_folder, rfolder, area):
    """First loop of plotMatchingIndex.m / plotWaveMatchingSession.m:
    per-session right-tail t-test of real vs permutations, the per-session
    significance ratio, and the per-session anovan p-values."""
    sig_ratio = []
    pp_anova = []
    for current_area in [1, 2, 4]:
        indx = T["Area"].str.contains(area[current_area - 1], na=False)
        current_T = T[indx.astype(bool)]
        ha_all = []
        pp_all = []
        for kk in range(len(current_T)):
            ops = get_session_info2(current_T, kk, data_folder)
            fname = f"{ops.mn}_{ops.tdb}_{ops.en}"
            flow_var, _phase_var, edges = _load_flow_var(rfolder, fname)
            flow_var1 = 1 - flow_var
            nan_index = np.isnan(flow_var1[0, :])
            flow_var1 = flow_var1[:, ~nan_index]

            # unbalanced two-way-anovan
            edges1 = edges[0, 1:][~nan_index]
            edges_matrix = np.tile(edges1, (flow_var1.shape[0], 1))
            permute_id_matrix = np.ones(flow_var1.shape)
            permute_id_matrix[0, :] = 0
            pp = anovan2(
                flow_var1.T.ravel(),  # MATLAB A(:) column-major
                edges_matrix.T.ravel(),
                permute_id_matrix.T.ravel(),
            )
            pp_all.append(np.asarray(pp))

            h = np.full(flow_var1.shape[1], np.nan)
            for ii in range(flow_var1.shape[1]):
                h[ii] = ttest2_right_1vN(flow_var1[0, ii], flow_var1[1:, ii])[0]
            ha = np.full(20, np.nan)
            ha[: h.size] = h
            ha_all.append(ha)
        ha_all = np.vstack(ha_all) if ha_all else np.zeros((0, 20))
        ha_all1 = ha_all[:, :8]
        ha_count = np.sum(~np.isnan(ha_all1), axis=1)
        ha_significance = np.nansum(ha_all1, axis=1)
        sig_ratio.append(ha_significance / ha_count)
        pp_anova.append(np.asarray(pp_all) if pp_all else np.zeros((0, 3)))
    return sig_ratio, pp_anova


def _matching_fig(T, data_folder, rfolder, area, sig_ratio, varname, fig_name, save_folder):
    """Shared second loop + figure of plotMatchingIndex.m (flow -> h4i,
    phase -> h4h)."""
    color2 = ["g", "r", "c", "m"]
    fig = plt.figure(figsize=(9, 6))
    edges = None
    real_all = []
    perm_all = []
    for count2, current_area in enumerate([1, 2, 4]):
        indx = T["Area"].str.contains(area[current_area - 1], na=False)
        current_T = T[indx.astype(bool)]
        n = len(current_T)
        var_real = np.full((n, 20), np.nan)
        var_permutation = np.full((n, 20), np.nan)
        for kk in range(n):
            ops = get_session_info2(current_T, kk, data_folder)
            fname = f"{ops.mn}_{ops.tdb}_{ops.en}"
            flow_var, phase_var, edges = _load_flow_var(rfolder, fname)
            var1 = 1 - (flow_var if varname == "flow" else phase_var)
            var_real[kk, :] = var1[0, :]
            var_permutation[kk, :] = np.mean(var1[1:, :], axis=0)
        current_sig = sig_ratio[count2]
        sig_indx = current_sig >= 0
        real_all.append(var_real[sig_indx, :])
        perm_all.append(var_permutation[sig_indx, :])

    for count1, current_area in enumerate([1, 2, 4]):
        ax = fig.add_subplot(1, 3, count1 + 1)
        real_temp = real_all[count1]
        perm_temp = perm_all[count1]
        sample_n = np.sum(~np.isnan(real_temp), axis=0)
        with np.errstate(invalid="ignore"):
            real_mean = np.nanmean(real_temp, axis=0)
            real_std = np.nanstd(real_temp, axis=0, ddof=1) / np.sqrt(sample_n)
            perm_mean = np.nanmean(perm_temp, axis=0)
            perm_std = np.nanstd(perm_temp, axis=0, ddof=1) / np.sqrt(sample_n)
        e = edges[0, :8]
        ax.errorbar(e, real_mean[:8], yerr=real_std[:8], color=color2[current_area - 1])
        ax.errorbar(e, perm_mean[:8], yerr=perm_std[:8], color="k")
        ax.set_xlim(0, 0.01)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1 + 1e-9, 0.2))
        ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1 + 1e-9, 0.2)])

        # unbalanced two-way-anovan over the first 8 columns
        real_temp2 = real_temp[:, :8]
        perm_temp2 = perm_temp[:, :8]
        both = np.vstack([real_temp2, perm_temp2])
        edges2 = edges[0, 1 : real_temp2.shape[1] + 1]
        edges_matrix2 = np.tile(edges2, (both.shape[0], 1))
        permute_id_matrix2 = np.ones(both.shape)
        permute_id_matrix2[: real_temp2.shape[0], :] = 0
        keep = ~np.isnan(both.T.ravel())
        _pp2 = anovan2(
            both.T.ravel()[keep],
            edges_matrix2.T.ravel()[keep],
            permute_id_matrix2.T.ravel()[keep],
        )

        for ii in range(8):
            h_i, _p_i = ttest_paired_h(real_temp2[:, ii], perm_temp2[:, ii])
            if not np.isnan(h_i) and h_i:
                ax.text(edges[0, ii], real_mean[ii] + 0.1, "*", fontsize=12)

    fig.savefig(save_folder / fig_name, bbox_inches="tight")
    return fig


def plotMatchingIndex(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotMatchingIndex.m

    Returns (h4h, h4i).
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    area = ["THAL", "STR", "CORTEX", "MB"]
    rfolder = data_folder / "ephys" / "flow_var"

    sig_ratio, _pp_anova = _sig_ratios(T, data_folder, rfolder, area)

    # flow (wave) matching index -> h4i (plotted/saved first in MATLAB)
    h4i = _matching_fig(
        T, data_folder, rfolder, area, sig_ratio, "flow",
        "Fig4i_flow_matchingIndex.pdf", save_folder,
    )

    # phase matching index -> h4h
    h4h = _matching_fig(
        T, data_folder, rfolder, area, sig_ratio, "phase",
        "Fig4h_phase_matchingIndex.pdf", save_folder,
    )
    return h4h, h4i
