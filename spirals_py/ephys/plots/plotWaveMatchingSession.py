"""Translated from ephys/plots/plotWaveMatchingSession.m
(Extended Data Fig.13f: wave matching index with 2-8Hz amplitude for all
sessions)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._ephys_helpers import anovan2, ttest2_right_1vN
from ._prediction_example_utils import _subplottight
from spirals_py.ephys.utils import get_session_info2
from spirals_py.ephys.plots.plotMatchingIndex import _load_flow_var


def plotWaveMatchingSession(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotWaveMatchingSession.m

    Returns hs13f.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    area = ["THAL", "STR", "CORTEX", "MB"]
    rfolder = data_folder / "ephys" / "flow_var"

    color1 = ["g", "r", "m"]
    sig_ratio = []
    flow_var_all = []
    pp_anova = []
    count2 = 1
    hs13f = plt.figure(figsize=(10, 5))
    for current_area in [1, 2, 4]:
        indx = T["Area"].str.contains(area[current_area - 1], na=False)
        current_T = T[indx.astype(bool)]
        ha_all = []
        pp_all = []
        flow_var_max = []
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
            flow_var_max.append(flow_var1[0, :].max())
            max_flow = flow_var1.max(axis=0)

            pos = kk + (count2 - 1) * 12
            ax2 = _subplottight(hs13f, 3, 12, pos)
            # MATLAB position tweaks
            x0, y0, w, hgt = ax2.get_position().bounds
            ax2.set_position([x0 + 0.018, y0 + 0.03, w - 0.025, hgt - 0.08])
            ax2.scatter(edges1, flow_var1[0, :], s=8, c=color1[count2 - 1])
            ax2.plot(edges1, flow_var1[0, :], color1[count2 - 1])
            for i in range(1, flow_var1.shape[0]):
                ax2.scatter(edges1, flow_var1[i, :], s=8, c="k")
                ax2.plot(edges1, flow_var1[i, :], "k")
            for i in range(edges1.size):
                if h[i] == 1 and edges1[i] <= 0.01:
                    ax2.text(edges1[i] - 0.0001, max_flow[i] + 0.2, "*", fontsize=12)
            ax2.set_xlim(0, 0.01)
            ax2.set_ylim(0, 1)
            ax2.set_yticklabels([])
            ax2.set_xticks([0, 0.005, 0.01])
            ax2.set_xticklabels(["0", "0.5", "1"])

        ha_all = np.vstack(ha_all) if ha_all else np.zeros((0, 20))
        ha_all1 = ha_all[:, :8]
        ha_count = np.sum(~np.isnan(ha_all1), axis=1)
        ha_significance = np.nansum(ha_all1, axis=1)
        sig_ratio.append(ha_significance / ha_count)
        flow_var_all.append(np.asarray(flow_var_max))
        pp_anova.append(np.asarray(pp_all) if pp_all else np.zeros((0, 3)))
        count2 += 1

    hs13f.savefig(save_folder / "FigS13f_wave_MatchingIndex.pdf", bbox_inches="tight")
    return hs13f
