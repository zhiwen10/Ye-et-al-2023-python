from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from spirals_py.task.plots.getMeanSpiralsDetection import _imresize
from spirals_py.task.plots.plotCorrectMapsFlow import (
    _get_cortex_atlas_path,
    _load_atlas1,
    _load_task_mask,
)
from spirals_py.task.preprocessing.taskTrace_upsample import taskTrace_upsample
from spirals_py.utils.atlas import overlayOutlines, plotOutline
from spirals_py.utils.io import load_outline_coords_h5


def _cbrewer2_ylorrd10():
    """cbrewer2('seq','YlOrRd',10), exported from MATLAB (see cbrewer2_YlOrRd_10.csv)."""
    return np.loadtxt(Path(__file__).with_name("cbrewer2_YlOrRd_10.csv"), delimiter=" ")


def plotCorrectMeanTrace2(data_folder, save_folder):
    """Translated from task/plots/plotCorrectMeanTrace2.m

    Left: registered mean image of an example session (ZYE_0091 first
    high-performance task session) with 8 ROI pixels on a circle. Right:
    mean widefield traces at those pixels across correct trials.

    Adaptations:
    - The MATLAB code loads U/V and warps them with the rfmap affine2d
      tform, but only the warped mean image is used in the figure; the
      rfmap file already stores mimgtransformed (same 1320x1140 output of
      imwarp), which is loaded directly instead.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    atlas1 = _load_atlas1(data_folder)
    maskPath, st = _get_cortex_atlas_path(data_folder)
    coords = load_outline_coords_h5(
        data_folder / "tables" / "isocortex_horizontal_projection_outline.mat"
    )

    # example session: first high-performance task session of ZYE_0091
    mn = "ZYE_0091"
    T_session = pd.read_excel(data_folder / "task" / "sessions" / f"{mn}.xlsx")
    T1 = T_session[T_session.label == "task"]
    T1 = T1[(T1.hit_left > 0.7) & (T1.hit_right > 0.7)]
    kk = 0
    tda = pd.Timestamp(T1.date.iloc[kk])
    en = int(T1.folder.iloc[kk])
    fname = f"{T1.MouseID.iloc[kk]}_{tda.strftime('%Y%m%d')}_{en}"

    downscale = 8
    with h5py.File(data_folder / "task" / "rfmap" / f"{fname}.mat", "r") as f:
        mimgtransformed = np.asarray(f["mimgtransformed"]).T  # 1320 x 1140
    mimgtransformed = mimgtransformed[::downscale, ::downscale]
    mimgtransformedRBG = mimgtransformed / mimgtransformed.max()
    mimgtransformedRGB = np.stack([mimgtransformedRBG] * 3, axis=-1)

    with h5py.File(
        data_folder / "task" / "task_mean_maps" / "task_mean_maps_all_mice.mat", "r"
    ) as f:
        trace_mean_all = np.asarray(f["trace_mean_all"]).transpose(3, 2, 1, 0)

    rate = 0.1
    t1 = np.arange(-1, 1 + 1 / 35, 1 / 35)
    tq = 1 + np.arange(int(round((t1.size - 1) / rate)) + 1) * rate  # MATLAB 1:rate:71
    qt = np.interp(tq, np.arange(1, t1.size + 1), t1)

    color2 = _cbrewer2_ylorrd10()
    nameList = ["VISp", "VISpm", "RSP", "SSp_ul", "SSp_ll", "SSp_bfd1", "SSp_bfd2", "VISrl"]
    frames_to_plot = 18
    first_frame = 36  # 1-based, as in MATLAB
    last_frame = first_frame + frames_to_plot
    t1a = t1[first_frame - 1]
    t1b = t1[last_frame - 1]
    first_frame1 = np.argmax(qt - t1a > 0)  # MATLAB find(...,1,'first')
    last_frame1 = np.argmax(qt - t1b > 0)

    scale3 = 5 / 8
    BW2 = _load_task_mask(data_folder, downscale)

    # 8 pixels on a circle of radius 17 around center [90,102] (1-based)
    center = np.array([90, 102])
    th2 = np.deg2rad(np.arange(91, 451, 45))
    px1, py1 = center[1], center[0]
    r = 17
    cx2 = np.round(r * np.cos(th2) + px1)
    cy2 = np.round(r * np.sin(th2) + py1)
    pixel = np.column_stack([cy2, cx2]).astype(int)  # (row, col), 1-based

    h5c = plt.figure(figsize=(4, 3))
    ax1 = h5c.add_subplot(1, 2, 1)
    low = np.percentile(mimgtransformedRGB, 2)
    high = np.percentile(mimgtransformedRGB, 98)
    ax1.imshow(
        np.clip((mimgtransformedRGB - low) / (high - low), 0, 1), alpha=BW2
    )
    overlayOutlines(coords, downscale, ax=ax1)
    plotOutline(maskPath[0:11], st, atlas1, None, scale3, "k", ax=ax1)
    ax1.scatter(pixel[:, 1] - 1, pixel[:, 0] - 1, s=16, c=color2[2:], marker="o")
    ax1.scatter(center[1] - 1, center[0] - 1, s=16, c="w")
    ax1.set_aspect("equal")
    ax1.axis("off")

    for kk in range(1):
        # -2:2 seconds, 141 samples
        trace_mean_current = trace_mean_all[:, :, :, kk]
        trace_mean_current = _imresize(trace_mean_current, (165, 143))
        freq = [2, 8]
        trace_mean3, traceFilt3, tracePhase3 = taskTrace_upsample(
            trace_mean_current, freq, rate
        )
        # only use -1:1 seconds; drop 350 upsampled frames on each side
        edge = int(35 / rate)
        trace2d = trace_mean3[:, :, edge:-edge]
        traceFilt2d = traceFilt3[:, :, edge:-edge]
        tracePhase1 = tracePhase3[:, :, edge:-edge]
        tracePhase1 = np.nan_to_num(tracePhase1)

        trace_raw = np.zeros((8, trace2d.shape[2]))
        for i in range(8):
            trace_raw[i, :] = trace2d[pixel[i, 0] - 1, pixel[i, 1] - 1, :]

        ax2 = h5c.add_subplot(1, 2, kk + 2)
        scalea = 1.5
        for i in range(8):
            ax2.plot(
                qt, trace_raw[i, :] * scalea * 100 + 2 * i, color=color2[i + 2], linewidth=1
            )
            locs, _ = find_peaks(trace_raw[i, :], prominence=0.0005)
            ax2.plot(
                qt[locs],
                trace_raw[i, locs] * scalea * 100 + 2 * i,
                "r.",
                markersize=3,
            )
        ax2.tick_params(labelsize=8)
        ax2.set_yticklabels([])
        for i in range(8):
            ax2.text(qt[310] - 0.1, 2 * i, nameList[i], color="k")
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.axvline(qt[first_frame1], linestyle="--", color="k")
        ax2.axvline(qt[last_frame1], linestyle="--", color="k")
        ax2.plot([qt[279] + 0.5, qt[279] + 0.5], [0, scalea], "r")
        ax2.set_xlim(-0.2, 0.6)
        ax2.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
        ax2.plot([0, 0.2], [0, 0], "k", linewidth=2)
    h5c.savefig(save_folder / "Fig5c_task_correct_mean_trace4.pdf", bbox_inches="tight")
    return h5c
