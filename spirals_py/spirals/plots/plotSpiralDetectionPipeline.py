"""Translated from spirals/plots/plotSpiralDetectionPipeline.m (Extended Data
Fig.2): illustration of the spiral detection pipeline on one example phase
frame of session ZYE_0012/2020-10-16/5 — padded phase map with search grids,
circle sampling at candidate radii, cumulative phase angles, candidate
centers, clustered/double-checked centers, and final spirals with radius
and direction.
"""

from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _angdiff,
    _checkClusterXY,
    _doubleCheckSpiralsAlgorithm,
    _inROI,
    _load_roi_vertices,
    _load_session_row,
    _matlab_round,
    _padZeros,
    _setSpiralDetectionParams,
    _spatialRefine,
    _spiralAlgorithm,
    _spiralRadiusCheck2,
    _unwrap_phases,
)
from spirals_py.spirals.plots.plotExampleOscillation import _cbrewer2
from spirals_py.spirals.preprocessing.spiralPhaseMap_freq import spiralPhaseMap_freq


def plotSpiralDetectionPipeline(T, data_folder, save_folder):
    """Translated from spirals/plots/plotSpiralDetectionPipeline.m; returns hs2."""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # use ZYE12 as an example
    kk = 7
    mn, tdb, en, fname, U, V, t, mimg, dV = _load_session_row(data_folder, T, kk)

    # set params for spiral detection
    freq = [2, 8]  # data filtering frequency range
    params = _setSpiralDetectionParams(U, t)
    # apply mask, this helps speed up spiral detection later
    roi_path = data_folder / "spirals" / "full_roi" / f"{mn}_{tdb}_{en}_roi.mat"
    roi = _load_roi_vertices(roi_path)
    tf = _inROI(roi, params["xx"].ravel(order="F"), params["yy"].ravel(order="F"))
    params["xxRoi"] = params["xx"].ravel(order="F")[tf]
    params["yyRoi"] = params["yy"].ravel(order="F")[tf]

    # look at these frame range for example spirals
    frame2 = np.arange(58744, 58944 + 1)
    rate1 = 1
    dV1 = dV[:50, frame2 - 1]
    t1 = t[frame2 - 1]
    trace2d2, traceAmp2, tracePhase2 = spiralPhaseMap_freq(
        U[:, :, :50], dV1, t1, params, freq, rate1
    )

    # frame 21 has a beautiful spiral as a good example
    iframe = 21
    pwi = 8
    tracePhase1 = tracePhase2[:, :, iframe - 1]
    tracePhase = _padZeros(tracePhase1, params["halfpadding"])
    pwAll1 = _spiralAlgorithm(tracePhase, params)

    greys = _cbrewer2("seq", "Greys", 9)
    rs = params["rs"]
    th = params["th"]
    cmap_c06 = cc.cm["CET_C6"]

    hs2, axs = plt.subplots(2, 3, figsize=(11, 6))

    # panel 1: padded example frame
    ax1 = axs[0, 0]
    ax1.imshow(tracePhase, cmap=cmap_c06)
    ax1.set_aspect("equal")
    ax1.scatter(
        params["xxRoi"] - 1, params["yyRoi"] - 1, s=0.5,
        facecolors="none", edgecolors=(0.5, 0.5, 0.5),
    )  # overlaid with searching grids
    px = pwAll1[pwi - 1, 0]
    py = pwAll1[pwi - 1, 1]
    ax1.scatter(px - 1, py - 1, s=8, c="k")  # example spiral center
    v = np.array([[500, 350], [600, 350], [600, 450], [500, 450]]) - 1  # zoom patch
    ax1.add_patch(plt.Polygon(v, closed=True, edgecolor="k", facecolor="none", linewidth=1))
    ax1.plot([400 - 1, 515 - 1], [700 - 1, 700 - 1], "k")

    # panel 2: phase map within the zoom patch
    ax2 = axs[0, 1]
    ax2.imshow(tracePhase, cmap=cmap_c06)
    ax2.set_aspect("equal")
    ax2.scatter(
        params["xxRoi"] - 1, params["yyRoi"] - 1, s=3,
        facecolors="none", edgecolors=(0.5, 0.5, 0.5),
    )  # searching grids
    ax2.scatter(px - 1, py - 1, s=24, c="k")  # example spiral center
    for rn in range(rs.size):
        r = rs[rn]
        cx = _matlab_round(r * np.cos(np.deg2rad(th)) + px).astype(int)
        cy = _matlab_round(r * np.sin(np.deg2rad(th)) + py).astype(int)
        ax2.scatter(cx - 1, cy - 1, s=24, c=greys[2 + 2 * rn], edgecolors="none")
        ax2.plot(
            np.concatenate([cx, cx[:1]]) - 1, np.concatenate([cy, cy[:1]]) - 1,
            color=greys[2 + 2 * rn], linewidth=2,
        )  # join the points on the circle
    ax2.set_xlim(500 - 1, 600 - 1)
    ax2.set_ylim(450 - 1, 350 - 1)  # MATLAB ylim([350,450])

    # panel 3: cumulative phase angles along a circle
    ax3 = axs[0, 2]
    for rn in range(rs.size):
        r = rs[rn]
        cx = _matlab_round(r * np.cos(np.deg2rad(th)) + px).astype(int)
        cy = _matlab_round(r * np.sin(np.deg2rad(th)) + py).astype(int)
        ph = tracePhase[cy - 1, cx - 1]  # MATLAB sub2ind
        ph2 = _unwrap_phases(ph)
        ph3 = np.abs(ph2 - ph2[0])
        ax3.scatter(np.arange(1, 11), ph3, s=24, c=greys[2 + 2 * rn], edgecolors="none")
    ax3.set_xlabel("Sampling points")
    ax3.set_ylabel("Cumulative phase angle")

    # panel 4: all candidate spirals on a single frame
    ax4 = axs[1, 0]
    ax4.imshow(tracePhase, cmap=cmap_c06)
    ax4.set_aspect("equal")
    ax4.scatter(pwAll1[:, 0] - 1, pwAll1[:, 1] - 1, s=8, c="k")

    # panel 5: clustered and double-checked spiral centers
    pwAll2 = _checkClusterXY(pwAll1, params["dThreshold"])
    pwAll3 = _doubleCheckSpiralsAlgorithm(tracePhase, pwAll2, params)
    pwAll4 = _spatialRefine(tracePhase, pwAll3, params)
    pwAll5 = _spiralRadiusCheck2(tracePhase, pwAll4, params)
    ax5 = axs[1, 1]
    ax5.imshow(tracePhase, cmap=cmap_c06)
    ax5.set_aspect("equal")
    ax5.scatter(pwAll3[:, 0] - 1, pwAll3[:, 1] - 1, s=8, c="k")

    # panel 6: final spirals with radius and direction
    ax6 = axs[1, 2]
    th2 = np.arange(1, 361, 5)
    ax6.imshow(tracePhase, cmap=cmap_c06)
    ax6.set_aspect("equal")
    ax6.scatter(pwAll5[:, 0] - 1, pwAll5[:, 1] - 1, s=8, c="k")
    for i in range(pwAll5.shape[0]):
        px1 = pwAll5[i, 0]
        py1 = pwAll5[i, 1]
        r = pwAll5[i, 2]
        cx2 = _matlab_round(r * np.cos(np.deg2rad(th2)) + px1)
        cy2 = _matlab_round(r * np.sin(np.deg2rad(th2)) + py1)
        if pwAll5[i, 3] == 1:  # counterclockwise, then color white
            color1 = "w"
        else:
            color1 = "k"  # clockwise, then color black
        ax6.plot(
            np.concatenate([cx2, cx2[:1]]) - 1, np.concatenate([cy2, cy2[:1]]) - 1,
            color=color1, linewidth=1,
        )  # draw the circle at max radius

    hs2.savefig(save_folder / "FigS2_detection-pipeline.png", dpi=300)
    plt.show()
    return hs2
