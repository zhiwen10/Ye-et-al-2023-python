"""Translated from spirals/plots/plotLFPspirals.m (Extended Data Fig.4bc/de).

Example phase maps and spiral organization in cortical LFP recorded on a
4-shank Neuropixels probe (ZYE_0020): probe-site traces/phase and their
48 x 8 shank-position maps.
"""

from pathlib import Path

import colorcet as cc
import h5py
import matplotlib.pyplot as plt
import numpy as np

from spirals_py.ephys.utils import chanMapReorder, loadChanMap
from spirals_py.spirals.plots._fig1_helpers_s3 import (
    colorSites,
    lfpPhasemap,
    makeProbePlot,
)


def plotLFPspirals(data_folder, save_folder):
    """Translated from spirals/plots/plotLFPspirals.m"""
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # ops.fproc1 / ops.chanMap overridden as in the MATLAB original
    with h5py.File(data_folder / "spirals" / "spirals_LFP" / "params.mat", "r") as f:
        ops = {
            "fs": float(np.asarray(f["ops"]["fs"]).ravel()[0]),
            "NT": float(np.asarray(f["ops"]["NT"]).ravel()[0]),
            "Nbatch": float(np.asarray(f["ops"]["Nbatch"]).ravel()[0]),
        }
    ops["fproc1"] = data_folder / "spirals" / "spirals_LFP" / "filtered_lfp_imec0.dat"
    ops["chanMap"] = data_folder / "ephys" / "config_files" / "NPtype24_hStripe_botRow0_ref1.mat"

    epochT = [1215, 1220]
    Fs = 35
    traceEphysMap1, phaseEphysMap1, ampEphysMap = lfpPhasemap(ops, epochT, Fs)
    chanMap1 = chanMapReorder(ops["chanMap"])  # (48, 8), 0-based channels

    n_samples = traceEphysMap1.shape[0]
    traceEphysMap = np.zeros((n_samples, 48, 8))
    phaseEphysMap = np.zeros((n_samples, 48, 8))
    for j in range(384):
        row, col = np.argwhere(chanMap1 == j)[0]
        phaseEphysMap[:, row, col] = phaseEphysMap1[:, j]
        traceEphysMap[:, row, col] = traceEphysMap1[:, j]

    traceEphysMap1 = traceEphysMap1 * 2.34 / 1000  # amplifier gain
    traceEphysMap = traceEphysMap * 2.34 / 1000

    immax = phaseEphysMap1.max()
    immin = phaseEphysMap1.min()
    maxRaw = traceEphysMap.max()
    minRaw = traceEphysMap.min()
    caxPhase = [immin, immax]
    caxRaw = [minRaw / 2, maxRaw / 2]
    colMapRaw = plt.get_cmap("viridis")(np.linspace(0, 1, 100))[:, :3]  # parula
    colMapPhase = cc.cm["CET_C6"](np.linspace(0, 1, 100))[:, :3]
    cmap_phase = cc.cm["CET_C6"]

    _, xcoords, ycoords, _, _ = loadChanMap(ops["chanMap"])
    cm = {"xcoords": xcoords, "ycoords": ycoords}

    hs4bc = plt.figure(figsize=(8, 5))
    kk = 37  # 1-based frame
    ax1 = hs4bc.add_subplot(2, 2, 1)
    ax1.set_facecolor("k")
    psRaw = makeProbePlot(ax1, cm, 12)
    colorSites(psRaw, traceEphysMap1[kk - 1, :], colMapRaw, caxRaw)
    ax1.set_axis_off()

    ax3 = hs4bc.add_subplot(2, 2, 3)
    ax3.set_facecolor("k")
    psPhase = makeProbePlot(ax3, cm, 12)
    colorSites(psPhase, phaseEphysMap1[kk - 1, :], colMapPhase, caxPhase)
    ax3.set_axis_off()

    ax2 = hs4bc.add_subplot(2, 2, 2)
    ax2.imshow(traceEphysMap[kk - 1], cmap="viridis",
               vmin=caxRaw[0], vmax=caxRaw[1], origin="lower")
    ax2.set_aspect("equal")
    ax2.set_axis_off()

    ax4 = hs4bc.add_subplot(2, 2, 4)
    ax4.imshow(phaseEphysMap[kk - 1], cmap=cmap_phase,
               vmin=caxPhase[0], vmax=caxPhase[1], origin="lower")
    ax4.set_aspect("equal")
    ax4.set_axis_off()

    hs4bc.savefig(save_folder / "FigS4bc_LFP_4shank_example.png")
    plt.show()

    hs4de = plt.figure(figsize=(8, 5), facecolor="k")
    for count1, kk in enumerate(range(27, 47), start=1):
        axa = hs4de.add_subplot(2, 20, count1)
        im = axa.imshow(traceEphysMap[kk - 1], cmap="viridis",
                        vmin=caxRaw[0], vmax=caxRaw[1], origin="lower")
        axa.set_aspect("equal")
        axa.set_axis_off()
        if count1 == 20:
            hs4de.colorbar(im, ax=axa)

        axb = hs4de.add_subplot(2, 20, count1 + 20)
        im2 = axb.imshow(phaseEphysMap[kk - 1], cmap=cmap_phase,
                         vmin=caxPhase[0], vmax=caxPhase[1], origin="lower")
        axb.set_aspect("equal")
        axb.set_axis_off()
        if count1 == 20:
            hs4de.colorbar(im2, ax=axb)

    hs4de.savefig(save_folder / "FigS4de_LFP_4shank_spirals.png")
    plt.show()
    return hs4bc, hs4de
