"""Translated from whisker/preprocessing/getSpiralsWhiskerMeanMaps.m"""

from pathlib import Path

import numpy as np

from spirals_py.spirals.plots._fig1_helpers_s1 import (
    _inROI,
    _load_roi_vertices,
    _padZeros,
)
from spirals_py.spirals.preprocessing.spiral_detection import (
    detect_padded_frames,
    getGroupingAlgorithm,
)
from spirals_py.utils.matio import load_mat_var, save_mat73

from ._whisker_utils import _bandpass_phase, _colorcet_c06, _imresize


def _whisker_detection_params(nt, xsize, ysize):
    """The inline params of getSpiralsWhiskerMeanMaps.m lines 59-85 (same
    fields as _setSpiralDetectionParams, but epochL 141, frameRange
    1:nt and the 660x570 coarse grids)."""
    params = {}
    params["downscale"] = 1
    params["lowpass"] = 0
    params["Fs"] = 35
    params["halfpadding"] = 120
    params["padding"] = 2 * params["halfpadding"]
    params["th"] = np.arange(1, 361, 36)
    params["rs"] = np.arange(10, 21, 5)
    params["gridsize"] = 10
    params["spiralRange"] = np.linspace(-np.pi, np.pi, 5)
    params["gsmooth"] = 0
    params["epochL"] = 141
    params["nt"] = nt
    params["frameRange"] = np.arange(1, nt + 1)
    params["dThreshold"] = 15
    params["rsRCheck"] = np.arange(10, 101, 10)
    params["pgridx"], params["pgridy"] = np.meshgrid(
        np.arange(21 - 10, 21 + 10 + 1), np.arange(21 - 10, 21 + 10 + 1)
    )
    params["xsize"] = xsize
    params["ysize"] = ysize
    xsizePadded = params["xsize"] + params["padding"]
    ysizePadded = params["ysize"] + params["padding"]
    xx, yy = np.meshgrid(
        np.arange(
            np.min(params["rs"]) + 1,
            xsizePadded - np.min(params["rs"]) - 1 + 1,
            params["gridsize"],
        ),
        np.arange(
            np.min(params["rs"]) + 1,
            ysizePadded - np.min(params["rs"]) - 1 + 1,
            params["gridsize"],
        ),
    )
    params["xx"] = xx
    params["yy"] = yy
    return params


def _draw_roi(tracePhase, halfpadding):
    """The interactive ROI of getSpiralsWhiskerMeanMaps.m lines 86-92:
    drawpolygon over imagesc of the zero-padded phase of frame 100 with
    the colorcet('C06') colormap (matplotlib PolygonSelector)."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    mimg1 = tracePhase[:, :, 100 - 1]
    mimg2 = _padZeros(mimg1, halfpadding)
    fig, ax = plt.subplots()
    ax.imshow(mimg2, cmap=_colorcet_c06())
    selector = PolygonSelector(ax, lambda verts: None)
    plt.show()
    return np.asarray(selector.verts, dtype=float)


def _save_roi(path, verts):
    """Save the polygon vertices as a (1,1) cell 'roi' whose 2xN dataset
    lives under #refs# (v7.3 layout, as required by _load_roi_vertices;
    save_mat73 writes cell elements at the HDF5 root instead)."""
    import h5py

    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        d = refs.create_dataset("r0", data=np.asarray(verts, dtype=float).T)
        d.attrs["MATLAB_class"] = np.bytes_("double")
        cell = np.empty((1, 1), dtype=h5py.ref_dtype)
        cell[0, 0] = d.ref
        c = f.create_dataset("roi", data=cell)
        c.attrs["MATLAB_class"] = np.bytes_("cell")


def _column_cell(cells):
    arr = np.empty((len(cells), 1), dtype=object)
    for i, c in enumerate(cells):
        arr[i, 0] = c
    return arr


def getSpiralsWhiskerMeanMaps(data_folder, save_folder):
    """Translated from whisker/preprocessing/getSpiralsWhiskerMeanMaps.m

    Detects spirals on the grand-mean whisker-evoked map of
    getWhiskerMeanMaps (wf_mean2, imresize'd to 660x570): per-pixel
    demean, butter(2,[2 8]/(35/2)) bandpass + filtfilt + hilbert phase,
    zero-padding by 120 px, the per-frame detection chain over the whole
    141-frame movie (frameStart = 1), radius >= 40 filter, unique rows,
    sortrows by frame, and the spatiotemporal grouping algorithm.
    archiveCell is saved as whisker_spirals_group_fftn.mat (M, 1 cell) in
    the v7.3 layout read by the whisker plot modules.

    The brain-mask ROI is drawn interactively the first time and cached
    as a (1, 1) cell 'roi' in whisker_evoked_map_roi.mat, so that
    _load_roi_vertices finds the 2xN vertex dataset under #refs# (the
    MATLAB original saves the drawpolygon vertices directly).

    Dead code skipped: the atlas/outline/st prologue and indexSSp2, the
    unused spiral_folder variable (whose 'sprials_grouping' spelling is a
    typo of 'spirals_grouping'), and the unused meanTrace2 output of the
    bandpass block.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    wf_mean2 = load_mat_var(save_folder / "whisker_spirals_mean_all.mat", "wf_mean2")
    wf_mean3 = _imresize(wf_mean2, (660, 570))

    _, tracePhase = _bandpass_phase(wf_mean3)
    params = _whisker_detection_params(
        tracePhase.shape[2], tracePhase.shape[0], tracePhase.shape[1]
    )

    roi_path = save_folder / "whisker_evoked_map_roi.mat"
    if roi_path.exists():
        roi = _load_roi_vertices(roi_path)
    else:
        roi = _draw_roi(tracePhase, params["halfpadding"])
        _save_roi(roi_path, roi)
    tf = _inROI(roi, params["xx"].ravel(), params["yy"].ravel())
    params["xxRoi"] = params["xx"][tf]
    params["yyRoi"] = params["yy"][tf]

    tracePhase1 = _padZeros(tracePhase, params["halfpadding"])
    pwAll = detect_padded_frames(tracePhase1, params, 1)

    # recalculate spiral 2d coordinates without padding
    pwAll[:, :2] = pwAll[:, :2] - params["halfpadding"]
    # only use spirals with radius > 40 pixels, based on 3d-fft
    filteredSpirals = pwAll[pwAll[:, 2] >= 40]
    # get rid of duplication, in case any; then sort by frame number
    filteredSpirals = np.unique(filteredSpirals, axis=0)
    filteredSpirals = filteredSpirals[np.argsort(filteredSpirals[:, 4], kind="stable")]
    archiveCell, test_stats = getGroupingAlgorithm(filteredSpirals)

    save_mat73(
        save_folder / "whisker_spirals_group_fftn.mat",
        {"archiveCell": _column_cell(archiveCell)},
    )
