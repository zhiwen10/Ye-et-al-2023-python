"""Private helpers for the Fig4/FigS13 ephys plot modules (plotProbeLocation,
plotSpiralsMatchingRate, plotMatchingIndex, plotVarMap, plotVarSummary,
plotWaveMatchingSession, sort_sprials_by_arousal,
plotSpiralsMatchingRate_Arousal).

Translated from MATLAB sources in YE-et-al-2023-spirals:
- ephys/utils/best_fit_line.m
- axons/utils/plotOutline_fill.m
- spirals/utils/nrrdread.m (via pynrrd; orientation matched to MATLAB output)
- spirals/utils/get_ssp_index.m + spirals/utils/select_area.m
- spirals/utils/density_color_plot.m + spirals/utils/remapColor.m
- revision2/prediction_motion_energy/phaseSpiralHistogram2.m
- revision2/prediction_motion_energy/spiralDensityBins2.m
Plus MATLAB statistics equivalents (ttest/ttest2/anovan) used by Fig4 g-i.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert

from spirals_py.utils.atlas import _contourc, makeSmoothCoords
from spirals_py.utils.io import loadStructureTree


def nrrdread(filename):
    """Translated from spirals/utils/nrrdread.m

    Returns (X, meta) where X matches the MATLAB nrrdread orientation:
    X = permute(reshape(stream, sizes), [2 1 3]), i.e. the first two axes
    of the raw NRRD data are swapped. (Adaptation: uses pynrrd.)
    """
    import nrrd

    data, header = nrrd.read(str(filename))
    X = np.asarray(data).swapaxes(0, 1)
    return X, header


def plotOutline_fill(
    areaPath, st, section, hemisphere, scale, fillcolor, faceAlpha=1, ax=None
):
    """Translated from axons/utils/plotOutline_fill.m"""
    if ax is None:
        ax = plt.gca()
    if isinstance(areaPath, str):
        areaPath = [areaPath]

    section = np.asarray(section, dtype=float)
    if scale != 1:
        import cv2

        section = cv2.resize(
            section,
            (int(section.shape[1] * scale), int(section.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST,
        )

    indx = []
    spath = st["structure_id_path"].astype(str)
    for p in areaPath:
        indx.extend(np.flatnonzero(spath.str.contains(p, regex=False, na=False)))
    idAll = st["id"].iloc[indx].to_numpy(dtype=float)

    area = np.isin(section, idAll).astype(float)
    if hemisphere == -1:  # keep left half
        area[:, section.shape[1] // 2 :] = 0
    elif hemisphere == 1:  # keep right half
        area[:, : section.shape[1] // 2] = 0

    c1 = _contourc(area > 0)
    coordsReg1 = makeSmoothCoords(c1)
    for c in coordsReg1:
        ax.fill(
            c["x"], c["y"], facecolor=fillcolor, edgecolor="k",
            linewidth=1, alpha=faceAlpha, zorder=0,
        )
    return ax


def best_fit_line(x, y, z):
    """Translated from ephys/utils/best_fit_line.m

    Returns (m, p, s): line through m with direction cosines p.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = x.size
    m = np.array([x.mean(), y.mean(), z.mean()])
    w = np.column_stack([x - m[0], y - m[1], z - m[2]])
    a = w.T @ w / n
    u, sv, _ = np.linalg.svd(a)
    p = u[:, 0]
    s = sv[1] + sv[2]  # MATLAB d(2,2) + d(3,3)
    return m, p, s


def ttest_paired_h(a, b):
    """MATLAB [h,p] = ttest(a, b) (paired), NaN-omitted."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = ~(np.isnan(a) | np.isnan(b))
    if keep.sum() < 2:
        return np.nan, np.nan
    res = stats.ttest_rel(a[keep], b[keep])
    return int(res.pvalue < 0.05), res.pvalue


def ttest2_h(a, b):
    """MATLAB [h,p] = ttest2(a, b) (two-sample), NaN-omitted."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    res = stats.ttest_ind(a, b)
    return int(res.pvalue < 0.05), res.pvalue


def ttest2_right_1vN(x, y):
    """MATLAB [h,p] = ttest2(x, y, 'Tail','right') with a scalar sample x.

    Pooled two-sample t statistic; with n1 = 1 the pooled variance is
    estimated from y alone (df = n2 - 1).
    """
    x = float(np.asarray(x).ravel()[0])
    y = np.asarray(y, dtype=float).ravel()
    n1, n2 = 1, y.size
    if n2 < 2 or np.isnan(x) or np.isnan(y).any():
        return np.nan, np.nan
    ssw = np.sum((y - y.mean()) ** 2)
    sp2 = ssw / (n1 + n2 - 2)
    denom = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    if denom == 0:
        return np.nan, np.nan
    t = (x - y.mean()) / denom
    p = stats.t.sf(t, n1 + n2 - 2)
    return int(p < 0.05), p


def anovan2(y, f1, f2):
    """MATLAB pp = anovan(y, {f1 f2}, 'model', 2) -> p-values for
    [f1, f2, f1:f2]. (Adaptation: statsmodels Type II SS; MATLAB anovan
    defaults to Type III — results are stored but not plotted.)"""
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = pd.DataFrame(
        {"y": np.asarray(y, dtype=float).ravel(),
         "f1": pd.Categorical(np.asarray(f1).ravel()),
         "f2": pd.Categorical(np.asarray(f2).ravel())}
    )
    model = smf.ols("y ~ f1 * f2", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    return aov["PR(>F)"].to_numpy()[:3]


# --- helpers for sort_sprials_by_arousal / plotSpiralsMatchingRate_Arousal ---


def _load_mat_var(path, varname):
    """Load one variable from a .mat file (v7 via scipy, v7.3 via h5py)."""
    try:
        m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return m[varname]
    except NotImplementedError:
        import h5py

        with h5py.File(str(path), "r") as f:
            a = np.asarray(f[varname][()])
        return a.transpose(*range(a.ndim)[::-1])


def _load_cell_array_h5(path, varname):
    """Load a v7.3 cell array (any shape) as an object ndarray of matrices
    with MATLAB orientation (each element transposed back)."""
    import h5py

    with h5py.File(str(path), "r") as f:
        refs = f[varname][()]
        out = np.empty(refs.shape, dtype=object)
        it = np.nditer(refs, flags=["multi_index", "refs_ok"])
        while not it.finished:
            i = it.multi_index
            out[i] = np.asarray(f[refs[i]]).T
            it.iternext()
    return out


def _ismember_rows_int(A, B):
    """MATLAB ismember(A, B, 'rows') for 1-based integer coordinates."""
    keys_B = {tuple(row) for row in np.asarray(B, dtype=np.int64)}
    mask = np.array(
        [tuple(row) in keys_B for row in np.asarray(A, dtype=np.int64)]
    )
    return mask


def get_ssp_index(data_folder):
    """Translated from spirals/utils/get_ssp_index.m (incl. select_area.m)

    Returns brain_index = [col, row] (1-based) of right-hemisphere SSp in
    the 10um horizontal projection atlas.
    """
    from ._prediction_example_utils import _load_outline_mat

    data_folder = Path(data_folder)
    projectedAtlas1, projectedTemplate1 = _load_outline_mat(data_folder)
    st = loadStructureTree(data_folder / "tables" / "structure_tree_safe_2017.csv")
    spath = st["structure_id_path"].astype(str)

    # only select cortex (children of /997/8/567/) in the atlas
    spath2 = spath.str.startswith("/997/8/567/")
    idFilt = np.flatnonzero(spath2.to_numpy()) + 1  # MATLAB st.index
    projectedAtlas1 = projectedAtlas1.copy()
    projectedTemplate1 = projectedTemplate1.copy()
    lia = np.isin(projectedAtlas1, idFilt)
    projectedAtlas1[~lia] = 0
    projectedTemplate1[~lia] = 0

    # select_area: SSp, right hemisphere, scale 1
    sensoryArea = "/997/8/567/688/695/315/453/322/"
    spath3 = spath.str.startswith(sensoryArea)
    idFilt1 = np.flatnonzero(spath3.to_numpy()) + 1
    lia1 = np.isin(projectedAtlas1, idFilt1)
    projectedAtlas1[~lia1] = 0
    projectedTemplate1[~lia1] = 0
    projectedAtlas1[:, : projectedAtlas1.shape[1] // 2] = 0  # right hemi

    rows, cols = np.nonzero(projectedAtlas1 > 0)
    brain_index = np.column_stack([cols + 1, rows + 1])
    return brain_index


def remapColor(A, lold, hold):
    """Translated from spirals/utils/remapColor.m"""
    A = np.asarray(A, dtype=float).ravel()
    Aremaped = np.zeros(A.shape)
    for i in range(A.size):
        newVal = 1 + (A[i] - lold) * (255 - 1) / (hold - lold)
        Aremaped[i] = np.floor(newVal + 0.5)
    return Aremaped


def density_color_plot(pwAllRaw, histbin):
    """Translated from spirals/utils/density_color_plot.m

    Returns unique_spirals (N x 3): unique (x, y) rows with a column-
    density count in column 3.
    """
    pwAllRaw = np.asarray(pwAllRaw, dtype=float)
    unique_xy = np.unique(pwAllRaw[:, :2], axis=0)
    unique_spirals = np.column_stack([unique_xy, np.zeros(len(unique_xy))])
    half = histbin / 2
    for k in range(unique_spirals.shape[0]):
        cx, cy = unique_spirals[k, 0], unique_spirals[k, 1]
        indx = (
            (pwAllRaw[:, 0] >= cx - half)
            & (pwAllRaw[:, 0] <= cx + half)
            & (pwAllRaw[:, 1] >= cy - half)
            & (pwAllRaw[:, 1] <= cy + half)
        )
        unique_spirals[k, 2] = np.flatnonzero(indx).size
    return unique_spirals


def phaseSpiralHistogram2(T, data_folder, n_bins, brain_index, low_freq_band):
    """Translated from
    revision2/prediction_motion_energy/phaseSpiralHistogram2.m

    Returns a list of lists (per session x phase bin) of spiral rows.
    """
    from ._prediction_example_utils import _load_tform, _transform_points_forward
    from spirals_py.ephys.utils import get_session_info2

    data_folder = Path(data_folder)
    Fs = 35.0
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    spirals_sort = [[None] * n_bins for _ in range(len(T))]

    for kk in range(len(T)):
        ops = get_session_info2(T, kk, data_folder)
        fname = f"{ops.mn}_{ops.tdb}_{ops.en}"
        t = np.load(Path(ops.session_root) / "svdTemporalComponents_corr.timestamps.npy")
        t = np.atleast_1d(t.squeeze())

        T_tform = _load_tform(data_folder / "ephys" / "rf_tform" / f"{fname}_tform.mat")
        image_energy2 = _load_mat_var(
            data_folder / "revision2" / "prediction_motion_energy" / f"{fname}_motion_energy.mat",
            "image_energy2",
        ).ravel()
        image_energy2 = np.asarray(image_energy2, dtype=float)
        image_energy2[np.isnan(image_energy2)] = 0
        if image_energy2.size < t.size:
            image_energy2 = np.concatenate(
                [image_energy2, np.zeros(t.size - image_energy2.size)]
            )
        elif image_energy2.size > t.size:
            image_energy2 = image_energy2[: t.size]
        signal2 = image_energy2

        f1, f2 = butter(4, np.asarray(low_freq_band) / (Fs / 2), btype="bandpass")
        padlen = 3 * (max(len(f1), len(f2)) - 1)
        meanTrace_low = filtfilt(f1, f2, signal2, padlen=padlen)
        tracePhase_low = np.angle(hilbert(meanTrace_low))

        try:
            archiveCell = _load_cell_array_h5(
                data_folder / "ephys" / "spirals_raw_fftn" / f"{fname}_spirals_group_fftn.mat",
                "archiveCell",
            )
        except (NotImplementedError, OSError, KeyError):
            archiveCell = np.asarray(
                loadmat(
                    str(data_folder / "ephys" / "spirals_raw_fftn" / f"{fname}_spirals_group_fftn.mat"),
                    squeeze_me=True, struct_as_record=False,
                )["archiveCell"],
                dtype=object,
            )
        archiveCell = archiveCell.ravel()
        spiral_duration = np.array([np.asarray(c).shape[0] for c in archiveCell])
        groupedCells = [np.asarray(c) for c in archiveCell[spiral_duration >= 2]]
        filteredSpirals = np.vstack(groupedCells).astype(float)

        u, v = _transform_points_forward(T_tform, filteredSpirals[:, 0], filteredSpirals[:, 1])
        filteredSpirals[:, 0] = np.floor(u + 0.5)
        filteredSpirals[:, 1] = np.floor(v + 0.5)
        lia = _ismember_rows_int(
            np.column_stack([filteredSpirals[:, 0], filteredSpirals[:, 1]]), brain_index
        )
        filteredSpirals = filteredSpirals[lia, :]

        for mm in range(n_bins):
            index0 = np.flatnonzero(
                (tracePhase_low >= phase_bins[mm]) & (tracePhase_low < phase_bins[mm + 1])
            )
            frames = index0 + 1  # MATLAB 1-based frame numbers
            a = np.isin(filteredSpirals[:, 4].astype(int), frames)
            spirals_sort[kk][mm] = filteredSpirals[a, :]
    return spirals_sort


def spiralDensityBins2(T, data_folder, spirals_sort):
    """Translated from revision2/prediction_motion_energy/spiralDensityBins2.m"""
    from spirals_py.ephys.utils import get_session_info2

    pixSize = 0.01  # mm/pix
    pixArea = pixSize**2
    ssp_index = get_ssp_index(data_folder)
    hist_bin = 40
    n_bins = len(spirals_sort[0]) if spirals_sort else 0
    count_sample = np.zeros((len(T), n_bins))

    for kk in range(len(T)):
        ops = get_session_info2(T, kk, data_folder)
        t = np.load(
            Path(ops.session_root) / "svdTemporalComponents_corr.timestamps.npy"
        )
        t = np.atleast_1d(t.squeeze())
        for m in range(n_bins):
            spirals_temp = spirals_sort[kk][m]
            if spirals_temp is None or len(spirals_temp) == 0:
                count_sample[kk, m] = 0
                continue
            unique_spirals = density_color_plot(spirals_temp, hist_bin)
            lia = _ismember_rows_int(unique_spirals[:, :2], ssp_index)
            unique_spirals = unique_spirals[lia, :]
            if unique_spirals.shape[0] == 0:
                count_sample[kk, m] = 0
                continue
            unique_spirals_unit = unique_spirals[:, 2] / (hist_bin * hist_bin * pixArea)
            unique_spirals_unit = (
                unique_spirals_unit / t.size * 35 * n_bins
            )  # spirals/(mm^2*s)
            count_sample[kk, m] = unique_spirals_unit.max()
    return count_sample


def filter_facevideo_table(T):
    """MATLAB T = T(logical(T.facevideo),:); T([14,19],:) = [];

    The facevideo column and rows 14/19 refer to the revision table; if
    the column is absent the table is returned unchanged.
    """
    if "facevideo" not in T.columns:
        return T
    T = T[T["facevideo"].astype(bool)]
    drop = [T.index[13], T.index[18]]
    return T.drop(index=drop)
