"""Private helpers shared by the Fig5 (task part) / FigS14 plot translations.

Sources (MATLAB):
- task/utils/getPhotodiodeTime.m and
  dependencies/spikes/analysis/helpers/schmittTimes.m (corrected variant:
  MATLAB allows logical indices shorter than the indexed array, selecting
  the first numel(mask) elements; the version in _task_helpers.py indexes
  with the full-length time base and crashes on unequal lengths)
- data loading patterns for the v7.3 files under task/task_outcome and
  task/trial_trace (arrays are transposed back to MATLAB orientation)
- task/sessions/*.xlsx table filtering shared by several plot functions
- MATLAB anovan(...,'model',2,'random',3) approximated with statsmodels
  Type-II ANOVA (the random-effect denominators are not reproduced; the
  result is not used for any figure)
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spirals_py.task.plots._task_helpers import _schmitt, load_h5_var, tsToT


# ------------------------------------------------------- v7.3 io
def load_task_freq_arrays(path, varnames):
    """Load variables from a *_freq_to*Hz.mat file (MATLAB v7.3).

    3-D arrays are stored (pixel, comp, time*trial) on disk... in fact
    (nTrials, comp, time); they are transposed back to MATLAB orientation
    (time, comp, trial). 2-D contrast_all (2, nTrials) becomes
    (nTrials, 2). Returns a dict of ndarrays.
    """
    out = {}
    with h5py.File(path, "r") as f:
        for v in varnames:
            a = np.asarray(f[v])
            if a.ndim == 3:
                a = a.transpose(2, 1, 0)
            elif a.ndim == 2:
                a = a.T
            out[v] = a
    return out


def load_task_outcome(data_folder, mn, varname="T_all"):
    """Load the T_all trial table from task/task_outcome/[mn]_task_outcome.mat."""
    from spirals_py.task.preprocessing.load_task_table import load_task_table

    return load_task_table(
        Path(data_folder) / "task" / "task_outcome" / f"{mn}_task_outcome.mat",
        varname,
    )


def load_high_perf_task_sessions(data_folder, mn):
    """Read task/sessions/[mn].xlsx and keep high-performance task sessions
    (label == "task" and hit_left/hit_right > 0.7)."""
    T_session = pd.read_excel(Path(data_folder) / "task" / "sessions" / f"{mn}.xlsx")
    T1 = T_session[T_session.label == "task"]
    T1 = T1[(T1.hit_left > 0.7) & (T1.hit_right > 0.7)]
    return T1.reset_index(drop=True)


# ------------------------------------------------------- wheel / photodiode
def schmittTimes2(t, sig, thresh):
    """Translated from dependencies/spikes/analysis/helpers/schmittTimes.m.

    Corrected variant of _task_helpers.schmittTimes: MATLAB
    t(schmittSig(1:end-1)==1 & schmittSig(2:end)==-1) applies the
    (end-1)-long logical mask to the first end-1 elements of t.
    """
    t = np.asarray(t, dtype=float).ravel()
    sig = np.asarray(sig, dtype=float).ravel()
    schmittSig = _schmitt(sig, thresh[0], thresh[1])
    down_mask = (schmittSig[:-1] == 1) & (schmittSig[1:] == -1)
    up_mask = (schmittSig[:-1] == -1) & (schmittSig[1:] == 1)
    flipsDown = t[:-1][down_mask]
    flipsUp = t[:-1][up_mask]
    flipTimes = np.sort(np.concatenate([flipsUp, flipsDown]))
    return flipTimes, flipsUp, flipsDown


def getPhotodiodeTime2(session_root, win):
    """Translated from task/utils/getPhotodiodeTime.m (v7.3 files via h5py,
    using the corrected schmittTimes2)."""
    session_root = Path(session_root)
    pd_sig = load_h5_var(session_root / "photodiode_raw.mat", "pd").ravel()
    tlTimes = load_h5_var(session_root / "photodiode_timestamps_Timeline.mat", "tlTimes")
    tt = tsToT(tlTimes, pd_sig.size)

    allPD, flipsUp, flipsDown = schmittTimes2(tt, pd_sig, [0.5, 0.8])
    flipsUp = flipsUp[(flipsUp >= win[0]) & (flipsUp <= win[1])]
    flipsUp = flipsUp[:-1]

    dff_allPD = np.diff(allPD)
    allPD1 = allPD.copy()
    indx = dff_allPD < 0.6
    indx = np.concatenate([[True], indx])  # MATLAB: indx = [1; indx]
    allPD1 = allPD1[~indx]
    return allPD1


# ------------------------------------------------------- stats
def anovan2_random(y, contrast, label, subject):
    """Approximation of MATLAB
    anovan(y, {contrast, label, subject}, 'model', 2, 'random', 3).

    Fits y ~ contrast + label + contrast:label with statsmodels (Type-II
    SS). MATLAB's random-effect denominator terms are not reproduced; the
    returned table is not used for any figure. Returns None on failure.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception:
        return None
    df = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=float).ravel(),
            "contrast": pd.Categorical(np.asarray(contrast).ravel().astype(str)),
            "label": pd.Categorical(np.asarray(label).ravel().astype(str)),
            "subj": pd.Categorical(np.asarray(subject).ravel().astype(str)),
        }
    )
    try:
        md = smf.ols("y ~ contrast + label + contrast:label", data=df).fit()
        return sm.stats.anova_lm(md, typ=2)
    except Exception:
        return None


# ------------------------------------------------------- plotting
def hit_rate_panels(fig, hit_rate_high, hit_rate_low, hit_rate_change,
                    hit_rate_change_mean, hit_rate_change_sem, xticklabels):
    """The two panels shared by plotHitRateSlow / plotHitRate2_8Hz."""
    ax = fig.add_subplot(1, 2, 1)
    for i in range(5):  # MATLAB 1:5
        x_hi = 4 * (i + 1)
        ax.scatter(np.full(4, x_hi), hit_rate_high[i, :], s=8, c="b")
        ax.scatter(np.full(4, x_hi - 2), hit_rate_low[i, :], s=8, c="m")
        for j in range(4):
            ax.plot(
                [x_hi, x_hi - 2],
                [hit_rate_high[i, j], hit_rate_low[i, j]],
                "k",
            )
    ax.set_xticks([3, 7, 11, 15, 19])
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1.0)

    ax = fig.add_subplot(1, 2, 2)
    for i in range(5):
        ax.scatter(np.full(4, 4 * (i + 1) - 1), hit_rate_change[i, :], s=8, c="k")
    ax.errorbar(
        4 * np.arange(1, 6) - 1,
        hit_rate_change_mean,
        yerr=hit_rate_change_sem,
        color="r",
        linewidth=2,
    )
    ax.axhline(0, linestyle="--", color="k")
    ax.set_ylim(-0.3, 0.3)
    ax.set_xticks([3, 7, 11, 15, 19])
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Hit rate change")
    return fig
