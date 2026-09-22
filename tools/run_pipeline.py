"""Execute every code cell of a pipeline notebook sequentially.

Usage: python tools/run_pipeline.py notebooks/pipeline4_ephys.ipynb
           [--run-detection] [--max-minutes N] [--redo] [--estimates]

Cells share one namespace (as in Jupyter); the first exception aborts the
run with a traceback and exit code 1.  A tqdm progress bar tracks cell
completion.  A cell is skipped when

- it is tagged ``detection`` (the slow spiral-detection steps) unless
  ``--run-detection`` is given,
- its function's estimated total runtime (ESTIMATES_MIN below, minutes
  for the whole notebook call) exceeds ``--max-minutes`` (default 10), or
- an identical cell already completed in an earlier run (tracked in
  <out_root>/.completed_cells.txt) unless ``--redo`` is given.

Skipped steps fall back to the release outputs via
spirals_py.utils.paths.release_twin, so downstream cells keep working.
All outputs go to the python output tree (SPIRALS_OUT_ROOT, see
spirals_py/utils/paths.py).
"""
import re
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nbformat
from tqdm import tqdm

# Estimated total runtime in minutes of one notebook-level call (all
# sessions), from the MATLAB run-time comments, observed runs and the
# 2026-09-21 desktop measurements.  Cells above the --max-minutes
# threshold are skipped.  None = no estimate (cell runs).
ESTIMATES_MIN = {
    # pipeline1 - spirals
    "getSpiralDetection": 900,          # ~1 h/session (MATLAB comment)
    "getSpiralDetectionFftnRaw": 900,
    "getSpiralDetectionFftnPermute": 900,
    "getSpiralsFFTnFreq": 2700,         # chains the detection drivers
    "getSpiralGrouping": 5,             # measured ~18 s/session
    "getSpiralGroupingScrambled": 50,   # 10 reps x 15 sessions
    "getSpiralDurationRatio": 13,       # measured ~52 s/session
    "getSpiralDensityMap": 8,
    "getSpiralsDensityLine": 6,
    "getTaperPowerMap3": 60,
    "getExamplePixelTrace_005_8Hz": 1,
    "getPowerBandRatio": 4,
    "setAlphaThreshold": 1,
    "getFFTNSpiralsMap": 6,
    "getFFTNSpiralsStats": 2,
    "getSpiralsGroup_freq": 20,         # 3 bands x 15 sessions
    "getSpiralsScrambled_freq": 100,
    "getDurationFreq": 2,
    "getSpiralDensityLine_all": 3,
    "getSpiralDensityByDuration": 35,   # 7 duration bins x 15 sessions
    "getSpiralDensityByRadius": 35,     # 7 radius bins x 15 sessions
    "getMotionEnergyIndex": 6,
    "getAmpIndex": 20,                  # band-pass per session x 15
    "getSpiralSpeed": 30,
    "getSpiralSpeedConcat": 2,
    "getSpeedForRadius": 1,
    # pipeline2 - axons
    "getAxonBiasTable": 15,
    "getSpiralsPhaseMap": 25,           # ~1.5 min/session (observed)
    "getSpiralsPhaseMapLeft": 12,
    "getSpiralsPhaseMap2": 25,
    "getAxonBiasTableMO": 5,
    "getAxonBiasTableMO2": 4,
    "getAxonBiasTableSSp2": 4,
    "getMOroi": 1,
    # pipeline3 - spirals_mirror
    "getReducedRankRegressionAP": 300,  # rank-50 CCA per session
    "getReducedRankRegressionHEMI": 300,
    "getExampleKernelAP": 3,
    "getExampleKernelHEMI": 3,
    "getAxonMapAP": 5,
    "getAxonMapHEMI": 5,
    "getAxonMapInjection": 4,
    "getMapsSession": 3,
    # pipeline4 - ephys
    "getdVPrediction": 130,             # ~10 min/session x 13
    "getdVPredictionPermute": 130,
    "getEphysROI": 1,                   # release ROI copied on first run
    "getSpiralsRaw": 780,               # detection within ROI
    "getEphysSpiralsGrouping": 3,
    "getSpiralsPrediction": 780,
    "getSpiralsPredictionPermute": 780,
    "getSpiralComparePredict": 8,
    "getSpiralComparePermute": 8,
    "getPhaseFlowMatchingIndex": 60,    # 21 flow reps per session
    "getVarOrderedByNeuron": 240,       # per-unit regressions
    # pipeline5 - whisker
    "getWhiskerMeanMaps": 5,
    "getSpiralsWhiskerMeanMaps": 20,    # detection on mean maps
    "getSpiralsPrePost": 3,
    "getSpiralsPeriStim": 4,
    "getWhiskerSingleTrials": 2,
    # pipeline6 - task
    "getPsychometricCurve": 8,
    "getTaskTrialOutcome": 6,
    "getMeanMapSession": 20,
    "getMeanMapsAll": 1,
    "getCorrectSpiralDensity": 2,
    "getTaskOnsetPhase": 15,
    "getTaskSpirals": 12,
    "getPassiveSpirals": 12,
    "getSprialCountByRadiusTime": 10,
    "getPassiveSpiralPrePost": 2,
    "getCorrectSpiralPrePost": 2,
}

_IMPORT_RE = re.compile(r"^from (spirals_py\.[\w.]+) import (\w+)", re.M)


def cell_function(cell):
    """Function called by a pipeline cell, or None (e.g. the setup cell)."""
    m = _IMPORT_RE.search(cell.source)
    if not m:
        return None
    func = m.group(2)
    for ln in cell.source.splitlines():
        if ln.startswith(func + "("):
            return func
    return None


def completed_path():
    from spirals_py.utils.paths import out_root

    p = out_root() / ".completed_cells.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_completed():
    p = completed_path()
    return set(p.read_text().split()) if p.exists() else set()


def cell_key(cell):
    import hashlib

    return hashlib.md5(cell.source.encode()).hexdigest()[:10]


def runnable_cells(cells, run_detection=False, max_minutes=10.0,
                   completed=frozenset(), redo=False):
    """Split cells into (to_run, skips) where skips is [(cell, reason)]."""
    to_run, skips = [], []
    for c in cells:
        func = cell_function(c)
        if not run_detection and "detection" in c.metadata.get("tags", []):
            skips.append((c, "detection (use --run-detection)"))
            continue
        est = ESTIMATES_MIN.get(func)
        if est is not None and est > max_minutes:
            skips.append((c, f"est. {est:g} min > {max_minutes:g} min "
                            f"(use --max-minutes)"))
            continue
        if func and not redo and cell_key(c) in completed:
            skips.append((c, "already completed (use --redo)"))
            continue
        to_run.append(c)
    return to_run, skips


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    run_detection = "--run-detection" in sys.argv
    redo = "--redo" in sys.argv
    max_minutes = 10.0
    if "--max-minutes" in sys.argv:
        max_minutes = float(sys.argv[sys.argv.index("--max-minutes") + 1])
    if "--estimates" in sys.argv:
        for k, v in sorted(ESTIMATES_MIN.items()):
            mark = "SKIP" if v is not None and v > max_minutes else "run "
            print(f"{mark} {k:32s} ~{v:g} min")
        return
    if not args:
        print(__doc__)
        sys.exit(2)
    nb_path = Path(args[0])
    nb = nbformat.read(nb_path, as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    to_run, skips = runnable_cells(
        cells, run_detection, max_minutes, load_completed(), redo
    )
    for c, reason in skips:
        func = cell_function(c) or "?"
        print(f"SKIP {nb_path.name}: {func} - {reason}", flush=True)
    ns = {"__name__": "__main__"}
    pbar = tqdm(to_run, desc=nb_path.stem, unit="cell", initial=0)
    done = 0
    for c in pbar:
        first = next(
            (ln for ln in c.source.splitlines() if ln.strip()), ""
        )
        pbar.set_postfix_str(first[:60], refresh=True)
        try:
            exec(compile(c.source, str(nb_path), "exec"), ns)
        except Exception:
            pbar.close()
            print(f"ERROR in cell '{first[:80]}' of {nb_path.name}", flush=True)
            traceback.print_exc()
            sys.exit(1)
        if cell_function(c):
            with completed_path().open("a") as fh:
                fh.write(cell_key(c) + "\n")
        done += 1
    print(f"DONE {nb_path.name}: {done} run, {len(skips)} skipped, "
          f"{len(cells)} total cells", flush=True)


if __name__ == "__main__":
    main()
