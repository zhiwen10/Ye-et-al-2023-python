"""Run every preprocessing pipeline notebook (1-6) in sequence.

Usage: python tools/run_all.py [pipeline ...]
           [--run-detection] [--max-minutes N] [--redo]

Without arguments runs pipeline1_spirals … pipeline6_task in order (any
existing notebook named pipeline*.ipynb in notebooks/ is picked up).
Pass pipeline names (e.g. ``pipeline1 pipeline4``) to run a subset.

Per-cell policy lives in tools/run_pipeline.py: detection-tagged cells,
cells whose estimated runtime exceeds --max-minutes (default 10) and
already-completed cells are skipped; skipped steps fall back to the
release outputs via spirals_py.utils.paths.release_twin.  A global tqdm
bar tracks overall progress; run_pipeline.py adds a per-notebook bar
with the current function.  All outputs go to the python output tree
(SPIRALS_OUT_ROOT, see spirals_py/utils/paths.py), never to the release.
"""
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
ORDER = [
    "pipeline1_spirals", "pipeline2_axons", "pipeline3_spirals_mirror",
    "pipeline4_ephys", "pipeline5_whisker", "pipeline6_task",
]

sys.path.insert(0, str(ROOT / "tools"))
import nbformat  # noqa: E402

import run_pipeline  # noqa: E402


def cell_count(nb_path, run_detection, max_minutes):
    nb = nbformat.read(nb_path, as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    to_run, _skips = run_pipeline.runnable_cells(
        cells, run_detection, max_minutes,
        run_pipeline.load_completed(), redo=False,
    )
    return len(to_run)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    run_detection = "--run-detection" in sys.argv
    redo = "--redo" in sys.argv
    max_minutes = 10.0
    if "--max-minutes" in sys.argv:
        max_minutes = float(sys.argv[sys.argv.index("--max-minutes") + 1])
    if argv:
        names = argv
    else:
        names = [p.stem for p in sorted(NB.glob("pipeline*.ipynb"))]
        names.sort(key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER))
    nbs = [NB / f"{n}.ipynb" for n in names]
    for nb in nbs:
        if not nb.exists():
            print(f"missing notebook: {nb}")
            sys.exit(2)
    counts = {nb: cell_count(nb, run_detection, max_minutes) for nb in nbs}
    total = sum(counts.values())
    if not total:
        print("nothing to run (all cells skipped or already completed)")
        return
    with tqdm(total=total, desc="all pipelines", unit="cell") as overall:
        for nb in nbs:
            cmd = [sys.executable, str(ROOT / "tools" / "run_pipeline.py"),
                   str(nb), "--max-minutes", str(max_minutes)]
            if run_detection:
                cmd.append("--run-detection")
            if redo:
                cmd.append("--redo")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=ROOT,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith("DONE"):
                    overall.update(counts[nb])
                    tqdm.write(line.rstrip())
                elif "it/s]" not in line and "it]]" not in line:
                    tqdm.write(line.rstrip())
            if proc.wait() != 0:
                overall.close()
                sys.exit(1)
    print("ALL PIPELINES DONE")


if __name__ == "__main__":
    main()
