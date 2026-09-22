# Ye et al. 2023 — Python translation

Python translation of the MATLAB code for **Ye et al. 2023** (cortical spirals
analysis), translated module by module from
[YE-et-al-2023-spirals](https://github.com/zhiwen10/YE-et-al-2023-spirals).

- `spirals_py/` — Python package mirroring the MATLAB repo structure
  (`utils`, `spirals`, `axons`, `spirals_mirror`, `ephys`, `whisker`, `task`).
  Function names are kept identical to the MATLAB originals
  (e.g. `loadUVt1`, `plotSpiralDuration`) for easy cross-referencing; each
  function's docstring points to the MATLAB file it was translated from.
- `notebooks/` — Jupyter notebooks translated from the top-level MATLAB scripts
  (`data_overview.m`, `figure1_spirals.m` … `figure6_task.m`), including the
  Extended Data figures. All notebooks execute end-to-end against the paper's
  data release. The `pipeline*_*.ipynb` notebooks translate the preprocessing
  pipelines and regenerate the intermediate `.mat` files the figure notebooks
  read (e.g. `pipeline6_task.ipynb` rewrites everything under `task/`).
- `notebooks/task_tutorial.ipynb` — doc-style tutorial on the task dataset:
  Part 1 explains the task design and data layout (with figures), Part 2 walks
  through one example session hands-on (behavior table, psychometric curve,
  task vs passive response movies, single-trial waves, session-epoch view).
  A good entry point before `figure6_task.ipynb`, e.g. for teaching.

## Installation (conda)

### 1. Install Anaconda/Miniconda

Install [Anaconda](https://www.anaconda.com/download) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don't have
it yet.

**(China users) configure the Tsinghua mirror first** — default channels can be
very slow. Create/edit `~/.condarc` (`C:\Users\<you>\.condarc` on Windows;
it is just a plain YAML text file named `.condarc`). The mirror is configured
per-user here rather than in `environment.yml`, because `channels:` entries in
`environment.yml` only apply during env creation, don't affect pip, and would
force a regional mirror on everyone:

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
ssl_verify: true
```

And for pip:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Create the environment

```bash
git clone https://gitee.com/ye-lab/Ye-et-al-2023-python.git
cd Ye-et-al-2023-python
conda env create -f environment.yml
conda activate spirals-py
```

This installs Python 3.11 with numpy, scipy, pandas, matplotlib, jupyterlab,
openpyxl, h5py, scikit-image, scikit-learn, tifffile, seaborn, statsmodels,
plus pip packages opencv-python, pynrrd and colorcet.

Equivalent manual setup (if you prefer not to use `environment.yml`):

```bash
conda create -n spirals-py python=3.11 numpy scipy pandas matplotlib \
    jupyterlab openpyxl h5py scikit-image scikit-learn tifffile seaborn \
    statsmodels -y
conda activate spirals-py
pip install opencv-python pynrrd colorcet
```

### 3. Install the `spirals_py` package (editable)

From the repository root:

```bash
pip install -e .
```

## Data

Download the data release from
[figshare](https://doi.org/10.6084/m9.figshare.27850707) and place it so the
folders are reachable at `D:\data` (same layout as the MATLAB code expects —
`D:\data\tables`, `D:\data\spirals`, `D:\data\ephys`, …). If your data lives
elsewhere, edit `DATA_FOLDER` in the first cell of each notebook.

Figures are saved under `D:\data\figures\Fig*`.

Note: six files in `ephys/rf_tform_4x/` are MATLAB v7 `.mat` files whose
`affine2d` objects Python cannot decode directly. Load them once in MATLAB and
re-save with `save('<name>_v73.mat', '-struct', 'S', '-v7.3')` next to the
originals; the loader (`spirals_py/ephys/plots/_prediction_example_utils.py`)
falls back to the `_v73` sibling automatically.

## Running the notebooks

```bash
conda activate spirals-py
cd Ye-et-al-2023-python
jupyter lab
```

Open any notebook in `notebooks/` and run the cells top to bottom.

| Notebook | Original MATLAB script |
|---|---|
| `data_overview.ipynb` | `data_overview.m` |
| `figure1_spirals.ipynb` | `figure1_spirals.m` (Fig. 1 + Ext. Data 1–9) |
| `figure2_axons.ipynb` | `figure2_axons.m` (Fig. 2 + Ext. Data 10) |
| `figure3_sprials_mirror.ipynb` | `figure3_sprials_mirror.m` (Fig. 3 + Ext. Data 11) |
| `figure4_ephys.ipynb` | `figure4_ephys.m` (Fig. 4 + Ext. Data 12–13) |
| `figure5_whisker.ipynb` | `figure5_whisker.m` (Fig. 5) |
| `figure6_task.ipynb` | `figure6_task.m` (Fig. 6 + Ext. Data 14–15) |
| `pipeline1_spirals.ipynb` | `pipeline1_spirals.m` (spiral preprocessing: regenerates the `spirals/` files; detection cells tagged) |
| `pipeline2_axons.ipynb` | `pipeline2_axons.m` (axon preprocessing: regenerates the `axons/` + `revision/axons/` files) |
| `pipeline3_spirals_mirror.ipynb` | `pipeline3_spirals_mirror.m` (mirror-symmetry preprocessing: regenerates the `spirals_mirror/` files) |
| `pipeline4_ephys.ipynb` | `pipeline4_ephys.m` (ephys preprocessing: regenerates the `ephys/` files) |
| `pipeline5_whisker.ipynb` | `pipeline5_whisker.m` (whisker preprocessing: regenerates the `whisker/` files) |
| `pipeline6_task.ipynb` | `pipeline6_task.m` (task preprocessing: regenerates the `task/*.mat` files) |

### Unattended full runs

```bash
python tools/run_pipeline.py notebooks/pipeline4_ephys.ipynb   # one pipeline
python tools/run_all.py                                        # all pipelines, in order
python tools/run_all.py pipeline1 pipeline4                    # subset
python tools/run_pipeline.py --estimates                       # runtime estimate table
```

- All pipeline outputs are written to a separate python tree — the MATLAB
  data release is never overwritten. Roots are resolved per platform
  (`spirals_py/utils/paths.py`): `SPIRALS_DATA_ROOT` (release inputs,
  default `D:\data` on Windows, `~/data` on macOS/Linux) and
  `SPIRALS_OUT_ROOT` (python outputs, default `D:\data_python` on Windows,
  `~/data_python` on macOS/Linux); export the variables to override.
  Chained intermediates fall back to the release copies via
  `spirals_py.utils.paths.release_twin` when they have not been regenerated.
- Cells tagged `detection` (the spiral-detection steps, ~1 h/session) are
  skipped by default; pass `--run-detection` to include them.
- Each function has a total-runtime estimate (`ESTIMATES_MIN` in
  `tools/run_pipeline.py`, from the MATLAB run-time comments, observed runs
  and desktop measurements); cells estimated above `--max-minutes` (default
  10) are skipped with a log line — raise the threshold to include them.
- Completed cells are tracked in `<out_root>/.completed_cells.txt` and skipped
  on later runs (`--redo` to rerun), so interrupted full runs resume.
- Progress bars: one per notebook cell (`tools/run_pipeline.py`), one overall
  (`tools/run_all.py`), plus per-session `tqdm` bars inside every
  preprocessing function. Logs: `D:\data_python\run_all.log` /
  `run_all.err.log` when launched detached.
