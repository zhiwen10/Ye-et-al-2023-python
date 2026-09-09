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
  data release.

## Installation (conda)

### 1. Install Anaconda/Miniconda

Install [Anaconda](https://www.anaconda.com/download) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don't have
it yet.

**(China users) configure the Tsinghua mirror first** — default channels can be
very slow. Create/edit `~/.condarc` (`C:\Users\<you>\.condarc` on Windows):

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
