"""Generate the figure notebooks for the Ye-et-al-2023-python translation.

Creates notebooks/figure1_spirals.ipynb, figure3_sprials_mirror.ipynb,
figure4_ephys.ipynb and figure6_task.ipynb, mirroring the MATLAB scripts
figure1_spirals.m, figure3_sprials_mirror.m, figure4_ephys.m, figure6_task.m.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

KERNEL = {
    "kernelspec": {
        "display_name": "Python (spirals-py)",
        "language": "python",
        "name": "spirals-py",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

SETUP_XLSX = """\
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_FOLDER = Path(r"D:\\data")
FIGURE_FOLDER = DATA_FOLDER / "figures"

# load session table
T = pd.read_excel(DATA_FOLDER / "tables" / "spiralSessions3.xlsx")
T"""

SETUP_CSV = """\
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_FOLDER = Path(r"D:\\data")
FIGURE_FOLDER = DATA_FOLDER / "figures"

# load widefield + ephys session table
T = pd.read_csv(DATA_FOLDER / "tables" / "spirals_ephys_sessions_new2.csv")
T"""

SETUP_PLAIN = """\
from pathlib import Path

import matplotlib.pyplot as plt

DATA_FOLDER = Path(r"D:\\data")
FIGURE_FOLDER = DATA_FOLDER / "figures"
"""


def cell(nb, kind, content):
    nb.cells.append(nbf.v4.new_markdown_cell(content) if kind == "md"
                    else nbf.v4.new_code_cell(content))


def call(import_path, func, args, comment):
    src = f"from {import_path} import {func}\n\n{comment}\n"
    lhs = ""
    if "{ret}" in comment:
        pass
    src += f"{func}({args})\nplt.show()"
    return src


def build(title, md_intro, setup, sections):
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    cell(nb, "md", f"# {title}\n\n{md_intro}")
    cell(nb, "code", setup)
    for header, entries in sections:
        cell(nb, "md", f"## {header}")
        for e in entries:
            if e[0] == "md":
                cell(nb, "md", e[1])
                continue
            _, imp, func, args, comment = e
            src = f"from {imp} import {func}\n\n"
            body = f"{func}({args})"
            if comment:
                src += f"# {comment}\n"
            src += body + "\nplt.show()"
            cell(nb, "code", src)
    return nb


S = "spirals_py.spirals.plots"
SM = "spirals_py.spirals_mirror.plots"
E = "spirals_py.ephys.plots"
TK = "spirals_py.task.plots"

# ---------------------------------------------------------------- figure 1
fig1_sections = [
    ("Figure 1", [
        ("code", f"{S}.plotSpiralTimeSeries3d", "plotSpiralTimeSeries3d",
         "DATA_FOLDER, save_folder", "plot example spiral time series & frame with optical flow"),
        ("code", f"{S}.plotSpiralTimeSeries", "plotSpiralTimeSeries",
         "DATA_FOLDER, save_folder", "plot example spiral frame"),
        ("code", f"{S}.plotSpiralSequence3", "plotSpiralSequence3",
         "DATA_FOLDER, save_folder", "plot example spiral sequence"),
        ("code", f"{S}.plotSpiralDuration", "plotSpiralDuration",
         "T, DATA_FOLDER, save_folder", "plot spiral duration ratio vs scrambled distribution"),
        ("code", f"{S}.plotSpiralDensityAllSessions", "plotSpiralDensityAllSessions",
         "DATA_FOLDER, save_folder", "plot spiral density map (combine all sessions)"),
        ("code", f"{S}.plotSpiralSpeedSummary", "plotSpiralSpeedSummary",
         "T, DATA_FOLDER, save_folder", "plot spiral speeds for all spiral radii"),
        ("code", f"{S}.makePressVideoExample", "makePressVideoExample",
         "DATA_FOLDER, save_folder", "make press video example"),
    ]),
    ("Extended Data Fig. 1", [
        ("code", f"{S}.plotExampleOscillation", "plotExampleOscillation",
         "DATA_FOLDER, save_folder", "plot example horizontal view and time series"),
        ("code", f"{S}.plotPowerSpectrum4", "plotPowerSpectrum4",
         "T, DATA_FOLDER, save_folder", "plot mean power map across 15 sessions"),
        ("code", f"{S}.plotExamplePowerSpectrum2", "plotExamplePowerSpectrum2",
         "T, DATA_FOLDER, save_folder", "plot example power spectrum"),
        ("code", f"{S}.plotPowerRatio3", "plotPowerRatio3",
         "DATA_FOLDER, save_folder", "plot power ratio"),
        ("code", f"{S}.plotPowerRatioRegression3", "plotPowerRatioRegression3",
         "T, DATA_FOLDER, save_folder", "plot power ratio regression"),
        ("code", f"{S}.plotExampleSpiral2b", "plotExampleSpiral2b",
         "T, DATA_FOLDER, save_folder", "plot LK_0003 example spirals"),
        ("code", f"{S}.plotExampleSpiralSpectrum2", "plotExampleSpiralSpectrum2",
         "T, DATA_FOLDER, save_folder", "plot LK_0003 power spectrum"),
        ("code", f"{S}.plotExampleSpiral3b", "plotExampleSpiral3b",
         "T, DATA_FOLDER, save_folder", "plot ZYE_0067 example spirals"),
        ("code", f"{S}.plotExampleSpiralSpectrum3", "plotExampleSpiralSpectrum3",
         "T, DATA_FOLDER, save_folder", "plot ZYE_0067 power spectrum"),
    ]),
    ("Extended Data Fig. 2", [
        ("code", f"{S}.plotSpiralDetectionPipeline", "plotSpiralDetectionPipeline",
         "T, DATA_FOLDER, save_folder", "plot spiral detection pipeline illustration"),
    ]),
    ("Extended Data Fig. 3", [
        ("code", None, None, None, None),  # placeholder replaced below
    ]),
    ("Extended Data Fig. 4", [
        ("code", f"{S}.plotLFPspirals", "plotLFPspirals",
         "DATA_FOLDER, save_folder", "plot example spirals in cortical LFP"),
    ]),
    ("Extended Data Fig. 5", [
        ("code", f"{S}.plotSpiralsBySession1", "plotSpiralsBySession1",
         "T, session_rows, DATA_FOLDER, save_folder", "plot visual rf mapping by session"),
        ("code", f"{S}.plotSpiralsBySession2", "plotSpiralsBySession2",
         "T, session_rows, DATA_FOLDER, save_folder", "plot spiral distribution by session"),
    ]),
    ("Extended Data Fig. 6", [
        ("code", f"{S}.plotSpiralDensityByRadius", "plotSpiralDensityByRadius",
         "DATA_FOLDER, save_folder", "plot spiral density maps across different radius"),
        ("code", f"{S}.plotSpiralDensityByDuration", "plotSpiralDensityByDuration",
         "DATA_FOLDER, save_folder", "plot spiral density maps across different durations"),
    ]),
    ("Extended Data Fig. 7", [
        ("code", f"{S}.plotExampleSpiralTrajectory", "plotExampleSpiralTrajectory",
         "T, DATA_FOLDER, save_folder", "plot example grouped spiral sequences"),
        ("code", f"{S}.plotSpiralDirectionRatio2", "plotSpiralDirectionRatio2",
         "T, DATA_FOLDER, save_folder", "plot CCW spirals ratio across sessions"),
        ("code", f"{S}.plotSpiralDensitySessionsMeanSEM", "plotSpiralDensitySessionsMeanSEM",
         "T, DATA_FOLDER, save_folder", "plot spiral density map mean and SEM across sessions"),
        ("code", f"{S}.plotSpiralsSymmetryRatio", "plotSpiralsSymmetryRatio",
         "T, DATA_FOLDER, save_folder", "plot spiral symmetry ratio across sessions"),
    ]),
    ("Extended Data Fig. 8", [
        ("code", f"{S}.plotSpiralSyncIndex", "plotSpiralSyncIndex",
         "T, DATA_FOLDER, save_folder", "plot example time series and example frame index"),
        ("code", f"{S}.plotMotionEnergyAmpX", "plotMotionEnergyAmpX",
         "T, DATA_FOLDER, save_folder", "plot 2-8Hz amp vs motion energy relationship"),
        ("code", f"{S}.plotAmpIndex", "plotAmpIndex",
         "DATA_FOLDER, save_folder", "plot 2-8Hz amp vs index across sessions"),
        ("code", f"{S}.plotMotionEnergyIndex", "plotMotionEnergyIndex",
         "DATA_FOLDER, save_folder", "plot motion energy vs index across sessions"),
        ("code", f"{S}.plotExamplePlaneWaveSeries2", "plotExamplePlaneWaveSeries2",
         "DATA_FOLDER, save_folder", "plot example plane wave"),
        ("code", f"{S}.plotWaveIndexAmp", "plotWaveIndexAmp",
         "DATA_FOLDER, save_folder", "plot plane wave index vs 2-8 Hz amp"),
        ("code", f"{S}.plotWaveRatio2", "plotWaveRatio2",
         "DATA_FOLDER, save_folder", "plot plane wave vs spiral wave ratio"),
        ("code", f"{S}.plotExamplePlaneWave", "plotExamplePlaneWave",
         "DATA_FOLDER, save_folder", "plot example plane wave symmetry"),
        ("code", f"{S}.plotSymmetry4", "plotSymmetry4",
         "DATA_FOLDER, save_folder", "plot plane wave angle distribution across frames"),
        ("code", f"{S}.plotBorderPlaneWave", "plotBorderPlaneWave",
         "DATA_FOLDER, save_folder", "plot plane wave angle distribution on the border"),
    ]),
    ("Extended Data Fig. 9", [
        ("code", f"{S}.plotSpiralSpeedExample", "plotSpiralSpeedExample",
         "DATA_FOLDER, save_folder", "plot example spiral speed calculation illustration"),
        ("code", f"{S}.plotSpeedForRadius", "plotSpeedForRadius",
         "50, DATA_FOLDER, save_folder", "plot all spiral speeds at 50-pixel radius"),
        ("code", f"{S}.plotSpeedForRadius", "plotSpeedForRadius",
         "100, DATA_FOLDER, save_folder", "plot all spiral speeds at 100-pixel radius"),
    ]),
]

# FigS3 needs the freq variable: build its cells manually
fig1_sections[3] = ("Extended Data Fig. 3", [
    ("md", "```python\nfreq = [2, 8]\n```\n(defined in the first cell below)"),
])
fig1_s3_extra = """\
freq = [2, 8]

from spirals_py.spirals.plots.plotExampleDataVsFft import plotExampleDataVsFft

# plot example epoch of data and 3d-fft
plotExampleDataVsFft(T, DATA_FOLDER, save_folder)
plt.show()"""

# ---------------------------------------------------------------- figure 3
fig3_sections = [
    ("Figure 3", [
        ("code", f"{SM}.plotCortexDivision", "plotCortexDivision",
         "DATA_FOLDER, save_folder", "plot division (AP, hemi)"),
        ("code", f"{SM}.plotVarianceExplained", "plotVarianceExplained",
         "T, DATA_FOLDER, save_folder", "plot variance explained from regression"),
        ("code", f"{SM}.plotExampleKernelHEMI", "plotExampleKernelHEMI",
         "DATA_FOLDER, save_folder", "plot example hemi regression kernel"),
        ("code", f"{SM}.plotExampleKernelAP", "plotExampleKernelAP",
         "DATA_FOLDER, save_folder", "plot example AP regression kernel"),
        ("code", f"{SM}.plotKernelMapsHEMI", "plotKernelMapsHEMI",
         "DATA_FOLDER, save_folder", "plot hemi kernel maps"),
        ("code", f"{SM}.plotKernelMapsAP", "plotKernelMapsAP",
         "DATA_FOLDER, save_folder", "plot AP kernel maps"),
        ("code", f"{SM}.plotMatchingIndexHEMI", "plotMatchingIndexHEMI",
         "DATA_FOLDER, save_folder", "plot hemi matching index"),
        ("code", f"{SM}.plotMatchingIndexAP", "plotMatchingIndexAP",
         "DATA_FOLDER, save_folder", "plot AP matching index"),
    ]),
    ("Extended Data Fig. 11", [
        ("code", f"{SM}.plotMapsSession", "plotMapsSession",
         "T, DATA_FOLDER, save_folder", "plot kernel maps across sessions"),
    ]),
]

# ---------------------------------------------------------------- figure 4
fig4_sections = [
    ("Figure 4", [
        ("code", f"{E}.plotSpiralPredictionExample", "plotSpiralPredictionExample",
         "T, DATA_FOLDER, save_folder", "plot example spiral prediction"),
        ("code", f"{E}.plotProbeLocation", "plotProbeLocation",
         "T, DATA_FOLDER, save_folder", "plot all probe locations in atlas space"),
        ("code", f"{E}.plotSpiralsMatchingRate", "plotSpiralsMatchingRate",
         "T, DATA_FOLDER, save_folder", "plot spiral matching ratio vs shuffle condition"),
        ("code", f"{E}.plotMatchingIndex", "plotMatchingIndex",
         "T, DATA_FOLDER, save_folder", "plot phase and wave matching index vs shuffle condition"),
    ]),
    ("Extended Data Fig. 12", [
        ("code", f"{E}.plotSpiralPredictionExample1", "plotSpiralPredictionExample1",
         "T, DATA_FOLDER, save_folder", "plot example spiral prediction from striatal spiking data"),
        ("code", f"{E}.plotSpiralPredictionExample2", "plotSpiralPredictionExample2",
         "T, DATA_FOLDER, save_folder", "plot example spiral prediction from midbrain spiking data"),
        ("code", f"{E}.plotWavePredictionExample1", "plotWavePredictionExample1",
         "T, DATA_FOLDER, save_folder", "plot example wave prediction from striatal spiking data"),
        ("code", f"{E}.plotWavePredictionExample2", "plotWavePredictionExample2",
         "T, DATA_FOLDER, save_folder", "plot example wave prediction from midbrain spiking data"),
    ]),
    ("Extended Data Fig. 13", [
        ("code", f"{E}.plotVarMap", "plotVarMap",
         "T, DATA_FOLDER, save_folder", "plot variance explained maps from prediction for all sessions"),
        ("code", f"{E}.plotVarSummary", "plotVarSummary",
         "T, DATA_FOLDER, save_folder", "plot variance explained summary and relationship with neuron N"),
        ("code", f"{E}.plotWaveMatchingSession", "plotWaveMatchingSession",
         "T, DATA_FOLDER, save_folder", "plot wave matching index with 2-8Hz amp for all sessions"),
    ]),
]

fig4_arousal = """\
from spirals_py.ephys.plots.sort_sprials_by_arousal import sort_sprials_by_arousal
from spirals_py.ephys.plots.plotSpiralsMatchingRate_Arousal import plotSpiralsMatchingRate_Arousal

# NOTE: these two cells require revision2 motion-energy data
# (revision2/prediction_motion_energy/*_motion_energy.mat and
# spiral_compare_sessions_arousal.mat) which are not part of the
# figshare Part1-4 download; guard with try/except.
try:
    sort_sprials_by_arousal(T, DATA_FOLDER, save_folder)
    plotSpiralsMatchingRate_Arousal(T, DATA_FOLDER, save_folder)
except FileNotFoundError as e:
    print(f"skipped (missing data): {e}")
plt.show()"""

# ---------------------------------------------------------------- figure 6
fig6_sections = [
    ("Figure 5 (task)", [
        ("code", f"{TK}.plotCorrectMapsFlow", "plotCorrectMapsFlow",
         "DATA_FOLDER, save_folder", "plot mean widefield map in correct trials"),
        ("code", f"{TK}.getMeanSpiralsDetection", "getMeanSpiralsDetection",
         "DATA_FOLDER, save_folder", "mean spirals detection"),
        ("code", f"{TK}.plotCorrectMeanTrace2", "plotCorrectMeanTrace2",
         "DATA_FOLDER, save_folder", "plot mean traces in correct trials"),
        ("code", f"{TK}.plotCorrectSpiralDensity", "plotCorrectSpiralDensity",
         "DATA_FOLDER, save_folder", "plot spiral density maps before and after onset"),
        ("code", f"{TK}.plotSpiralRateAll", "plotSpiralRateAll",
         "DATA_FOLDER, save_folder", "plot spiral rate change around onset time"),
        ("code", f"{TK}.plotMeanTraceExampleSlow", "plotMeanTraceExampleSlow",
         "DATA_FOLDER, save_folder", "plot mean traces between 0.05-2Hz"),
        ("code", f"{TK}.plotTraceExample2_8Hz", "plotTraceExample2_8Hz",
         "DATA_FOLDER, save_folder", "plot mean traces between 2-8Hz"),
        ("code", f"{TK}.plotHitRateSlow", "plotHitRateSlow",
         "DATA_FOLDER, save_folder", "psychometric curve with pre-stim 0.05-2Hz power"),
        ("code", f"{TK}.plotHitRate2_8Hz", "plotHitRate2_8Hz",
         "DATA_FOLDER, save_folder", "psychometric curve with pre-stim 2-8Hz phase"),
    ]),
    ("Extended Data Fig. 14", [
        ("code", f"{TK}.plotPsychometricCurve", "plotPsychometricCurve",
         "DATA_FOLDER, save_folder", "plot psychometric curve"),
        ("code", f"{TK}.plotTaskSessionEpochExample", "plotTaskSessionEpochExample",
         "DATA_FOLDER, save_folder", "plot example session with low-arousal states and miss"),
        ("code", f"{TK}.plotMeanTraceAcrossContrasts", "plotMeanTraceAcrossContrasts",
         "DATA_FOLDER, save_folder", "plot mean responses across contrasts"),
        ("code", f"{TK}.plotExamplePhase2_8Hz", "plotExamplePhase2_8Hz",
         "DATA_FOLDER, save_folder", "plot example 2-8Hz around onset"),
    ]),
    ("Extended Data Fig. 15", [
        ("code", f"{TK}.plotMeanMapsAll", "plotMeanMapsAll",
         "DATA_FOLDER, save_folder", "plot mean wf maps in different trial types"),
        ("code", f"{TK}.plotMeanTraceAll", "plotMeanTraceAll",
         "DATA_FOLDER, save_folder", "plot traces in different trial types"),
        ("code", f"{TK}.plotSpiralTrialExample", "plotSpiralTrialExample",
         "DATA_FOLDER, save_folder", "plot single trial maps"),
        ("code", f"{TK}.plotPassiveSpiralPrePost", "plotPassiveSpiralPrePost",
         "DATA_FOLDER, save_folder", "plot spiral density maps in passive viewing"),
        ("code", f"{TK}.plotCorrectSpiralPrePost", "plotCorrectSpiralPrePost",
         "DATA_FOLDER, save_folder", "plot spiral density maps in active task"),
        ("code", f"{TK}.plotSpiralCountOverTime", "plotSpiralCountOverTime",
         "DATA_FOLDER, save_folder", "plot spiral density maps across radius in task"),
    ]),
]

SAVE_XLSX = 'save_folder = FIGURE_FOLDER / "{name}"\nsave_folder.mkdir(parents=True, exist_ok=True)'
SESSION_ROWS = "session_rows = list(range(6))  # MATLAB 1:6 (plot 6/15 sessions)"

TP = "spirals_py.task.preprocessing"


def pipeline_cell(func, comment=None, extra="", args=None):
    """One preprocessing call per cell, mirroring pipeline6_task.m."""
    args = args or "DATA_FOLDER, save_folder"
    src = f"from {TP}.{func} import {func}\n\n"
    if extra:
        src += extra + "\n"
    if comment:
        src += f"# {comment}\n"
    src += f"{func}({args})"
    return src


SPIRALS_SAVE = 'save_folder = DATA_FOLDER / "task" / "spirals"'


def make_pipeline6():
    """notebooks/pipeline6_task.ipynb, mirroring pipeline6_task.m."""
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    nbc = nb.cells
    nbc.append(nbf.v4.new_markdown_cell(
        "# pipeline6_task\n\nPython translation of `pipeline6_task.m` from Ye et al. 2023 "
        "(spirals): task preprocessing pipeline. Regenerates every `task/*.mat` file "
        "consumed by `notebooks/figure6_task.ipynb` from the raw data release.\n\n"
        "No figures are produced; progress is printed per mouse/session."))
    nbc.append(nbf.v4.new_code_cell(
        '# pipeline task\nfrom pathlib import Path\n\nDATA_FOLDER = Path(r"D:\\data")'))
    sections = [
        ("get psychometric curve and trial outcome", [
            ("getPsychometricCurve",
             'save_folder = DATA_FOLDER / "task" / "psychometric_curve"',
             "get psychometric curves for all subjects"),
            ("getTaskTrialOutcome",
             'save_folder = DATA_FOLDER / "task" / "task_outcome"',
             "get task trial outcome for each trial"),
        ]),
        ("get mean maps for different trial types", [
            ("getMeanMapSession",
             'save_folder = DATA_FOLDER / "task" / "task_mean_maps"',
             "get mean map for all contrast and trial types across sessions"),
            ("getMeanMapsAll",
             'save_folder = DATA_FOLDER / "task" / "task_mean_maps"',
             "average mean maps at [-2,2]s around stim onset for 3 trial types"),
        ]),
        ("get spirals for different trial types", [
            ("getCorrectSpiralDensity", SPIRALS_SAVE, None),
        ]),
        ("get phase and amplitude around stim onset in VISp", [
            ("getTaskOnsetPhase", SPIRALS_SAVE + "\n\nfreq1 = [0.05, 2]",
             "get phase and amplitude around onset time",
             "DATA_FOLDER, save_folder, freq1"),
            ("getTaskOnsetPhase", SPIRALS_SAVE + "\n\nfreq2 = [2, 8]",
             "get phase and amplitude around onset time",
             "DATA_FOLDER, save_folder, freq2"),
        ]),
        ("spirals during each trial", [
            ("getTaskSpirals", SPIRALS_SAVE,
             "task spirals at [-2,2]s around onset time in each trial"),
            ("getPassiveSpirals", SPIRALS_SAVE,
             "passive spirals at [-2,2]s around onset time in each trial"),
            ("getSprialCountByRadiusTime", SPIRALS_SAVE,
             "sort task spirals by radius and time around stim onset"),
            ("getPassiveSpiralPrePost", SPIRALS_SAVE,
             "concatenate all passive spirals before and after stim onset"),
            ("getCorrectSpiralPrePost", SPIRALS_SAVE,
             "concatenate all task spirals before and after stim onset"),
        ]),
    ]
    for header, entries in sections:
        nbc.append(nbf.v4.new_markdown_cell(f"## {header}"))
        for entry in entries:
            func, extra, comment, *args = entry
            nbc.append(nbf.v4.new_code_cell(
                pipeline_cell(func, comment, extra, args[0] if args else None)))
    nbf.write(nb, NB / "pipeline6_task.ipynb")


def make_fig1():
    nb = build(
        "figure1_spirals",
        "Python translation of `figure1_spirals.m` from Ye et al. 2023 (spirals).\n\n"
        "Covers main Fig. 1 and Extended Data Figs. 1-9 (spiral waves in widefield data).",
        SETUP_XLSX, fig1_sections,
    )
    # insert save_folder + freq cells after each section header
    return nb


def main():
    import sys

    # selective generation, e.g. `python tools/make_notebooks.py pipeline6`,
    # to avoid rewriting the (executed) figure notebooks
    if len(sys.argv) > 1:
        if "pipeline6" in sys.argv[1:]:
            make_pipeline6()
            print("written:", NB / "pipeline6_task.ipynb")
        return

    # figure1
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    nbc = nb.cells
    nbc.append(nbf.v4.new_markdown_cell(
        "# figure1_spirals\n\nPython translation of `figure1_spirals.m` from Ye et al. 2023 "
        "(spirals). Covers main Fig. 1 and Extended Data Figs. 1-9."))
    nbc.append(nbf.v4.new_code_cell(SETUP_XLSX))
    fig_names = ["Fig1", "FigS1", "FigS2", "FigS3", "FigS4", "FigS5", "FigS6", "FigS7", "FigS8", "FigS9"]
    for (header, entries), name in zip(fig1_sections, fig_names):
        nbc.append(nbf.v4.new_markdown_cell(f"## {header}"))
        nbc.append(nbf.v4.new_code_cell(SAVE_XLSX.format(name=name)))
        if name == "FigS3":
            nbc.append(nbf.v4.new_code_cell(fig1_s3_extra))
            for func, args, comment in [
                ("plotMapDataVsFftn", "DATA_FOLDER, save_folder, freq",
                 "plot density map (combine all sessions) for data and 3d-fft"),
                ("plotScatterDataVsFftn", "T, DATA_FOLDER, save_folder, freq",
                 "plot peak density across sessions for data and 3d-fft"),
            ]:
                src = f"from spirals_py.spirals.plots.{func} import {func}\n\n# {comment}\n{func}({args})\nplt.show()"
                nbc.append(nbf.v4.new_code_cell(src))
            continue
        if name == "FigS5":
            nbc.append(nbf.v4.new_code_cell(SESSION_ROWS))
        for e in entries:
            if e[0] == "md":
                nbc.append(nbf.v4.new_markdown_cell(e[1]))
                continue
            _, imp, func, args, comment = e
            src = f"from {imp} import {func}\n\n# {comment}\n{func}({args})\nplt.show()"
            nbc.append(nbf.v4.new_code_cell(src))
    nbf.write(nb, NB / "figure1_spirals.ipynb")

    # figure3
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    nbc = nb.cells
    nbc.append(nbf.v4.new_markdown_cell(
        "# figure3_sprials_mirror\n\nPython translation of `figure3_sprials_mirror.m` from Ye et al. 2023 "
        "(spirals). Main Fig. 3 (mirror-symmetric spiral decomposition) and Extended Data Fig. 11."))
    nbc.append(nbf.v4.new_code_cell(SETUP_XLSX))
    for (header, entries), name in zip(fig3_sections, ["Fig3", "FigS11"]):
        nbc.append(nbf.v4.new_markdown_cell(f"## {header}"))
        nbc.append(nbf.v4.new_code_cell(SAVE_XLSX.format(name=name)))
        for _, imp, func, args, comment in entries:
            src = f"from {imp} import {func}\n\n# {comment}\n{func}({args})\nplt.show()"
            nbc.append(nbf.v4.new_code_cell(src))
    nbf.write(nb, NB / "figure3_sprials_mirror.ipynb")

    # figure4
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    nbc = nb.cells
    nbc.append(nbf.v4.new_markdown_cell(
        "# figure4_ephys\n\nPython translation of `figure4_ephys.m` from Ye et al. 2023 (spirals). "
        "Main Fig. 4 (spiral prediction from subcortical spiking) and Extended Data Figs. 12-13."))
    nbc.append(nbf.v4.new_code_cell(SETUP_CSV))
    for (header, entries), name in zip(fig4_sections, ["Fig4", "FigS12", "FigS13"]):
        nbc.append(nbf.v4.new_markdown_cell(f"## {header}"))
        nbc.append(nbf.v4.new_code_cell(SAVE_XLSX.format(name=name)))
        for _, imp, func, args, comment in entries:
            src = f"from {imp} import {func}\n\n# {comment}\n{func}({args})\nplt.show()"
            nbc.append(nbf.v4.new_code_cell(src))
        if name == "FigS13":
            nbc.append(nbf.v4.new_markdown_cell(
                "### Arousal-stratified spiral matching\n\n"
                "Requires revision2 motion-energy data (not part of the figshare Part1-4 download)."))
            nbc.append(nbf.v4.new_code_cell(fig4_arousal))
    nbf.write(nb, NB / "figure4_ephys.ipynb")

    # figure6
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    nbc = nb.cells
    nbc.append(nbf.v4.new_markdown_cell(
        "# figure6_task\n\nPython translation of `figure6_task.m` from Ye et al. 2023 (spirals). "
        "Visual-motor behavior task: main figure (saved under Fig5 folder names, as in the MATLAB "
        "script) and Extended Data Figs. 14-15."))
    nbc.append(nbf.v4.new_code_cell(SETUP_PLAIN))
    for (header, entries), name in zip(fig6_sections, ["Fig5", "FigS14", "FigS15"]):
        nbc.append(nbf.v4.new_markdown_cell(f"## {header}"))
        nbc.append(nbf.v4.new_code_cell(SAVE_XLSX.format(name=name)))
        for _, imp, func, args, comment in entries:
            src = f"from {imp} import {func}\n\n# {comment}\n{func}({args})\nplt.show()"
            nbc.append(nbf.v4.new_code_cell(src))
    nbf.write(nb, NB / "figure6_task.ipynb")

    # pipeline6
    make_pipeline6()

    print("written:", [p.name for p in sorted(NB.glob("figure*.ipynb"))])


if __name__ == "__main__":
    main()
