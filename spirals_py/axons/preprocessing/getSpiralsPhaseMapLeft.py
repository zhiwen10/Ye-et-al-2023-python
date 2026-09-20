"""Translated from revision2/plane_wave/getSpiralsPhaseMapLeft.m

Left-hemisphere variant of getSpiralsPhaseMap.m for sessions 8-15:
spirals with radius >= 70 and direction 0 (the MATLAB source rewrites
column 4 to -1 where it is 0 and keeps those rows; here the original
direction column is compared to 0 directly) inside the ROI 250 <= x <=
350, 500 <= y <= 650.  The phase maps are normalized at pixel (70, 48)
(the MATLAB comment says (70, 95), the code uses (70, 48)) and saved as
spiral_phase_all_norm in <fname>_mean_flow2_left.mat.

Unlike getSpiralsPhaseMap.m, the frameTemp bounds guard
(frameTemp(1) > 0 & frameTemp(end) < size(dV, 2)) is already present
in this MATLAB source.  Skipped dead code: the mimgtransformed warp,
which no downstream code reads.
"""

from pathlib import Path

from spirals_py.axons.preprocessing.getSpiralsPhaseMap import _session_phase_maps
from spirals_py.utils.matio import save_mat73


def getSpiralsPhaseMapLeft(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for kk in range(7, 15):
        fname, spiral_phase_all_norm = _session_phase_maps(
            T, data_folder, kk, radius=70, direction=0,
            roi=(250, 350, 500, 650), ref_pixel=(70, 48),
        )
        save_mat73(
            save_folder / f"{fname}_mean_flow2_left.mat",
            {"spiral_phase_all_norm": spiral_phase_all_norm},
        )
        print(f"getSpiralsPhaseMapLeft: {fname} ({kk + 1}/15)")
