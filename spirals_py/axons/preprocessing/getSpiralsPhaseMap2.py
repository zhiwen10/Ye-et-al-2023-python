"""Translated from axons/preprocessing/getSpiralsPhaseMap2.m

Large-spiral variant of getSpiralsPhaseMap.m over the 15 spiral-imaging
sessions: spirals with radius >= 100, direction 1 and 750 <= x <= 850,
600 <= y <= 800; phase maps normalized at pixel (70, 95) and saved as
spiral_phase_all_norm in <fname>_mean_flow_large.mat.

Fixed MATLAB bugs (both documented here, code otherwise unchanged):
- the Windows backslash path literals 'spirals\\svd' and 'spirals\\
  rf_tform' do not resolve on other platforms; forward slashes are
  used, matching the paths every other variant builds;
- like getSpiralsPhaseMap.m there is no frameTemp bounds check; the
  same guard as getSpiralsPhaseMapLeft.m is added so spirals starting
  within the first 35 frames or running past the end of dV are
  skipped instead of erroring.

Skipped dead code of the MATLAB source: the introductory outline
figure (overlayOutlines plot), BW1, mimg and the outline / maskPath
loads.
"""

from pathlib import Path

from tqdm import tqdm

from spirals_py.axons.preprocessing.getSpiralsPhaseMap import _session_phase_maps
from spirals_py.utils.matio import save_mat73


def getSpiralsPhaseMap2(T, data_folder, save_folder):
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for kk in tqdm(range(15), desc="getSpiralsPhaseMap2"):
        fname, spiral_phase_all_norm = _session_phase_maps(
            T, data_folder, kk, radius=100, direction=1,
            roi=(750, 850, 600, 800), ref_pixel=(70, 95),
        )
        save_mat73(
            save_folder / f"{fname}_mean_flow_large.mat",
            {"spiral_phase_all_norm": spiral_phase_all_norm},
        )
        print(f"getSpiralsPhaseMap2: {fname} ({kk + 1}/15)")
