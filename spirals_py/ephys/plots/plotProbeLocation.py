"""Translated from ephys/plots/plotProbeLocation.m
(Figure 4f: all probe track locations projected onto 2-D atlas views)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._ephys_helpers import best_fit_line, nrrdread, plotOutline_fill
from ._prediction_example_utils import _get_cortex_atlas_path
from spirals_py.ephys.utils import get_session_info2

# cbrewer2('qual','Set1',9)
_SET1 = np.array(
    [
        [228, 26, 28],
        [55, 126, 184],
        [77, 175, 74],
        [152, 78, 163],
        [255, 127, 0],
        [255, 255, 51],
        [166, 86, 40],
        [247, 129, 191],
        [153, 153, 153],
    ]
) / 255.0


def plotProbeLocation(T, data_folder, save_folder):
    """Original MATLAB file: ephys/plots/plotProbeLocation.m

    Returns h4f.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    mainfolder = data_folder / "ephys" / "probe_location"
    atlas, metaAVGT = nrrdread(data_folder / "tables" / "annotation_50.nrrd")
    _, st = _get_cortex_atlas_path(data_folder)

    area = ["CORTEX", "THAL", "STR", "MB"]
    color_id = [2, 3, 1, 4]  # blue, green, red, purple in Set1 order
    _maskPaths, _st = _get_cortex_atlas_path(data_folder)

    # 3 subareas
    maskPath = [
        "/997/8/343/1129/549/",  # Th
        "/997/8/567/623/477/",  # STR
        "/997/8/343/313/",  # midbrain
        "/997/8/567/688/695/315/",  # isocortex
        "/997/8/567/688/695/698/",  # olfactory
    ]
    root1 = "/997/"

    atlas = np.asarray(atlas, dtype=float)
    atlas1 = [
        np.squeeze(atlas[80, :, :]).T,  # atlas(80,:,:) transposed
        np.squeeze(atlas[:, :, 80]),
        np.squeeze(atlas[:, 120, :]),
    ]

    order_2d = np.array([[3, 3, 1], [1, 2, 2]])  # per-view (x, y) axes
    axis_lim = np.array([264, 264, 264])

    h4f = plt.figure(figsize=(3, 6))
    for iplot in range(3):
        ax = h4f.add_subplot(3, 1, iplot + 1)
        scale3 = 1
        plotOutline_fill(root1, st, atlas1[iplot], None, scale3, "w", ax=ax)
        plotOutline_fill(maskPath[0], st, atlas1[iplot], None, scale3, [0.5, 0.5, 0.5], ax=ax)
        plotOutline_fill(maskPath[1], st, atlas1[iplot], None, scale3, [0.5, 0.5, 0.5], ax=ax)
        plotOutline_fill(maskPath[2], st, atlas1[iplot], None, scale3, [0.5, 0.5, 0.5], ax=ax)
        plotOutline_fill(maskPath[3:5], st, atlas1[iplot], None, scale3, "w", ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.invert_yaxis()  # set(gca, 'YDir','reverse')
        ax.set_xlim(-20 + 20, axis_lim[0] + 20)
        ax.set_ylim(-20 + 20, axis_lim[1] + 20)

        for current_area in [2, 3, 4]:
            indx = T["Area"].str.contains(area[current_area - 1], na=False)
            current_T = T[indx.astype(bool)]
            for kk in range(len(current_T)):
                ops = get_session_info2(current_T, kk, data_folder)
                fname = f"{ops.mn}_{ops.tdb}_{ops.en}"
                probefolder_full = mainfolder / fname
                if not probefolder_full.exists():
                    continue
                for pointcsv in sorted(probefolder_full.glob("*.csv")):
                    pointRaw = np.loadtxt(pointcsv, delimiter=",", ndmin=2)
                    pointRaw = np.floor(pointRaw / 2 + 0.5)  # MATLAB round
                    pointRaw = pointRaw[:, [1, 0, 2]]
                    if current_area == 4 and kk >= 4:  # MATLAB kk >= 5
                        pointRaw[:, 0] = 228 - pointRaw[:, 0]
                    m, p, _s = best_fit_line(
                        pointRaw[:, 0], pointRaw[:, 1], pointRaw[:, 2]
                    )
                    # ensure proper orientation: want 0 at the top of the brain
                    # and positive distance goes down into the brain
                    if p[1] < 0:
                        p = -p
                    # determine "origin" at top of brain: step upwards along
                    # tract direction until tip of brain / past cortex
                    in_brain = True
                    vector_length_old = np.linalg.norm(m - pointRaw[0, :])
                    while in_brain:
                        m = m - p / 5
                        vector_length_new = np.linalg.norm(m - pointRaw[0, :])
                        if vector_length_new > vector_length_old:
                            in_brain = False
                        vector_length_old = vector_length_new
                    tip_index_max = int(np.argmax(pointRaw[:, 1]))
                    tip_index_min = int(np.argmin(pointRaw[:, 1]))
                    reference_probe_length_tip = np.sqrt(
                        np.sum(
                            (pointRaw[tip_index_max, :] - pointRaw[tip_index_min, :]) ** 2
                        )
                    )
                    probe_length_histo = np.floor(reference_probe_length_tip + 0.5)

                    o1 = order_2d[0, iplot] - 1
                    o2 = order_2d[1, iplot] - 1
                    span = np.array([0.0, probe_length_histo])
                    ax.plot(
                        m[o1] + p[o1] * span,
                        m[o2] + p[o2] * span,
                        color=_SET1[color_id[current_area - 1] - 1],
                        linewidth=1,
                    )

    h4f.savefig(save_folder / "Fig4f_probe_projection_2d.pdf", bbox_inches="tight")
    return h4f
