from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _datestr(val, fmt):
    ts = pd.Timestamp(val)
    return ts.strftime(fmt)


def _fmt_num(x):
    f = float(x)
    return str(int(f)) if f.is_integer() else str(f)


def loadChanMap(cmIn):
    """Translated from utils/loadChanMap.m

    cmIn is a path to a Kilosort channel map .mat file (or a dict with the
    fields chanMap, xcoords, ycoords, kcoords, connected).
    """
    if isinstance(cmIn, (str, Path)):
        try:
            m = loadmat(cmIn, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            # v7.3 (HDF5) channel map: read datasets directly
            import h5py

            m = {}
            with h5py.File(cmIn, "r") as f:
                for k in f.keys():
                    if not k.startswith("#"):
                        m[k] = np.asarray(f[k]).T.squeeze()
        cmIn = {
            k: np.atleast_1d(np.asarray(v).squeeze())
            for k, v in m.items()
            if not k.startswith("__")
        }

    if "chanMap" not in cmIn or cmIn["chanMap"].size == 0:
        raise ValueError("The provided channel map must contain 'chanMap'")
    chanMap = np.atleast_1d(cmIn["chanMap"].ravel())

    if "xcoords" not in cmIn or cmIn["xcoords"].size == 0 or cmIn["xcoords"].size != chanMap.size:
        xcoords = np.zeros(chanMap.size)
    else:
        xcoords = np.atleast_1d(cmIn["xcoords"].ravel()).astype(float)

    if "ycoords" not in cmIn or cmIn["ycoords"].size == 0 or cmIn["ycoords"].size != chanMap.size:
        ycoords = np.arange(1, chanMap.size + 1, dtype=float)
    else:
        ycoords = np.atleast_1d(cmIn["ycoords"].ravel()).astype(float)

    if "kcoords" not in cmIn or cmIn["kcoords"].size == 0 or cmIn["kcoords"].size != chanMap.size:
        kcoords = np.ones(chanMap.size)
    else:
        kcoords = np.atleast_1d(cmIn["kcoords"].ravel())

    NchanTOT = chanMap.size
    if "connected" in cmIn and cmIn["connected"].size:
        connected = np.atleast_1d(cmIn["connected"].ravel()).astype(bool)
        chanMap = chanMap[connected]
        xcoords = xcoords[connected]
        ycoords = ycoords[connected]
        kcoords = kcoords[connected]
        NchanTOT = int(connected.sum())

    return chanMap, xcoords, ycoords, kcoords, NchanTOT


def _ismember_rows(A, B):
    """Index of each row of A in B (0-based), or -1 if not present."""
    keys_B = {tuple(row): i for i, row in enumerate(np.asarray(B))}
    return np.array([keys_B.get(tuple(row), -1) for row in np.asarray(A)])


def chanMapReorder(Map):
    """Translated from utils/chanMapReorder.m

    Returns 0-based channel indices (MATLAB chanMap1 minus 1), reshaped
    column-major like the original.
    """
    chanMap, xc, yc, kcoords, NchanTOTdefault = loadChanMap(Map)
    Map = str(Map)

    if "NPtype24_hStripe" in Map:
        xc1 = xc + 282
        yc1 = (yc + 15) / 15 + 1
        mapReal = np.column_stack([xc1, yc1])
        map1 = np.column_stack([np.zeros(48), np.arange(1, 49)])
        map2 = []
        for i in range(1, 9):
            mapTemp = map1.copy()
            if i % 2 == 0:
                mapTemp[:, 0] = mapTemp[:, 0] + 32 + 250 * (i / 2 - 1)
            else:
                mapTemp[:, 0] = mapTemp[:, 0] + 250 * (i - 1) / 2
            map2.append(mapTemp)
        map2 = np.vstack(map2)
        b = _ismember_rows(map2, mapReal)
        chanMap1 = b.reshape((48, 8), order="F")
    elif "NPtype24_doubleLengthStripe" in Map:
        yc1 = np.where(yc <= 705, yc / 15 + 1, (yc - 7.5) / 15 + 1)
        xc1 = np.where((xc - 32) % 250 == 0, xc - 32, xc)
        mapReal = np.column_stack([xc1, yc1])
        map1 = np.column_stack([np.zeros(96), np.arange(1, 97)])
        map2 = []
        for i in range(1, 5):
            mapTemp = map1.copy()
            mapTemp[:, 0] = mapTemp[:, 0] + 250 * (i - 1)
            map2.append(mapTemp)
        map2 = np.vstack(map2)
        b = _ismember_rows(map2, mapReal)
        chanMap1 = b.reshape((96, 4), order="F")
    elif "NPtype24_quadrupleLengthStripe" in Map:
        yc1 = np.where(yc <= 1400, yc / 15 + 1, (yc - 15) / 15 + 1)
        xc1 = np.where((xc + 32) % 250 == 0, xc + 32, xc)
        yc1 = yc1 / 2 + 1
        mapReal = np.column_stack([xc1, yc1])
        map1 = np.column_stack([np.zeros(96), np.arange(1, 97)])
        map2 = []
        for i in range(1, 5):
            mapTemp = map1.copy()
            mapTemp[:, 0] = mapTemp[:, 0] + 250 * (i - 2)
            map2.append(mapTemp)
        map2 = np.vstack(map2)
        b = _ismember_rows(map2, mapReal)
        chanMap1 = b.reshape((96, 4), order="F")
    elif "neuropixPhase3B1" in Map:
        yc1 = yc / 20
        yc1 = np.where(yc1 % 2 == 0, yc1 - 1, yc1)
        xc1 = (xc - 11) / 16 + 1
        yc1 = (yc1 + 1) / 2
        mapReal = np.column_stack([xc1, yc1])
        map1 = np.column_stack([np.zeros(96), np.arange(1, 97)])
        map2 = []
        for i in range(1, 5):
            mapTemp = map1.copy()
            mapTemp[:, 0] = mapTemp[:, 0] + i
            map2.append(mapTemp)
        map2 = np.vstack(map2)
        b = _ismember_rows(map2, mapReal)
        chanMap1 = b.reshape((96, 4), order="F")
        chanMap1[47, 1] = 383  # MATLAB chanMap1(48,2) = 384 (1-based)
    elif "NPtype21_botRow0" in Map:
        yc1 = (yc + 30) / 15
        xc1 = (xc + 282) / 32 + 1
        mapReal = np.column_stack([xc1, yc1])
        chanMap1 = np.zeros((192, 2))
        for i in range(384):
            chanMap1[int(mapReal[i, 1]) - 1, int(mapReal[i, 0]) - 1] = i
    else:
        raise ValueError(f"Unrecognized channel map: {Map}")

    return chanMap1


def get_session_info2(T, kk, data_folder):
    """Translated from ephys/utils/get_session_info2.m

    T is a pandas DataFrame of the session table; kk is a 0-based row index.
    Returns a SimpleNamespace mirroring the MATLAB ops struct.
    """
    row = T.iloc[kk]
    ops = SimpleNamespace()
    ops.mn = row["MouseID"]
    ops.tda = row["date"]
    ops.en = row["folder"]
    ops.imec = row["imec"]
    ops.probeName = row["probeName"]
    ops.doubleLength = row["doubleLength"]
    ops.td = _datestr(ops.tda, "%Y-%m-%d")
    ops.tdb = _datestr(ops.tda, "%Y%m%d")
    ops.fname = f"{ops.mn}_{ops.tdb}_{_fmt_num(ops.en)}"

    data_folder = Path(data_folder)
    subfolder = f"{ops.mn}_{ops.tdb}_{_fmt_num(ops.en)}"
    ops.session_root = data_folder / "ephys" / "svd_spikes" / subfolder

    if ops.doubleLength == 1:
        ops.chanMap = data_folder / "ephys" / "config_files" / "NPtype24_doubleLengthStripe_botRow0_ref0.mat"
    elif ops.doubleLength == 0:
        ops.chanMap = data_folder / "ephys" / "config_files" / "NPtype24_quadrupleLengthStripe_botRow0_ref0.mat"
    elif ops.doubleLength == 2:
        ops.chanMap = data_folder / "ephys" / "config_files" / "neuropixPhase3B1_kilosortChanMap.mat"
    elif ops.doubleLength == 3:
        ops.chanMap = data_folder / "ephys" / "config_files" / "NPtype24_hStripe_botRow0_ref1.mat"
    elif ops.doubleLength == 4:
        ops.chanMap = data_folder / "ephys" / "config_files" / "NPtype21_botRow0.mat"

    ops.chanMap1 = chanMapReorder(ops.chanMap)
    ops.curation = row["Curation"]
    return ops


def readClusterGroupsCSV(filename):
    """Translated from ephys/utils/readClusterGroupsCSV.m

    cgs: 0 = noise, 1 = mua, 2 = good, 3 = unsorted.
    """
    filename = Path(filename)
    with open(filename) as fid:
        lines = [ln.split() for ln in fid.read().splitlines() if ln.strip()]
    # files are whitespace- or comma-separated with a header row
    if len(lines[0]) < 2:
        lines = [ln[0].split(",") for ln in lines]
    col1 = [ln[0] for ln in lines[1:]]
    col2 = [ln[1] for ln in lines[1:]]

    cids = np.array([int(float(c)) for c in col1 if c.replace(".", "", 1).isdigit()])
    labels = [l for c, l in zip(col1, col2) if c.replace(".", "", 1).isdigit()]
    cgs = np.zeros(cids.size, dtype=int)
    cgs[np.array([l == "mua" for l in labels])] = 1
    cgs[np.array([l == "good" for l in labels])] = 2
    cgs[np.array([l == "unsorted" for l in labels])] = 3
    return cids, cgs


def loadKSdir2(ksDir, params=None):
    """Translated from ephys/utils/loadKSdir2.m

    params: dict with keys 'excludeNoise' (default True) and 'loadPCs'
    (default False). Returns a SimpleNamespace mirroring spikeStruct.
    """
    if params is None:
        params = {}
    params.setdefault("excludeNoise", True)
    params.setdefault("loadPCs", False)

    ksDir = Path(ksDir)
    ss = np.load(ksDir / "spike_times.npy").ravel()
    st = ss.astype(float) / 30000  # sampling rate: 30k Hz
    spikeTemplates = np.load(ksDir / "spike_templates.npy").ravel()  # zero-indexed
    clu = np.load(ksDir / "spike_clusters.npy").ravel()
    temps = np.load(ksDir / "templates.npy")
    winv = np.load(ksDir / "whitening_mat_inv.npy")

    spikeTimes = np.load(ksDir / "spikeTimes.npy").ravel()
    spikeAmps = np.load(ksDir / "spikeAmps.npy").ravel()
    spikeDepths = np.load(ksDir / "spikeDepths.npy").ravel()
    spikeSites = np.load(ksDir / "spikeSites.npy").ravel()

    cgsFile = None
    for name in ("cluster_groups.csv", "cluster_group.tsv", "cluster_KSLabel.tsv"):
        if (ksDir / name).exists():
            cgsFile = ksDir / name
            break

    cids = None
    cgs = None
    if cgsFile is not None:
        cids, cgs = readClusterGroupsCSV(cgsFile)

        if params["excludeNoise"]:
            noiseClusters = cids[cgs == 0]
            st = st[~np.isin(clu, noiseClusters)]
            spikeTemplates = spikeTemplates[~np.isin(clu, noiseClusters)]
            clu = clu[~np.isin(clu, noiseClusters)]
            cgs = cgs[~np.isin(cids, noiseClusters)]
            cids = cids[~np.isin(cids, noiseClusters)]

    coords = np.load(ksDir / "channel_positions.npy")
    ycoords = coords[:, 1]
    xcoords = coords[:, 0]

    spikeStruct = SimpleNamespace()
    spikeStruct.st = st
    spikeStruct.spikeTemplates = spikeTemplates
    spikeStruct.clu = clu
    spikeStruct.cgs = cgs
    spikeStruct.cids = cids
    spikeStruct.xcoords = xcoords
    spikeStruct.ycoords = ycoords
    spikeStruct.temps = temps
    spikeStruct.winv = winv
    spikeStruct.spikeAmps = spikeAmps
    spikeStruct.spikeDepths = spikeDepths
    spikeStruct.spikeSites = spikeSites
    spikeStruct.spikeTimes = spikeTimes
    spikeStruct.gcluster = cids[cgs == 2] if cids is not None else None
    return spikeStruct


def get_wf2ephysT2(ops, t):
    """Translated from ephys/utils/get_wf2ephysT2.m"""
    syncTL = np.load(Path(ops.session_root) / "tl_sync.npy").ravel()
    syncProbe = np.load(Path(ops.session_root) / f"{ops.probeName}_sync.npy").ravel()
    if syncProbe.size > syncTL.size:
        syncProbe = syncProbe[: syncTL.size]

    # MATLAB interp1 returns NaN outside the syncTL range; np.interp clamps
    WF2ephysT1 = np.interp(t, syncTL, syncProbe)
    WF2ephysT1[(t < syncTL[0]) | (t > syncTL[-1])] = np.nan
    return syncTL, syncProbe, WF2ephysT1


def get_MUA_bin(sp, WF2ephysT):
    """Translated from ephys/utils/get_MUA_bin.m"""
    spikeT = []
    for m in range(len(sp.gcluster)):
        incl = (sp.spikeAmps > 20) & (sp.clu == sp.gcluster[m])
        spikeT.append(sp.spikeTimes[incl].ravel())

    MUA = np.zeros((len(WF2ephysT), len(sp.gcluster)))
    for m in range(len(sp.gcluster)):
        n, _ = np.histogram(spikeT[m], bins=WF2ephysT)
        MUA[:, m] = np.concatenate([[0], n])

    MUA = MUA.T
    MUA_std = (MUA - MUA.mean(axis=1, keepdims=True)) / np.nanstd(MUA, axis=1, keepdims=True)
    MUA_std = MUA_std[MUA_std.std(axis=1) > 0]
    return MUA_std
