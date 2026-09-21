from pathlib import Path

from tqdm import tqdm

from spirals_py.task.preprocessing._task_io import save_mat
from spirals_py.task.preprocessing._task_utils import TASK_MICE, load_task_sessions
from spirals_py.task.preprocessing.getTrialID import getTrialID
from spirals_py.task.preprocessing.getTrialResult import getTrialResult


def getTaskTrialOutcome(data_folder, save_folder):
    """Translated from task/preprocessing/getTaskTrialOutcome.m

    Task trial outcome table (getTrialResult) and trial ID table
    (getTrialID) for each mouse, saved as
    [mouse]_task_outcome.mat / [mouse]_task_trial_ID.mat.

    The tables are stored as MATLAB v5 structs of column arrays
    (MATLAB tables cannot be written from Python); load_mat_table and
    load_task_table read them back.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for mn in tqdm(TASK_MICE, desc="getTaskTrialOutcome"):
        T1 = load_task_sessions(data_folder, mn)
        T_all = getTrialResult(T1, data_folder)
        save_mat(save_folder / f"{mn}_task_outcome.mat", {"T_all": T_all})
        trial_all = getTrialID(T1, data_folder)
        save_mat(save_folder / f"{mn}_task_trial_ID.mat", {"trial_all": trial_all})
