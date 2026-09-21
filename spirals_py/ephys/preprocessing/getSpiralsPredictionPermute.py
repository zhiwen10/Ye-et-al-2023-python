from pathlib import Path

from tqdm import tqdm

from spirals_py.ephys.preprocessing.getSpiralsRaw import _spiral_detection_session
from spirals_py.ephys.utils import get_session_info2
from spirals_py.utils.matio import load_mat_var
from spirals_py.utils.paths import out_root, release_twin


def getSpiralsPredictionPermute(T, data_folder, save_folder):
    """Reconstructed counterpart of ephys/preprocessing/
    getSpiralsPrediction.m for the permuted predictions (no standalone
    MATLAB file exists in pipeline4; reconstructed from
    revision2/spiral_prediction/getSpiralDetectionPredict.m + the
    pipeline4 flow, as consumed by getSpiralComparePermute.m).

    Same as getSpiralsPrediction but the movie is the permuted-unit
    prediction loaded from ephys/dv_permute/<fname>_dv_predict.mat; the
    detected pwAll is saved as
    ephys/spirals_predict_permute/<fname>_spirals_predicted_permute.mat.
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for kk in tqdm(range(len(T)), desc="getSpiralsPredictionPermute"):
        ops = get_session_info2(T, kk, data_folder)
        fname = ops.fname
        dV_predict = load_mat_var(
            release_twin(
                out_root() / "ephys" / "dv_permute" / f"{fname}_dv_predict.mat",
                data_folder,
            ),
            "dV_predict",
        )
        _spiral_detection_session(
            ops, data_folder, dV_predict, save_folder,
            f"{fname}_spirals_predicted_permute.mat",
        )
        print(f"getSpiralsPredictionPermute: {fname} ({kk + 1}/{len(T)})")
