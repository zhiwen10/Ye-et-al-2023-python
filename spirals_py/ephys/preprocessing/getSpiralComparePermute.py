from spirals_py.ephys.preprocessing.getSpiralComparePredict import _compare_sessions


def getSpiralComparePermute(T, data_folder, save_folder):
    """Translated from ephys/preprocessing/getSpiralComparePermute.m

    Identical to getSpiralComparePredict but reading the permuted
    predictions (ephys/spirals_predict_permute/
    <fname>_spirals_predicted_permute.mat); saves
    spiral_compare_sessions_neighbor_permute.mat with
    spiral_{left,right}_match_all_perm.  See getSpiralComparePredict for
    the matching rule and the traceAmp data-release note.
    """
    _compare_sessions(
        T, data_folder, save_folder,
        predict_subfolder="spirals_predict_permute",
        predicted_suffix="spirals_predicted_permute.mat",
        var_suffix="_perm",
        out_name="spiral_compare_sessions_neighbor_permute.mat",
    )
