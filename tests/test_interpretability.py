import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from fireml.interpretability import grouped_permutation_importance


class InvertedFeatureModel:
    """Small deterministic frame-based model for permutation unit tests."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = frame["anti_signal"].to_numpy(dtype=float)
        return np.column_stack([1.0 - probability, probability])


def test_grouped_permutation_is_reproducible_complete_and_non_mutating():
    labels = np.array([0] * 20 + [1] * 20)
    frame = pd.DataFrame({
        "anti_signal": np.concatenate([
            np.linspace(0.75, 0.95, 20),
            np.linspace(0.05, 0.25, 20),
        ]),
        "ignored_feature": np.arange(40),
    })
    before = frame.copy(deep=True)
    first = grouped_permutation_importance(
        InvertedFeatureModel(), frame, labels, repeats=20, seed=99
    )
    second = grouped_permutation_importance(
        InvertedFeatureModel(), frame, labels, repeats=20, seed=99
    )
    assert_frame_equal(first, second)
    assert_frame_equal(frame, before)
    assert set(first["feature"]) == set(frame.columns)
    assert first["feature"].is_unique
    anti_signal = first[first["feature"] == "anti_signal"].iloc[0]
    assert anti_signal["mean_pr_auc_decrease"] < 0
    ignored = first[first["feature"] == "ignored_feature"].iloc[0]
    assert ignored["mean_pr_auc_decrease"] == 0
