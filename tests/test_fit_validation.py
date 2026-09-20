"""Fit validation without fitting any real estimator or requiring a GPU."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from fireml import modelling, robustness


class FakeBooster:
    def __init__(self, device):
        self.device = device

    def save_config(self):
        return json.dumps({"learner": {"generic_param": {"device": self.device}}})


class FakeEstimator:
    def __init__(self, device):
        self.device = device

    def get_booster(self):
        return FakeBooster(self.device)


class FakePipeline:
    def __init__(self, device="cuda:0", convergence_warning=False):
        self.named_steps = {"model": FakeEstimator(device)}
        self.convergence_warning = convergence_warning
        self.fit_indices = None
        self.prediction_calls = 0

    def fit(self, features, target):
        self.fit_indices = features.index.tolist()
        if self.convergence_warning:
            warnings.warn("Synthetic convergence failure", ConvergenceWarning)
        return self

    def predict_proba(self, features):
        self.prediction_calls += 1
        probability = np.array([0.2, 0.8])
        return np.column_stack([1 - probability, probability])


@pytest.fixture
def frame_and_split():
    return (
        pd.DataFrame({"field": ["a", "b"] * 3, "LARGER_FIRE": [0, 1] * 3}),
        {"train": np.array([0, 1]), "validation": np.array([2, 3]), "test": np.array([4, 5])},
    )


@pytest.mark.parametrize("caller", [modelling._fit_validation, robustness._fit_validation_then_test])
@pytest.mark.parametrize("failure", ["cpu_fallback", "convergence"])
def test_invalid_fits_fail_before_validation_or_test_predictions(monkeypatch, frame_and_split, caller, failure):
    frame, split = frame_and_split
    pipeline = FakePipeline(device="cpu", convergence_warning=failure == "convergence")
    module = modelling if caller is modelling._fit_validation else robustness
    monkeypatch.setattr(module, "make_model_pipeline", lambda *args: pipeline)
    expected = ConvergenceWarning if failure == "convergence" else RuntimeError
    family = "logistic_regression" if failure == "convergence" else "xgboost"
    with pytest.raises(expected):
        caller(frame, split, ["field"], family, {}, 1, 1, "cuda")
    assert pipeline.fit_indices == [0, 1]
    assert pipeline.prediction_calls == 0


@pytest.mark.parametrize("requested,actual", [("cuda", "cuda:0"), ("cuda:1", "cuda:1"), ("cpu", "cpu")])
def test_valid_device_allows_validation_prediction(monkeypatch, frame_and_split, requested, actual):
    frame, split = frame_and_split
    pipeline = FakePipeline(device=actual)
    monkeypatch.setattr(modelling, "make_model_pipeline", lambda *args: pipeline)
    returned, probability, metrics = modelling._fit_validation(
        frame, split, ["field"], "xgboost", {}, 1, 1, requested,
    )
    assert returned is pipeline
    assert probability.tolist() == [0.2, 0.8]
    assert metrics["pr_auc"] == 1.0
    assert pipeline.prediction_calls == 1


def test_warning_filter_is_local_to_fit(frame_and_split):
    frame, _ = frame_and_split
    before = list(warnings.filters)
    with pytest.raises(ConvergenceWarning):
        modelling.fit_pipeline_checked(
            FakePipeline(convergence_warning=True), frame[["field"]], frame.LARGER_FIRE,
            family="logistic_regression", device="cpu",
        )
    assert warnings.filters == before
