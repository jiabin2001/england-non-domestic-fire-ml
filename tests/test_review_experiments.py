"""Review protocol checks with synthetic records and fake estimators only."""

import json
import runpy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from fireml import review_experiments as review


@pytest.fixture
def saved_case():
    frame = pd.DataFrame({
        "SOURCE_ROW_ID": np.arange(100, 112),
        "FINANCIAL_YEAR": ["2010/11"] * 4 + ["2011/12"] * 4 + ["2012/13"] * 4,
        "LARGER_FIRE": [0, 1] * 6,
        "BUILDING_TYPE": ["a", "b"] * 6,
        "OCCUPIED_TIME": ["yes"] * 12,
        "ITEM_IGNITED": ["item"] * 12,
        "FIRE_SIZE_ON_ARRIVAL": ["small", "large"] * 6,
    }, index=np.arange(20, 32))
    partitions = {
        "temporal": {"train": [20, 21, 22, 23], "validation": [24, 25, 26, 27], "test": [28, 29, 30, 31]},
        "random": {"train": [22, 28, 21, 29], "validation": [24, 30, 25, 31], "test": [20, 26, 23, 27]},
    }
    rows = []
    for design, split in partitions.items():
        for part, indices in split.items():
            for index in indices:
                row = frame.loc[index]
                rows.append({
                    "design": design, "partition": part, "cohort_index": index,
                    "source_row_id": row.SOURCE_ROW_ID, "financial_year": row.FINANCIAL_YEAR,
                    "larger_fire": row.LARGER_FIRE,
                })
    references = {}
    for key in review.REFERENCE_PREDICTIONS:
        design = key.split("_", 1)[0]
        prediction = frame.loc[partitions[design]["test"]].copy().iloc[::-1]
        prediction["probability"] = np.where(prediction.LARGER_FIRE, 0.8, 0.2)
        references[key] = prediction
    selection = {
        "split_years": {
            "temporal_train": ["2010/11"], "temporal_validation": ["2011/12"],
            "temporal_test": ["2012/13"],
        },
        "xgboost_device": "cuda", "seed": 42,
        "selected_family_by_block": {design: {"B": "xgboost", "C": "xgboost"} for design in partitions},
        "selected_hyperparameters": {
            "random": {"xgboost": {"max_depth": 3}},
            "temporal": {"xgboost": {"max_depth": 6}},
        },
    }
    return frame, pd.DataFrame(rows), references, selection


def test_saved_assignments_preserve_order_and_accept_reordered_reference(saved_case):
    splits = review.validate_saved_partitions(*saved_case)
    assert splits["random"]["train"].tolist() == [22, 28, 21, 29]
    assert splits["temporal"]["test"].tolist() == [28, 29, 30, 31]


@pytest.mark.parametrize("column,replacement", [
    ("source_row_id", 999), ("financial_year", "2099/00"), ("larger_fire", 1),
])
def test_every_assignment_identity_field_is_checked(saved_case, column, replacement):
    frame, assignments, references, selection = saved_case
    # Change a training row, which is not covered by core test predictions.
    assignments.loc[0, column] = replacement
    with pytest.raises(ValueError, match="differs from saved assignments"):
        review.validate_saved_partitions(frame, assignments, references, selection)


def test_duplicate_assignment_and_changed_reference_are_rejected(saved_case):
    frame, assignments, references, selection = saved_case
    duplicated = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="partition the exact cohort"):
        review.validate_saved_partitions(frame, duplicated, references, selection)
    references["temporal_C"].loc[31, "LARGER_FIRE"] = 0
    with pytest.raises(ValueError, match="different LARGER_FIRE"):
        review.validate_saved_partitions(frame, assignments, references, selection)


def test_four_fixed_experiments_remove_only_occupancy_and_do_not_reselect(saved_case):
    selection = saved_case[-1]
    audit = {"feature_blocks": {"B": ["BUILDING_TYPE", "OCCUPIED_TIME", "ITEM_IGNITED"]}}
    cfg = {"baseline_logistic_c": 1.0, "xgboost_device": "cuda"}
    plan = review.experiment_plan(audit, selection, cfg)
    assert tuple(item["experiment_id"] for item in plan) == review.EXPERIMENT_IDS
    assert [item["columns"] for item in plan[:2]] == [["BUILDING_TYPE", "ITEM_IGNITED"]] * 2
    assert plan[0]["parameters"] == {"max_depth": 3}
    assert plan[1]["parameters"] == {"max_depth": 6}
    assert [item["columns"] for item in plan[2:]] == [["BUILDING_TYPE"], ["FIRE_SIZE_ON_ARRIVAL"]]
    assert all(item["parameters"] == {"C": 1.0} for item in plan[2:])
    assert [item["reference_key"] for item in plan] == ["random_B", "temporal_B", "temporal_B", "temporal_C"]


def test_fit_uses_training_only_and_freezes_threshold_before_test(tmp_path, monkeypatch, saved_case):
    frame, _, _, selection = saved_case
    split = {"train": np.array([20, 21, 22, 23]), "validation": np.array([24, 25, 26, 27]), "test": np.array([28, 29, 30, 31])}
    events = []

    class FakePipeline:
        def fit(self, features, target):
            events.append(("fit", features.index.tolist(), features.columns.tolist()))
            assert target.index.tolist() == split["train"].tolist()
            return self

        def predict_proba(self, features):
            events.append(("predict", features.index.tolist()))
            values = np.array([0.2, 0.8, 0.3, 0.7])
            return np.column_stack([1 - values, values])

    def threshold(labels, probability):
        events.append(("threshold", labels.tolist()))
        return 0.6

    monkeypatch.setattr(review, "make_model_pipeline", lambda *args: FakePipeline())
    monkeypatch.setattr(review, "choose_f1_threshold", threshold)
    monkeypatch.setattr(review.joblib, "dump", lambda obj, path, **kwargs: path.write_bytes(b"fake fitted model"))
    (tmp_path / "models").mkdir()
    (tmp_path / "predictions").mkdir()
    experiment = review.experiment_plan(
        {"feature_blocks": {"B": ["BUILDING_TYPE", "OCCUPIED_TIME"]}}, selection,
        {"baseline_logistic_c": 1.0, "xgboost_device": "cuda"},
    )[2]
    result = review._fit_one(frame, split, experiment, output=tmp_path, seed=42, n_jobs=1, device="cuda")
    assert events == [
        ("fit", [20, 21, 22, 23], ["BUILDING_TYPE"]),
        ("predict", [24, 25, 26, 27]), ("threshold", [0, 1, 0, 1]),
        ("predict", [28, 29, 30, 31]),
    ]
    assert len(result["metrics"]) == 2
    assert all(row["threshold"] == 0.6 for row in result["metrics"])
    saved = pd.read_parquet(tmp_path / "predictions/building_type_temporal_test.parquet")
    assert saved.SOURCE_ROW_ID.tolist() == [108, 109, 110, 111]
    review._verify_completed_experiment(tmp_path, result)


@pytest.mark.parametrize("failure", ["convergence", "cpu_fallback"])
def test_invalid_fit_rejected_before_predictions(tmp_path, monkeypatch, saved_case, failure):
    frame = saved_case[0]

    class FakeBooster:
        def save_config(self):
            return json.dumps({"learner": {"generic_param": {"device": "cpu"}}})

    class FakeEstimator:
        def get_booster(self):
            return FakeBooster()

    class FakePipeline:
        named_steps = {"model": FakeEstimator()}

        def fit(self, features, target):
            if failure == "convergence":
                warnings.warn("did not converge", ConvergenceWarning)
            return self

        def predict_proba(self, features):
            pytest.fail("Invalid fit must fail before predicting or persisting a model.")

    monkeypatch.setattr(review, "make_model_pipeline", lambda *args: FakePipeline())
    experiment = {
        "experiment_id": "synthetic", "columns": ["BUILDING_TYPE"],
        "model": "logistic_regression" if failure == "convergence" else "xgboost",
        "parameters": {},
    }
    expected = ConvergenceWarning if failure == "convergence" else RuntimeError
    with pytest.raises(expected):
        review._fit_one(frame, {"train": np.array([20, 21])}, experiment, output=tmp_path, seed=1, n_jobs=1, device="cuda")
    assert not (tmp_path / "models").exists()


def test_completed_fit_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/review_experiments.yaml").write_text("output_dir: outputs/diagnostics\n")
    output = tmp_path / "outputs/diagnostics"
    output.mkdir(parents=True)
    receipt = output / "experiment_receipt.json"
    receipt.write_text('{"status": "fits_completed"}')
    before = receipt.read_bytes()
    monkeypatch.setattr(review, "_prepare_inputs", lambda *args: pytest.fail("Completed output must fail before loading research inputs."))
    with pytest.raises(FileExistsError, match="already complete"):
        review.run_review_fits(resume=True)
    assert receipt.read_bytes() == before


def test_uncertainty_only_cli_never_calls_fit(monkeypatch):
    calls = []
    monkeypatch.setattr(review, "run_review_fits", lambda **kwargs: pytest.fail("Uncertainty-only must not fit."))
    monkeypatch.setattr(review, "run_review_uncertainty", lambda **kwargs: calls.append(kwargs) or Path("synthetic-output"))
    monkeypatch.setattr("sys.argv", ["08_run_review_experiments.py", "--uncertainty-only"])
    script = Path(__file__).resolve().parents[1] / "scripts/08_run_review_experiments.py"
    runpy.run_path(str(script), run_name="__main__")
    assert calls == [{"overwrite": False}]
