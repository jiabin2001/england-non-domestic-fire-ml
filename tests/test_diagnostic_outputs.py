"""Opt-in checks of published diagnostics; no cohort, ODS, fitting, or GPU needed."""

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve

from fireml.config import ROOT, load_yaml


pytestmark = pytest.mark.artifact

EXPERIMENTS = {
    "b_minus_occupied_random": ("random", "B"),
    "b_minus_occupied_temporal": ("temporal", "B"),
    "building_type_temporal": ("temporal", "B"),
    "arrival_size_temporal": ("temporal", "C"),
}
SPLIT_COMPARISON = "b_minus_occupied_random_minus_temporal"
PAIRED_DIFFERENCE = "paired new-minus-reference average-precision difference"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


@pytest.fixture(scope="module")
def diagnostic_outputs():
    directory = ROOT / "outputs/diagnostics"
    return {
        "directory": directory,
        "receipt": read_json(directory / "experiment_receipt.json"),
        "uncertainty_receipt": read_json(directory / "uncertainty_receipt.json"),
        "performance": pd.read_csv(directory / "performance.csv"),
        "uncertainty": pd.read_csv(directory / "uncertainty.csv"),
        "selection": read_json(ROOT / "outputs/metrics/model_selection.json"),
        "audit": read_json(ROOT / "outputs/metrics/audit_receipt.json"),
        "assignments": pd.read_csv(ROOT / "outputs/tables/split_assignments.csv"),
    }


def test_diagnostic_receipts_account_for_exactly_four_completed_fits_and_hashed_artifacts(diagnostic_outputs):
    saved = diagnostic_outputs
    directory, receipt = saved["directory"], saved["receipt"]
    assert receipt["status"] == "fits_completed"
    assert set(receipt["experiments"]) == set(EXPERIMENTS)
    assert receipt["performance_sha256"] == digest(directory / "performance.csv")
    assert receipt["identity"]["config"] == read_json(directory / "config_snapshot.json")
    assert receipt["identity"]["config"] == load_yaml("config/review_experiments.yaml")
    assert receipt["identity"]["estimator_seed"] == saved["selection"]["seed"]
    assert receipt["identity"]["estimator_seed"] == load_yaml("config/analysis.yaml")["random_seed"]
    assert receipt["training_environment"]["xgboost_device"] == "cuda"
    assert receipt["training_environment"]["n_jobs"] == receipt["identity"]["n_jobs"]

    expected_models, expected_predictions = set(), set()
    for experiment_id, entry in receipt["experiments"].items():
        assert entry["status"] == "completed"
        expected_paths = {
            f"models/{experiment_id}.joblib",
            f"predictions/{experiment_id}_validation.parquet",
            f"predictions/{experiment_id}_test.parquet",
        }
        assert set(entry["artifact_sha256"]) == expected_paths
        for relative, expected in entry["artifact_sha256"].items():
            assert digest(directory / relative) == expected, relative
        expected_models.add(f"{experiment_id}.joblib")
        expected_predictions.update(f"{experiment_id}_{part}.parquet" for part in ("validation", "test"))
        if experiment_id.startswith("b_minus_occupied_"):
            assert entry["actual_model_device"].startswith("cuda")
            assert len(entry["booster_config_sha256"]) == 64
        else:
            assert entry["actual_model_device"].startswith("cpu")
    assert {path.name for path in (directory / "models").glob("*.joblib")} == expected_models
    assert {path.name for path in (directory / "predictions").glob("*.parquet")} == expected_predictions

    performance = saved["performance"]
    assert len(performance) == 8
    assert not performance.duplicated(["experiment_id", "split_role"]).any()
    assert set(performance.experiment_id) == set(EXPERIMENTS)
    assert set(performance.split_role) == {"validation", "test"}
    log = [json.loads(line) for line in (directory / "fit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    for event in ("fit_started", "validation_threshold_frozen", "fit_completed"):
        observed = [entry["experiment_id"] for entry in log if entry["event"] == event]
        assert sorted(observed) == sorted(EXPERIMENTS), event


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
def test_diagnostic_protocol_matches_core_settings_and_requested_information(experiment_id, diagnostic_outputs):
    saved = diagnostic_outputs
    receipt, selection = saved["receipt"], saved["selection"]
    plan = receipt["identity"]["experiment_plan"]
    assert len(plan) == len(EXPERIMENTS)
    assert {entry["experiment_id"] for entry in plan} == set(EXPERIMENTS)
    experiment = next(entry for entry in plan if entry["experiment_id"] == experiment_id)
    design, block = EXPERIMENTS[experiment_id]
    assert experiment["design"] == design
    assert experiment["reference_block"] == block
    assert experiment["reference_key"] == f"{design}_{block}"
    if experiment_id.startswith("b_minus_occupied_"):
        full_columns = saved["audit"]["feature_blocks"]["B"]
        assert "OCCUPIED_TIME" in full_columns
        assert experiment["columns"] == [column for column in full_columns if column != "OCCUPIED_TIME"]
        assert experiment["model"] == selection["selected_family_by_block"][design]["B"]
        assert experiment["parameters"] == selection["selected_hyperparameters"][design][experiment["model"]]
    else:
        expected_column = "BUILDING_TYPE" if experiment_id == "building_type_temporal" else "FIRE_SIZE_ON_ARRIVAL"
        assert experiment["columns"] == [expected_column]
        assert experiment["model"] == "logistic_regression"
        assert experiment["parameters"] == {"C": 1.0}
    reference = receipt["reference_predictions"][experiment["reference_key"]]
    assert reference["path"] == f"outputs/metrics/predictions_{design}_block_{block}.parquet"
    assert reference["sha256"] == digest(ROOT / reference["path"])
    for _, row in saved["performance"].loc[saved["performance"].experiment_id.eq(experiment_id)].iterrows():
        assert row["design"] == design
        assert row["model"] == experiment["model"]
        assert int(row["feature_count"]) == len(experiment["columns"])
        assert json.loads(row["parameters"]) == experiment["parameters"]


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_diagnostic_predictions_match_saved_assignments_and_recompute_metrics(experiment_id, partition, diagnostic_outputs):
    saved = diagnostic_outputs
    design, reference_block = EXPERIMENTS[experiment_id]
    prediction = pd.read_parquet(saved["directory"] / f"predictions/{experiment_id}_{partition}.parquet")
    assignments = saved["assignments"]
    expected = assignments.loc[assignments.design.eq(design) & assignments.partition.eq(partition)]
    assert prediction.SOURCE_ROW_ID.is_unique and prediction.cohort_index.is_unique
    for actual_column, expected_column in (
        ("SOURCE_ROW_ID", "source_row_id"), ("cohort_index", "cohort_index"),
        ("FINANCIAL_YEAR", "financial_year"), ("LARGER_FIRE", "larger_fire"),
    ):
        np.testing.assert_array_equal(prediction[actual_column], expected[expected_column])
    assert set(prediction.LARGER_FIRE) == {0, 1}
    assert np.isfinite(prediction.probability).all() and prediction.probability.between(0, 1).all()
    training_ids = assignments.loc[assignments.design.eq(design) & assignments.partition.eq("train"), "source_row_id"]
    assert set(prediction.SOURCE_ROW_ID).isdisjoint(training_ids)
    metric_rows = saved["performance"].loc[
        saved["performance"].experiment_id.eq(experiment_id) & saved["performance"].split_role.eq(partition)
    ]
    assert len(metric_rows) == 1
    row = metric_rows.iloc[0]
    assert int(row["n"]) == len(prediction)
    assert int(row["positive_count"]) == int(prediction.LARGER_FIRE.sum())
    assert row["positive_prevalence"] == pytest.approx(prediction.LARGER_FIRE.mean(), rel=0, abs=1e-12)
    assert row["pr_auc"] == pytest.approx(
        average_precision_score(prediction.LARGER_FIRE, prediction.probability), rel=0, abs=1e-12
    )
    threshold = saved["receipt"]["experiments"][experiment_id]["threshold"]
    assert row["threshold"] == pytest.approx(threshold, rel=0, abs=1e-12)
    np.testing.assert_array_equal(prediction.prediction, (prediction.probability >= threshold).astype(int))
    if partition == "validation":
        precision, recall, thresholds = precision_recall_curve(prediction.LARGER_FIRE, prediction.probability)
        denominator = precision[:-1] + recall[:-1]
        f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
        assert threshold == pytest.approx(float(thresholds[np.argmax(f1)]), rel=0, abs=1e-12)
    else:
        core = pd.read_parquet(ROOT / f"outputs/metrics/predictions_{design}_block_{reference_block}.parquet")
        assert core.SOURCE_ROW_ID.is_unique and set(core.SOURCE_ROW_ID) == set(prediction.SOURCE_ROW_ID)
        aligned = core.set_index("SOURCE_ROW_ID").loc[prediction.SOURCE_ROW_ID]
        for column in ("LARGER_FIRE", "FINANCIAL_YEAR"):
            np.testing.assert_array_equal(prediction[column], aligned[column])


def test_diagnostic_uncertainty_receipt_identifies_its_actual_inputs_and_output(diagnostic_outputs):
    saved = diagnostic_outputs
    directory, receipt = saved["directory"], saved["uncertainty_receipt"]
    cfg = saved["receipt"]["identity"]["config"]
    assert receipt["record_type"] == "post_review_fixed_model_uncertainty"
    assert receipt["bootstrap_repeats"] == cfg["bootstrap_repeats"] == 10000
    assert receipt["bootstrap_seed"] == cfg["bootstrap_seed"] == 20260921
    assert receipt["output_sha256"] == digest(directory / "uncertainty.csv")
    assert receipt["code_sha256"] == digest(ROOT / "src/fireml/review_uncertainty.py")
    expected_inputs = {f"predictions/{name}_test.parquet" for name in EXPERIMENTS}
    expected_inputs.update(entry["path"] for entry in saved["receipt"]["reference_predictions"].values())
    assert set(receipt["input_sha256"]) == expected_inputs
    for relative, expected in receipt["input_sha256"].items():
        base = ROOT if relative.startswith("outputs/") else directory
        assert digest(base / relative) == expected, relative
    table = saved["uncertainty"]
    assert len(table) == 15
    assert set(table.experiment_id) == set(EXPERIMENTS) | {SPLIT_COMPARISON}
    assert table.groupby("experiment_id").size().eq(3).all()
    assert table.bootstrap_repeats.eq(receipt["bootstrap_repeats"]).all()
    assert table.bootstrap_seed.eq(receipt["bootstrap_seed"]).all()
    assert np.isfinite(table[["point_estimate", "ci_lower_95", "ci_upper_95"]]).all().all()
    assert table.ci_lower_95.le(table.ci_upper_95).all()
    ap = table.loc[table.estimand.eq("average precision")]
    differences = table.loc[~table.estimand.eq("average precision")]
    assert ap[["point_estimate", "ci_lower_95", "ci_upper_95"]].ge(0).all().all()
    assert ap[["point_estimate", "ci_lower_95", "ci_upper_95"]].le(1).all().all()
    assert differences[["point_estimate", "ci_lower_95", "ci_upper_95"]].ge(-1).all().all()
    assert differences[["point_estimate", "ci_lower_95", "ci_upper_95"]].le(1).all().all()


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
def test_paired_intervals_use_same_records_and_new_minus_reference_points(experiment_id, diagnostic_outputs):
    saved = diagnostic_outputs
    design, block = EXPERIMENTS[experiment_id]
    prediction = pd.read_parquet(saved["directory"] / f"predictions/{experiment_id}_test.parquet")
    reference = pd.read_parquet(ROOT / f"outputs/metrics/predictions_{design}_block_{block}.parquet")
    observed = float(average_precision_score(prediction.LARGER_FIRE, prediction.probability))
    reference_ap = float(average_precision_score(reference.LARGER_FIRE, reference.probability))
    rows = saved["uncertainty"].loc[saved["uncertainty"].experiment_id.eq(experiment_id)]
    assert rows.design.eq(design).all()
    assert rows.paired_test_n.eq(len(prediction)).all()
    assert rows.positive_count.eq(prediction.LARGER_FIRE.sum()).all()
    np.testing.assert_allclose(rows.positive_prevalence, prediction.LARGER_FIRE.mean(), rtol=0, atol=1e-12)
    assert rows.overlap_identifier.eq("SOURCE_ROW_ID").all()
    assert rows.bootstrap_method.str.startswith("paired class-stratified").all()
    new_row = rows.loc[rows.estimand.eq("average precision") & rows.model.eq(experiment_id)]
    reference_row = rows.loc[rows.estimand.eq("average precision") & rows.model.eq(f"core_{design}_{block}")]
    difference = rows.loc[rows.estimand.eq(PAIRED_DIFFERENCE)]
    assert len(new_row) == len(reference_row) == len(difference) == 1
    assert new_row.iloc[0].point_estimate == pytest.approx(observed, rel=0, abs=1e-12)
    assert reference_row.iloc[0].point_estimate == pytest.approx(reference_ap, rel=0, abs=1e-12)
    assert difference.iloc[0].point_estimate == pytest.approx(observed - reference_ap, rel=0, abs=1e-12)
    assert difference.iloc[0].reference_model == f"core_{design}_{block}"
    # No conclusion about the difference is inferred from overlapping AP intervals;
    # percentile intervals also need not contain the observed point estimate.


def test_partial_pair_intervals_match_overlap_prevalences_and_split_difference(diagnostic_outputs):
    saved = diagnostic_outputs
    random = pd.read_parquet(saved["directory"] / "predictions/b_minus_occupied_random_test.parquet")
    temporal = pd.read_parquet(saved["directory"] / "predictions/b_minus_occupied_temporal_test.parquet")
    shared = pd.Index(random.SOURCE_ROW_ID).intersection(temporal.SOURCE_ROW_ID)
    np.testing.assert_array_equal(
        random.set_index("SOURCE_ROW_ID").loc[shared, "LARGER_FIRE"],
        temporal.set_index("SOURCE_ROW_ID").loc[shared, "LARGER_FIRE"],
    )
    rows = saved["uncertainty"].loc[saved["uncertainty"].experiment_id.eq(SPLIT_COMPARISON)]
    assert rows.overlap_n.eq(len(shared)).all()
    assert rows.random_test_n.eq(len(random)).all()
    assert rows.temporal_test_n.eq(len(temporal)).all()
    assert rows.random_positive_count.eq(random.LARGER_FIRE.sum()).all()
    assert rows.temporal_positive_count.eq(temporal.LARGER_FIRE.sum()).all()
    assert rows.bootstrap_method.str.contains("shared records resampled jointly").all()
    for design, frame in (("random", random), ("temporal", temporal)):
        np.testing.assert_allclose(rows[f"{design}_positive_prevalence"], frame.LARGER_FIRE.mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(rows[f"overlap_fraction_{design}_test"], len(shared) / len(frame), rtol=0, atol=1e-12)
        ap_row = rows.loc[rows.design.eq(design)]
        assert len(ap_row) == 1
        assert ap_row.iloc[0].point_estimate == pytest.approx(
            average_precision_score(frame.LARGER_FIRE, frame.probability), rel=0, abs=1e-12
        )
    difference = rows.loc[rows.design.eq("random-minus-temporal")]
    assert len(difference) == 1
    expected = average_precision_score(random.LARGER_FIRE, random.probability) - average_precision_score(temporal.LARGER_FIRE, temporal.probability)
    assert difference.iloc[0].point_estimate == pytest.approx(expected, rel=0, abs=1e-12)


def test_complete_execution_contains_diagnostics_and_consistent_computation_provenance(diagnostic_outputs):
    saved = diagnostic_outputs
    execution = read_json(ROOT / "outputs/metrics/execution.json")
    expected = [
        "source import/cache", "audit", "cohort", "data checksums",
        "core model selection and fitting", "core uncertainty", "grouped permutation",
        "temporal and split robustness", "diagnostic fits", "diagnostic uncertainty",
        "reports and figures",
    ]
    assert execution["status"] == "complete"
    assert [stage["name"] for stage in execution["stages"]] == expected
    assert all(stage["status"] == "complete" for stage in execution["stages"])
    previous_finish = datetime.fromisoformat(execution["started_at_utc"])
    for stage in execution["stages"]:
        started = datetime.fromisoformat(stage["started_at_utc"])
        finished = datetime.fromisoformat(stage["finished_at_utc"])
        assert previous_finish <= started <= finished
        previous_finish = finished
    assert previous_finish <= datetime.fromisoformat(execution["finished_at_utc"])
    by_name = {stage["name"]: stage for stage in execution["stages"]}
    for stage_name, completed_at in (
        ("diagnostic fits", saved["receipt"]["completed_at_utc"]),
        ("diagnostic uncertainty", saved["uncertainty_receipt"]["completed_at_utc"]),
    ):
        stage = by_name[stage_name]
        assert datetime.fromisoformat(stage["started_at_utc"]) <= datetime.fromisoformat(completed_at) <= datetime.fromisoformat(stage["finished_at_utc"])
    for relative, expected_hash in saved["receipt"]["identity"]["code_sha256"].items():
        assert execution["code_and_config_sha256"][relative] == expected_hash, relative
