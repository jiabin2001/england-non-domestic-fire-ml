import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fireml.config import ROOT
from fireml.features import resolve_blocks


def test_metrics_in_legal_ranges():
    for name in ("random_validation_performance.csv", "temporal_validation_performance.csv"):
        data = pd.read_csv(ROOT / "outputs/tables" / name)
        for column in ("positive_prevalence", "pr_auc", "roc_auc", "recall", "precision", "f1", "balanced_accuracy", "brier_score", "threshold"):
            assert data[column].between(0, 1).all(), column


def test_required_output_table_schemas():
    schemas = {
        "cohort_flow.csv": {"stage", "records_excluded", "records_remaining"},
        "target_mapping.csv": {"raw_category", "main_mapping", "main_action"},
        "feature_policy.csv": {"raw_field", "block_a", "block_b", "block_c", "leakage_risk"},
        "annual_incident_prevalence.csv": {"FINANCIAL_YEAR", "incident_count", "larger_fire_prevalence"},
        "expanding_window_performance.csv": {
            "test_year", "positive_prevalence", "pr_auc", "pr_auc_baseline",
            "pr_auc_absolute_lift", "normalized_pr_auc", "roc_auc",
        },
        "sensitivity_analysis_results.csv": {"analysis", "test_period", "pr_auc"},
        "bootstrap_confidence_intervals.csv": {
            "estimand", "design", "point_estimate", "ci_lower_95", "ci_upper_95",
            "bootstrap_repeats", "bootstrap_seed", "bootstrap_method",
            "random_test_n", "temporal_test_n", "overlap_n",
            "overlap_fraction_random_test", "overlap_fraction_temporal_test",
            "overlap_identifier",
        },
        "grouped_permutation_importance.csv": {
            "feature", "baseline_pr_auc", "mean_permuted_pr_auc",
            "mean_pr_auc_decrease", "std_pr_auc_decrease", "permutation_p02_5",
            "permutation_p97_5", "permutation_repeats", "permutation_seed",
        },
        "pr_auc_prevalence_context.csv": {
            "design", "block", "model", "positive_prevalence", "pr_auc",
            "pr_auc_baseline", "pr_auc_absolute_lift", "normalized_pr_auc",
            "roc_auc", "brier_score",
        },
    }
    for name, expected in schemas.items():
        assert expected.issubset(pd.read_csv(ROOT / "outputs/tables" / name).columns), name


def test_pre_and_post_test_records_have_separate_semantics():
    pre_path = ROOT / "outputs/metrics/pre_test_model_config.json"
    post_path = ROOT / "outputs/metrics/post_test_evaluation_receipt.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    post = json.loads(post_path.read_text(encoding="utf-8"))
    assert pre["record_type"] == "internal_within_run_pre_test_configuration"
    assert post["record_type"] == "internal_within_run_post_test_evaluation_receipt"
    forbidden_pre_fields = {
        "temporal_test_consulted", "test_evaluation_count_this_run",
        "test_evaluated_at_utc", "runtime_environment",
    }
    assert forbidden_pre_fields.isdisjoint(pre)
    assert post["test_evaluation_count_this_run"] == 1
    assert datetime.fromisoformat(pre["generated_at_utc"]) <= datetime.fromisoformat(
        post["test_evaluated_at_utc"]
    )
    assert post["pre_test_config_sha256"] == hashlib.sha256(pre_path.read_bytes()).hexdigest()
    assert not (ROOT / "outputs/metrics/locked_model_config.json").exists()


def test_all_ten_figures_have_png_and_pdf():
    png = sorted((ROOT / "outputs/figures").glob("*.png"))
    pdf = sorted((ROOT / "outputs/figures").glob("*.pdf"))
    assert len(png) == 10
    assert len(pdf) == 10
    assert all(path.stat().st_size > 0 for path in png + pdf)


def test_new_outputs_are_nonempty_and_manifested():
    required = {
        "outputs/tables/bootstrap_confidence_intervals.csv",
        "outputs/tables/grouped_permutation_importance.csv",
        "outputs/tables/pr_auc_prevalence_context.csv",
        "outputs/figures/09_bootstrap_pr_auc_ci.png",
        "outputs/figures/09_bootstrap_pr_auc_ci.pdf",
        "outputs/figures/10_grouped_permutation_importance.png",
        "outputs/figures/10_grouped_permutation_importance.pdf",
        "outputs/metrics/pre_test_model_config.json",
        "outputs/metrics/post_test_evaluation_receipt.json",
        "outputs/metrics/data_archive_manifest.json",
    }
    manifest = set(json.loads(
        (ROOT / "outputs/metrics/output_manifest.json").read_text(encoding="utf-8")
    ))
    assert required.issubset(manifest)
    for relative in required:
        assert (ROOT / relative).stat().st_size > 0


def test_core_pr_auc_points_are_unchanged_and_drive_bootstrap():
    expected = {"random": 0.6577880127085883, "temporal": 0.6419521680039855}
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    for design, expected_value in expected.items():
        performance = pd.read_csv(ROOT / f"outputs/tables/{design}_validation_performance.csv")
        observed = performance[(performance.block == "B") & (performance.model == "xgboost")].iloc[0].pr_auc
        predictions = pd.read_parquet(ROOT / f"outputs/metrics/predictions_{design}_block_B.parquet")
        calculated = average_precision_score(predictions["LARGER_FIRE"], predictions["probability"])
        ci_point = bootstrap[(bootstrap.estimand == "PR-AUC") & (bootstrap.design == design)].iloc[0].point_estimate
        assert np.isclose(observed, expected_value, rtol=0, atol=1e-6)
        assert np.isclose(calculated, expected_value, rtol=0, atol=1e-6)
        assert np.isclose(ci_point, expected_value, rtol=0, atol=1e-6)
        assert np.isclose(observed, calculated, rtol=0, atol=1e-12)
        assert np.isclose(calculated, ci_point, rtol=0, atol=1e-12)


def test_bootstrap_overlap_matches_saved_test_identifiers():
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    difference = bootstrap[bootstrap.design == "random-minus-temporal"].iloc[0]
    random = pd.read_parquet(ROOT / "outputs/metrics/predictions_random_block_B.parquet")
    temporal = pd.read_parquet(ROOT / "outputs/metrics/predictions_temporal_block_B.parquet")
    source_overlap = len(set(random["SOURCE_ROW_ID"]) & set(temporal["SOURCE_ROW_ID"]))
    index_overlap = len(set(random.index) & set(temporal.index))
    assert source_overlap == index_overlap == int(difference.overlap_n)
    assert np.isclose(
        difference.overlap_fraction_random_test, source_overlap / len(random), rtol=0, atol=1e-12
    )
    assert np.isclose(
        difference.overlap_fraction_temporal_test, source_overlap / len(temporal), rtol=0, atol=1e-12
    )
    assert "covariance not modelled" in difference.bootstrap_method


def test_grouped_importance_uses_every_block_b_field_and_saved_baseline(cohort):
    importance = pd.read_csv(ROOT / "outputs/tables/grouped_permutation_importance.csv")
    expected_features = resolve_blocks(cohort.columns)["B"]
    assert importance["feature"].is_unique
    assert set(importance["feature"]) == set(expected_features)
    temporal = pd.read_csv(ROOT / "outputs/tables/temporal_validation_performance.csv")
    core = temporal[(temporal.block == "B") & (temporal.model == "xgboost")].iloc[0]
    assert np.allclose(importance["baseline_pr_auc"], core.pr_auc, rtol=0, atol=1e-12)
    assignments = pd.read_csv(ROOT / "outputs/tables/split_assignments.csv")
    assigned = assignments[
        (assignments.design == "temporal") & (assignments.partition == "test")
    ]["cohort_index"].to_numpy()
    predictions = pd.read_parquet(ROOT / "outputs/metrics/predictions_temporal_block_B.parquet")
    assert np.array_equal(predictions.index.to_numpy(), assigned)


def test_expanding_window_has_valid_prevalence_context():
    expanding = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    assert np.allclose(expanding["pr_auc_baseline"], expanding["positive_prevalence"])
    assert np.allclose(
        expanding["pr_auc_absolute_lift"],
        expanding["pr_auc"] - expanding["positive_prevalence"],
    )
    expected_normalized = expanding["pr_auc_absolute_lift"] / (
        1.0 - expanding["positive_prevalence"]
    )
    assert np.allclose(expanding["normalized_pr_auc"], expected_normalized)
    assert expanding["normalized_pr_auc"].between(0, 1).all()


def test_data_archive_manifest_matches_local_recovery_files():
    path = ROOT / "outputs/metrics/data_archive_manifest.json"
    archive = json.loads(path.read_text(encoding="utf-8"))
    assert archive["external_archive_required"] is True
    assert len(archive["files"]) == 3
    for item in archive["files"]:
        local = ROOT / item["path"]
        assert item["exists"] is True
        assert item["size_bytes"] == local.stat().st_size
        assert item["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()


def test_documentation_does_not_overstate_internal_run_records():
    paths = [
        ROOT / "README.md",
        ROOT / "reports/final_analysis_report.md",
        ROOT / "reports/methods_receipt.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    forbidden = (
        "proof that configuration was locked",
        "proof of preregistration",
        "externally preregistered",
        "external preregistration",
        "mandatory roof-positive",
    )
    assert all(phrase not in text for phrase in forbidden)
    required = (
        "internal procedural safeguard",
        "not an externally timestamped preregistration",
        "not fully independent",
        "not confidence-interval limits",
        "official fire0304-aligned roof-positive definition",
        "fixed descriptive threshold of 0.5",
        "checksums verify retained files but cannot recover",
    )
    assert all(phrase in text for phrase in required)
