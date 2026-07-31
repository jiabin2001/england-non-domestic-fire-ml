import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fireml.config import ROOT, load_yaml
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
        "sensitivity_analysis_results.csv": {
            "analysis", "test_period", "positive_prevalence", "pr_auc",
            "pr_auc_baseline", "pr_auc_absolute_lift", "normalized_pr_auc",
        },
        "bootstrap_confidence_intervals.csv": {
            "estimand", "design", "point_estimate", "ci_lower_95", "ci_upper_95",
            "bootstrap_repeats", "bootstrap_seed", "bootstrap_method",
            "random_test_n", "temporal_test_n", "overlap_n",
            "overlap_fraction_random_test", "overlap_fraction_temporal_test",
            "overlap_identifier",
        },
        "grouped_permutation_importance.csv": {
            "feature", "baseline_pr_auc", "mean_permuted_pr_auc",
            "mean_pr_auc_decrease", "std_pr_auc_decrease",
            "permutation_repeats", "permutation_seed",
        },
        "pr_auc_prevalence_context.csv": {
            "design", "block", "model", "positive_prevalence", "pr_auc",
            "pr_auc_baseline", "pr_auc_absolute_lift", "normalized_pr_auc",
            "roc_auc", "brier_score",
        },
        "cross_block_split_difference.csv": {
            "block", "random_ap", "temporal_ap", "ap_difference",
            "absolute_lift_difference", "normalized_ap_difference",
            "roc_auc_difference",
        },
    }
    for name, expected in schemas.items():
        assert expected.issubset(pd.read_csv(ROOT / "outputs/tables" / name).columns), name


def test_model_selection_record_contains_reproducible_settings():
    cfg = load_yaml("config/analysis.yaml")
    selection_path = ROOT / "outputs/metrics/model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["record_type"] == "model_selection_record"
    assert "average precision" in selection["selection_metric"].lower()
    assert set(selection["selected_family_by_block"]) == {"random", "temporal"}
    assert set(selection["validation_thresholds"]) == {"random", "temporal"}
    assert selection["seed"] == cfg["random_seed"]
    assert selection["xgboost_device"] == cfg["xgboost_device"]
    assert not (ROOT / "outputs/metrics/pre_test_model_config.json").exists()
    assert not (ROOT / "outputs/metrics/post_test_evaluation_receipt.json").exists()
    assert not (ROOT / "outputs/metrics/locked_model_config.json").exists()


def test_all_figures_have_matching_png_and_pdf_files():
    expected = {
        "01_study_workflow", "02_annual_count_prevalence",
        "03_random_vs_temporal_pr_auc", "04_expanding_window_performance",
        "05_information_block_comparison", "06_best_temporal_confusion_matrix",
        "07_best_temporal_calibration", "08_building_type_subgroups",
        "09_bootstrap_pr_auc_ci", "10_grouped_permutation_importance",
    }
    png = sorted((ROOT / "outputs/figures").glob("*.png"))
    pdf = sorted((ROOT / "outputs/figures").glob("*.pdf"))
    assert {path.stem for path in png} == expected
    assert {path.stem for path in pdf} == expected
    assert all(path.stat().st_size > 0 for path in png + pdf)


def test_new_outputs_are_nonempty_and_manifested():
    required = {
        "outputs/tables/bootstrap_confidence_intervals.csv",
        "outputs/tables/grouped_permutation_importance.csv",
        "outputs/tables/pr_auc_prevalence_context.csv",
        "outputs/tables/cross_block_split_difference.csv",
        "outputs/figures/09_bootstrap_pr_auc_ci.png",
        "outputs/figures/09_bootstrap_pr_auc_ci.pdf",
        "outputs/figures/10_grouped_permutation_importance.png",
        "outputs/figures/10_grouped_permutation_importance.pdf",
        "outputs/metrics/model_selection.json",
        "outputs/metrics/data_archive_manifest.json",
    }
    manifest = set(json.loads(
        (ROOT / "outputs/metrics/output_manifest.json").read_text(encoding="utf-8")
    ))
    assert required.issubset(manifest)
    for relative in required:
        assert (ROOT / relative).stat().st_size > 0


def test_core_average_precision_points_are_internally_consistent():
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    for design in ("random", "temporal"):
        performance = pd.read_csv(ROOT / f"outputs/tables/{design}_validation_performance.csv")
        observed = performance[(performance.block == "B") & (performance.model == "xgboost")].iloc[0].pr_auc
        predictions = pd.read_parquet(ROOT / f"outputs/metrics/predictions_{design}_block_B.parquet")
        calculated = average_precision_score(predictions["LARGER_FIRE"], predictions["probability"])
        ci_point = bootstrap[
            (bootstrap.estimand == "average precision") & (bootstrap.design == design)
        ].iloc[0].point_estimate
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
    assert "partially paired" in difference.bootstrap_method
    assert "shared records resampled jointly" in difference.bootstrap_method


def test_bootstrap_run_settings_and_method():
    cfg = load_yaml("config/analysis.yaml")
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    assert (bootstrap["bootstrap_repeats"] == cfg["bootstrap_repeats"]).all()
    assert (bootstrap["bootstrap_seed"] == cfg["bootstrap_seed"]).all()
    assert bootstrap["bootstrap_method"].str.contains("partially paired").all()


def test_random_stability_varies_split_seed_only():
    cfg = load_yaml("config/analysis.yaml")
    stability = pd.read_csv(ROOT / "outputs/tables/random_seed_stability.csv")
    assert set(stability["split_seed"]) == set(cfg["random_stability_seeds"])
    assert stability["estimator_seed"].nunique() == 1
    assert int(stability["estimator_seed"].iloc[0]) == cfg["random_seed"]


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
