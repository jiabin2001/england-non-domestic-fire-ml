import json

import pandas as pd

from fireml.config import ROOT


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
        "expanding_window_performance.csv": {"test_year", "pr_auc", "roc_auc"},
        "sensitivity_analysis_results.csv": {"analysis", "test_period", "pr_auc"},
    }
    for name, expected in schemas.items():
        assert expected.issubset(pd.read_csv(ROOT / "outputs/tables" / name).columns), name


def test_temporal_test_was_evaluated_once_after_lock():
    lock = json.loads((ROOT / "outputs/metrics/locked_model_config.json").read_text(encoding="utf-8"))
    assert lock["temporal_test_consulted"] is True
    assert lock["test_evaluation_count_this_run"] == 1
    assert "locked_at_utc" in lock and "test_evaluated_at_utc" in lock


def test_all_eight_figures_have_png_and_pdf():
    png = sorted((ROOT / "outputs/figures").glob("*.png"))
    pdf = sorted((ROOT / "outputs/figures").glob("*.pdf"))
    assert len(png) == 8
    assert len(pdf) == 8

