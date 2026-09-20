"""Reporting regressions using synthetic artifacts only."""

import hashlib
import json
from pathlib import Path
import runpy

import pandas as pd
import pytest

from fireml import reporting


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.01, 0.02, "lay entirely above zero"),
        (-0.03, -0.01, "lay entirely below zero"),
        (-0.01, 0.02, "included zero"),
        (0.0, 0.02, "included zero"),
    ],
)
def test_interval_description_handles_both_directions(lower, upper, expected):
    assert reporting._interval_relation(lower, upper) == expected


def test_ranking_separates_test_winner_from_validation_selection():
    models = pd.DataFrame([
        {"model": "xgboost", "pr_auc": 0.50},
        {"model": "logistic_regression", "pr_auc": 0.70},
        {"model": "random_forest", "pr_auc": 0.60},
    ])
    text = reporting._model_ranking_text(models, "random_forest")
    assert "Logistic Regression 0.700; Random Forest 0.600; XGBoost 0.500" in text
    assert "validation-selected family was Random Forest" in text
    assert "not used to reselect the model" in text


def test_annual_note_uses_saved_selection_window():
    expanding = pd.DataFrame({"test_year": ["2018/19", "2019/20", "2020/21", "2021/22"]})
    text = reporting._annual_evaluation_note(
        expanding, {"temporal_validation_years": ["2018/19", "2019/20"]}
    )
    assert "2018/19–2019/20 folds are development-period descriptive" in text
    assert "2020/21–2021/22 folds occur after that selection window" in text
    assert "not a nested annual model-selection procedure" in text


def test_report_missing_inputs_fail_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    monkeypatch.setattr(
        reporting, "ensure_output_dirs",
        lambda: pytest.fail("Missing saved artifacts must be detected before writes."),
    )
    with pytest.raises(FileNotFoundError, match="requires saved analysis artifacts") as error:
        reporting.build_report()
    assert "predictions_temporal_block_B.parquet" in str(error.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("random_family", "temporal_family", "bootstrap_label", "message"),
    [
        ("xgboost", "random_forest", "validation-selected XGBoost", "same selected Block B family"),
        ("random_forest", "random_forest", "validation-selected XGBoost", "bootstrap model labels disagree"),
    ],
)
def test_mismatched_selected_families_fail_preflight(
    tmp_path, monkeypatch, random_family, temporal_family, bootstrap_label, message
):
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    (tmp_path / "outputs/metrics").mkdir(parents=True)
    (tmp_path / "outputs/tables").mkdir(parents=True)
    (tmp_path / "outputs/metrics/model_selection.json").write_text(json.dumps({
        "selected_family_by_block": {
            "random": {"B": random_family}, "temporal": {"B": temporal_family},
        },
    }))
    pd.DataFrame({"model": [bootstrap_label]}).to_csv(
        tmp_path / "outputs/tables/bootstrap_confidence_intervals.csv", index=False
    )
    with pytest.raises(ValueError, match=message):
        reporting._validate_report_selection()


def test_render_keeps_training_receipts_and_records_render_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    for relative in reporting.REPORT_INPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic placeholder")
    historical = {
        "runtime_environment": b'{"packages": {"pandas": "historical-version"}}\n',
        "data_archive_manifest": b'{"files": [{"sha256": "original-source-checksum"}]}\n',
        "model_selection": b'{"selected_family": "historical-selection"}\n',
    }
    for name, content in historical.items():
        (tmp_path / f"outputs/metrics/{name}.json").write_bytes(content)

    # All analysis readers/plotters are replaced: this test runs no research analysis.
    monkeypatch.setattr(reporting, "ensure_output_dirs", lambda: None)
    monkeypatch.setattr(reporting, "_validate_report_selection", lambda: None)
    for function in (
        "_workflow_figure", "_annual_figure", "_random_temporal_figure",
        "_expanding_figure", "_block_figure", "_confusion_and_calibration",
        "_subgroup_figure", "_bootstrap_figure", "_grouped_permutation_figure",
        "_write_random_stability_summary", "_write_final_report", "_write_methods_receipt",
    ):
        monkeypatch.setattr(reporting, function, lambda *args: None)
    monkeypatch.setattr(
        reporting, "write_data_archive_manifest",
        lambda: pytest.fail("Report rendering must never rescan or overwrite data provenance."),
    )
    monkeypatch.setattr(reporting.importlib.metadata, "version", lambda package: "current-version")
    reporting.build_report()

    context = json.loads((tmp_path / "outputs/metrics/report_environment.json").read_text())
    assert context["record_type"] == "report_render_environment"
    assert context["packages"]["pandas"] == "current-version"
    assert "not training provenance" in context["scope"]
    for name, content in historical.items():
        relative = f"outputs/metrics/{name}.json"
        assert (tmp_path / relative).read_bytes() == content
        assert context["input_receipts"][relative]["sha256"] == hashlib.sha256(content).hexdigest()


def test_report_only_entrypoint_calls_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(reporting, "build_report", lambda: calls.append("render"))
    monkeypatch.setattr("sys.argv", ["07_render_report.py"])
    script = Path(__file__).resolve().parents[1] / "scripts/07_render_report.py"
    runpy.run_path(str(script), run_name="__main__")
    assert calls == ["render"]


def test_report_only_entrypoint_reports_missing_artifacts(monkeypatch, capsys):
    def missing():
        raise FileNotFoundError("Missing saved predictions; restore matching artifacts.")

    monkeypatch.setattr(reporting, "build_report", missing)
    monkeypatch.setattr("sys.argv", ["07_render_report.py"])
    script = Path(__file__).resolve().parents[1] / "scripts/07_render_report.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 1
    assert "Missing saved predictions" in capsys.readouterr().err


def test_final_report_renders_changed_outcomes_from_synthetic_artifacts(tmp_path, monkeypatch):
    """Exercise the real template without touching the saved research artifacts."""
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    monkeypatch.setattr(reporting, "load_yaml", lambda path: {"excluded_main_years": {}})
    (tmp_path / "reports").mkdir()
    (tmp_path / "outputs/tables").mkdir(parents=True)
    years = ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23", "2023/24"]
    families = ["logistic_regression", "random_forest", "xgboost"]
    receipts = {
        "data/raw/source_metadata.json": {"official_page_updated": "2026-07-22", "sha256": "synthetic"},
        "outputs/metrics/audit_receipt.json": {
            "logical_rows": 100, "main_years": years,
            "temporal_train_years": years[:2], "temporal_validation_years": years[2:4],
            "temporal_test_years": years[4:], "feature_blocks": {block: ["field"] for block in "ABC"},
        },
        "outputs/metrics/cohort_receipt.json": {"rows": 90, "positive_count": 30, "positive_prevalence": 1 / 3},
        "outputs/metrics/model_selection.json": {
            "selected_hyperparameters": {
                design: {family: {} for family in families} for design in ("random", "temporal")
            },
            "selected_family_by_block": {
                design: {block: "random_forest" for block in "ABC"}
                for design in ("random", "temporal")
            },
        },
    }
    for relative, receipt in receipts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt))
    performance = pd.DataFrame([
        {
            "design": design, "block": block, "model": family, "n": 10,
            "positive_count": 3, "positive_prevalence": 0.3,
            "pr_auc": ap - (0.1 if design == "random" else 0) - (0.1 if block == "C" else 0),
            "roc_auc": 0.8, "f1": 0.5, "brier_score": 0.2,
        }
        for design in ("random", "temporal") for block in "ABC"
        for family, ap in zip(families, [0.7, 0.6, 0.5])
    ])
    bootstrap = pd.DataFrame([
        {
            "estimand": "average precision" if design != "random-minus-temporal" else "difference",
            "design": design, "point_estimate": point, "ci_lower_95": point - 0.01,
            "ci_upper_95": point + 0.01, "overlap_n": 2,
            "overlap_fraction_random_test": 0.2, "overlap_fraction_temporal_test": 0.2,
        }
        for design, point in [("random", 0.5), ("temporal", 0.6), ("random-minus-temporal", -0.1)]
    ])
    expanding = pd.DataFrame([
        {"test_year": year, "n": 10, "positive_prevalence": prevalence, "pr_auc": ap,
         "pr_auc_absolute_lift": ap - prevalence, "normalized_pr_auc": (ap - prevalence) / (1 - prevalence),
         "roc_auc": 0.8}
        for year, prevalence, ap in zip(years[2:], [0.3, 0.4, 0.3, 0.2], [0.5, 0.4, 0.6, 0.7])
    ])
    tables = {
        f"{design}_validation_performance": performance[performance.design.eq(design)]
        for design in ("random", "temporal")
    }
    tables.update({
        "block_comparison": performance[performance.model.eq("random_forest")],
        "expanding_window_performance": expanding,
        "sensitivity_analysis_results": pd.DataFrame([
            {"analysis": analysis, "test_period": "2022/23–2023/24", "n": 10,
             "positive_prevalence": 0.3, "pr_auc": 0.5, "roc_auc": 0.7, "f1": 0.5}
            for analysis in ["main_temporal_definition", "roofs_roof_spaces_positive"]
        ]),
        "bootstrap_confidence_intervals": bootstrap,
        "pr_auc_prevalence_context": performance.assign(
            pr_auc_absolute_lift=lambda frame: frame.pr_auc - frame.positive_prevalence,
            normalized_pr_auc=lambda frame: (frame.pr_auc - frame.positive_prevalence) / (1 - frame.positive_prevalence),
        ),
        "grouped_permutation_importance": pd.DataFrame([
            {"feature": "field", "mean_pr_auc_decrease": 0.1, "std_pr_auc_decrease": 0.01, "permutation_repeats": 30}
        ]),
        "building_type_subgroup_performance": pd.DataFrame([
            {"building_type": "synthetic", "pr_auc": 0.5, "positive_prevalence": 0.3,
             "recall": 0.4, "positive_count": 3, "n": 10}
        ]),
        "cohort_flow": pd.DataFrame([
            {"stage": stage, "records_excluded": 1}
            for stage in ["Remove exact duplicates", "Exclude late calls", "Apply binary target mapping"]
        ]),
        "year_coverage": pd.DataFrame({"financial_year": years}),
        "split_assignments": pd.DataFrame([
            {"design": design, "partition": partition}
            for design in ("random", "temporal") for partition in ("train", "validation", "test")
        ]),
    })
    monkeypatch.setattr(reporting.pd, "read_csv", lambda path: tables[Path(path).stem].copy())
    stability = reporting._summarize_random_stability(
        pd.DataFrame({"split_seed": [1, 2], "estimator_seed": [1, 1], "pr_auc": [0.49, 0.51]}),
        0.6, [1, 2], 1,
    ).iloc[0]
    reporting._write_final_report(stability)
    report = (tmp_path / "reports/final_analysis_report.md").read_text(encoding="utf-8")
    assert "interval lay entirely below zero" in report
    assert "Logistic Regression 0.700; Random Forest 0.600; XGBoost 0.500" in report
    assert "validation-selected family was Random Forest" in report
    assert "C-minus-B difference of -0.100" in report
    assert "2020/21–2021/22 folds are development-period descriptive" in report
    assert "2022/23–2023/24 folds occur after that selection window" in report
    assert "`OCCUPIED_TIME`" in report
    assert "XGBoost had the highest" not in report
    assert "no_rerun_revision" not in report
    assert "historical" not in report


@pytest.fixture
def diagnostic_results():
    performance, uncertainty, plan = [], [], []
    for experiment_id, design, family, block, point, reference in (
        ("b_minus_occupied_random", "random", "xgboost", "B", 0.62, 0.60),
        ("b_minus_occupied_temporal", "temporal", "xgboost", "B", 0.57, 0.61),
        ("building_type_temporal", "temporal", "logistic_regression", "B", 0.51, 0.61),
        ("arrival_size_temporal", "temporal", "logistic_regression", "C", 0.93, 0.94),
    ):
        performance.append({
            "experiment_id": experiment_id, "split_role": "test", "design": design,
            "model": family, "reference_block": block, "pr_auc": point,
        })
        plan.append({"experiment_id": experiment_id, "columns": ["synthetic"]})
        for model, estimand, value in (
            (experiment_id, "average precision", point),
            (f"core_{design}_{block}", "average precision", reference),
            (experiment_id, "paired new-minus-reference average-precision difference", point - reference),
        ):
            uncertainty.append({
                "experiment_id": experiment_id, "design": design, "model": model,
                "estimand": estimand, "point_estimate": value,
                "ci_lower_95": value - 0.005, "ci_upper_95": value + 0.005,
            })
    uncertainty.append({
        "experiment_id": "b_minus_occupied_random_minus_temporal", "design": "random-minus-temporal",
        "model": "b_minus_occupied", "estimand": "random-minus-temporal average-precision difference",
        "point_estimate": 0.05, "ci_lower_95": 0.03, "ci_upper_95": 0.07,
    })
    return {
        "performance": pd.DataFrame(performance), "uncertainty": pd.DataFrame(uncertainty),
        "receipt": {"status": "fits_completed", "identity": {"input_sha256": {}, "experiment_plan": plan}},
        "uncertainty_receipt": {"bootstrap_repeats": 1000, "bootstrap_seed": 4, "method_notes": "Synthetic method note."},
    }


def test_diagnostics_report_uses_actual_signed_results_and_interpretation(diagnostic_results):
    report = reporting._diagnostics_report_section(diagnostic_results)
    methods = reporting._diagnostics_methods_section(diagnostic_results)
    assert "+0.0200 [+0.0150, +0.0250]" in report
    assert "-0.0400 [-0.0450, -0.0350]" in report
    assert "+0.0500 [+0.0300, +0.0700]" in report
    assert "1,000 bootstrap repeats" in report
    assert "change both information and model family" in report
    assert "no joint difference-of-differences interval" in report
    assert "not the presence or amount of leakage" in report
    assert "seed: 4" in methods
    assert "no_rerun_revision" not in report + methods


def test_diagnostic_figures_plot_the_supplied_intervals(diagnostic_results, monkeypatch):
    figures = {}
    monkeypatch.setattr(reporting, "_save", lambda figure, stem: figures.update({stem: figure}))
    reporting._diagnostic_figures(diagnostic_results)
    assert set(figures) == {"11_occupancy_ablation", "12_simple_baselines"}
    occupancy = figures["11_occupancy_ablation"]
    # Four actual AP points and both signed paired differences are plotted, not placeholders.
    assert [float(line.get_xdata()[0]) for line in occupancy.axes[0].lines] == [0.60, 0.62, 0.61, 0.57]
    assert [float(line.get_xdata()[0]) for line in occupancy.axes[1].lines[:2]] == pytest.approx([0.02, -0.04])
    baseline = figures["12_simple_baselines"]
    assert [float(line.get_xdata()[0]) for line in baseline.axes[0].lines] == [0.61, 0.51, 0.94, 0.93]
    for figure in figures.values():
        reporting.plt.close(figure)


def test_incomplete_diagnostic_directory_fails_before_render(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    (tmp_path / "outputs/diagnostics").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Incomplete diagnostics"):
        reporting._load_diagnostics()


def test_diagnostics_loader_rejects_changed_core_reference(tmp_path, monkeypatch, diagnostic_results):
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    directory = tmp_path / "outputs/diagnostics"
    directory.mkdir(parents=True)
    for name in ("performance", "uncertainty"):
        diagnostic_results[name].to_csv(directory / f"{name}.csv", index=False)
    receipt = diagnostic_results["receipt"]
    receipt["performance_sha256"] = hashlib.sha256((directory / "performance.csv").read_bytes()).hexdigest()
    reference = tmp_path / "outputs/metrics/model_selection.json"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"matching reference")
    receipt["identity"]["input_sha256"] = {
        "outputs/metrics/model_selection.json": hashlib.sha256(reference.read_bytes()).hexdigest(),
    }
    uncertainty_receipt = diagnostic_results["uncertainty_receipt"]
    uncertainty_receipt["output_sha256"] = hashlib.sha256((directory / "uncertainty.csv").read_bytes()).hexdigest()
    (directory / "experiment_receipt.json").write_text(json.dumps(receipt))
    (directory / "uncertainty_receipt.json").write_text(json.dumps(uncertainty_receipt))
    assert reporting._load_diagnostics() is not None
    reference.write_bytes(b"new core run")
    with pytest.raises(ValueError, match="Diagnostics reference input changed"):
        reporting._load_diagnostics()
