from __future__ import annotations

import hashlib
import json
import os
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / "data/interim/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .config import ROOT, ensure_output_dirs, load_yaml
from .evaluation import calibration_points


MODEL_LABELS = {
    "dummy": "Dummy prior",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Rendering consumes saved analysis artifacts; none of these are acquired or fitted here.
REPORT_INPUTS = (
    "data/raw/source_metadata.json",
    "outputs/metrics/audit_receipt.json",
    "outputs/metrics/cohort_receipt.json",
    "outputs/metrics/model_selection.json",
    "outputs/metrics/runtime_environment.json",
    "outputs/metrics/data_archive_manifest.json",
    "outputs/metrics/predictions_temporal_block_B.parquet",
    *(f"outputs/tables/{name}.csv" for name in (
        "annual_incident_prevalence", "random_validation_performance",
        "temporal_validation_performance", "expanding_window_performance",
        "block_comparison", "building_type_subgroup_performance",
        "bootstrap_confidence_intervals", "grouped_permutation_importance",
        "random_seed_stability", "sensitivity_analysis_results",
        "pr_auc_prevalence_context", "cohort_flow", "year_coverage", "split_assignments",
    )),
)


def _require_report_inputs() -> None:
    missing = [relative for relative in REPORT_INPUTS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Report-only rendering requires saved analysis artifacts. Missing:\n- "
            + "\n- ".join(missing)
            + "\nRestore the matching saved artifacts. This command does not run the analysis."
        )


def _validate_report_selection() -> None:
    """Reject saved results that cannot support this same-family comparison report."""
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    families = selection["selected_family_by_block"]
    random_family, temporal_family = families["random"]["B"], families["temporal"]["B"]
    if random_family != temporal_family:
        raise ValueError(
            "Report-only rendering requires the same selected Block B family in both "
            f"designs; saved selection has random={random_family}, temporal={temporal_family}. "
            "The selected-pipeline bootstrap cannot be labelled as a same-family comparison."
        )
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    accepted_labels = {
        temporal_family, MODEL_LABELS[temporal_family],
        f"validation-selected {MODEL_LABELS[temporal_family]}",
    }
    if not bootstrap["model"].isin(accepted_labels).all():
        raise ValueError(
            "Saved bootstrap model labels disagree with the selected Block B family. "
            "Restore matching artifacts; rendering will not relabel uncertainty results."
        )


def _write_report_environment() -> None:
    packages = {}
    for package in (
        "numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "matplotlib", "PyYAML",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    historical_receipts = {}
    for name in ("runtime_environment", "data_archive_manifest", "model_selection"):
        path = ROOT / f"outputs/metrics/{name}.json"
        historical_receipts[str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "existing artifact preserved; historical accuracy not reverified",
        }
    context = {
        "record_type": "report_render_environment",
        "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "scope": (
            "Rendering saved artifacts only. Recomputes confusion/calibration and table "
            "summaries from saved predictions/results; no data acquisition, model fitting, "
            "bootstrap or permutation analysis. These versions are not training provenance."
        ),
        "historical_receipts": historical_receipts,
    }
    (ROOT / "outputs/metrics/report_environment.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )


def _annual_evaluation_note(expanding: pd.DataFrame, audit: dict) -> str:
    validation_years = audit["temporal_validation_years"]
    cutoff = max(validation_years)
    years = expanding["test_year"].astype(str).tolist()
    development = [year for year in years if year <= cutoff]
    later = [year for year in years if year > cutoff]
    note = (
        f"Annual models reuse the family and settings selected using "
        f"{_year_span(validation_years)}. "
    )
    if development:
        note += (
            f"The {_year_span(development)} folds are development-period descriptive results, "
            "not independent temporal validation: model selection had access to outcomes "
            "from their period even though each model fits only earlier years. "
        )
    if later:
        note += (
            f"The {_year_span(later)} folds occur after that selection window and are "
            "later-year evaluations under the fixed selected settings. "
        )
    return note + "These folds are not a nested annual model-selection procedure."


def _interval_relation(lower: float, upper: float) -> str:
    if lower > 0:
        return "lay entirely above zero"
    if upper < 0:
        return "lay entirely below zero"
    return "included zero"


def _model_ranking_text(main_models: pd.DataFrame, selected_family: str) -> str:
    ranked = main_models.sort_values("pr_auc", ascending=False)
    scores = "; ".join(
        f"{MODEL_LABELS[row.model]} {row.pr_auc:.3f}"
        for row in ranked.itertuples(index=False)
    )
    return (
        f"Temporal Block B AP, ordered by observed test score, was: {scores}. "
        f"The validation-selected family was {MODEL_LABELS[selected_family]}. "
        "No pairwise model-difference interval was estimated; this ranking does not "
        "establish a statistically superior family and is not used to reselect the model."
    )


def _year_span(years: list[str] | pd.Series) -> str:
    values = list(years)
    return values[0] if len(values) == 1 else f"{values[0]}–{values[-1]}"


def _excluded_at_stage(flow: pd.DataFrame, stage: str) -> int:
    matches = flow.loc[flow["stage"].eq(stage), "records_excluded"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cohort-flow row for {stage!r}.")
    return int(matches.iloc[0])


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for extension in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        fig.savefig(
            ROOT / f"outputs/figures/{stem}.{extension}",
            dpi=220,
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(fig)


def write_data_archive_manifest() -> dict:
    """Record current data checksums during an explicit analysis run, never rendering."""
    cfg = load_yaml("config/analysis.yaml")
    paths = (
        ("official source ODS", ROOT / cfg["raw_path"]),
        ("one-time imported raw Parquet", ROOT / cfg["parquet_path"]),
        ("main analysis cohort Parquet", ROOT / cfg["cohort_path"]),
    )
    files = []
    for role, path in paths:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append({
                "role": role,
                "path": relative,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            })
        else:
            files.append({
                "role": role,
                "path": relative,
                "exists": False,
                "size_bytes": None,
                "sha256": None,
            })
    manifest = {
        "purpose": "checksums for a separate durable dissertation data deposit",
        "external_archive_required": True,
        "git_policy": "raw and reproducibility Parquet data are intentionally excluded from Git",
        "files": files,
    }
    (ROOT / "outputs/metrics/data_archive_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _workflow_figure() -> None:
    labels = [
        "Official ODS\nimmutable raw file", "One-time import\nand audit", "Main cohort\n2010/11–2023/24",
        "Train / validation\nmodel selection", "Random and temporal\nholdout evaluation",
        "Expanding windows,\nsensitivity, subgroup",
    ]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.4); ax.axis("off")
    for i, label in enumerate(labels):
        x = 0.15 + i * 1.98
        patch = FancyBboxPatch((x, 0.65), 1.6, 1.05, boxstyle="round,pad=0.04", fc="#e8f1f5", ec="#315b6d")
        ax.add_patch(patch); ax.text(x + 0.8, 1.18, label, ha="center", va="center", fontsize=8.5)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + 1.94, 1.18), xytext=(x + 1.62, 1.18), arrowprops={"arrowstyle": "->", "color": "#315b6d"})
    ax.set_title("Study workflow: validation selection and holdout evaluation", fontsize=12)
    _save(fig, "01_study_workflow")


def _annual_figure() -> None:
    annual = pd.read_csv(ROOT / "outputs/tables/annual_incident_prevalence.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(annual))
    ax.bar(x, annual["incident_count"], color="#7ba6b5", label="Incidents")
    ax.set_ylabel("Incident count"); ax.set_xticks(x, annual["FINANCIAL_YEAR"], rotation=45, ha="right")
    second = ax.twinx(); second.plot(x, annual["larger_fire_prevalence"], color="#a43c3c", marker="o", label="Larger-fire prevalence")
    second.set_ylabel("Larger-fire prevalence"); second.set_ylim(0, max(0.4, annual["larger_fire_prevalence"].max() * 1.15))
    ax.set_title("Annual incident count and larger-fire prevalence (main cohort)")
    _save(fig, "02_annual_count_prevalence")


def _random_temporal_figure() -> None:
    random = pd.read_csv(ROOT / "outputs/tables/random_validation_performance.csv")
    temporal = pd.read_csv(ROOT / "outputs/tables/temporal_validation_performance.csv")
    random = random[(random.block == "B") & (random.model != "dummy")]
    temporal = temporal[(temporal.block == "B") & (temporal.model != "dummy")]
    models = list(random.model)
    x = np.arange(len(models)); width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width/2, random.pr_auc, width, label="Random holdout", color="#4c78a8")
    ax.bar(x + width/2, temporal.set_index("model").loc[models, "pr_auc"], width, label="Temporal holdout", color="#f58518")
    ax.set_xticks(x, [MODEL_LABELS[m] for m in models]); ax.set_ylim(0, 0.75); ax.set_ylabel("Average precision")
    ax.set_title("Random versus temporal performance: Block B"); ax.legend(frameon=False)
    _save(fig, "03_random_vs_temporal_pr_auc")


def _expanding_figure() -> None:
    data = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    family = selection["selected_family_by_block"]["temporal"]["B"]
    cutoff = max(audit["temporal_validation_years"])
    labels = [f"{year}*" if year <= cutoff else year for year in data.test_year]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(labels, data.pr_auc, marker="o", color="#4c78a8", label="Average precision")
    axes[0].plot(labels, data.pr_auc_baseline, marker="s", color="#a43c3c", label="Positive prevalence")
    axes[0].set_ylim(0.2, 0.75); axes[0].set_ylabel("Average precision / prevalence")
    axes[0].set_title("Average precision and its prevalence baseline")
    axes[1].plot(labels, data.normalized_pr_auc, marker="o", color="#f58518", label="Normalized AP")
    axes[1].plot(labels, data.roc_auc, marker="s", color="#54a24b", label="ROC-AUC")
    axes[1].set_ylim(0.45, 0.88); axes[1].set_ylabel("Prevalence-context / ROC area")
    axes[1].set_title("Prevalence-context discrimination")
    for ax in axes:
        ax.set_xlabel("Annual evaluation (* development-period descriptive)")
        ax.legend(frameon=False)
    fig.suptitle(f"Expanding-window annual performance: fixed Block B {MODEL_LABELS[family]} settings")
    _save(fig, "04_expanding_window_performance")


def _block_figure() -> None:
    data = pd.read_csv(ROOT / "outputs/tables/block_comparison.csv")
    designs = ["random", "temporal"]; blocks = ["A", "B", "C"]
    x = np.arange(3); width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, design, color in [(-width/2, "random", "#4c78a8"), (width/2, "temporal", "#f58518")]:
        subset = data[data.design == design].set_index("block").loc[blocks]
        ax.bar(x + offset, subset.pr_auc, width, label=design.title(), color=color)
    ax.set_xticks(x, ["A: structural/context", "B: retrospective incident", "C: retrospective\n+ arrival-state"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Average precision"); ax.set_title("Information-block comparison (validation-selected family)")
    ax.legend(frameon=False)
    _save(fig, "05_information_block_comparison")


def _confusion_and_calibration() -> None:
    prediction = pd.read_parquet(ROOT / "outputs/metrics/predictions_temporal_block_B.parquet")
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    family = selection["selected_family_by_block"]["temporal"]["B"]
    threshold = float(selection["validation_thresholds"]["temporal"]["B"][family])
    y = prediction["LARGER_FIRE"].to_numpy(); pred = prediction["prediction"].to_numpy()
    matrix = pd.crosstab(pd.Series(y, name="actual"), pd.Series(pred, name="predicted"), dropna=False).reindex(index=[0,1], columns=[0,1], fill_value=0)
    matrix.to_csv(ROOT / "outputs/tables/best_temporal_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(5, 4.4)); image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="black")
    ax.set_xticks([0,1], ["Predicted 0", "Predicted 1"]); ax.set_yticks([0,1], ["Actual 0", "Actual 1"])
    ax.set_title(f"Temporal Block B confusion matrix\nthreshold={threshold:.3f}")
    fig.colorbar(image, ax=ax, fraction=0.046)
    _save(fig, "06_best_temporal_confusion_matrix")

    predicted, observed = calibration_points(y, prediction["probability"].to_numpy())
    calibration = pd.DataFrame({"mean_predicted_probability": predicted, "observed_fraction_positive": observed})
    calibration.to_csv(ROOT / "outputs/tables/best_temporal_calibration_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.8)); ax.plot([0,1],[0,1], linestyle="--", color="#666666", label="Ideal")
    ax.plot(predicted, observed, marker="o", color="#4c78a8", label=f"Block B {MODEL_LABELS[family]}")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed positive fraction"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Temporal Block B calibration"); ax.legend(frameon=False)
    _save(fig, "07_best_temporal_calibration")


def _subgroup_figure() -> None:
    data = pd.read_csv(ROOT / "outputs/tables/building_type_subgroup_performance.csv").sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(data.building_type, data.pr_auc, color="#72a06a")
    ax.scatter(data.positive_prevalence, np.arange(len(data)), color="#a43c3c", label="Positive prevalence", zorder=3)
    ax.set_xlim(0, 0.85); ax.set_xlabel("Average precision (bars) / positive prevalence (points)")
    ax.set_title("Major building-type subgroup performance (temporal Block B)"); ax.legend(frameon=False, loc="lower right")
    _save(fig, "08_building_type_subgroups")


def _bootstrap_figure() -> None:
    data = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    pr_auc = data[data["estimand"] == "average precision"].set_index("design").loc[["random", "temporal"]]
    difference = data[data["design"] == "random-minus-temporal"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(2)
    points = pr_auc["point_estimate"].to_numpy()
    errors = np.vstack([
        points - pr_auc["ci_lower_95"].to_numpy(),
        pr_auc["ci_upper_95"].to_numpy() - points,
    ])
    axes[0].errorbar(points, y, xerr=errors, fmt="o", color="#315b6d", capsize=4)
    axes[0].set_yticks(y, ["Random", "Temporal"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Average precision")
    axes[0].set_title("Fixed-model AP (95% bootstrap CI)")
    point = float(difference["point_estimate"])
    diff_error = np.array([[
        point - float(difference["ci_lower_95"]),
    ], [
        float(difference["ci_upper_95"]) - point,
    ]])
    axes[1].errorbar(point, 0, xerr=diff_error, fmt="o", color="#a43c3c", capsize=4)
    axes[1].axvline(0, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_yticks([0], ["Random − temporal"])
    axes[1].set_xlabel("Average-precision difference")
    axes[1].set_title("Partially paired holdout comparison")
    _save(fig, "09_bootstrap_pr_auc_ci")


def _grouped_permutation_figure() -> None:
    data = pd.read_csv(ROOT / "outputs/tables/grouped_permutation_importance.csv")
    data = data.sort_values("mean_pr_auc_decrease", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.barh(
        data["feature"],
        data["mean_pr_auc_decrease"],
        xerr=data["std_pr_auc_decrease"],
        color="#4c78a8",
        alpha=0.9,
        error_kw={"elinewidth": 0.8, "capsize": 2},
    )
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel("Mean AP decrease (error bars: permutation SD)")
    ax.set_title("Grouped permutation importance: model dependence, not causal effect")
    _save(fig, "10_grouped_permutation_importance")


def _format_rows(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for record in frame[columns].to_dict("records"):
        cells = []
        for column in columns:
            value = record[column]
            cells.append(f"{value:.{decimals}f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def _human_join(values: list[str]) -> str:
    if len(values) < 2:
        return "".join(values)
    return f"{', '.join(values[:-1])} and {values[-1]}"


def _summarize_random_stability(
    stability: pd.DataFrame,
    temporal_ap: float,
    configured_seeds: list[int],
    estimator_seed: int,
) -> pd.DataFrame:
    expected_seeds = [int(value) for value in configured_seeds]
    observed_seeds = stability["split_seed"].astype(int).tolist()
    if observed_seeds != expected_seeds:
        raise ValueError(
            "random_seed_stability.csv must contain each configured split seed once, "
            "in configuration order."
        )
    if not stability["estimator_seed"].eq(estimator_seed).all():
        raise ValueError("Random split stability rows must use the configured estimator seed.")

    differences = stability["pr_auc"] - temporal_ap
    return pd.DataFrame([{
        "split_count": len(stability),
        "estimator_seed": estimator_seed,
        "temporal_ap_reference": temporal_ap,
        "random_ap_median": stability["pr_auc"].median(),
        "random_ap_q1": stability["pr_auc"].quantile(0.25),
        "random_ap_q3": stability["pr_auc"].quantile(0.75),
        "random_ap_min": stability["pr_auc"].min(),
        "random_ap_max": stability["pr_auc"].max(),
        "ap_difference_median": differences.median(),
        "ap_difference_q1": differences.quantile(0.25),
        "ap_difference_q3": differences.quantile(0.75),
        "ap_difference_min": differences.min(),
        "ap_difference_max": differences.max(),
        "ap_difference_range_width": differences.max() - differences.min(),
        "positive_difference_count": int(differences.gt(0).sum()),
        "positive_difference_fraction": differences.gt(0).mean(),
    }])


def _write_random_stability_summary() -> pd.Series:
    cfg = load_yaml("config/analysis.yaml")
    selection = json.loads(
        (ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8")
    )
    family = selection["selected_family_by_block"]["temporal"]["B"]
    temporal = pd.read_csv(ROOT / "outputs/tables/temporal_validation_performance.csv")
    temporal_ap = temporal[
        temporal["block"].eq("B") & temporal["model"].eq(family)
    ].iloc[0].pr_auc
    stability = pd.read_csv(ROOT / "outputs/tables/random_seed_stability.csv")
    summary = _summarize_random_stability(
        stability,
        float(temporal_ap),
        cfg["random_stability_seeds"],
        int(cfg["random_seed"]),
    )
    summary.to_csv(ROOT / "outputs/tables/random_seed_stability_summary.csv", index=False)
    return summary.iloc[0]


def _write_final_report(stability_summary: pd.Series) -> None:
    cfg = load_yaml("config/analysis.yaml")
    metadata = json.loads((ROOT / "data/raw/source_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    cohort = json.loads((ROOT / "outputs/metrics/cohort_receipt.json").read_text(encoding="utf-8"))
    temporal = pd.read_csv(ROOT / "outputs/tables/temporal_validation_performance.csv")
    random = pd.read_csv(ROOT / "outputs/tables/random_validation_performance.csv")
    block = pd.read_csv(ROOT / "outputs/tables/block_comparison.csv")
    expanding = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    sensitivity = pd.read_csv(ROOT / "outputs/tables/sensitivity_analysis_results.csv")
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    prevalence_context = pd.read_csv(ROOT / "outputs/tables/pr_auc_prevalence_context.csv")
    importance = pd.read_csv(ROOT / "outputs/tables/grouped_permutation_importance.csv")
    subgroup = pd.read_csv(ROOT / "outputs/tables/building_type_subgroup_performance.csv")
    flow = pd.read_csv(ROOT / "outputs/tables/cohort_flow.csv")
    year_coverage = pd.read_csv(ROOT / "outputs/tables/year_coverage.csv")
    assignments = pd.read_csv(ROOT / "outputs/tables/split_assignments.csv")
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    shared_configuration_families = [
        family for family in ("logistic_regression", "random_forest", "xgboost")
        if selection["selected_hyperparameters"]["temporal"][family]
        == selection["selected_hyperparameters"]["random"][family]
    ]
    differing_configuration_families = [
        family for family in ("logistic_regression", "random_forest", "xgboost")
        if family not in shared_configuration_families
    ]
    if shared_configuration_families:
        shared_configuration_sentence = (
            f"{_human_join([MODEL_LABELS[family] for family in shared_configuration_families])} "
            "selected identical hyperparameters under random and temporal development designs."
        )
    else:
        shared_configuration_sentence = (
            "No model family selected identical hyperparameters under the random and temporal "
            "development designs."
        )
    if differing_configuration_families:
        differing_configuration_details = "; ".join(
            f"{MODEL_LABELS[family]}: random "
            f"`{json.dumps(selection['selected_hyperparameters']['random'][family], sort_keys=True)}` "
            f"versus temporal "
            f"`{json.dumps(selection['selected_hyperparameters']['temporal'][family], sort_keys=True)}`"
            for family in differing_configuration_families
        )
        differing_configuration_sentence = (
            f"Selected settings differed for {differing_configuration_details}."
        )
    else:
        differing_configuration_sentence = "No selected settings differed between the two designs."
    main_models = temporal[(temporal.block == "B") & (temporal.model != "dummy")].sort_values("pr_auc", ascending=False)
    core_family = selection["selected_family_by_block"]["temporal"]["B"]
    if (
        selection["selected_hyperparameters"]["temporal"][core_family]
        == selection["selected_hyperparameters"]["random"][core_family]
    ):
        core_configuration_sentence = (
            f"The primary {MODEL_LABELS[core_family]} RQ1 contrast therefore uses the same "
            "selected configuration in both designs."
        )
    else:
        core_configuration_sentence = (
            f"The primary {MODEL_LABELS[core_family]} RQ1 contrast uses different selected "
            "configurations in the two designs and should be interpreted accordingly."
        )
    temporal_core = temporal[(temporal.block == "B") & (temporal.model == core_family)].iloc[0]
    random_core = random[(random.block == "B") & (random.model == core_family)].iloc[0]
    random_ci = bootstrap[(bootstrap.estimand == "average precision") & (bootstrap.design == "random")].iloc[0]
    temporal_ci = bootstrap[(bootstrap.estimand == "average precision") & (bootstrap.design == "temporal")].iloc[0]
    difference_ci = bootstrap[bootstrap.design == "random-minus-temporal"].iloc[0]
    bootstrap_difference_width = float(difference_ci.ci_upper_95 - difference_ci.ci_lower_95)
    stability_difference_width = float(stability_summary.ap_difference_range_width)
    context_core = prevalence_context[prevalence_context.model == core_family].copy()
    importance_top = importance.nlargest(5, "mean_pr_auc_decrease").copy()
    selected_temporal = block[block.design == "temporal"].set_index("block")
    rq3_gain = selected_temporal.loc["C", "pr_auc"] - selected_temporal.loc["B", "pr_auc"]
    expanding_spearman = expanding[["positive_prevalence", "pr_auc"]].corr(method="spearman").iloc[0, 1]
    expanding_lift_range = expanding.pr_auc_absolute_lift.max() - expanding.pr_auc_absolute_lift.min()
    expanding_normalized_range = expanding.normalized_pr_auc.max() - expanding.normalized_pr_auc.min()
    expanding_roc_range = expanding.roc_auc.max() - expanding.roc_auc.min()
    difference_relation = _interval_relation(difference_ci.ci_lower_95, difference_ci.ci_upper_95)
    family_comparison = random[(random.block == "B") & (random.model != "dummy")][
        ["model", "pr_auc", "roc_auc", "brier_score"]
    ].merge(
        temporal[(temporal.block == "B") & (temporal.model != "dummy")][
            ["model", "pr_auc", "roc_auc", "brier_score"]
        ],
        on="model",
        suffixes=("_random", "_temporal"),
    )
    family_comparison["ap_difference"] = (
        family_comparison["pr_auc_random"] - family_comparison["pr_auc_temporal"]
    )
    family_comparison["roc_auc_difference"] = (
        family_comparison["roc_auc_random"] - family_comparison["roc_auc_temporal"]
    )
    family_difference_text = ", ".join(
        f"{MODEL_LABELS[row.model]} {row.ap_difference:+.3f}"
        for row in family_comparison.itertuples(index=False)
    )
    if family_comparison["ap_difference"].gt(0).all():
        family_ap_direction_text = f"All {len(family_comparison)} Block B model families favoured random splitting in AP"
    elif family_comparison["ap_difference"].lt(0).all():
        family_ap_direction_text = f"All {len(family_comparison)} Block B model families favoured temporal splitting in AP"
    else:
        family_ap_direction_text = "The Block B model-family AP differences were mixed in direction"
    primary_split_family_ap_text = (
        family_ap_direction_text[0].lower() + family_ap_direction_text[1:]
    )
    if family_comparison["roc_auc_difference"].gt(0).all():
        family_roc_direction_text = "were also positive"
    elif family_comparison["roc_auc_difference"].lt(0).all():
        family_roc_direction_text = "were all negative"
    else:
        family_roc_direction_text = "were mixed in sign or included zero"
    stability_count = int(stability_summary.split_count)
    stability_positive_count = int(stability_summary.positive_difference_count)
    if stability_positive_count == stability_count:
        stability_direction_text = f"all {stability_count} differences were positive"
        stability_interpretation = (
            "Every saved assignment had higher random-holdout AP for this retrospective "
            "Block B task under the fixed model settings."
        )
    elif stability_positive_count > stability_count / 2:
        stability_direction_text = (
            f"{stability_positive_count} of {stability_count} differences were positive"
        )
        stability_interpretation = (
            "A majority of saved assignments had higher random-holdout AP for this retrospective "
            "Block B task, but the direction was not uniform across assignments."
        )
    else:
        stability_direction_text = (
            f"{stability_positive_count} of {stability_count} differences were positive"
        )
        stability_interpretation = (
            "The repeated assignments do not support a directionally consistent random-split "
            "optimism effect in this retrospective Block B task."
        )
    cross_block_rows = []
    for block_name in ("A", "B", "C"):
        random_row = random[(random.block == block_name) & (random.model == core_family)].iloc[0]
        temporal_row = temporal[(temporal.block == block_name) & (temporal.model == core_family)].iloc[0]
        random_lift = random_row.pr_auc - random_row.positive_prevalence
        temporal_lift = temporal_row.pr_auc - temporal_row.positive_prevalence
        random_normalized = random_lift / (1.0 - random_row.positive_prevalence)
        temporal_normalized = temporal_lift / (1.0 - temporal_row.positive_prevalence)
        cross_block_rows.append({
            "block": block_name,
            "random_ap": random_row.pr_auc,
            "temporal_ap": temporal_row.pr_auc,
            "ap_difference": random_row.pr_auc - temporal_row.pr_auc,
            "absolute_lift_difference": random_lift - temporal_lift,
            "normalized_ap_difference": random_normalized - temporal_normalized,
            "roc_auc_difference": random_row.roc_auc - temporal_row.roc_auc,
        })
    cross_block_table = pd.DataFrame(cross_block_rows)
    cross_block_table.to_csv(
        ROOT / "outputs/tables/cross_block_split_difference.csv", index=False
    )
    cross_block = cross_block_table.set_index("block")
    block_a_lift = (
        selected_temporal.loc["A", "pr_auc"]
        - selected_temporal.loc["A", "positive_prevalence"]
    )
    block_b_lift = (
        selected_temporal.loc["B", "pr_auc"]
        - selected_temporal.loc["B", "positive_prevalence"]
    )
    sensitivity_context = sensitivity.copy()
    sensitivity_context["ap_absolute_lift"] = (
        sensitivity_context["pr_auc"] - sensitivity_context["positive_prevalence"]
    )
    sensitivity_context["normalized_ap"] = sensitivity_context["ap_absolute_lift"] / (
        1.0 - sensitivity_context["positive_prevalence"]
    )
    recent_expanding = expanding[
        expanding["test_year"].isin(audit["temporal_test_years"])
    ]
    recent_weighted_ap = float(np.average(recent_expanding.pr_auc, weights=recent_expanding.n))
    official_update = pd.Timestamp(metadata["official_page_updated"])
    official_update_text = f"{official_update.day} {official_update.strftime('%B %Y')}"
    excluded_year_text = "; ".join(
        f"{year}: {reason.rstrip('.')}"
        for year, reason in cfg["excluded_main_years"].items()
    )
    split_sizes = assignments.groupby(["design", "partition"]).size()
    random_split_sizes = "/".join(
        f"{int(split_sizes.loc[('random', part)]):,}"
        for part in ("train", "validation", "test")
    )
    subgroup_low = subgroup.loc[subgroup.pr_auc.idxmin()]
    subgroup_high = subgroup.loc[subgroup.pr_auc.idxmax()]
    zero_recall_subgroups = subgroup.loc[subgroup["recall"].eq(0)].sort_values(
        ["n", "building_type"], ascending=[False, True]
    )
    if zero_recall_subgroups.empty:
        subgroup_threshold_text = (
            "Every retained building-type subgroup had non-zero recall at the single global "
            "validation-F1 threshold."
        )
    else:
        zero_recall_subgroup = zero_recall_subgroups.iloc[0]
        subgroup_threshold_text = (
            "At the single global validation-F1 threshold, the largest retained subgroup with "
            f"zero recall was {zero_recall_subgroup.building_type}, with "
            f"{int(zero_recall_subgroup.positive_count)} positives among "
            f"{int(zero_recall_subgroup.n):,} incidents."
        )
    annual_evaluation_note = _annual_evaluation_note(expanding, audit)
    model_ranking_text = _model_ranking_text(main_models, core_family)
    report = f"""# Final analysis report

## Study definition

This retrospective prediction study uses the official **Other Building Fires Dataset** for England. “Non-domestic building fires” is the dissertation's analytical wording for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings and can include hotels, hostels, care homes and student halls; it is not limited to buildings without accommodation functions.

The analysis predicts incident-level final fire spread among already-recorded primary fires. It is not a causal study, annual building fire-risk model, fire-physics simulation or real-time FRS deployment tool. Block B contains retrospectively recorded incident information; Block C combines retrospective incident information plus arrival-state information. Its inherited Block B investigation fields have not been established as available at first arrival.

This report is rendered from saved analysis artifacts. Rendering does not rerun training or validate historical results under revised code. See [no-rerun revision notes](no_rerun_revision.md) for the status of the reviewed historical snapshot and the unchanged numerical artifacts.

## Data and cohort

The official ODS was updated {official_update_text}. The raw-file SHA-256 is `{metadata['sha256']}`. The audit found {audit['logical_rows']:,} logical incident rows across {_year_span(year_coverage['financial_year'])}. The defined main period is {_year_span(audit['main_years'])}. Excluded years are {excluded_year_text}.

After year restriction, {_excluded_at_stage(flow, 'Remove exact duplicates'):,} exact duplicates, {_excluded_at_stage(flow, 'Exclude late calls'):,} late calls and {_excluded_at_stage(flow, 'Apply binary target mapping'):,} roof-space or otherwise unmappable target records were removed sequentially. The main cohort contains {cohort['rows']:,} incidents, {cohort['positive_count']:,} larger fires ({cohort['positive_prevalence']:.1%}).

The main target follows the unambiguous room→floor→whole-building ordering and excludes `Roofs/ Roof spaces`, which cannot be placed uniquely on that scale. Official sources are inconsistent: the current Fire statistics definitions omit roofs from the larger-fire list, while FIRE0304-linked detailed releases include them. The roof-positive sensitivity therefore tests the alternative published convention rather than treating either source as uniquely authoritative.

## Validation and modelling

Temporal train/validation/test years are {_year_span(audit['temporal_train_years'])}, {_year_span(audit['temporal_validation_years'])} and {_year_span(audit['temporal_test_years'])}. The stratified random comparator has exactly the same {random_split_sizes} sample sizes. All imputing and encoding were pipeline-fitted on training data only. Compact hyperparameter and family selection used validation average precision (AP), calculated with scikit-learn's non-interpolated `average_precision_score`; analytical classification thresholds maximised validation F1. The selected train-fitted pipeline and its validation-derived threshold were then evaluated once on the corresponding holdout, with no train+validation refit or test-set retuning.

{shared_configuration_sentence} {differing_configuration_sentence} {core_configuration_sentence}

The main model comparison is Block B, the retrospective incident-information model:

{_format_rows(main_models.assign(model=main_models.model.map(MODEL_LABELS)).rename(columns={'pr_auc': 'average_precision'}), ['model','n','positive_count','positive_prevalence','average_precision','roc_auc','f1','brier_score'])}

## Research questions

### RQ1 — Random versus temporal validation

For the validation-selected Block B {MODEL_LABELS[core_family]}, random holdout AP was {random_core.pr_auc:.3f} (95% bootstrap CI {random_ci.ci_lower_95:.3f}–{random_ci.ci_upper_95:.3f}) and temporal holdout AP was {temporal_core.pr_auc:.3f} ({temporal_ci.ci_lower_95:.3f}–{temporal_ci.ci_upper_95:.3f}). The random-minus-temporal point difference was {difference_ci.point_estimate:+.3f}, with a 95% partially paired bootstrap interval of {difference_ci.ci_lower_95:+.4f} to {difference_ci.ci_upper_95:+.4f}. The holdouts overlap by {int(difference_ci.overlap_n):,} records ({difference_ci.overlap_fraction_random_test:.1%} of random and {difference_ci.overlap_fraction_temporal_test:.1%} of temporal holdout); those records were resampled jointly, while design-specific records were resampled independently within outcome and membership strata. The interval {difference_relation}.

Across-assignment stability was assessed using {stability_count} saved split seeds. With the estimator seed held fixed, they produced a median random-minus-temporal AP difference of {stability_summary.ap_difference_median:+.3f} (IQR {stability_summary.ap_difference_q1:+.3f} to {stability_summary.ap_difference_q3:+.3f}; range {stability_summary.ap_difference_min:+.3f} to {stability_summary.ap_difference_max:+.3f}); {stability_direction_text}. {stability_interpretation}

The fixed-split bootstrap interval and across-split point range address different uncertainty sources. The former conditions on the primary splits, fitted models and selected settings; the latter describes changes across the saved assignments. Their widths were {bootstrap_difference_width:.3f} and {stability_difference_width:.3f}, respectively. The quantities are dependent and neither is a joint interval or a measure of total uncertainty across test sampling and split assignment.

For the primary split, {primary_split_family_ap_text} ({family_difference_text}), and the corresponding ROC-AUC differences {family_roc_direction_text}. The magnitude and direction of split differences should not be treated as universal or operationally important without a decision-specific cost analysis. The repeated splits are an empirical sensitivity analysis under fixed model settings, not a second bootstrap interval or {stability_count} independent datasets.

The following comparison holds the model family fixed at {MODEL_LABELS[core_family]} across information blocks:

{_format_rows(cross_block.reset_index(), ['block','random_ap','temporal_ap','ap_difference','absolute_lift_difference','normalized_ap_difference','roc_auc_difference'])}

Block A's random-minus-temporal AP difference was {cross_block.loc['A','ap_difference']:+.3f}; Block C's was {cross_block.loc['C','ap_difference']:+.3f}. For Block C, the absolute AP-lift difference after subtracting each holdout's prevalence was {cross_block.loc['C','absolute_lift_difference']:+.4f}, and the normalized AP difference was {cross_block.loc['C','normalized_ap_difference']:+.3f}. These are descriptive comparisons; the Block B uncertainty interval cannot be transferred to the other blocks. They do not establish that temporal stability of any particular field caused the pattern.

The expanding-window models trained on all years available before each test year achieved a sample-size-weighted mean annual AP of {recent_weighted_ap:.3f} in {_year_span(audit['temporal_test_years'])}, compared with {temporal_core.pr_auc:.3f} for the main train-through-{audit['temporal_train_years'][-1]} model on the combined holdout. The difference of {recent_weighted_ap - temporal_core.pr_auc:+.3f} is not a clean decomposition of training recency: the annual models use different training sets and a weighted mean of annual AP is not the pooled holdout AP.

AP's no-information baseline is approximately the positive prevalence. The random and temporal {MODEL_LABELS[core_family]} holdouts had prevalences of {random_core.positive_prevalence:.3f} and {temporal_core.positive_prevalence:.3f}, respectively, so their AP values are interpreted with prevalence context:

{_format_rows(context_core.rename(columns={'pr_auc': 'average_precision', 'pr_auc_absolute_lift': 'ap_absolute_lift', 'normalized_pr_auc': 'normalized_ap'}), ['design','positive_prevalence','average_precision','ap_absolute_lift','normalized_ap','roc_auc','brier_score'])}

Normalized AP is an auxiliary prevalence-relative summary, not a replacement primary metric. AP {temporal_core.pr_auc:.3f} is a ranking summary, not “{temporal_core.pr_auc:.1%} accuracy.”

### RQ2 — Best later-year model

{model_ranking_text}

Grouped permutation of each original Block B field on the exact {_year_span(audit['temporal_test_years'])} temporal test set gave the following five largest mean AP decreases:

{_format_rows(importance_top, ['feature','mean_pr_auc_decrease','std_pr_auc_decrease'])}

With only {int(importance.permutation_repeats.iloc[0])} permutations, the table reports the mean and sample standard deviation; empirical 2.5th and 97.5th percentiles are too coarsely resolved to interpret. This analysis measures the fitted model's dependence on each recorded field, not a causal effect. High importance does not mean that a variable causes greater fire spread. Correlated or overlapping fields can share importance; in particular, `CAUSE_OF_FIRE`, `SOURCE_OF_IGNITION` and `ITEM_IGNITED` may encode overlapping information. Results apply only to this fitted pipeline, feature set and temporal test set, and negative values are retained rather than truncated.

### Structural/context information

On the same temporal holdout, Block A's {len(audit['feature_blocks']['A'])} structural/context fields achieved AP {selected_temporal.loc['A','pr_auc']:.3f}, an absolute lift of {block_a_lift:.3f} above prevalence. Block B's lift was {block_b_lift:.3f} using {len(audit['feature_blocks']['B'])} fields. This is a descriptive nested-block comparison, not an operational-utility estimate, because some Block A fields are retrospectively recorded.

### RQ3 — Retrospective incident information plus arrival-state information

For validation-selected families, temporal AP was {selected_temporal.loc['B','pr_auc']:.3f} in Block B and {selected_temporal.loc['C','pr_auc']:.3f} in Block C, a C-minus-B difference of {rq3_gain:+.3f}. `FIRE_SIZE_ON_ARRIVAL` precedes final `SPREAD_OF_FIRE`, but is a highly proximal state variable. Block C also inherits Block B cause/ignition fields that may be revised after investigation, so the complete feature set is not established as available at arrival. This is a retrospective incident information plus arrival-state model; a deployable arrival-time model would require an independently verified feature-availability policy and new evaluation.

## Temporal stability and sensitivity

{annual_evaluation_note}

{_format_rows(expanding.rename(columns={'pr_auc': 'average_precision', 'pr_auc_absolute_lift': 'ap_absolute_lift', 'normalized_pr_auc': 'normalized_ap'}), ['test_year','positive_prevalence','average_precision','ap_absolute_lift','normalized_ap','roc_auc'])}

Across these {len(expanding)} annual folds, AP ranged from {expanding.pr_auc.min():.3f} to {expanding.pr_auc.max():.3f}; its Spearman correlation with prevalence was {expanding_spearman:.3f}. ROC-AUC had range width {expanding_roc_range:.3f}, AP absolute lift {expanding_lift_range:.3f}, and normalized AP {expanding_normalized_range:.3f}. These descriptive summaries mix development-period and later-year evaluations and cannot be interpreted as independent validation across all years or identify why prevalence changed.

Expanding-window F1, precision, recall and balanced accuracy use a fixed descriptive threshold of 0.5 and are not directly comparable with the main table's validation-F1 operating point. The {_year_span(audit['temporal_validation_years'])} window was used to select settings and thresholds; annual prevalence alone does not establish any policy or COVID effect.

{_format_rows(sensitivity_context[['analysis','test_period','n','positive_prevalence','pr_auc','ap_absolute_lift','normalized_ap','roc_auc','f1']].rename(columns={'pr_auc': 'average_precision'}), ['analysis','test_period','n','positive_prevalence','average_precision','ap_absolute_lift','normalized_ap','roc_auc','f1'])}

Across target, cohort and new-year checks, normalized AP ranged from {sensitivity_context.normalized_ap.min():.3f} to {sensitivity_context.normalized_ap.max():.3f}. Roof-positive absolute lift was {sensitivity_context.loc[sensitivity_context.analysis == 'roofs_roof_spaces_positive','ap_absolute_lift'].iloc[0]:.3f}, compared with {sensitivity_context.loc[sensitivity_context.analysis == 'main_temporal_definition','ap_absolute_lift'].iloc[0]:.3f} for the main definition. Changing the target also changes prevalence and the estimand, so these values alone do not establish a better model. Across {stability_count} split assignments with a fixed estimator seed, random-holdout AP had median {stability_summary.random_ap_median:.3f} (IQR {stability_summary.random_ap_q1:.3f}–{stability_summary.random_ap_q3:.3f}) and ranged from {stability_summary.random_ap_min:.3f} to {stability_summary.random_ap_max:.3f}.

Building-type subgroup AP ranged from {subgroup_low.pr_auc:.3f} for {subgroup_low.building_type} (prevalence {subgroup_low.positive_prevalence:.3f}) to {subgroup_high.pr_auc:.3f} for {subgroup_high.building_type} ({subgroup_high.positive_prevalence:.3f}). {subgroup_threshold_text} These are descriptive diagnostics of the saved model and global threshold, not evidence that building type causes fire spread or that the remaining fields lack within-group signal.

## Limitations

- The public file has no incident identifier or exact date/month, limiting dependence checks and finer temporal validation.
- Incident fields may reflect officer judgement; cause/ignition fields may be revised after investigation, and delay fields may be estimated.
- Block B is retrospective and not strictly dispatch-time information.
- Block C includes both retrospective fields and an arrival-state field close to the final outcome. Its score does not establish deployability at arrival; the contribution of that state field requires a simple baseline or ablation to quantify.
- `OCCUPIED_TIME` can include occupants in buildings to which the fire spread, potentially encoding already-realised spread. No removal ablation has been run, so its effect on the saved scores is unknown.
- Average precision is prevalence-sensitive; cross-split and subgroup comparisons require their respective positive prevalences.
- Hyperparameters and analytical thresholds were selected using {_year_span(audit['temporal_validation_years'])}. Annual folds within or before that window are development-period descriptive results, not independent temporal validation.
- Bootstrap intervals condition on the fixed splits, fitted models and selected settings; they do not represent repeated end-to-end model-selection uncertainty.
- `FRS_TERRITORY` is an available pre-incident geographic field excluded by scope rather than outcome leakage; no territory-inclusive sensitivity was run, so its incremental predictive value is unknown.
- Subgroup and permutation results are descriptive model diagnostics, not evidence of differential or variable-level causal effects.
- Temporal performance differences do not by themselves identify why distributions changed.
- The publisher URL can be replaced in future. Checksums verify retained files but cannot recover them; the ODS and reproducibility Parquet files require a separate durable institutional deposit.

## Reproducibility

All tables, figures, fitted selected pipelines, split assignments, model-selection settings, software versions and method decisions are saved under `outputs/` and `reports/`. `python scripts/06_build_report.py` runs the full analysis, including model fitting. To render saved artifacts only, use `python scripts/07_render_report.py`; it recomputes confusion/calibration and table summaries but performs no acquisition, model fitting, bootstrap or permutation analysis. Historical `runtime_environment.json`, `model_selection.json` and `data_archive_manifest.json` remain unchanged; the current render environment is recorded separately in `report_environment.json`. Their preservation does not independently verify historical provenance.
"""
    (ROOT / "reports/final_analysis_report.md").write_text(report, encoding="utf-8")


def _write_methods_receipt(stability_summary: pd.Series) -> None:
    cfg = load_yaml("config/analysis.yaml")
    metadata = json.loads((ROOT / "data/raw/source_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    cohort = json.loads((ROOT / "outputs/metrics/cohort_receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    runtime = json.loads((ROOT / "outputs/metrics/runtime_environment.json").read_text(encoding="utf-8"))
    render_environment = json.loads((ROOT / "outputs/metrics/report_environment.json").read_text(encoding="utf-8"))
    archive = json.loads((ROOT / "outputs/metrics/data_archive_manifest.json").read_text(encoding="utf-8"))
    expanding = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    overlap = bootstrap[bootstrap.design == "random-minus-temporal"].iloc[0]
    bootstrap_difference_width = float(overlap.ci_upper_95 - overlap.ci_lower_95)
    stability_difference_width = float(stability_summary.ap_difference_range_width)
    policy = load_yaml("config/feature_policy.yaml")
    core_family = selection["selected_family_by_block"]["temporal"]["B"]
    annual_evaluation_note = _annual_evaluation_note(expanding, audit)
    shared_configuration_families = [
        family for family in ("logistic_regression", "random_forest", "xgboost")
        if selection["selected_hyperparameters"]["temporal"][family]
        == selection["selected_hyperparameters"]["random"][family]
    ]
    differing_configurations = {
        family: {
            "random": selection["selected_hyperparameters"]["random"][family],
            "temporal": selection["selected_hyperparameters"]["temporal"][family],
        }
        for family in ("logistic_regression", "random_forest", "xgboost")
        if family not in shared_configuration_families
    }
    manifest = sorted(set([
        str(path.relative_to(ROOT)).replace("\\", "/")
        for base in (ROOT / "outputs", ROOT / "reports")
        for path in base.rglob("*") if path.is_file()
    ] + ["reports/methods_receipt.md", "outputs/metrics/output_manifest.json"]))
    receipt = f"""# Methods receipt

Rendered from saved artifacts. Rendering does not rerun the analysis or verify unknown historical provenance. See [no-rerun revision notes](no_rerun_revision.md) for the reviewed snapshot's status.

## Source

- Official dataset: Other Building Fires Dataset
- Source page: {cfg['source_page_url']}
- Guidance: {cfg['guidance_url']}
- Download URL: {metadata['download_url']}
- Download timestamp (raw-file mtime, if recorded): {metadata.get('download_timestamp_utc', 'not recorded')}
- Metadata recorded: {metadata.get('download_recorded_at_utc', 'not recorded')}
- Official page update: {metadata['official_page_updated']}
- File size: {metadata['file_size_bytes']} bytes
- SHA-256: `{metadata['sha256']}`
- ODS sheets: {json.dumps(metadata['all_sheets'])}
- Data sheet: {metadata['data_sheet']}
- Data archive manifest: `{json.dumps(archive['files'])}`
- The source ODS and reproducibility Parquet files are excluded from Git and require a separate durable institution-controlled deposit. Checksums verify retained files but cannot recover them after a publisher URL is replaced; this manifest is not itself an archive.

## Cohort and target

- Main years: {', '.join(audit['main_years'])}
- Exclusions, in order: outside main years; complete duplicate rows; `LATE_CALL=yes`; main-target exclusions (`Roofs/ Roof spaces` and any unmappable/missing category).
- Final rows: {cohort['rows']}; positives: {cohort['positive_count']}; prevalence: {cohort['positive_prevalence']:.8f}.
- Exact target mapping: `{json.dumps(cohort['target_mapping'])}`.
- Main-estimand rationale: the room→floor→whole-building ordering maps six categories unambiguously; `Roofs/ Roof spaces` is not assigned because it cannot be located unambiguously on that ordering.
- Roof-definition sensitivity: the source definitions documented for this study differ on roof spaces. The main target excludes roofs; the saved sensitivity maps them to 1. Consult the sensitivity table for its observed prevalence.

## Feature blocks

- Block A — structural and context: `{audit['feature_blocks']['A']}`
- Block B — retrospective incident information: `{audit['feature_blocks']['B']}`
- Block C — retrospective incident information plus arrival-state information: `{audit['feature_blocks']['C']}`
- Block C inherits retrospective cause/ignition fields from Block B; availability of the complete set at arrival has not been established. Its arrival-state field is close to the outcome, and no simple arrival-state-only baseline or ablation quantifies that field's contribution here.
- `OCCUPIED_TIME` may include occupants in buildings to which the fire spread, potentially encoding already-realised spread; no removal ablation has quantified its effect on these saved results.
- All retained predictors are treated as categorical/banded fields. Missing/blank values become `Missing/Unknown`; one-hot encoding uses `handle_unknown='ignore'`. No rare-category merger is applied.
- `RESPONSE_TIME` is used; its redundant code field is not used.
- Leakage blacklist: `{policy['leakage_blacklist']}`
- Always excluded from predictors: `{policy['always_excluded']}` plus `E_CODE_TERRITORY`.
- `FRS_TERRITORY` is excluded as a scope/geographic-generalisation decision, not classified as outcome leakage; no territory-inclusive sensitivity was estimated in this run.

## Validation

- Temporal train: {', '.join(audit['temporal_train_years'])}
- Temporal validation: {', '.join(audit['temporal_validation_years'])}
- Temporal test: {', '.join(audit['temporal_test_years'])}
- Random comparator: stratified sampling with exactly matching train/validation/test counts.
- Primary seed: {cfg['random_seed']}; stability seeds: {cfg['random_stability_seeds']}.
- Selection metric: validation average precision (`average_precision_score`). Threshold: validation F1 maximum, an analytical operating point rather than an operational optimum.
- Legacy output columns and file stems named `pr_auc` store this non-interpolated AP value; no trapezoidal precision–recall curve area is calculated.
- Threshold provenance: the selected train-fitted pipeline and its validation-derived threshold are evaluated on test without a train+validation refit or test-set retuning.
- {annual_evaluation_note}
- Expanding-window thresholded metrics use fixed threshold 0.5 and are not directly comparable with main-table thresholded metrics. Annual AP, prevalence-relative summaries and ROC-AUC are the intended temporal-stability comparisons.

## Hyperparameters and thresholds

- Full candidate ranges and results: `reports/hyperparameter_plan.md` and `outputs/tables/hyperparameter_search_results.csv`.
- Selected parameters: `{json.dumps(selection['selected_hyperparameters'])}`
- Families with identical random and temporal selected hyperparameters: `{shared_configuration_families}`.
- Families with differing selected hyperparameters, including their design-specific settings: `{json.dumps(differing_configurations, sort_keys=True)}`
- Validation-selected family by block: `{json.dumps(selection['selected_family_by_block'])}`
- Validation thresholds: `{json.dumps(selection['validation_thresholds'])}`
- XGBoost device: `{selection['xgboost_device']}`; tree method: `hist`. The configured device is `{cfg['xgboost_device']}` and must match the recorded selection for a result-reproducing rerun.

## Fixed-model uncertainty and prevalence context

- Average-precision intervals use {cfg['bootstrap_repeats']:,} partially paired, class-and-membership-stratified bootstrap repeats with seed {cfg['bootstrap_seed']}.
- Random and temporal test sets overlap by {int(overlap.overlap_n):,} records ({overlap.overlap_fraction_random_test:.6%} of random test and {overlap.overlap_fraction_temporal_test:.6%} of temporal test), based on `SOURCE_ROW_ID` and cross-checked against cohort index.
- Shared records are resampled jointly in both holdouts; random-only and temporal-only records are resampled independently. Outcome class and observed overlap membership counts remain fixed.
- Percentile limits are the 2.5th and 97.5th percentiles. They condition on fixed splits, fitted models and selected settings; repeated end-to-end selection is outside their scope.
- The repeat count controls Monte Carlo error in these fixed-model percentile limits; it does not address split-assignment or model-selection uncertainty.
- AP baseline, absolute lift and normalized AP are prevalence-context diagnostics. Normalized AP is auxiliary and does not replace the primary AP definition.

## Split-assignment stability

- The {int(stability_summary.split_count)} saved split seeds are checked against `config/analysis.yaml`. Complete seed-level results are in `outputs/tables/random_seed_stability.csv`; seed-selection history is not reconstructed during report rendering.
- With estimator seed {cfg['random_seed']} and selected random-design hyperparameters fixed, the median random-minus-temporal AP difference was {stability_summary.ap_difference_median:+.6f} (IQR {stability_summary.ap_difference_q1:+.6f} to {stability_summary.ap_difference_q3:+.6f}; range {stability_summary.ap_difference_min:+.6f} to {stability_summary.ap_difference_max:+.6f}). Positive differences occurred for {int(stability_summary.positive_difference_count)} of {int(stability_summary.split_count)} assignments.
- The primary fixed-split bootstrap interval width was {bootstrap_difference_width:.6f}; the across-split point range width was {stability_difference_width:.6f}. They describe dependent, different uncertainty sources and are not combined into a joint interval.
- This is an empirical split-assignment sensitivity analysis, not a Monte Carlo bootstrap or a repetition of end-to-end family/hyperparameter selection. It does not require a second bootstrap; the separate fixed-model bootstrap remains configured at {cfg['bootstrap_repeats']:,} repeats.

## Grouped permutation importance

- The saved permutation results evaluate the validation-selected temporal Block B {MODEL_LABELS[core_family]} pipeline on the saved {_year_span(audit['temporal_test_years'])} test indices.
- Each of the {len(audit['feature_blocks']['B'])} original Block B fields is permuted as a whole before the complete fitted preprocessing-and-model pipeline. This automatically groups all one-hot columns derived from that field.
- Each field uses {cfg['permutation_repeats']} repeats with seed {cfg['permutation_seed']}; importance is baseline AP minus permuted AP, with negative values retained.
- Because {cfg['permutation_repeats']} repeats do not resolve tail quantiles well, permutation variability is summarised by the sample standard deviation rather than empirical 2.5th/97.5th percentiles.
- Importance measures model dependence, not a causal effect, and may be shared across correlated or overlapping fields.

## Existing historical runtime record

The following record is preserved unchanged. Earlier reporting code could overwrite its package list; preservation cannot verify whether that happened historically. Unknown training metadata is not backfilled from the current environment.

```json
{json.dumps(runtime, indent=2)}
```

## Current report-render environment

These versions describe this rendering process only, not the original training environment.

```json
{json.dumps(render_environment, indent=2)}
```

## Output manifest

""" + "\n".join(f"- `{item}`" for item in manifest)
    (ROOT / "reports/methods_receipt.md").write_text(receipt, encoding="utf-8")
    (ROOT / "outputs/metrics/output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_report() -> None:
    """Render saved artifacts, recomputing only confusion/calibration and table summaries.

    Does not acquire data, fit models, or rerun bootstrap/permutation analyses.
    Historical training/source receipts are read-only inputs to this operation.
    """
    _require_report_inputs()
    _validate_report_selection()
    ensure_output_dirs()
    _write_report_environment()
    _workflow_figure(); _annual_figure(); _random_temporal_figure(); _expanding_figure()
    _block_figure(); _confusion_and_calibration(); _subgroup_figure()
    _bootstrap_figure(); _grouped_permutation_figure()
    stability_summary = _write_random_stability_summary()
    _write_final_report(stability_summary)
    _write_methods_receipt(stability_summary)
