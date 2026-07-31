from __future__ import annotations

import hashlib
import json
import os
import importlib.metadata
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


def _write_data_archive_manifest() -> dict:
    """Record recovery-critical data checksums without placing data in Git."""
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(data.test_year, data.pr_auc, marker="o", color="#4c78a8", label="Average precision")
    axes[0].plot(data.test_year, data.pr_auc_baseline, marker="s", color="#a43c3c", label="Positive prevalence")
    axes[0].set_ylim(0.2, 0.75); axes[0].set_ylabel("Average precision / prevalence")
    axes[0].set_title("Average precision and its prevalence baseline")
    axes[1].plot(data.test_year, data.normalized_pr_auc, marker="o", color="#f58518", label="Normalized AP")
    axes[1].plot(data.test_year, data.roc_auc, marker="s", color="#54a24b", label="ROC-AUC")
    axes[1].set_ylim(0.45, 0.88); axes[1].set_ylabel("Prevalence-context / ROC area")
    axes[1].set_title("Prevalence-context discrimination")
    for ax in axes:
        ax.set_xlabel("One-year temporal test")
        ax.legend(frameon=False)
    fig.suptitle("Expanding-window annual performance: fixed Block B XGBoost settings")
    _save(fig, "04_expanding_window_performance")


def _block_figure() -> None:
    data = pd.read_csv(ROOT / "outputs/tables/block_comparison.csv")
    designs = ["random", "temporal"]; blocks = ["A", "B", "C"]
    x = np.arange(3); width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, design, color in [(-width/2, "random", "#4c78a8"), (width/2, "temporal", "#f58518")]:
        subset = data[data.design == design].set_index("block").loc[blocks]
        ax.bar(x + offset, subset.pr_auc, width, label=design.title(), color=color)
    ax.set_xticks(x, ["A: structural/context", "B: retrospective incident", "C: first-arrival"])
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
    ax.plot(predicted, observed, marker="o", color="#4c78a8", label="Block B XGBoost")
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


def _write_final_report() -> None:
    cfg = load_yaml("config/analysis.yaml")
    metadata = json.loads((ROOT / "data/raw/source_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    cohort = json.loads((ROOT / "outputs/metrics/cohort_receipt.json").read_text(encoding="utf-8"))
    temporal = pd.read_csv(ROOT / "outputs/tables/temporal_validation_performance.csv")
    random = pd.read_csv(ROOT / "outputs/tables/random_validation_performance.csv")
    block = pd.read_csv(ROOT / "outputs/tables/block_comparison.csv")
    expanding = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    sensitivity = pd.read_csv(ROOT / "outputs/tables/sensitivity_analysis_results.csv")
    stability = pd.read_csv(ROOT / "outputs/tables/random_seed_stability.csv")
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
    shared_configuration_labels = [MODEL_LABELS[family] for family in shared_configuration_families]
    shared_configuration_text = " and ".join(shared_configuration_labels)
    main_models = temporal[(temporal.block == "B") & (temporal.model != "dummy")].sort_values("pr_auc", ascending=False)
    core_family = selection["selected_family_by_block"]["temporal"]["B"]
    temporal_core = temporal[(temporal.block == "B") & (temporal.model == core_family)].iloc[0]
    random_core = random[(random.block == "B") & (random.model == core_family)].iloc[0]
    random_ci = bootstrap[(bootstrap.estimand == "average precision") & (bootstrap.design == "random")].iloc[0]
    temporal_ci = bootstrap[(bootstrap.estimand == "average precision") & (bootstrap.design == "temporal")].iloc[0]
    difference_ci = bootstrap[bootstrap.design == "random-minus-temporal"].iloc[0]
    context_core = prevalence_context[prevalence_context.model == core_family].copy()
    importance_top = importance.head(5).copy()
    selected_temporal = block[block.design == "temporal"].set_index("block")
    rq3_gain = selected_temporal.loc["C", "pr_auc"] - selected_temporal.loc["B", "pr_auc"]
    expanding_spearman = expanding[["positive_prevalence", "pr_auc"]].corr(method="spearman").iloc[0, 1]
    expanding_lift_range = expanding.pr_auc_absolute_lift.max() - expanding.pr_auc_absolute_lift.min()
    expanding_normalized_range = expanding.normalized_pr_auc.max() - expanding.normalized_pr_auc.min()
    expanding_roc_range = expanding.roc_auc.max() - expanding.roc_auc.min()
    difference_relation = (
        "lay entirely above zero"
        if difference_ci.ci_lower_95 > 0
        else "included zero"
    )
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
    stability_differences = stability["pr_auc"] - float(temporal_core.pr_auc)
    stability_difference_text = ", ".join(f"{value:+.3f}" for value in stability_differences)
    stability_difference_range = float(stability_differences.max() - stability_differences.min())
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
    cross_block = pd.DataFrame(cross_block_rows).set_index("block")
    block_a_lift = (
        selected_temporal.loc["A", "pr_auc"]
        - selected_temporal.loc["A", "positive_prevalence"]
    )
    block_b_lift = (
        selected_temporal.loc["B", "pr_auc"]
        - selected_temporal.loc["B", "positive_prevalence"]
    )
    block_a_lift_share = block_a_lift / block_b_lift
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
    prison = subgroup[subgroup.building_type.eq("Prison")].iloc[0]
    report = f"""# Final analysis report

## Study definition

This retrospective prediction study uses the official **Other Building Fires Dataset** for England. “Non-domestic building fires” is the dissertation's analytical wording for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings and can include hotels, hostels, care homes and student halls; it is not limited to buildings without accommodation functions.

The analysis predicts incident-level final fire spread among already-recorded primary fires. It is not a causal study, annual building fire-risk model, fire-physics simulation or real-time FRS deployment tool. Block B contains retrospectively recorded incident information; Block C is a separately interpreted first-arrival prognostic model.

## Data and cohort

The official ODS was updated {official_update_text}. The raw-file SHA-256 is `{metadata['sha256']}`. The audit found {audit['logical_rows']:,} logical incident rows across {_year_span(year_coverage['financial_year'])}. The defined main period is {_year_span(audit['main_years'])}. Excluded years are {excluded_year_text}.

After year restriction, {_excluded_at_stage(flow, 'Remove exact duplicates'):,} exact duplicates, {_excluded_at_stage(flow, 'Exclude late calls'):,} late calls and {_excluded_at_stage(flow, 'Apply binary target mapping'):,} roof-space or otherwise unmappable target records were removed sequentially. The main cohort contains {cohort['rows']:,} incidents, {cohort['positive_count']:,} larger fires ({cohort['positive_prevalence']:.1%}).

The main target follows the unambiguous room→floor→whole-building ordering and excludes `Roofs/ Roof spaces`, which cannot be placed uniquely on that scale. Official sources are inconsistent: the current Fire statistics definitions omit roofs from the larger-fire list, while FIRE0304-linked detailed releases include them. The roof-positive sensitivity therefore tests the alternative published convention rather than treating either source as uniquely authoritative.

## Validation and modelling

Temporal train/validation/test years are {_year_span(audit['temporal_train_years'])}, {_year_span(audit['temporal_validation_years'])} and {_year_span(audit['temporal_test_years'])}. The stratified random comparator has exactly the same {random_split_sizes} sample sizes. All imputing and encoding were pipeline-fitted on training data only. Compact hyperparameter and family selection used validation average precision (AP), calculated with scikit-learn's non-interpolated `average_precision_score`; analytical classification thresholds maximised validation F1. The selected train-fitted pipeline and its validation-derived threshold were then evaluated once on the corresponding holdout, with no train+validation refit or test-set retuning.

{shared_configuration_text} selected identical hyperparameters under random and temporal development designs. In particular, the primary XGBoost RQ1 contrast is not confounded by comparing different XGBoost configurations; Logistic Regression selected different regularisation strengths (`C=1.0` random versus `C=0.1` temporal).

The main model comparison is Block B, the retrospective incident-information model:

{_format_rows(main_models.assign(model=main_models.model.map(MODEL_LABELS)).rename(columns={'pr_auc': 'average_precision'}), ['model','n','positive_count','positive_prevalence','average_precision','roc_auc','f1','brier_score'])}

## Research questions

### RQ1 — Random versus temporal validation

For the validation-selected Block B XGBoost, random holdout AP was {random_core.pr_auc:.3f} (95% bootstrap CI {random_ci.ci_lower_95:.3f}–{random_ci.ci_upper_95:.3f}) and temporal holdout AP was {temporal_core.pr_auc:.3f} ({temporal_ci.ci_lower_95:.3f}–{temporal_ci.ci_upper_95:.3f}). The random-minus-temporal point difference was {difference_ci.point_estimate:+.3f}, with a 95% partially paired bootstrap interval of {difference_ci.ci_lower_95:+.4f} to {difference_ci.ci_upper_95:+.4f}. The holdouts overlap by {int(difference_ci.overlap_n):,} records ({difference_ci.overlap_fraction_random_test:.1%} of each holdout); those records were resampled jointly, while design-specific records were resampled independently within outcome and membership strata. The interval {difference_relation}.

These intervals condition on the fixed splits, fitted models and selected settings. They represent test-sample uncertainty and the observed overlap covariance, but not variability from repeating the full selection procedure.

All three Block B model families favoured random splitting in AP ({family_difference_text}), and their ROC-AUC differences were also positive. With the estimator seed held fixed, the three random split assignments produced random-minus-temporal AP differences of {stability_difference_text}. Together, these results support a small and directionally consistent random-split optimism effect in this retrospective Block B task. Its exact magnitude varies with the split and should not be treated as a universal or operationally important bias without a decision-specific cost analysis.

The direction is not universal across information blocks, even for the same XGBoost family:

{_format_rows(cross_block.reset_index(), ['block','random_ap','temporal_ap','ap_difference','absolute_lift_difference','normalized_ap_difference','roc_auc_difference'])}

Block A showed a smaller random advantage ({cross_block.loc['A','ap_difference']:+.3f}); Block C reversed in raw AP ({cross_block.loc['C','ap_difference']:+.3f}). For Block C, the absolute AP-lift difference after subtracting each holdout's prevalence was effectively zero ({cross_block.loc['C','absolute_lift_difference']:+.4f}), while normalized AP slightly favoured the temporal holdout ({cross_block.loc['C','normalized_ap_difference']:+.3f}). This confines the inferentially supported positive finding to the retrospective Block B specification and shows that split effects depend on the information set. It does not by itself establish that temporal stability of any particular field caused the pattern.

The expanding-window models trained on all years available before each test year achieved a sample-size-weighted mean annual AP of {recent_weighted_ap:.3f} in {_year_span(audit['temporal_test_years'])}, compared with {temporal_core.pr_auc:.3f} for the main train-through-{audit['temporal_train_years'][-1]} model on the combined two-year holdout. The {recent_weighted_ap - temporal_core.pr_auc:+.3f} gap suggests that training recency is not a large explanation here, but it is not a clean decomposition: the annual models use different training sets and a weighted mean of annual AP is not the pooled two-year AP.

AP's no-information baseline is approximately the positive prevalence. The random and temporal XGBoost holdouts had prevalences of {random_core.positive_prevalence:.3f} and {temporal_core.positive_prevalence:.3f}, respectively, so their AP values are interpreted with prevalence context:

{_format_rows(context_core.rename(columns={'pr_auc': 'average_precision', 'pr_auc_absolute_lift': 'ap_absolute_lift', 'normalized_pr_auc': 'normalized_ap'}), ['design','positive_prevalence','average_precision','ap_absolute_lift','normalized_ap','roc_auc','brier_score'])}

Normalized AP is an auxiliary prevalence-relative summary, not a replacement primary metric. AP {temporal_core.pr_auc:.3f} is a ranking summary, not “{temporal_core.pr_auc:.1%} accuracy.”

### RQ2 — Best later-year model

XGBoost had the highest temporal Block B AP ({temporal_core.pr_auc:.3f}), followed by Random Forest ({main_models.iloc[1].pr_auc:.3f}) and Logistic Regression ({main_models.iloc[2].pr_auc:.3f}). The margins are small and no pairwise model-difference interval was estimated, so XGBoost is described only as the highest-performing evaluated family.

Grouped permutation of each original Block B field on the exact 2022/23–2023/24 temporal test set gave the following five largest mean AP decreases:

{_format_rows(importance_top, ['feature','mean_pr_auc_decrease','std_pr_auc_decrease'])}

With only {int(importance.permutation_repeats.iloc[0])} permutations, the table reports the mean and sample standard deviation; empirical 2.5th and 97.5th percentiles are too coarsely resolved to interpret. This analysis measures the fitted model's dependence on each recorded field, not a causal effect. High importance does not mean that a variable causes greater fire spread. Correlated or overlapping fields can share importance; in particular, `CAUSE_OF_FIRE`, `SOURCE_OF_IGNITION` and `ITEM_IGNITED` may encode overlapping information. Results apply only to this fitted pipeline, feature set and temporal test set, and negative values are retained rather than truncated.

### RQ3 — First-arrival information

On the same temporal holdout, Block A's {len(audit['feature_blocks']['A'])} structural/context fields achieved AP {selected_temporal.loc['A','pr_auc']:.3f}, an absolute lift of {block_a_lift:.3f} above prevalence. That is {block_a_lift_share:.1%} of Block B's {block_b_lift:.3f} lift using {len(audit['feature_blocks']['B'])} fields. This is a descriptive nested-block comparison, not an operational-utility estimate, because some Block A fields are retrospectively recorded.

For validation-selected families, temporal AP rose from {selected_temporal.loc['B','pr_auc']:.3f} in Block B to {selected_temporal.loc['C','pr_auc']:.3f} in Block C, an absolute gain of {rq3_gain:.3f}. `FIRE_SIZE_ON_ARRIVAL` is temporally prior to final `SPREAD_OF_FIRE`, but it is a highly proximal state variable. The Block C result is therefore first-arrival prognosis, not pre-incident building risk and not evidence of deployability before crews arrive.

## Temporal stability and sensitivity

{_format_rows(expanding.rename(columns={'pr_auc': 'average_precision', 'pr_auc_absolute_lift': 'ap_absolute_lift', 'normalized_pr_auc': 'normalized_ap'}), ['test_year','positive_prevalence','average_precision','ap_absolute_lift','normalized_ap','roc_auc'])}

Across these {len(expanding)} later-year folds, AP ranged from {expanding.pr_auc.min():.3f} to {expanding.pr_auc.max():.3f} and had the same rank ordering as prevalence (Spearman {expanding_spearman:.3f}). ROC-AUC varied by only {expanding_roc_range:.3f}, AP absolute lift by {expanding_lift_range:.3f}, and normalized AP by {expanding_normalized_range:.3f}. This supports stable later-year ranking performance while showing that much of the raw AP movement accompanies a changing prevalence baseline; four annual folds cannot identify why prevalence changed.

Expanding-window F1, precision, recall and balanced accuracy use a fixed descriptive threshold of 0.5 and are not directly comparable with the main table's validation-F1 operating point. The 2020/21–2021/22 validation window overlaps the COVID-disrupted period, and 2020/21 has the highest expanding-window prevalence ({expanding.iloc[0].positive_prevalence:.3f}); this may affect selected settings and thresholds. No policy or COVID attribution is made.

{_format_rows(sensitivity_context[['analysis','test_period','n','positive_prevalence','pr_auc','ap_absolute_lift','normalized_ap','roc_auc','f1']].rename(columns={'pr_auc': 'average_precision'}), ['analysis','test_period','n','positive_prevalence','average_precision','ap_absolute_lift','normalized_ap','roc_auc','f1'])}

Across target, cohort and new-year checks, normalized AP ranged only from {sensitivity_context.normalized_ap.min():.3f} to {sensitivity_context.normalized_ap.max():.3f}. The roof-positive definition had higher raw AP but slightly lower absolute lift ({sensitivity_context.loc[sensitivity_context.analysis == 'roofs_roof_spaces_positive','ap_absolute_lift'].iloc[0]:.3f}) than the main definition ({sensitivity_context.loc[sensitivity_context.analysis == 'main_temporal_definition','ap_absolute_lift'].iloc[0]:.3f}); it should not be read as unambiguously better performance. Across the three split assignments with a fixed estimator seed, random-holdout AP ranged from {stability.pr_auc.min():.3f} to {stability.pr_auc.max():.3f}.

Building-type subgroup AP ranged from {subgroup_low.pr_auc:.3f} for {subgroup_low.building_type} (prevalence {subgroup_low.positive_prevalence:.3f}) to {subgroup_high.pr_auc:.3f} for {subgroup_high.building_type} ({subgroup_high.positive_prevalence:.3f}). At the single global validation-F1 threshold, the Prison subgroup contained {int(prison.positive_count)} positives among {int(prison.n):,} incidents but received no positive predictions (recall and precision both zero). This is evidence that the global analytical threshold does not transfer uniformly across prevalence-defined subgroups; it is not evidence that building type causes fire spread or that the remaining fields lack within-group signal.

## Limitations

- The public file has no incident identifier or exact date/month, limiting dependence checks and finer temporal validation.
- Incident fields may reflect officer judgement; cause/ignition fields may be revised after investigation, and delay fields may be estimated.
- Block B is retrospective and not strictly dispatch-time information.
- Block C's exceptional performance is dominated by proximity to the final outcome and must remain a separate prognostic scenario.
- Average precision is prevalence-sensitive; cross-split and subgroup comparisons require their respective positive prevalences.
- Hyperparameters and analytical thresholds were selected using 2020/21–2021/22, a validation window that overlaps the COVID-disrupted period and includes an unusually high-prevalence first year.
- Bootstrap intervals condition on the fixed splits, fitted models and selected settings; they do not represent repeated end-to-end model-selection uncertainty.
- `FRS_TERRITORY` is an available pre-incident geographic field excluded by scope rather than outcome leakage; no territory-inclusive sensitivity was run, so its incremental predictive value is unknown.
- Subgroup and permutation results are descriptive model diagnostics, not evidence of differential or variable-level causal effects.
- Temporal performance differences do not by themselves identify why distributions changed.
- The publisher URL can be replaced in future. Checksums verify retained files but cannot recover them; the ODS and reproducibility Parquet files require a separate durable institutional deposit.

## Reproducibility

All tables, figures, fitted selected pipelines, split assignments, model-selection settings, software versions and method decisions are saved under `outputs/` and `reports/`. From an existing ODS or Parquet cache, run `python scripts/06_build_report.py` after installing the pinned project environment.
"""
    (ROOT / "reports/final_analysis_report.md").write_text(report, encoding="utf-8")


def _write_methods_receipt() -> None:
    cfg = load_yaml("config/analysis.yaml")
    metadata = json.loads((ROOT / "data/raw/source_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    cohort = json.loads((ROOT / "outputs/metrics/cohort_receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    runtime = json.loads((ROOT / "outputs/metrics/runtime_environment.json").read_text(encoding="utf-8"))
    archive = json.loads((ROOT / "outputs/metrics/data_archive_manifest.json").read_text(encoding="utf-8"))
    expanding = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    stability = pd.read_csv(ROOT / "outputs/tables/random_seed_stability.csv")
    bootstrap = pd.read_csv(ROOT / "outputs/tables/bootstrap_confidence_intervals.csv")
    overlap = bootstrap[bootstrap.design == "random-minus-temporal"].iloc[0]
    temporal_point = bootstrap[
        (bootstrap.estimand == "average precision") & (bootstrap.design == "temporal")
    ].iloc[0].point_estimate
    stability_differences = stability["pr_auc"] - temporal_point
    stability_difference_range = stability_differences.max() - stability_differences.min()
    policy = load_yaml("config/feature_policy.yaml")
    shared_configuration_families = [
        family for family in ("logistic_regression", "random_forest", "xgboost")
        if selection["selected_hyperparameters"]["temporal"][family]
        == selection["selected_hyperparameters"]["random"][family]
    ]
    manifest = sorted(set([
        str(path.relative_to(ROOT)).replace("\\", "/")
        for base in (ROOT / "outputs", ROOT / "reports")
        for path in base.rglob("*") if path.is_file()
    ] + ["reports/methods_receipt.md", "outputs/metrics/output_manifest.json"]))
    receipt = f"""# Methods receipt

## Source

- Official dataset: Other Building Fires Dataset
- Source page: {cfg['source_page_url']}
- Guidance: {cfg['guidance_url']}
- Download URL: {metadata['download_url']}
- Download timestamp (raw-file mtime): {metadata.get('download_timestamp_utc', metadata['download_recorded_at_utc'])}
- Metadata recorded: {metadata['download_recorded_at_utc']}
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
- Roof-definition sensitivity: current Fire statistics definitions omit roofs from the larger-fire list, while FIRE0304-linked detailed releases include them. The main target excludes roofs; the sensitivity maps them to 1 and reproduces the published approximately 26% 2024/25 proportion.

## Feature blocks

- Block A — structural and context: `{audit['feature_blocks']['A']}`
- Block B — retrospective incident information: `{audit['feature_blocks']['B']}`
- Block C — first-arrival prognostic: `{audit['feature_blocks']['C']}`
- All retained predictors are treated as categorical/banded fields. Missing/blank values become `Missing/Unknown`; one-hot encoding uses `handle_unknown='ignore'`. No rare-category merger was required because the largest field has 82 disclosed categories and sparse one-hot encoding remained tractable.
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
- The validation years 2020/21–2021/22 overlap the COVID-disrupted period. The first expanding-window year has positive prevalence {expanding.iloc[0].positive_prevalence:.6f}; this design feature may affect selection and thresholds but does not identify a COVID effect.
- Expanding-window thresholded metrics use fixed threshold 0.5 and are not directly comparable with main-table thresholded metrics. Annual AP, prevalence-relative summaries and ROC-AUC are the intended temporal-stability comparisons.

## Hyperparameters and thresholds

- Full candidate ranges and results: `reports/hyperparameter_plan.md` and `outputs/tables/hyperparameter_search_results.csv`.
- Selected parameters: `{json.dumps(selection['selected_hyperparameters'])}`
- Families with identical random and temporal selected hyperparameters: `{shared_configuration_families}`. This includes Random Forest and the primary XGBoost comparator; Logistic Regression differs (`C=1.0` random, `C=0.1` temporal).
- Validation-selected family by block: `{json.dumps(selection['selected_family_by_block'])}`
- Validation thresholds: `{json.dumps(selection['validation_thresholds'])}`
- XGBoost device: `{selection['xgboost_device']}`; tree method: `hist`. The configured device is `{cfg['xgboost_device']}` and must match the recorded selection for a result-reproducing rerun.

## Fixed-model uncertainty and prevalence context

- Average-precision intervals use {cfg['bootstrap_repeats']:,} partially paired, class-and-membership-stratified bootstrap repeats with seed {cfg['bootstrap_seed']}.
- Random and temporal test sets overlap by {int(overlap.overlap_n):,} records ({overlap.overlap_fraction_random_test:.6%} of random test and {overlap.overlap_fraction_temporal_test:.6%} of temporal test), based on `SOURCE_ROW_ID` and cross-checked against cohort index.
- Shared records are resampled jointly in both holdouts; random-only and temporal-only records are resampled independently. Outcome class and observed overlap membership counts remain fixed.
- Percentile limits are the 2.5th and 97.5th percentiles. They condition on fixed splits, fitted models and selected settings; repeated end-to-end selection is outside their scope.
- The repeat count controls Monte Carlo error in these fixed-model percentile limits; it does not address split-assignment or model-selection uncertainty.
- With estimator seed {cfg['random_seed']} fixed, split seeds {cfg['random_stability_seeds']} give random-minus-temporal AP differences of {', '.join(f'{value:+.3f}' for value in stability_differences)} (range {stability_difference_range:.3f}).
- AP baseline, absolute lift and normalized AP are prevalence-context diagnostics. Normalized AP is auxiliary and does not replace the primary AP definition.

## Grouped permutation importance

- The validation-selected Temporal Block B XGBoost pipeline is evaluated on the exact saved 2022/23–2023/24 test indices.
- Each of the 16 original Block B fields is permuted as a whole before the complete fitted preprocessing-and-model pipeline. This automatically groups all one-hot columns derived from that field.
- Each field uses {cfg['permutation_repeats']} repeats with seed {cfg['permutation_seed']}; importance is baseline AP minus permuted AP, with negative values retained.
- Because {cfg['permutation_repeats']} repeats do not resolve tail quantiles well, permutation variability is summarised by the sample standard deviation rather than empirical 2.5th/97.5th percentiles.
- Importance measures model dependence, not a causal effect, and may be shared across correlated or overlapping fields.

## Software

```json
{json.dumps(runtime, indent=2)}
```

## Output manifest

""" + "\n".join(f"- `{item}`" for item in manifest)
    (ROOT / "reports/methods_receipt.md").write_text(receipt, encoding="utf-8")
    (ROOT / "outputs/metrics/output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_report() -> None:
    ensure_output_dirs()
    runtime_path = ROOT / "outputs/metrics/runtime_environment.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    packages = [
        "numpy", "pandas", "pyarrow", "odfpy", "scikit-learn", "xgboost",
        "scipy", "matplotlib", "joblib", "PyYAML", "pytest",
    ]
    runtime["packages"] = {
        package: importlib.metadata.version(package) for package in packages
    }
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    _write_data_archive_manifest()
    _workflow_figure(); _annual_figure(); _random_temporal_figure(); _expanding_figure()
    _block_figure(); _confusion_and_calibration(); _subgroup_figure()
    _bootstrap_figure(); _grouped_permutation_figure()
    _write_final_report(); _write_methods_receipt()
