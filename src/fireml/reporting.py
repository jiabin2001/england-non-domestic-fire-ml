from __future__ import annotations

import json
import os
import importlib.metadata
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / "data/interim/matplotlib"))

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


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"outputs/figures/{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _workflow_figure() -> None:
    labels = [
        "Official ODS\nimmutable raw file", "One-time import\nand audit", "Main cohort\n2010/11–2023/24",
        "Train / validation\nconfiguration lock", "Random and temporal\nholdout evaluation",
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
    ax.set_title("Study workflow: development decisions precede holdout evaluation", fontsize=12)
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
    ax.set_xticks(x, [MODEL_LABELS[m] for m in models]); ax.set_ylim(0, 0.75); ax.set_ylabel("PR-AUC")
    ax.set_title("Random versus temporal performance: Block B"); ax.legend(frameon=False)
    _save(fig, "03_random_vs_temporal_pr_auc")


def _expanding_figure() -> None:
    data = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(data.test_year, data.pr_auc, marker="o", color="#4c78a8", label="PR-AUC")
    ax.plot(data.test_year, data.roc_auc, marker="s", color="#54a24b", label="ROC-AUC")
    ax.set_ylim(0.5, 0.9); ax.set_ylabel("Area under curve"); ax.set_xlabel("One-year temporal test")
    ax.set_title("Expanding-window annual performance: locked Block B XGBoost"); ax.legend(frameon=False)
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
    ax.set_ylim(0, 1.05); ax.set_ylabel("PR-AUC"); ax.set_title("Information-block comparison (validation-selected family)")
    ax.legend(frameon=False)
    _save(fig, "05_information_block_comparison")


def _confusion_and_calibration() -> None:
    prediction = pd.read_parquet(ROOT / "outputs/metrics/predictions_temporal_block_B.parquet")
    lock = json.loads((ROOT / "outputs/metrics/locked_model_config.json").read_text(encoding="utf-8"))
    family = lock["selected_family_by_block"]["temporal"]["B"]
    threshold = float(lock["thresholds"]["temporal"]["B"][family])
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
    ax.set_xlim(0, 0.85); ax.set_xlabel("PR-AUC (bars) / positive prevalence (points)")
    ax.set_title("Major building-type subgroup performance (temporal Block B)"); ax.legend(frameon=False, loc="lower right")
    _save(fig, "08_building_type_subgroups")


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
    temporal = pd.read_csv(ROOT / "outputs/tables/temporal_validation_performance.csv")
    random = pd.read_csv(ROOT / "outputs/tables/random_validation_performance.csv")
    block = pd.read_csv(ROOT / "outputs/tables/block_comparison.csv")
    expanding = pd.read_csv(ROOT / "outputs/tables/expanding_window_performance.csv")
    sensitivity = pd.read_csv(ROOT / "outputs/tables/sensitivity_analysis_results.csv")
    stability = pd.read_csv(ROOT / "outputs/tables/random_seed_stability.csv")
    cohort = json.loads((ROOT / "outputs/metrics/cohort_receipt.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "outputs/metrics/locked_model_config.json").read_text(encoding="utf-8"))
    main_models = temporal[(temporal.block == "B") & (temporal.model != "dummy")].sort_values("pr_auc", ascending=False)
    core_family = lock["selected_family_by_block"]["temporal"]["B"]
    temporal_core = temporal[(temporal.block == "B") & (temporal.model == core_family)].iloc[0]
    random_core = random[(random.block == "B") & (random.model == core_family)].iloc[0]
    selected_temporal = block[block.design == "temporal"].set_index("block")
    rq3_gain = selected_temporal.loc["C", "pr_auc"] - selected_temporal.loc["B", "pr_auc"]
    report = f"""# Final analysis report

## Study definition

This retrospective prediction study uses the official **Other Building Fires Dataset** for England. “Non-domestic building fires” is the dissertation's analytical wording for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings and can include hotels, hostels, care homes and student halls; it is not limited to buildings without accommodation functions.

The analysis predicts incident-level final fire spread among already-recorded primary fires. It is not a causal study, annual building fire-risk model, fire-physics simulation or real-time FRS deployment tool. Block B contains retrospectively recorded incident information; Block C is a separately interpreted first-arrival prognostic model.

## Data and cohort

The official ODS was updated 22 July 2026. The raw-file SHA-256 is `560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595`. The audit found 245,207 logical incident rows across 2010/11–2025/26. The locked main period is 2010/11–2023/24; 2024/25 is excluded because Suffolk submissions are incomplete for part of that year, and 2025/26 is excluded because it spans the IRS-to-FaRDaP transition.

After year restriction, 20 exact duplicates, 3,063 late calls and 6,081 `Roofs/ Roof spaces` target records were removed sequentially. The main cohort contains {cohort['rows']:,} incidents, {cohort['positive_count']:,} larger fires ({cohort['positive_prevalence']:.1%}).

## Validation and modelling

Temporal train/validation/test years are 2010/11–2019/20, 2020/21–2021/22 and 2022/23–2023/24. The stratified random comparator has exactly the same 159,533/23,824/25,814 sample sizes. All imputing and encoding were pipeline-fitted on development data only. Compact hyperparameter selection used validation PR-AUC; analytical classification thresholds maximised validation F1. The temporal holdout was evaluated once after configuration locking.

The main model comparison is Block B, the retrospective incident-information model:

{_format_rows(main_models.assign(model=main_models.model.map(MODEL_LABELS)), ['model','n','positive_count','positive_prevalence','pr_auc','roc_auc','f1','brier_score'])}

## Research questions

### RQ1 — Random versus temporal validation

For the validation-selected Block B XGBoost, random holdout PR-AUC was {random_core.pr_auc:.3f} and temporal holdout PR-AUC was {temporal_core.pr_auc:.3f}, a random-minus-temporal difference of {random_core.pr_auc - temporal_core.pr_auc:+.3f}. Logistic Regression and Random Forest also had higher random than temporal Block B PR-AUC. Thus random splitting modestly overestimated later-year discrimination in the core retrospective scenario, although the magnitude depends on information block and should not be generalised to every deployment definition.

### RQ2 — Best later-year model

XGBoost had the highest temporal Block B PR-AUC ({temporal_core.pr_auc:.3f}), followed by Random Forest ({main_models.iloc[1].pr_auc:.3f}) and Logistic Regression ({main_models.iloc[2].pr_auc:.3f}). The margins are small relative to the much larger gain from adding information, so the result supports XGBoost within this prespecified comparison rather than a universal algorithm ranking.

### RQ3 — First-arrival information

For validation-selected families, temporal PR-AUC rose from {selected_temporal.loc['B','pr_auc']:.3f} in Block B to {selected_temporal.loc['C','pr_auc']:.3f} in Block C, an absolute gain of {rq3_gain:.3f}. `FIRE_SIZE_ON_ARRIVAL` is temporally prior to final `SPREAD_OF_FIRE`, but it is a highly proximal state variable. The Block C result is therefore first-arrival prognosis, not pre-incident building risk and not evidence of deployability before crews arrive.

## Temporal stability and sensitivity

Expanding-window annual PR-AUC ranged from {expanding.pr_auc.min():.3f} to {expanding.pr_auc.max():.3f}; 2023/24 was {expanding.iloc[-1].pr_auc:.3f}. No policy or COVID attribution is made because this design establishes performance variation, not its cause.

{_format_rows(sensitivity[['analysis','test_period','n','positive_prevalence','pr_auc','roc_auc','f1']], ['analysis','test_period','n','positive_prevalence','pr_auc','roc_auc','f1'])}

The mandatory roof-positive definition increased temporal Block B PR-AUC to {sensitivity.loc[sensitivity.analysis == 'roofs_roof_spaces_positive','pr_auc'].iloc[0]:.3f}. Reintroducing late calls produced {sensitivity.loc[sensitivity.analysis == 'include_late_calls','pr_auc'].iloc[0]:.3f}. A separate 2024/25 check excluding Suffolk produced {sensitivity.loc[sensitivity.analysis == 'include_2024_25_exclude_suffolk','pr_auc'].iloc[0]:.3f}; it remains secondary because the main time window was locked before modelling. Across three prespecified random seeds, PR-AUC ranged from {stability.pr_auc.min():.3f} to {stability.pr_auc.max():.3f}.

## Limitations

- The public file has no incident identifier or exact date/month, limiting dependence checks and finer temporal validation.
- Incident fields may reflect officer judgement; cause/ignition fields may be revised after investigation, and delay fields may be estimated.
- Block B is retrospective and not strictly dispatch-time information.
- Block C's exceptional performance is dominated by proximity to the final outcome and must remain a separate prognostic scenario.
- Subgroup PR-AUC is prevalence-sensitive; subgroup comparisons are descriptive and not evidence of differential causal effects.
- Temporal performance differences do not by themselves identify why distributions changed.

## Reproducibility

All tables, figures, fitted selected pipelines, split assignments, lock receipt, software versions and exact method decisions are saved under `outputs/` and `reports/`. From an existing ODS or Parquet cache, run `python scripts/06_build_report.py` after installing the pinned project environment.
"""
    (ROOT / "reports/final_analysis_report.md").write_text(report, encoding="utf-8")


def _write_methods_receipt() -> None:
    cfg = load_yaml("config/analysis.yaml")
    metadata = json.loads((ROOT / "data/raw/source_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    cohort = json.loads((ROOT / "outputs/metrics/cohort_receipt.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "outputs/metrics/locked_model_config.json").read_text(encoding="utf-8"))
    runtime = json.loads((ROOT / "outputs/metrics/runtime_environment.json").read_text(encoding="utf-8"))
    policy = load_yaml("config/feature_policy.yaml")
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

## Cohort and target

- Main years: {', '.join(audit['main_years'])}
- Exclusions, in order: outside main years; complete duplicate rows; `LATE_CALL=yes`; main-target exclusions (`Roofs/ Roof spaces` and any unmappable/missing category).
- Final rows: {cohort['rows']}; positives: {cohort['positive_count']}; prevalence: {cohort['positive_prevalence']:.8f}.
- Exact target mapping: `{json.dumps(cohort['target_mapping'])}`.
- Roof sensitivity: map `Roofs/ Roof spaces` to 1.

## Feature blocks

- Block A — structural and context: `{audit['feature_blocks']['A']}`
- Block B — retrospective incident information: `{audit['feature_blocks']['B']}`
- Block C — first-arrival prognostic: `{audit['feature_blocks']['C']}`
- All retained predictors are treated as categorical/banded fields. Missing/blank values become `Missing/Unknown`; one-hot encoding uses `handle_unknown='ignore'`. No rare-category merger was required because the largest field has 82 disclosed categories and sparse one-hot encoding remained tractable.
- `RESPONSE_TIME` is used; its redundant code field is not used.
- Leakage blacklist: `{policy['leakage_blacklist']}`
- Always excluded from predictors: `{policy['always_excluded']}` plus `E_CODE_TERRITORY`.

## Validation

- Temporal train: {', '.join(audit['temporal_train_years'])}
- Temporal validation: {', '.join(audit['temporal_validation_years'])}
- Temporal test: {', '.join(audit['temporal_test_years'])}
- Random comparator: stratified sampling with exactly matching train/validation/test counts.
- Primary seed: {cfg['random_seed']}; stability seeds: {cfg['random_stability_seeds']}.
- Selection metric: validation PR-AUC. Threshold: validation F1 maximum, an analytical operating point rather than an operational optimum.
- Temporal test evaluation count in the locked run: {lock['test_evaluation_count_this_run']}.

## Hyperparameters and thresholds

- Full candidate ranges and results: `reports/hyperparameter_plan.md` and `outputs/tables/hyperparameter_search_results.csv`.
- Selected parameters: `{json.dumps(lock['selected_hyperparameters'])}`
- Validation-selected family by block: `{json.dumps(lock['selected_family_by_block'])}`
- Locked thresholds: `{json.dumps(lock['thresholds'])}`
- XGBoost device: `{lock['xgboost_device']}`; tree method: `hist`.

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
    _workflow_figure(); _annual_figure(); _random_temporal_figure(); _expanding_figure()
    _block_figure(); _confusion_and_calibration(); _subgroup_figure()
    _write_final_report(); _write_methods_receipt()
