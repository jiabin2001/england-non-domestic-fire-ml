from __future__ import annotations

import json
import platform
import subprocess
import time
import warnings
from typing import Any

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import ROOT, ensure_output_dirs, load_yaml
from .evaluation import choose_f1_threshold, classification_metrics
from .features import assert_no_leakage, resolve_blocks
from .preprocessing import make_catboost_preprocessor, make_preprocessor
from .splits import assignment_frame, make_random_split_like, make_temporal_split


FAMILIES = ("logistic_regression", "random_forest", "xgboost", "catboost")


def detect_xgb_device() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return "cuda" if result.returncode == 0 and result.stdout.strip() else "cpu"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "cpu"


def resolve_xgb_device(configured: str) -> str:
    """Resolve an explicit device, with a visible auto-detection fallback."""
    configured = configured.lower()
    if configured not in {"cpu", "cuda", "auto"}:
        raise ValueError("xgboost_device must be one of: cpu, cuda, auto.")
    if configured == "cpu":
        return "cpu"
    detected = detect_xgb_device()
    if configured == "cuda" and detected != "cuda":
        raise RuntimeError("analysis.yaml requires CUDA, but no NVIDIA GPU was detected.")
    if configured == "auto":
        warnings.warn(
            f"xgboost_device=auto selected {detected}; hardware can change fitted results.",
            RuntimeWarning,
            stacklevel=2,
        )
        return detected
    return configured


def candidate_grid() -> dict[str, list[dict[str, Any]]]:
    return {
        "logistic_regression": [
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ],
        "random_forest": [
            {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1, "max_features": "sqrt"},
        ],
        "xgboost": [
            {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0},
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0},
            {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0},
            {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        ],
        "catboost": [
            {"iterations": 100, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3.0},
            {"iterations": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
            {"iterations": 200, "learning_rate": 0.1, "depth": 4, "l2_leaf_reg": 3.0},
            {"iterations": 200, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 10.0},
        ],
    }


def default_config(family: str) -> dict[str, Any]:
    return candidate_grid()[family][1 if family == "logistic_regression" else 0]


def make_estimator(family: str, parameters: dict[str, Any], seed: int, n_jobs: int, device: str):
    if family == "dummy":
        return DummyClassifier(strategy="prior", random_state=seed)
    if family == "logistic_regression":
        return LogisticRegression(
            C=float(parameters["C"]), solver="lbfgs", max_iter=500,
            random_state=seed,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            **parameters, random_state=seed, n_jobs=n_jobs,
        )
    if family == "xgboost":
        return XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if family == "catboost":
        return CatBoostClassifier(
            **parameters,
            cat_features=None,
            loss_function="Logloss",
            random_seed=seed,
            thread_count=n_jobs,
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,
        )
    raise KeyError(family)


def make_model_pipeline(columns: list[str], family: str, parameters: dict[str, Any], seed: int, n_jobs: int, device: str) -> Pipeline:
    assert_no_leakage({"active": columns})
    preprocessor = (
        make_catboost_preprocessor(columns)
        if family == "catboost"
        else make_preprocessor(columns)
    )
    estimator = make_estimator(family, parameters, seed, n_jobs, device)
    if family == "catboost":
        estimator.set_params(cat_features=columns)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def _fit_validation(
    frame: pd.DataFrame,
    split: dict[str, np.ndarray],
    columns: list[str],
    family: str,
    parameters: dict[str, Any],
    seed: int,
    n_jobs: int,
    device: str,
) -> tuple[Pipeline, np.ndarray, dict]:
    pipeline = make_model_pipeline(columns, family, parameters, seed, n_jobs, device)
    start = time.perf_counter()
    pipeline.fit(frame.loc[split["train"], columns], frame.loc[split["train"], "LARGER_FIRE"])
    seconds = time.perf_counter() - start
    probability = pipeline.predict_proba(frame.loc[split["validation"], columns])[:, 1]
    threshold = choose_f1_threshold(frame.loc[split["validation"], "LARGER_FIRE"].to_numpy(), probability)
    metrics = classification_metrics(frame.loc[split["validation"], "LARGER_FIRE"].to_numpy(), probability, threshold)
    metrics["fit_seconds"] = float(seconds)
    return pipeline, probability, metrics


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_hyperparameter_plan(baseline: pd.DataFrame, grid: dict[str, list[dict[str, Any]]], device: str) -> None:
    baseline_view = baseline[["design", "model", "pr_auc", "fit_seconds"]]
    baseline_lines = ["| design | model | validation average precision | fit seconds |", "|---|---|---:|---:|"]
    baseline_lines.extend(
        f"| {row.design} | {row.model} | {row.pr_auc:.4f} | {row.fit_seconds:.2f} |"
        for row in baseline_view.itertuples(index=False)
    )
    baseline_text = "\n".join(baseline_lines)
    report = f"""# Hyperparameter plan

## Status and selection boundary

Candidate configurations are deliberately compact and are selected only by validation-set average precision, calculated with scikit-learn's `average_precision_score`. Holdout performance does not alter the candidate set. The selected configuration is a practical comparison setting, not a claim of theoretical optimality.

XGBoost runtime device: `{device}` (`hist` tree method). CUDA is used only when `nvidia-smi` confirms an available GPU; otherwise execution falls back to CPU. CatBoost is fixed to CPU because its GPU training is non-deterministic; this keeps estimator randomness fixed during the split-assignment sensitivity analysis.

## Baseline observations

{baseline_text}

## Logistic Regression

- `C` controls inverse L2 regularisation strength. scikit-learn's default is 1.0.
- Candidates: 0.1, 1.0 and 10.0. They cover stronger, default and weaker regularisation on a log scale without changing the algorithm or feature information.
- `penalty='l2'`, `solver='lbfgs'` and `max_iter=500` remain fixed. The larger iteration limit is convergence control, not model-complexity tuning.
- This compact range is fair because it gives the linear model meaningful regularisation flexibility while avoiding a much larger search budget than the tree models.

## Random Forest

- `n_estimators` controls Monte Carlo stability; software default is 100. Candidates use 100–300 trees.
- `max_depth` controls tree depth; default is unlimited (`None`). One capped value (20) tests a materially simpler forest.
- `min_samples_leaf` controls terminal-node regularisation; default is 1. Values 1 and 5 compare default with smoother leaves.
- `max_features` remains at the software classification default, `sqrt`, so feature subsampling is not separately optimised.
- Four joint configurations test default, more trees, larger leaves and capped depth. More configurations could favour this family merely by search volume, so the budget is intentionally limited.

## XGBoost

- The scikit-learn wrapper exposes `None` for many constructor defaults; the default-equivalent explicit configuration used here is 100 trees, learning rate 0.3, depth 6, minimum child weight 1, and full row/column sampling.
- `n_estimators` and `learning_rate` control boosting length and shrinkage. Configurations compare 100×0.3 with 200–300×0.1.
- `max_depth` controls interaction complexity; depth 3 is compared with the default-equivalent depth 6.
- `min_child_weight` regularises small child nodes; 1 and 5 are compared.
- `subsample` and `colsample_bytree` are 1.0 by default; a single 0.8/0.8 configuration checks moderate stochastic regularisation.
- Four joint configurations keep XGBoost's search budget equal to Random Forest's and avoid a large optimisation exercise.

## CatBoost

- CatBoost receives the original categorical fields directly after train-safe conversion of missing values to `Missing/Unknown`; it does not receive one-hot encoded features.
- Four joint configurations vary boosting length/shrinkage, tree depth and L2 leaf regularisation while keeping the search budget equal to Random Forest and XGBoost.
- Training is fixed to CPU with `loss_function='Logloss'`, the common estimator seed and four threads. The validation-selection metric remains external non-interpolated average precision, exactly as for the other families.

## Candidate configurations

```json
{json.dumps(grid, indent=2)}
```

## Stability rule

All candidate validation AP values are retained. The selected configuration's margin over adjacent candidates is reported in `hyperparameter_stability_summary.csv`. Expanding-window and sensitivity analyses reuse the selected configuration.
"""
    (ROOT / "reports/hyperparameter_plan.md").write_text(report, encoding="utf-8")


def run_core_models() -> dict[str, Any]:
    ensure_output_dirs()
    cfg = load_yaml("config/analysis.yaml")
    frame = pd.read_parquet(ROOT / cfg["cohort_path"])
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    blocks = resolve_blocks(frame.columns)
    seed, n_jobs = int(cfg["random_seed"]), int(cfg["n_jobs"])
    device = resolve_xgb_device(str(cfg.get("xgboost_device", "auto")))
    temporal = make_temporal_split(
        frame, audit["temporal_train_years"], audit["temporal_validation_years"], audit["temporal_test_years"]
    )
    random = make_random_split_like(frame, temporal, seed)
    splits = {"temporal": temporal, "random": random}
    assignment_frame(frame, splits).to_csv(ROOT / "outputs/tables/split_assignments.csv", index=False)

    # Baselines on the prespecified central Block B, using validation only.
    baseline_rows = []
    for design, split in splits.items():
        for family in ("dummy",) + FAMILIES:
            params = {} if family == "dummy" else default_config(family)
            _, _, metrics = _fit_validation(frame, split, blocks["B"], family, params, seed, n_jobs, device)
            baseline_rows.append({"design": design, "block": "B", "model": family, **metrics, "parameters": json.dumps(params)})
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(ROOT / "outputs/tables/baseline_validation_performance.csv", index=False)
    grid = candidate_grid()
    write_hyperparameter_plan(baseline, grid, device)

    # Formal compact selection on Block B only; no test labels are consulted.
    search_rows = []
    for design, split in splits.items():
        for family in FAMILIES:
            for candidate_id, params in enumerate(grid[family], start=1):
                _, _, metrics = _fit_validation(frame, split, blocks["B"], family, params, seed, n_jobs, device)
                search_rows.append({
                    "design": design, "block": "B", "model": family,
                    "candidate_id": candidate_id, "parameters": json.dumps(params), **metrics,
                })
    search = pd.DataFrame(search_rows)
    search.to_csv(ROOT / "outputs/tables/hyperparameter_search_results.csv", index=False)
    chosen: dict[str, dict[str, dict[str, Any]]] = {}
    for design in splits:
        chosen[design] = {}
        for family in FAMILIES:
            subset = search[(search["design"] == design) & (search["model"] == family)]
            best = subset.sort_values(["pr_auc", "candidate_id"], ascending=[False, True]).iloc[0]
            chosen[design][family] = json.loads(best["parameters"])

    # With family hyperparameters fixed, obtain block-specific validation metrics,
    # analytical F1 thresholds and the validation-selected family for each block.
    development_rows = []
    selected_by_block: dict[str, dict[str, str]] = {}
    for design, split in splits.items():
        selected_by_block[design] = {}
        for block, columns in blocks.items():
            for family in ("dummy",) + FAMILIES:
                params = {} if family == "dummy" else chosen[design][family]
                _, _, metrics = _fit_validation(frame, split, columns, family, params, seed, n_jobs, device)
                development_rows.append({
                    "design": design, "block": block, "model": family,
                    "parameters": json.dumps(params), **metrics,
                })
            candidates = [row for row in development_rows if row["design"] == design and row["block"] == block and row["model"] != "dummy"]
            selected_by_block[design][block] = max(candidates, key=lambda row: row["pr_auc"])["model"]
    development = pd.DataFrame(development_rows)
    development.to_csv(ROOT / "outputs/tables/development_validation_performance.csv", index=False)

    selection = {
        "record_type": "model_selection_record",
        "selection_metric": "validation average precision (sklearn average_precision_score)",
        "threshold_rule": "maximum validation F1",
        "xgboost_device": device,
        "catboost_task_type": "CPU",
        "rq1_comparator_family": selected_by_block["temporal"]["B"],
        "selected_hyperparameters": chosen,
        "selected_family_by_block": selected_by_block,
        "validation_thresholds": {
            design: {
                block: {
                    row["model"]: row["threshold"]
                    for row in development_rows if row["design"] == design and row["block"] == block
                }
                for block in blocks
            }
            for design in splits
        },
        "seed": seed,
        "split_years": {
            "temporal_train": audit["temporal_train_years"],
            "temporal_validation": audit["temporal_validation_years"],
            "temporal_test": audit["temporal_test_years"],
            "random_comparator": "target-stratified with temporal-matched partition sizes",
        },
    }
    selection_path = ROOT / "outputs/metrics/model_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, default=_json_ready), encoding="utf-8"
    )
    # Remove deprecated within-run receipt files; the selection record contains only
    # the configuration required to reproduce the fitted models and thresholds.
    (ROOT / "outputs/metrics/pre_test_model_config.json").unlink(missing_ok=True)
    (ROOT / "outputs/metrics/post_test_evaluation_receipt.json").unlink(missing_ok=True)
    (ROOT / "outputs/metrics/locked_model_config.json").unlink(missing_ok=True)

    for pattern in ("best_*.joblib", "rq1_*.joblib"):
        for stale_model in (ROOT / "outputs/models").glob(pattern):
            stale_model.unlink()

    # Validation chooses hyperparameters, family and threshold. The test model remains
    # train-fitted so the validation-derived threshold applies to the same fitted model.
    test_rows = []
    for design, split in splits.items():
        for block, columns in blocks.items():
            for family in ("dummy",) + FAMILIES:
                params = {} if family == "dummy" else chosen[design][family]
                pipeline = make_model_pipeline(columns, family, params, seed, n_jobs, device)
                start = time.perf_counter()
                pipeline.fit(
                    frame.loc[split["train"], columns],
                    frame.loc[split["train"], "LARGER_FIRE"],
                )
                seconds = time.perf_counter() - start
                probability = pipeline.predict_proba(frame.loc[split["test"], columns])[:, 1]
                threshold = float(selection["validation_thresholds"][design][block][family])
                metrics = classification_metrics(frame.loc[split["test"], "LARGER_FIRE"].to_numpy(), probability, threshold)
                row = {
                    "design": design, "split_role": "test", "block": block, "model": family,
                    "parameters": json.dumps(params), "fit_seconds": float(seconds), **metrics,
                }
                test_rows.append(row)
                selected_for_design = family == selected_by_block[design][block]
                rq1_comparator = block == "B" and family == selection["rq1_comparator_family"]
                if selected_for_design:
                    model_path = ROOT / f"outputs/models/best_{design}_block_{block}_{family}.joblib"
                    joblib.dump(pipeline, model_path, compress=3)
                if rq1_comparator and not selected_for_design:
                    model_path = ROOT / f"outputs/models/rq1_{design}_block_B_{family}.joblib"
                    joblib.dump(pipeline, model_path, compress=3)
                if rq1_comparator or (block != "B" and selected_for_design):
                    prediction = frame.loc[split["test"], ["SOURCE_ROW_ID", "FINANCIAL_YEAR", "BUILDING_TYPE", "LARGER_FIRE"]].copy()
                    prediction["probability"] = probability
                    prediction["prediction"] = (probability >= threshold).astype("int8")
                    prediction.to_parquet(ROOT / f"outputs/metrics/predictions_{design}_block_{block}.parquet", index=True)
    performance = pd.DataFrame(test_rows)
    random_perf = performance[performance["design"] == "random"].reset_index(drop=True)
    temporal_perf = performance[performance["design"] == "temporal"].reset_index(drop=True)
    random_perf.to_csv(ROOT / "outputs/tables/random_validation_performance.csv", index=False)
    temporal_perf.to_csv(ROOT / "outputs/tables/temporal_validation_performance.csv", index=False)

    keys = ["block", "model"]
    metric_columns = ["pr_auc", "roc_auc", "recall", "precision", "f1", "balanced_accuracy", "brier_score"]
    comparison = random_perf[keys + metric_columns].merge(
        temporal_perf[keys + metric_columns], on=keys, suffixes=("_random", "_temporal")
    )
    for metric in metric_columns:
        comparison[f"{metric}_random_minus_temporal"] = comparison[f"{metric}_random"] - comparison[f"{metric}_temporal"]
    comparison.to_csv(ROOT / "outputs/tables/random_to_temporal_difference.csv", index=False)

    block_rows = []
    for design in splits:
        for block in blocks:
            family = selected_by_block[design][block]
            selected = performance[(performance["design"] == design) & (performance["block"] == block) & (performance["model"] == family)].iloc[0]
            block_rows.append(selected.to_dict())
    pd.DataFrame(block_rows).to_csv(ROOT / "outputs/tables/block_comparison.csv", index=False)

    stability = search.copy()
    stability["best_pr_auc"] = stability.groupby(["design", "model"])["pr_auc"].transform("max")
    stability["delta_from_best_pr_auc"] = stability["pr_auc"] - stability["best_pr_auc"]
    stability["selected"] = np.isclose(stability["pr_auc"], stability["best_pr_auc"])
    stability.to_csv(ROOT / "outputs/tables/hyperparameter_stability_summary.csv", index=False)

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "xgboost_device": device,
        "catboost_task_type": "CPU",
        "n_jobs": n_jobs,
    }
    (ROOT / "outputs/metrics/runtime_environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return {
        "selection": selection,
        "blocks": blocks,
        "splits": splits,
        "performance": performance,
    }
