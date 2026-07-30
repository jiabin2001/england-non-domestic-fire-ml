from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ROOT, ensure_output_dirs, load_yaml
from .features import resolve_blocks


def load_temporal_test_data(
    cohort_path: Path | None = None,
    assignments_path: Path | None = None,
    predictions_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load Block B test data and verify it against saved split and prediction records."""
    cohort = pd.read_parquet(
        cohort_path or ROOT / "data/processed/analysis_cohort.parquet"
    )
    assignments = pd.read_csv(
        assignments_path or ROOT / "outputs/tables/split_assignments.csv"
    )
    predictions = pd.read_parquet(
        predictions_path or ROOT / "outputs/metrics/predictions_temporal_block_B.parquet"
    )
    assigned_test = assignments[
        (assignments["design"] == "temporal")
        & (assignments["partition"] == "test")
    ].copy()
    indices = assigned_test["cohort_index"].to_numpy(dtype=int)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Temporal test assignments are empty or contain duplicate indices.")
    if not np.array_equal(predictions.index.to_numpy(dtype=int), indices):
        raise ValueError("Saved temporal predictions do not match split-assignment indices.")

    test = cohort.loc[indices]
    checks = (
        np.array_equal(
            test["SOURCE_ROW_ID"].to_numpy(dtype=int),
            assigned_test["source_row_id"].to_numpy(dtype=int),
        ),
        np.array_equal(
            test["FINANCIAL_YEAR"].astype(str).to_numpy(),
            assigned_test["financial_year"].astype(str).to_numpy(),
        ),
        np.array_equal(
            test["LARGER_FIRE"].to_numpy(dtype=int),
            assigned_test["larger_fire"].to_numpy(dtype=int),
        ),
        np.array_equal(
            test["SOURCE_ROW_ID"].to_numpy(dtype=int),
            predictions["SOURCE_ROW_ID"].to_numpy(dtype=int),
        ),
    )
    if not all(checks):
        raise ValueError("Cohort, split assignments and temporal predictions are misaligned.")

    features = resolve_blocks(cohort.columns)["B"]
    x_test = test[features].copy(deep=True)
    y_test = test["LARGER_FIRE"].to_numpy(dtype=int, copy=True)
    saved_probability = predictions["probability"].to_numpy(dtype=float, copy=True)
    return x_test, y_test, saved_probability, indices


def grouped_permutation_importance(
    model: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    repeats: int = 30,
    seed: int = 20260821,
    baseline_probability: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Permute one original input field at a time and score the complete pipeline."""
    if repeats < 1:
        raise ValueError("Permutation repeats must be positive.")
    labels = np.asarray(y_test, dtype=int)
    if labels.size != len(x_test):
        raise ValueError("Features and labels must have the same number of rows.")
    if baseline_probability is None:
        baseline = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    else:
        baseline = np.asarray(baseline_probability, dtype=float)
    if baseline.shape != labels.shape:
        raise ValueError("Baseline probabilities must align with labels.")
    baseline_pr_auc = float(average_precision_score(labels, baseline))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int | float]] = []

    for feature in x_test.columns:
        original_values = x_test[feature].to_numpy(copy=True)
        working = x_test.copy(deep=True)
        permuted_scores = np.empty(repeats, dtype=float)
        for repeat in range(repeats):
            working[feature] = original_values[rng.permutation(len(original_values))]
            probability = np.asarray(model.predict_proba(working)[:, 1], dtype=float)
            permuted_scores[repeat] = average_precision_score(labels, probability)
        decreases = baseline_pr_auc - permuted_scores
        lower, upper = np.quantile(decreases, [0.025, 0.975])
        rows.append({
            "feature": feature,
            "baseline_pr_auc": baseline_pr_auc,
            "mean_permuted_pr_auc": float(permuted_scores.mean()),
            "mean_pr_auc_decrease": float(decreases.mean()),
            "std_pr_auc_decrease": float(decreases.std(ddof=1)) if repeats > 1 else 0.0,
            "permutation_p02_5": float(lower),
            "permutation_p97_5": float(upper),
            "permutation_repeats": repeats,
            "permutation_seed": seed,
            "design": "temporal",
            "block": "B",
            "model": "validation-selected XGBoost",
            "evaluation_set": "2022/23-2023/24 temporal test set",
        })
    return pd.DataFrame(rows).sort_values(
        "mean_pr_auc_decrease", ascending=False
    ).reset_index(drop=True)


def run_grouped_permutation_analysis() -> pd.DataFrame:
    """Explain the fixed Temporal Block B XGBoost pipeline on its saved test set."""
    ensure_output_dirs()
    cfg = load_yaml("config/analysis.yaml")
    pre_test_path = ROOT / "outputs/metrics/pre_test_model_config.json"
    pre_test = json.loads(pre_test_path.read_text(encoding="utf-8"))
    selected = pre_test["selected_family_by_block"]["temporal"]["B"]
    if selected != "xgboost":
        raise ValueError(
            f"Temporal Block B selected family is {selected!r}, not the required XGBoost."
        )

    x_test, y_test, saved_probability, _ = load_temporal_test_data()
    model = joblib.load(
        ROOT / "outputs/models/best_temporal_block_B_xgboost.joblib"
    )
    # The fitted trees are unchanged; CPU inference matches the sparse CPU input and
    # avoids XGBoost's much slower cross-device DMatrix fallback during 480 predictions.
    model.set_params(model__device="cpu")
    model_features = list(model.feature_names_in_)
    if model_features != list(x_test.columns):
        raise ValueError("Saved pipeline input fields do not match the current Block B policy.")
    reproduced_probability = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    if not np.allclose(reproduced_probability, saved_probability, rtol=1e-6, atol=5e-7):
        raise ValueError("Saved model no longer reproduces the fixed temporal test predictions.")

    importance = grouped_permutation_importance(
        model,
        x_test,
        y_test,
        repeats=int(cfg["permutation_repeats"]),
        seed=int(cfg["permutation_seed"]),
        baseline_probability=saved_probability,
    )
    importance.to_csv(
        ROOT / "outputs/tables/grouped_permutation_importance.csv", index=False
    )
    return importance
