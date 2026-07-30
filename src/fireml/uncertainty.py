from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ROOT, ensure_output_dirs, load_yaml


def stratified_resample_indices(
    y_true: np.ndarray | pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample positives and negatives separately, preserving both class counts."""
    labels = np.asarray(y_true, dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size + negative.size != labels.size:
        raise ValueError("Bootstrap labels must be binary values 0 and 1.")
    if positive.size == 0 or negative.size == 0:
        raise ValueError("PR-AUC bootstrap requires both outcome classes.")
    return np.concatenate([
        rng.choice(negative, size=negative.size, replace=True),
        rng.choice(positive, size=positive.size, replace=True),
    ])


def stratified_bootstrap_pr_auc(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return stratified bootstrap draws of test-set average precision."""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probability, dtype=float)
    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have identical shapes.")
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive.")
    draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = stratified_resample_indices(labels, rng)
        draws[repeat] = average_precision_score(labels[indices], probabilities[indices])
    return draws


def build_bootstrap_ci_table(
    random_predictions: pd.DataFrame,
    temporal_predictions: pd.DataFrame,
    repeats: int = 2000,
    seed: int = 20260811,
) -> pd.DataFrame:
    """Summarise fixed-model PR-AUC uncertainty for the two independent holdouts."""
    required = {"LARGER_FIRE", "probability"}
    for design, frame in (("random", random_predictions), ("temporal", temporal_predictions)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{design} predictions missing columns: {sorted(missing)}")

    child_seeds = np.random.SeedSequence(seed).spawn(2)
    random_rng, temporal_rng = (np.random.default_rng(child) for child in child_seeds)
    random_y = random_predictions["LARGER_FIRE"].to_numpy(dtype=int, copy=True)
    random_probability = random_predictions["probability"].to_numpy(dtype=float, copy=True)
    temporal_y = temporal_predictions["LARGER_FIRE"].to_numpy(dtype=int, copy=True)
    temporal_probability = temporal_predictions["probability"].to_numpy(dtype=float, copy=True)
    random_draws = stratified_bootstrap_pr_auc(
        random_y, random_probability, repeats, random_rng
    )
    temporal_draws = stratified_bootstrap_pr_auc(
        temporal_y, temporal_probability, repeats, temporal_rng
    )

    random_point = float(average_precision_score(random_y, random_probability))
    temporal_point = float(average_precision_score(temporal_y, temporal_probability))
    difference_draws = random_draws - temporal_draws

    def row(
        estimand: str,
        design: str,
        point: float,
        draws: np.ndarray,
        method: str,
    ) -> dict[str, str | int | float]:
        lower, upper = np.quantile(draws, [0.025, 0.975])
        return {
            "estimand": estimand,
            "design": design,
            "block": "B",
            "model": "validation-selected XGBoost",
            "point_estimate": point,
            "ci_lower_95": float(lower),
            "ci_upper_95": float(upper),
            "bootstrap_repeats": repeats,
            "bootstrap_seed": seed,
            "bootstrap_method": method,
        }

    return pd.DataFrame([
        row(
            "PR-AUC",
            "random",
            random_point,
            random_draws,
            "stratified percentile bootstrap",
        ),
        row(
            "PR-AUC",
            "temporal",
            temporal_point,
            temporal_draws,
            "stratified percentile bootstrap",
        ),
        row(
            "random-minus-temporal PR-AUC difference",
            "random-minus-temporal",
            random_point - temporal_point,
            difference_draws,
            "conditional independent stratified percentile bootstrap",
        ),
    ])


def add_pr_auc_prevalence_context(performance: pd.DataFrame) -> pd.DataFrame:
    """Add prevalence-relative PR-AUC summaries without changing primary metrics."""
    result = performance.copy(deep=True)
    prevalence = result["positive_prevalence"].astype(float)
    pr_auc = result["pr_auc"].astype(float)
    result["pr_auc_baseline"] = prevalence
    result["pr_auc_absolute_lift"] = pr_auc - prevalence
    denominator = 1.0 - prevalence
    result["normalized_pr_auc"] = np.divide(
        result["pr_auc_absolute_lift"],
        denominator,
        out=np.full(len(result), np.nan, dtype=float),
        where=denominator.to_numpy() != 0,
    )
    return result


def build_pr_auc_prevalence_context(
    random_performance: pd.DataFrame,
    temporal_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact context table for the main Block B model families."""
    combined = pd.concat([random_performance, temporal_performance], ignore_index=True)
    combined = combined[
        (combined["block"] == "B")
        & combined["model"].isin(("logistic_regression", "random_forest", "xgboost"))
    ].copy()
    contextual = add_pr_auc_prevalence_context(combined)
    columns = [
        "design", "block", "model", "n", "positive_count",
        "positive_prevalence", "pr_auc", "pr_auc_baseline",
        "pr_auc_absolute_lift", "normalized_pr_auc", "roc_auc", "brier_score",
    ]
    return contextual[columns].sort_values(["design", "model"]).reset_index(drop=True)


def run_uncertainty_analysis() -> dict[str, pd.DataFrame]:
    """Generate bootstrap and prevalence-context tables from fixed test outputs."""
    ensure_output_dirs()
    cfg = load_yaml("config/analysis.yaml")
    random_predictions = pd.read_parquet(
        ROOT / "outputs/metrics/predictions_random_block_B.parquet"
    )
    temporal_predictions = pd.read_parquet(
        ROOT / "outputs/metrics/predictions_temporal_block_B.parquet"
    )
    bootstrap = build_bootstrap_ci_table(
        random_predictions,
        temporal_predictions,
        repeats=int(cfg["bootstrap_repeats"]),
        seed=int(cfg["bootstrap_seed"]),
    )
    bootstrap.to_csv(
        ROOT / "outputs/tables/bootstrap_confidence_intervals.csv", index=False
    )

    random_performance = pd.read_csv(
        ROOT / "outputs/tables/random_validation_performance.csv"
    )
    temporal_performance = pd.read_csv(
        ROOT / "outputs/tables/temporal_validation_performance.csv"
    )
    prevalence_context = build_pr_auc_prevalence_context(
        random_performance, temporal_performance
    )
    prevalence_context.to_csv(
        ROOT / "outputs/tables/pr_auc_prevalence_context.csv", index=False
    )
    return {"bootstrap": bootstrap, "prevalence_context": prevalence_context}
