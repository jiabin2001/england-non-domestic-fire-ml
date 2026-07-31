from __future__ import annotations

import json

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
        raise ValueError("Average-precision bootstrap requires both outcome classes.")
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


def holdout_overlap_summary(
    random_predictions: pd.DataFrame,
    temporal_predictions: pd.DataFrame,
) -> dict[str, str | int | float]:
    """Count random/temporal test overlap and cross-check both saved identifiers."""
    if "SOURCE_ROW_ID" in random_predictions and "SOURCE_ROW_ID" in temporal_predictions:
        random_ids = pd.Index(random_predictions["SOURCE_ROW_ID"])
        temporal_ids = pd.Index(temporal_predictions["SOURCE_ROW_ID"])
        if not random_ids.is_unique or not temporal_ids.is_unique:
            raise ValueError("SOURCE_ROW_ID must be unique within each saved test set.")
        overlap_n = len(random_ids.intersection(temporal_ids))
        identifier = "SOURCE_ROW_ID (cross-checked against cohort index)"
        random_index = pd.Index(random_predictions.index)
        temporal_index = pd.Index(temporal_predictions.index)
        if not random_index.is_unique or not temporal_index.is_unique:
            raise ValueError("Cohort index must be unique within each saved test set.")
        index_overlap_n = len(random_index.intersection(temporal_index))
        if index_overlap_n != overlap_n:
            raise ValueError("SOURCE_ROW_ID and cohort-index overlap counts disagree.")
    else:
        random_index = pd.Index(random_predictions.index)
        temporal_index = pd.Index(temporal_predictions.index)
        if not random_index.is_unique or not temporal_index.is_unique:
            raise ValueError("Cohort index must be unique within each saved test set.")
        overlap_n = len(random_index.intersection(temporal_index))
        identifier = "cohort index"

    random_n, temporal_n = len(random_predictions), len(temporal_predictions)
    if random_n == 0 or temporal_n == 0:
        raise ValueError("Saved test prediction sets must be non-empty.")
    return {
        "random_test_n": random_n,
        "temporal_test_n": temporal_n,
        "overlap_n": overlap_n,
        "overlap_fraction_random_test": overlap_n / random_n,
        "overlap_fraction_temporal_test": overlap_n / temporal_n,
        "overlap_identifier": identifier,
    }


def partially_paired_bootstrap_pr_auc(
    random_predictions: pd.DataFrame,
    temporal_predictions: pd.DataFrame,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap two partially overlapping holdouts while preserving their covariance.

    Outcome class and overlap membership are treated as fixed strata. Records shared
    by both holdouts are resampled once per replicate and used in both scores; the
    design-specific records are resampled independently.
    """
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive.")
    has_source_id = (
        "SOURCE_ROW_ID" in random_predictions.columns
        and "SOURCE_ROW_ID" in temporal_predictions.columns
    )
    identifier = "SOURCE_ROW_ID" if has_source_id else None
    random_ids = pd.Index(
        random_predictions[identifier] if identifier else random_predictions.index
    )
    temporal_ids = pd.Index(
        temporal_predictions[identifier] if identifier else temporal_predictions.index
    )
    if not random_ids.is_unique or not temporal_ids.is_unique:
        raise ValueError("Holdout identifiers must be unique within each design.")

    common_ids = random_ids.intersection(temporal_ids, sort=False)
    random_overlap = random_ids.get_indexer(common_ids)
    temporal_overlap = temporal_ids.get_indexer(common_ids)
    random_y = random_predictions["LARGER_FIRE"].to_numpy(dtype=int, copy=True)
    temporal_y = temporal_predictions["LARGER_FIRE"].to_numpy(dtype=int, copy=True)
    if not np.array_equal(random_y[random_overlap], temporal_y[temporal_overlap]):
        raise ValueError("Outcome labels disagree for records shared by both holdouts.")

    random_unique_mask = ~random_ids.isin(common_ids)
    temporal_unique_mask = ~temporal_ids.isin(common_ids)
    strata: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for label in (0, 1):
        overlap_label = random_y[random_overlap] == label
        random_shared = random_overlap[overlap_label]
        temporal_shared = temporal_overlap[overlap_label]
        random_only = np.flatnonzero(random_unique_mask & (random_y == label))
        temporal_only = np.flatnonzero(temporal_unique_mask & (temporal_y == label))
        if random_shared.size + random_only.size == 0:
            raise ValueError(f"Random holdout has no observations for class {label}.")
        if temporal_shared.size + temporal_only.size == 0:
            raise ValueError(f"Temporal holdout has no observations for class {label}.")
        strata.append((random_shared, temporal_shared, random_only, temporal_only))

    random_probability = random_predictions["probability"].to_numpy(dtype=float, copy=True)
    temporal_probability = temporal_predictions["probability"].to_numpy(dtype=float, copy=True)
    random_draws = np.empty(repeats, dtype=float)
    temporal_draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        random_parts: list[np.ndarray] = []
        temporal_parts: list[np.ndarray] = []
        for random_shared, temporal_shared, random_only, temporal_only in strata:
            if random_shared.size:
                shared_draw = rng.integers(0, random_shared.size, size=random_shared.size)
                random_parts.append(random_shared[shared_draw])
                temporal_parts.append(temporal_shared[shared_draw])
            if random_only.size:
                random_parts.append(
                    random_only[rng.integers(0, random_only.size, size=random_only.size)]
                )
            if temporal_only.size:
                temporal_parts.append(
                    temporal_only[rng.integers(0, temporal_only.size, size=temporal_only.size)]
                )
        random_indices = np.concatenate(random_parts)
        temporal_indices = np.concatenate(temporal_parts)
        random_draws[repeat] = average_precision_score(
            random_y[random_indices], random_probability[random_indices]
        )
        temporal_draws[repeat] = average_precision_score(
            temporal_y[temporal_indices], temporal_probability[temporal_indices]
        )
    return random_draws, temporal_draws


def build_bootstrap_ci_table(
    random_predictions: pd.DataFrame,
    temporal_predictions: pd.DataFrame,
    repeats: int = 100000,
    seed: int = 20260731,
    model_label: str = "validation-selected model",
) -> pd.DataFrame:
    """Summarise fixed-model average-precision uncertainty for overlapping holdouts."""
    required = {"LARGER_FIRE", "probability"}
    for design, frame in (("random", random_predictions), ("temporal", temporal_predictions)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{design} predictions missing columns: {sorted(missing)}")

    overlap = holdout_overlap_summary(random_predictions, temporal_predictions)
    rng = np.random.default_rng(seed)
    random_y = random_predictions["LARGER_FIRE"].to_numpy(dtype=int, copy=True)
    random_probability = random_predictions["probability"].to_numpy(dtype=float, copy=True)
    temporal_y = temporal_predictions["LARGER_FIRE"].to_numpy(dtype=int, copy=True)
    temporal_probability = temporal_predictions["probability"].to_numpy(dtype=float, copy=True)
    random_draws, temporal_draws = partially_paired_bootstrap_pr_auc(
        random_predictions, temporal_predictions, repeats, rng
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
            "model": model_label,
            "point_estimate": point,
            "ci_lower_95": float(lower),
            "ci_upper_95": float(upper),
            "bootstrap_repeats": repeats,
            "bootstrap_seed": seed,
            "bootstrap_method": method,
            **overlap,
        }

    return pd.DataFrame([
        row(
            "average precision",
            "random",
            random_point,
            random_draws,
            "partially paired class-and-membership-stratified percentile bootstrap",
        ),
        row(
            "average precision",
            "temporal",
            temporal_point,
            temporal_draws,
            "partially paired class-and-membership-stratified percentile bootstrap",
        ),
        row(
            "random-minus-temporal average-precision difference",
            "random-minus-temporal",
            random_point - temporal_point,
            difference_draws,
            "partially paired class-and-membership-stratified percentile bootstrap; shared records resampled jointly",
        ),
    ])


def add_pr_auc_prevalence_context(performance: pd.DataFrame) -> pd.DataFrame:
    """Add prevalence-relative average-precision summaries without changing primary metrics."""
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
        & combined["model"].ne("dummy")
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
    selection = json.loads(
        (ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8")
    )
    family = selection.get(
        "rq1_comparator_family",
        selection["selected_family_by_block"]["temporal"]["B"],
    )
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
        model_label=f"validation-selected {family}",
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
