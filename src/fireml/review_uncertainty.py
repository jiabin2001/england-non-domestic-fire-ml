"""Fixed-model bootstrap comparisons for the post-review supplementary experiments.

These functions consume predictions and return tables; they never fit models, read
research data, or write outputs. SOURCE_ROW_ID, rather than a saved DataFrame index,
defines record identity across prediction files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .uncertainty import stratified_resample_indices


METHOD_NOTES = (
    "Average precision is sklearn's non-interpolated average_precision_score. "
    "Intervals are two-sided 95% percentile bootstrap intervals conditional on the "
    "fitted models and observed class counts. Same-test model comparisons align "
    "SOURCE_ROW_ID and resample positive and negative records separately, using "
    "the same sampled records for both models; differences are new minus reference. "
    "Random-minus-temporal comparisons additionally fix overlap membership: shared "
    "records are sampled jointly within outcome class, and design-only records are "
    "sampled independently. Thus test sizes, prevalences, and overlap counts stay "
    "fixed. AP calculations group tied scores before accumulating precision; "
    "bootstrap multiplicities are exact observation weights, not score bins. "
    "Difference intervals come directly from paired differences, not from comparing "
    "separate AP intervals. No model refitting, hyperparameter selection, threshold "
    "selection, year-level resampling, or multiplicity adjustment is included. "
    "These exploratory post-review analyses use previously examined test periods "
    "and do not create a new untouched or prospective validation set."
)


def _validate_settings(repeats: int, seed: int) -> None:
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("Bootstrap repeats must be a positive integer.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer.")


def _validated_predictions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = ["SOURCE_ROW_ID", "LARGER_FIRE", "probability"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} predictions have duplicate column names.")
    if frame.empty:
        raise ValueError(f"{name} predictions must be nonempty.")
    ids = pd.Index(frame["SOURCE_ROW_ID"])
    if not ids.is_unique or ids.hasnans:
        raise ValueError(f"{name} SOURCE_ROW_ID must be unique and nonmissing.")
    labels = frame["LARGER_FIRE"]
    # Check before integer conversion: fractional labels must not become 0 or 1.
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{name} outcome labels must be binary values 0 and 1.")
    if labels.nunique() != 2:
        raise ValueError(f"{name} average-precision bootstrap requires both outcome classes.")
    try:
        probability = frame["probability"].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} probabilities must be finite numbers in [0, 1].") from exc
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"{name} probabilities must be finite numbers in [0, 1].")
    return pd.DataFrame({
        "SOURCE_ROW_ID": ids.to_numpy(copy=True),
        "LARGER_FIRE": labels.to_numpy(dtype=np.int8, copy=True),
        "probability": probability,
    })


class _WeightedAveragePrecision:
    """Cache exact score-tie groups so bootstrap replicates need no new sort."""

    def __init__(self, labels: np.ndarray, probability: np.ndarray):
        self.order = np.argsort(-probability, kind="stable")
        sorted_probability = probability[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability) != 0) + 1]
        self.sorted_labels = labels[self.order]

    def __call__(self, multiplicities: np.ndarray) -> float:
        weights = multiplicities[self.order]
        group_total = np.add.reduceat(weights, self.starts)
        group_positive = np.add.reduceat(weights * self.sorted_labels, self.starts)
        cumulative_total = np.cumsum(group_total, dtype=float)
        precision = np.divide(
            np.cumsum(group_positive, dtype=float), cumulative_total,
            out=np.zeros_like(cumulative_total), where=cumulative_total > 0,
        )
        return float(np.dot(group_positive, precision) / group_positive.sum())


def _paired_ap_draws(
    labels: np.ndarray,
    probability: np.ndarray,
    reference_probability: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    score = _WeightedAveragePrecision(labels, probability)
    reference_score = _WeightedAveragePrecision(labels, reference_probability)
    draws = np.empty(repeats)
    reference_draws = np.empty(repeats)
    for repeat in range(repeats):
        indices = stratified_resample_indices(labels, rng)
        weights = np.bincount(indices, minlength=len(labels))
        draws[repeat] = score(weights)
        reference_draws[repeat] = reference_score(weights)
    return draws, reference_draws


def _partially_paired_ap_draws(
    random: pd.DataFrame,
    temporal: pd.DataFrame,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use the original partial-pair strata/draw order, with cached weighted AP.

    Equivalence to uncertainty.partially_paired_bootstrap_pr_auc is checked on
    synthetic tied scores with disjoint, partially shared, and identical holdouts.
    This private helper expects the validated frames produced above.
    """
    random_ids = pd.Index(random["SOURCE_ROW_ID"])
    temporal_ids = pd.Index(temporal["SOURCE_ROW_ID"])
    common_ids = random_ids.intersection(temporal_ids, sort=False)
    random_overlap = random_ids.get_indexer(common_ids)
    temporal_overlap = temporal_ids.get_indexer(common_ids)
    random_y = random["LARGER_FIRE"].to_numpy()
    temporal_y = temporal["LARGER_FIRE"].to_numpy()
    if not np.array_equal(random_y[random_overlap], temporal_y[temporal_overlap]):
        raise ValueError("Outcome labels disagree for SOURCE_ROW_ID records shared by both holdouts.")
    random_unique = ~random_ids.isin(common_ids)
    temporal_unique = ~temporal_ids.isin(common_ids)
    strata = []
    for label in (0, 1):
        shared_label = random_y[random_overlap] == label
        strata.append((
            random_overlap[shared_label], temporal_overlap[shared_label],
            np.flatnonzero(random_unique & (random_y == label)),
            np.flatnonzero(temporal_unique & (temporal_y == label)),
        ))
    random_score = _WeightedAveragePrecision(random_y, random["probability"].to_numpy())
    temporal_score = _WeightedAveragePrecision(temporal_y, temporal["probability"].to_numpy())
    random_draws = np.empty(repeats)
    temporal_draws = np.empty(repeats)
    for repeat in range(repeats):
        random_parts, temporal_parts = [], []
        for random_shared, temporal_shared, random_only, temporal_only in strata:
            if random_shared.size:
                shared_draw = rng.integers(0, random_shared.size, size=random_shared.size)
                random_parts.append(random_shared[shared_draw])
                temporal_parts.append(temporal_shared[shared_draw])
            if random_only.size:
                random_parts.append(random_only[rng.integers(0, random_only.size, size=random_only.size)])
            if temporal_only.size:
                temporal_parts.append(temporal_only[rng.integers(0, temporal_only.size, size=temporal_only.size)])
        random_weights = np.bincount(np.concatenate(random_parts), minlength=len(random))
        temporal_weights = np.bincount(np.concatenate(temporal_parts), minlength=len(temporal))
        random_draws[repeat] = random_score(random_weights)
        temporal_draws[repeat] = temporal_score(temporal_weights)
    overlap = {
        "random_test_n": len(random),
        "temporal_test_n": len(temporal),
        "random_positive_count": int(random_y.sum()),
        "temporal_positive_count": int(temporal_y.sum()),
        "random_positive_prevalence": float(random_y.mean()),
        "temporal_positive_prevalence": float(temporal_y.mean()),
        "overlap_n": len(common_ids),
        "overlap_fraction_random_test": len(common_ids) / len(random),
        "overlap_fraction_temporal_test": len(common_ids) / len(temporal),
        "overlap_identifier": "SOURCE_ROW_ID",
    }
    return random_draws, temporal_draws, overlap


def _interval_row(
    *, estimand: str, design: str, model: str, reference_model: str,
    point: float, draws: np.ndarray, repeats: int, seed: int, method: str,
    **metadata,
) -> dict:
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimand": estimand, "design": design, "model": model,
        "reference_model": reference_model, "point_estimate": float(point),
        "ci_lower_95": float(lower), "ci_upper_95": float(upper),
        "bootstrap_repeats": repeats, "bootstrap_seed": seed,
        "bootstrap_method": method, **metadata,
    }


def paired_model_ap_intervals(
    new_predictions: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    *, design: str, model: str, reference_model: str,
    repeats: int = 10000, seed: int = 20260921,
) -> pd.DataFrame:
    """Return two AP intervals and the paired new-minus-reference AP interval."""
    _validate_settings(repeats, seed)
    new = _validated_predictions(new_predictions, "new model")
    reference = _validated_predictions(reference_predictions, "reference model")
    locations = pd.Index(reference["SOURCE_ROW_ID"]).get_indexer(new["SOURCE_ROW_ID"])
    if len(new) != len(reference) or (locations < 0).any():
        raise ValueError("Paired model comparison requires identical SOURCE_ROW_ID sets.")
    reference = reference.iloc[locations].reset_index(drop=True)
    labels = new["LARGER_FIRE"].to_numpy()
    if not np.array_equal(labels, reference["LARGER_FIRE"].to_numpy()):
        raise ValueError("Outcome labels disagree after aligning paired SOURCE_ROW_ID records.")
    probability = new["probability"].to_numpy()
    reference_probability = reference["probability"].to_numpy()
    draws, reference_draws = _paired_ap_draws(
        labels, probability, reference_probability, repeats, np.random.default_rng(seed)
    )
    point = float(average_precision_score(labels, probability))
    reference_point = float(average_precision_score(labels, reference_probability))
    common = {
        "design": design, "repeats": repeats, "seed": seed,
        "method": "paired class-stratified percentile bootstrap; same records resampled jointly",
        "n": len(new), "positive_count": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "paired_test_n": len(new), "overlap_identifier": "SOURCE_ROW_ID",
    }
    return pd.DataFrame([
        _interval_row(estimand="average precision", model=model, reference_model="",
                      point=point, draws=draws, **common),
        _interval_row(estimand="average precision", model=reference_model, reference_model="",
                      point=reference_point, draws=reference_draws, **common),
        _interval_row(estimand="paired new-minus-reference average-precision difference",
                      model=model, reference_model=reference_model,
                      point=point - reference_point, draws=draws - reference_draws, **common),
    ])


def split_ap_intervals(
    random_predictions: pd.DataFrame,
    temporal_predictions: pd.DataFrame,
    *, model: str, repeats: int = 10000, seed: int = 20260921,
) -> pd.DataFrame:
    """Return both design AP intervals and a partially paired random-minus-temporal interval."""
    _validate_settings(repeats, seed)
    random = _validated_predictions(random_predictions, "random")
    temporal = _validated_predictions(temporal_predictions, "temporal")
    random_draws, temporal_draws, overlap = _partially_paired_ap_draws(
        random, temporal, repeats, np.random.default_rng(seed)
    )
    random_point = float(average_precision_score(random["LARGER_FIRE"], random["probability"]))
    temporal_point = float(average_precision_score(temporal["LARGER_FIRE"], temporal["probability"]))
    common = {
        "model": model, "reference_model": "", "repeats": repeats, "seed": seed,
        "method": "partially paired class-and-membership-stratified percentile bootstrap; shared records resampled jointly",
        **overlap,
    }
    rows = []
    for design, frame, point, draws in (
        ("random", random, random_point, random_draws),
        ("temporal", temporal, temporal_point, temporal_draws),
    ):
        rows.append(_interval_row(
            estimand="average precision", design=design, point=point, draws=draws,
            n=len(frame), positive_count=int(frame["LARGER_FIRE"].sum()),
            positive_prevalence=float(frame["LARGER_FIRE"].mean()), **common,
        ))
    rows.append(_interval_row(
        estimand="random-minus-temporal average-precision difference",
        design="random-minus-temporal", point=random_point - temporal_point,
        draws=random_draws - temporal_draws, **common,
    ))
    return pd.DataFrame(rows)
