import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from fireml.uncertainty import (
    add_pr_auc_prevalence_context,
    build_bootstrap_ci_table,
    stratified_resample_indices,
)


def _prediction_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    random_ids = np.arange(40)
    temporal_ids = np.arange(30, 70)
    random = pd.DataFrame({
        "SOURCE_ROW_ID": random_ids,
        "LARGER_FIRE": (random_ids % 4 == 0).astype(int),
        "probability": np.linspace(0.02, 0.95, random_ids.size),
    }, index=np.arange(40))
    temporal = pd.DataFrame({
        "SOURCE_ROW_ID": temporal_ids,
        "LARGER_FIRE": (temporal_ids % 4 == 0).astype(int),
        "probability": np.linspace(0.05, 0.85, temporal_ids.size) ** 1.2,
    }, index=np.arange(30, 70))
    return random, temporal


def test_stratified_resampling_preserves_class_counts():
    labels = np.array([0] * 17 + [1] * 5)
    indices = stratified_resample_indices(labels, np.random.default_rng(7))
    sampled = labels[indices]
    assert len(sampled) == len(labels)
    assert int(sampled.sum()) == 5
    assert int((sampled == 0).sum()) == 17


def test_bootstrap_is_reproducible_bounded_and_non_mutating():
    random, temporal = _prediction_frames()
    random_before = random.copy(deep=True)
    temporal_before = temporal.copy(deep=True)
    first = build_bootstrap_ci_table(random, temporal, repeats=80, seed=123)
    second = build_bootstrap_ci_table(random, temporal, repeats=80, seed=123)
    assert_frame_equal(first, second)
    assert_frame_equal(random, random_before)
    assert_frame_equal(temporal, temporal_before)
    pr_auc_rows = first[first["estimand"] == "average precision"]
    assert pr_auc_rows["ci_lower_95"].between(0, 1).all()
    assert pr_auc_rows["ci_upper_95"].between(0, 1).all()
    assert (first["ci_lower_95"] <= first["ci_upper_95"]).all()
    difference = first[first["design"] == "random-minus-temporal"].iloc[0]
    assert np.isfinite(difference["ci_lower_95"])
    assert np.isfinite(difference["ci_upper_95"])
    assert "partially paired" in difference["bootstrap_method"]
    assert "shared records resampled jointly" in difference["bootstrap_method"]
    assert difference["overlap_n"] == 10
    assert difference["overlap_fraction_random_test"] == 0.25
    assert difference["overlap_fraction_temporal_test"] == 0.25


def test_prevalence_context_handles_all_positive_edge_case():
    performance = pd.DataFrame({
        "positive_prevalence": [0.25, 1.0],
        "pr_auc": [0.65, 1.0],
    })
    contextual = add_pr_auc_prevalence_context(performance)
    assert contextual.loc[0, "pr_auc_baseline"] == 0.25
    assert contextual.loc[0, "pr_auc_absolute_lift"] == 0.40
    assert np.isclose(contextual.loc[0, "normalized_pr_auc"], 0.40 / 0.75)
    assert np.isnan(contextual.loc[1, "normalized_pr_auc"])
