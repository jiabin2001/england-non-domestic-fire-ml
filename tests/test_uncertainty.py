import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score

import fireml.uncertainty as uncertainty

from fireml.uncertainty import (
    add_pr_auc_prevalence_context,
    build_bootstrap_ci_table,
    stratified_resample_indices,
    stratified_bootstrap_pr_auc,
    partially_paired_bootstrap_pr_auc,
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


@pytest.mark.parametrize("score_kind", ["unique", "tied", "constant"])
def test_presorted_single_bootstrap_matches_sklearn_and_preserves_rng(score_kind):
    labels = (np.arange(40) % 4 == 0).astype(int)
    if score_kind == "unique":
        probability = np.linspace(0.01, 0.99, len(labels))
    elif score_kind == "tied":
        probability = (np.arange(len(labels)) % 7) / 6
    else:
        probability = np.full(len(labels), 0.4)
    actual_rng, reference_rng = np.random.default_rng(88), np.random.default_rng(88)
    actual = stratified_bootstrap_pr_auc(labels, probability, 40, actual_rng)
    expected = []
    for _ in range(40):
        indices = stratified_resample_indices(labels, reference_rng)
        expected.append(average_precision_score(labels[indices], probability[indices]))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state


@pytest.mark.parametrize("offset", [0, 20, 50])
def test_presorted_partial_bootstrap_matches_sklearn_for_each_draw(monkeypatch, offset):
    def prediction_frame(start):
        identifiers = np.arange(start, start + 40)
        return pd.DataFrame({
            "SOURCE_ROW_ID": identifiers,
            "LARGER_FIRE": (identifiers % 4 == 0).astype(int),
            "probability": (identifiers % 7) / 6,
        }, index=identifiers)

    random, temporal = prediction_frame(0), prediction_frame(offset).iloc[::-1].copy()
    temporal["probability"] = 1 - temporal["probability"]
    actual_rng, reference_rng = np.random.default_rng(91), np.random.default_rng(91)
    actual = partially_paired_bootstrap_pr_auc(random, temporal, 40, actual_rng)

    class SklearnResampledAP:
        def __init__(self, labels, probability):
            self.labels = labels
            self.probability = probability

        def __call__(self, multiplicities):
            indices = np.repeat(np.arange(len(self.labels)), multiplicities)
            assert len(indices) == len(self.labels)
            assert self.labels[indices].sum() == self.labels.sum()
            return average_precision_score(self.labels[indices], self.probability[indices])

    monkeypatch.setattr(uncertainty, "_PresortedAveragePrecision", SklearnResampledAP)
    expected = partially_paired_bootstrap_pr_auc(random, temporal, 40, reference_rng)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-14)
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state


def test_bootstrap_labels_describe_actual_selected_pipelines():
    random, temporal = _prediction_frames()
    labels = {
        "random": "validation-selected Logistic Regression",
        "temporal": "validation-selected Random Forest",
    }
    result = build_bootstrap_ci_table(random, temporal, repeats=10, seed=42, model_labels=labels)
    assert result.iloc[0]["model"] == labels["random"]
    assert result.iloc[1]["model"] == labels["temporal"]
    assert result.iloc[2]["model"] == f"random: {labels['random']}; temporal: {labels['temporal']}"
    assert not result["model"].str.contains("XGBoost").any()
    unspecified = build_bootstrap_ci_table(random, temporal, repeats=10, seed=42)
    assert unspecified["model"].eq("validation-selected pipeline").all()
