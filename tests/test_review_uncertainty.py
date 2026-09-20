import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score

from fireml.review_uncertainty import (
    _paired_ap_draws,
    _partially_paired_ap_draws,
    _WeightedAveragePrecision,
    paired_model_ap_intervals,
    split_ap_intervals,
)
from fireml.uncertainty import partially_paired_bootstrap_pr_auc, stratified_resample_indices


def predictions(start=0, size=24):
    ids = np.arange(start, start + size)
    return pd.DataFrame({
        "SOURCE_ROW_ID": ids,
        "LARGER_FIRE": (ids % 3 == 0).astype(int),
        "probability": (ids % 5) / 4.0,
    }, index=np.arange(100, 100 + size) * 7)


@pytest.mark.parametrize("constant_score", [False, True])
def test_cached_weighted_ap_matches_sklearn_with_ties_and_zero_weight_groups(constant_score):
    frame = predictions()
    labels = frame["LARGER_FIRE"].to_numpy()
    probability = frame["probability"].to_numpy(copy=True)
    if constant_score:
        probability[:] = 0.5
    cached_score = _WeightedAveragePrecision(labels, probability)
    rng = np.random.default_rng(47)
    for _ in range(30):
        # Deliberately allow entire score groups to receive no observations.
        counts = rng.integers(0, 3, size=len(frame))
        counts[0] = counts[1] = 1
        actual = cached_score(counts)
        expected = average_precision_score(labels, probability, sample_weight=counts)
        assert actual == pytest.approx(expected, abs=1e-14)


def test_paired_draws_match_sklearn_on_exact_same_stratified_resamples():
    frame = predictions()
    labels = frame["LARGER_FIRE"].to_numpy()
    probability = frame["probability"].to_numpy()
    reference = 1.0 - probability
    actual, actual_reference = _paired_ap_draws(
        labels, probability, reference, 40, np.random.default_rng(71)
    )
    rng = np.random.default_rng(71)
    expected, expected_reference = [], []
    for _ in range(40):
        indices = stratified_resample_indices(labels, rng)
        assert int(labels[indices].sum()) == int(labels.sum())
        expected.append(average_precision_score(labels[indices], probability[indices]))
        expected_reference.append(average_precision_score(labels[indices], reference[indices]))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual_reference, expected_reference, rtol=0, atol=1e-14)


@pytest.mark.parametrize("start", [0, 12, 30])
def test_optimized_partial_draws_match_original_resampling_with_any_overlap(start):
    random = predictions()
    temporal = predictions(start).iloc[::-1].copy()
    temporal["probability"] = 1.0 - temporal["probability"]
    expected = partially_paired_bootstrap_pr_auc(
        random, temporal, 40, np.random.default_rng(23)
    )
    actual_random, actual_temporal, metadata = _partially_paired_ap_draws(
        random, temporal, 40, np.random.default_rng(23)
    )
    np.testing.assert_allclose(actual_random, expected[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual_temporal, expected[1], rtol=0, atol=1e-14)
    assert metadata["overlap_n"] == max(0, 24 - start)
    assert metadata["random_positive_count"] == 8
    assert metadata["temporal_positive_count"] == 8


def test_paired_intervals_align_source_ids_and_preserve_inputs():
    new = predictions()
    reference = new.sample(frac=1, random_state=3).reset_index(drop=True)
    before, reference_before = new.copy(deep=True), reference.copy(deep=True)
    kwargs = dict(design="temporal", model="fixed ablation", reference_model="saved full model", repeats=40, seed=15)
    first = paired_model_ap_intervals(new, reference, **kwargs)
    assert_frame_equal(first, paired_model_ap_intervals(new, reference, **kwargs))
    difference = first.iloc[2]
    assert difference["point_estimate"] == 0
    assert difference["ci_lower_95"] == difference["ci_upper_95"] == 0
    assert first["paired_test_n"].eq(len(new)).all()
    assert first["positive_prevalence"].eq(1 / 3).all()
    assert "fixed ablation" in first["model"].tolist()
    assert not first["model"].str.contains("validation-selected").any()
    assert_frame_equal(new, before)
    assert_frame_equal(reference, reference_before)


def test_paired_difference_uses_joint_draws_and_new_minus_reference_sign():
    new, reference = predictions(), predictions()
    new["probability"] = 0.1 + 0.8 * new["LARGER_FIRE"]
    reference["probability"] = 0.5
    result = paired_model_ap_intervals(
        new, reference, design="random", model="perfect", reference_model="constant", repeats=30
    )
    difference = result.iloc[2]
    assert difference["point_estimate"] == pytest.approx(2 / 3)
    assert difference["ci_lower_95"] == pytest.approx(2 / 3)
    assert difference["ci_upper_95"] == pytest.approx(2 / 3)


def test_split_intervals_are_reproducible_ignore_dataframe_indices_and_record_overlap():
    random, temporal = predictions(), predictions(12).reset_index(drop=True)
    before, temporal_before = random.copy(deep=True), temporal.copy(deep=True)
    kwargs = dict(model="fixed B without occupancy", repeats=40, seed=56)
    first = split_ap_intervals(random, temporal, **kwargs)
    assert_frame_equal(first, split_ap_intervals(random, temporal, **kwargs))
    assert first["overlap_n"].eq(12).all()
    assert first["overlap_identifier"].eq("SOURCE_ROW_ID").all()
    assert first["random_positive_prevalence"].eq(1 / 3).all()
    difference = first.iloc[2]
    assert difference["point_estimate"] == first.iloc[0]["point_estimate"] - first.iloc[1]["point_estimate"]
    assert first["ci_lower_95"].le(first["ci_upper_95"]).all()
    assert_frame_equal(random, before)
    assert_frame_equal(temporal, temporal_before)


@pytest.mark.parametrize("column,value", [
    ("LARGER_FIRE", 0.4), ("LARGER_FIRE", 2), ("LARGER_FIRE", np.nan),
    ("probability", np.inf), ("probability", np.nan),
    ("probability", -0.1), ("probability", 1.1),
    ("SOURCE_ROW_ID", np.nan),
])
def test_prediction_validation_rejects_invalid_values_before_casting(column, value):
    frame = predictions().astype({column: float})
    frame.iloc[0, frame.columns.get_loc(column)] = value
    with pytest.raises(ValueError, match="binary|finite|nonmissing"):
        split_ap_intervals(frame, predictions(12), model="fixed", repeats=5)


def test_duplicate_source_ids_are_rejected():
    frame = predictions()
    frame.iloc[0, frame.columns.get_loc("SOURCE_ROW_ID")] = frame.iloc[1]["SOURCE_ROW_ID"]
    with pytest.raises(ValueError, match="unique"):
        split_ap_intervals(frame, predictions(12), model="fixed", repeats=5)


def test_paired_comparison_requires_same_records_and_labels():
    kwargs = dict(design="temporal", model="new", reference_model="old", repeats=5)
    with pytest.raises(ValueError, match="identical SOURCE_ROW_ID"):
        paired_model_ap_intervals(predictions(), predictions(3), **kwargs)
    wrong_labels = predictions()
    wrong_labels.iloc[0, wrong_labels.columns.get_loc("LARGER_FIRE")] = 0
    with pytest.raises(ValueError, match="labels disagree"):
        paired_model_ap_intervals(predictions(), wrong_labels, **kwargs)
    with pytest.raises(ValueError, match="labels disagree"):
        split_ap_intervals(predictions(), wrong_labels, model="fixed", repeats=5)


def test_bootstrap_requires_both_classes_and_positive_repeat_count():
    frame = predictions()
    frame["LARGER_FIRE"] = 0
    with pytest.raises(ValueError, match="both outcome classes"):
        split_ap_intervals(frame, predictions(12), model="fixed", repeats=5)
    for repeats in (0, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            split_ap_intervals(predictions(), predictions(12), model="fixed", repeats=repeats)
