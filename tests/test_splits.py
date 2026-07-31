import numpy as np
import pytest

from fireml.splits import assert_disjoint, make_random_split_like, make_temporal_split


def _temporal(cohort, audit_receipt):
    return make_temporal_split(
        cohort,
        audit_receipt["temporal_train_years"],
        audit_receipt["temporal_validation_years"],
        audit_receipt["temporal_test_years"],
    )


def test_temporal_partitions_are_disjoint_and_ordered(cohort, audit_receipt):
    split = _temporal(cohort, audit_receipt)
    assert_disjoint(split)
    train_year = cohort.loc[split["train"], "FINANCIAL_YEAR"].str[:4].astype(int).max()
    validation_year = cohort.loc[split["validation"], "FINANCIAL_YEAR"].str[:4].astype(int).min()
    test_year = cohort.loc[split["test"], "FINANCIAL_YEAR"].str[:4].astype(int).min()
    assert train_year < validation_year < test_year


def test_random_matches_temporal_sizes_and_is_reproducible(cohort, audit_receipt):
    temporal = _temporal(cohort, audit_receipt)
    first = make_random_split_like(cohort, temporal, 20260801)
    second = make_random_split_like(cohort, temporal, 20260801)
    assert {key: len(value) for key, value in first.items()} == {key: len(value) for key, value in temporal.items()}
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_temporal_split_rejects_validation_year_after_test(cohort):
    with pytest.raises(ValueError, match="strictly ordered"):
        make_temporal_split(
            cohort,
            ["2010/11"],
            ["2012/13", "2023/24"],
            ["2022/23"],
        )
