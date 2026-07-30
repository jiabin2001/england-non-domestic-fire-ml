import pandas as pd
import pytest

from fireml.features import assert_no_leakage, resolve_blocks
from fireml.preprocessing import make_preprocessor


def test_preprocessor_learns_categories_from_training_only():
    train = pd.DataFrame({"field": ["a", "b", None]})
    test = pd.DataFrame({"field": ["never_seen"]})
    transformer = make_preprocessor(["field"])
    transformer.fit(train)
    categories = transformer.named_transformers_["categorical"].named_steps["onehot"].categories_[0]
    assert "never_seen" not in categories
    assert transformer.transform(test).shape[0] == 1


def test_leakage_blacklist_fails_fast():
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage({"bad": ["DAY_OF_WEEK", "SPREAD_OF_FIRE"]})


def test_target_and_year_not_in_predictors(cohort):
    blocks = resolve_blocks(cohort.columns)
    for columns in blocks.values():
        assert "LARGER_FIRE" not in columns
        assert "SPREAD_OF_FIRE" not in columns
        assert "FINANCIAL_YEAR" not in columns
        assert "FRS_TERRITORY" not in columns

