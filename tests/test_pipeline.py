import pandas as pd
import pytest

from fireml.features import assert_no_leakage, resolve_blocks
from fireml.modelling import candidate_grid, make_model_pipeline
from fireml.preprocessing import make_catboost_preprocessor, make_preprocessor


def test_preprocessor_learns_categories_from_training_only():
    train = pd.DataFrame({"field": ["a", "b", None]})
    test = pd.DataFrame({"field": ["never_seen"]})
    transformer = make_preprocessor(["field"])
    transformer.fit(train)
    categories = transformer.named_transformers_["categorical"].named_steps["onehot"].categories_[0]
    assert "never_seen" not in categories
    assert transformer.transform(test).shape[0] == 1


def test_catboost_preprocessor_preserves_native_string_categories():
    frame = pd.DataFrame({
        "first": ["a", None, "b"],
        "second": ["x", "y", None],
    })
    transformed = make_catboost_preprocessor(list(frame.columns)).fit_transform(frame)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.columns.tolist() == frame.columns.tolist()
    assert transformed.shape == frame.shape
    assert transformed.loc[1, "first"] == "Missing/Unknown"
    assert transformed.loc[2, "second"] == "Missing/Unknown"
    assert transformed.map(type).eq(str).all().all()


def test_catboost_pipeline_uses_cpu_and_native_categorical_fields():
    frame = pd.DataFrame({
        "first": ["a", "b", None, "a", "c", "b"],
        "second": ["x", "x", "y", None, "y", "x"],
    })
    labels = [0, 1, 0, 1, 1, 0]
    pipeline = make_model_pipeline(
        list(frame.columns),
        "catboost",
        candidate_grid()["catboost"][0],
        seed=42,
        n_jobs=1,
        device="cpu",
    )
    pipeline.fit(frame, labels)
    params = pipeline.named_steps["model"].get_params()
    assert params["task_type"] == "CPU"
    assert params["cat_features"] == list(frame.columns)
    assert pipeline.predict_proba(frame).shape == (len(frame), 2)


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
