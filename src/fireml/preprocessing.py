from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Create a train-fitted categorical preprocessor for disclosed banded fields."""
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing/Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
    ])
    return ColumnTransformer(
        [("categorical", categorical, columns)],
        remainder="drop",
        sparse_threshold=1.0,
    )


def _prepare_catboost_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Preserve original fields while giving CatBoost explicit string categories."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("CatBoost preprocessing requires a pandas DataFrame.")
    return (
        frame.loc[:, columns]
        .astype("string")
        .fillna("Missing/Unknown")
        .astype(str)
    )


def make_catboost_preprocessor(columns: list[str]) -> FunctionTransformer:
    """Create a train-safe native-categorical transformer without one-hot encoding."""
    return FunctionTransformer(
        _prepare_catboost_frame,
        kw_args={"columns": columns},
        validate=False,
        feature_names_out="one-to-one",
    )
