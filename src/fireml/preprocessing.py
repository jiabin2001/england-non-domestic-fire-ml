from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

