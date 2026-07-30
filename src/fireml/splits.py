from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_temporal_split(
    frame: pd.DataFrame,
    train_years: list[str],
    validation_years: list[str],
    test_years: list[str],
) -> dict[str, np.ndarray]:
    split = {
        "train": frame.index[frame["FINANCIAL_YEAR"].isin(train_years)].to_numpy(),
        "validation": frame.index[frame["FINANCIAL_YEAR"].isin(validation_years)].to_numpy(),
        "test": frame.index[frame["FINANCIAL_YEAR"].isin(test_years)].to_numpy(),
    }
    assert_disjoint(split)
    if not (max(map(_year_start, train_years)) < min(map(_year_start, validation_years)) < min(map(_year_start, test_years))):
        raise ValueError("Temporal split years are not strictly ordered.")
    return split


def make_random_split_like(frame: pd.DataFrame, reference: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    indices = frame.index.to_numpy()
    target = frame["LARGER_FIRE"].to_numpy()
    dev_idx, test_idx = train_test_split(
        indices,
        test_size=len(reference["test"]),
        stratify=target,
        random_state=seed,
    )
    dev_target = frame.loc[dev_idx, "LARGER_FIRE"].to_numpy()
    train_idx, validation_idx = train_test_split(
        dev_idx,
        test_size=len(reference["validation"]),
        stratify=dev_target,
        random_state=seed,
    )
    split = {"train": np.sort(train_idx), "validation": np.sort(validation_idx), "test": np.sort(test_idx)}
    assert_disjoint(split)
    assert_matching_sizes(split, reference)
    return split


def assert_disjoint(split: dict[str, np.ndarray]) -> None:
    names = list(split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if np.intersect1d(split[left], split[right]).size:
                raise ValueError(f"Overlapping records in {left} and {right}.")


def assert_matching_sizes(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    for name in ("train", "validation", "test"):
        if len(left[name]) != len(right[name]):
            raise ValueError(f"Split size mismatch for {name}.")


def assignment_frame(frame: pd.DataFrame, splits: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for design, parts in splits.items():
        for partition, indices in parts.items():
            subset = frame.loc[indices]
            rows.extend({
                "design": design,
                "partition": partition,
                "cohort_index": int(index),
                "source_row_id": int(row.SOURCE_ROW_ID),
                "financial_year": row.FINANCIAL_YEAR,
                "larger_fire": int(row.LARGER_FIRE),
            } for index, row in subset.iterrows())
    return pd.DataFrame(rows)


def _year_start(value: str) -> int:
    return int(value.split("/")[0])

