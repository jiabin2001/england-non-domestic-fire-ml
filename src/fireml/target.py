from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


NEGATIVE_CANONICAL = {
    "no fire damage",
    "limited to item first ignited",
    "limited to item 1st ignited",
    "limited to room of origin",
}
POSITIVE_CANONICAL = {
    "limited to floor of origin",
    "limited to floor of origin (not whole building)",
    "limited to 2 floors",
    "affecting more than 2 floors",
    "whole building/ affecting more than 2 floors",
    "whole building",
}
ROOF_CANONICAL = {"roofs and spaces", "roofs/ roof spaces"}
EXCLUDED_CANONICAL = ROOF_CANONICAL | {"unknown", "not known", "don't know"}


def canonicalise(value: object) -> str | None:
    if pd.isna(value):
        return None
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def infer_mapping(actual_values: Iterable[object], roofs_positive: bool = False) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    unknown: list[str] = []
    for raw in actual_values:
        if pd.isna(raw):
            continue
        text = str(raw)
        canonical = canonicalise(text)
        if canonical in NEGATIVE_CANONICAL:
            result[text] = 0
        elif canonical in POSITIVE_CANONICAL:
            result[text] = 1
        elif canonical in ROOF_CANONICAL:
            result[text] = 1 if roofs_positive else None
        elif canonical in EXCLUDED_CANONICAL:
            result[text] = None
        else:
            unknown.append(text)
    if unknown:
        raise ValueError(f"Unrecognised SPREAD_OF_FIRE categories: {unknown}")
    return result


def map_target(series: pd.Series, roofs_positive: bool = False) -> tuple[pd.Series, dict[str, int | None]]:
    mapping = infer_mapping(series.dropna().unique(), roofs_positive=roofs_positive)
    mapped = series.map(mapping).astype("Int8")
    return mapped, mapping
