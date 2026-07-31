from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def _load_yaml_cached(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_yaml(relative_path: str) -> dict[str, Any]:
    """Load immutable-on-disk configuration without repeated YAML I/O.

    A defensive copy prevents one caller from mutating the cached configuration seen
    by another caller.
    """
    return deepcopy(_load_yaml_cached(relative_path))


def ensure_output_dirs() -> None:
    for relative in (
        "data/raw",
        "data/interim",
        "data/processed",
        "outputs/tables",
        "outputs/figures",
        "outputs/metrics",
        "outputs/models",
        "reports",
    ):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
