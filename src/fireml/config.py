from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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

