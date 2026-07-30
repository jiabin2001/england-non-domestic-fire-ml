import json

import pandas as pd
import pytest

from fireml.config import ROOT


@pytest.fixture(scope="session")
def cohort():
    return pd.read_parquet(ROOT / "data/processed/analysis_cohort.parquet")


@pytest.fixture(scope="session")
def audit_receipt():
    return json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))

