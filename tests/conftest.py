import pandas as pd
import pytest

from fireml.config import load_yaml


@pytest.fixture(scope="session")
def cohort():
    """Small synthetic records; unit tests never require the research cohort."""
    policy = load_yaml("config/feature_policy.yaml")
    columns = [column for block in policy["blocks"].values() for column in block["candidates"]]
    rows = []
    for year in range(2010, 2024):
        for incident in range(12):
            rows.append({
                **{column: f"category_{incident % 3}" for column in columns},
                "SOURCE_ROW_ID": len(rows),
                "FINANCIAL_YEAR": f"{year}/{str(year + 1)[-2:]}",
                "LARGER_FIRE": int(incident % 3 == 0),
                "SPREAD_OF_FIRE": "synthetic outcome",
                "FRS_TERRITORY": "synthetic territory",
            })
    frame = pd.DataFrame(rows)
    # Nonconsecutive labels also exercise label-based rather than positional splits.
    frame.index = pd.Index(range(1000, 1000 + 3 * len(frame), 3))
    return frame


@pytest.fixture(scope="session")
def audit_receipt(cohort):
    years = sorted(cohort["FINANCIAL_YEAR"].unique())
    return {
        "temporal_train_years": years[:-4],
        "temporal_validation_years": years[-4:-2],
        "temporal_test_years": years[-2:],
    }

