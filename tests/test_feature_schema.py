import pandas as pd
import pytest

from fireml.config import load_yaml
from fireml.features import build_feature_policy, resolve_blocks


def configured_columns():
    policy = load_yaml("config/feature_policy.yaml")
    return [column for block in policy["blocks"].values() for column in block["candidates"]]


def test_default_feature_membership_is_preserved():
    blocks = resolve_blocks(configured_columns())
    assert {name: len(columns) for name, columns in blocks.items()} == {"A": 5, "B": 16, "C": 19}
    assert blocks["B"][:5] == blocks["A"]
    assert blocks["C"][:16] == blocks["B"]
    assert "OCCUPIED_TIME" in blocks["B"]


@pytest.mark.parametrize("missing_field", ["BUILDING_TYPE", "ITEM_IGNITED", "FIRE_SIZE_ON_ARRIVAL"])
def test_missing_predictor_prevents_silent_experiment_change(missing_field):
    columns = [column for column in configured_columns() if column != missing_field]
    with pytest.raises(ValueError, match=missing_field):
        resolve_blocks(columns)


def test_generated_policy_explains_retained_occupancy_and_arrival_timing_risks():
    frame = pd.DataFrame({column: ["example"] for column in configured_columns()})
    policy = build_feature_policy(frame).set_index("raw_field")
    occupancy = policy.loc["OCCUPIED_TIME"]
    assert occupancy["block_b"] and occupancy["block_c"]
    assert not occupancy["excluded"]
    assert "outcome proxy" in occupancy["leakage_risk"]
    assert "buildings to which the fire spread" in occupancy["official_meaning"]
    assert "No removal sensitivity analysis has been run" in occupancy["reason"]
    assert "inherits investigative fields" in policy.loc["FIRE_SIZE_ON_ARRIVAL", "reason"]
