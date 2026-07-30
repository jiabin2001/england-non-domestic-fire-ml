import pandas as pd
import pytest

from fireml.target import infer_mapping, map_target


def test_observed_target_mapping_is_complete():
    values = [
        "No fire damage", "Limited to item 1st ignited", "Limited to room of origin",
        "Limited to floor of origin (not whole building)", "Limited to 2 floors",
        "Whole Building/ Affecting more than 2 floors", "Roofs/ Roof spaces",
    ]
    mapping = infer_mapping(values)
    assert set(mapping) == set(values)
    assert set(value for value in mapping.values() if value is not None) == {0, 1}
    assert mapping["Roofs/ Roof spaces"] is None


def test_unknown_target_category_raises():
    with pytest.raises(ValueError, match="Unrecognised"):
        map_target(pd.Series(["No fire damage", "invented category"]))


def test_roof_sensitivity_maps_positive():
    mapping = infer_mapping(["Roofs/ Roof spaces"], roofs_positive=True)
    assert mapping["Roofs/ Roof spaces"] == 1

