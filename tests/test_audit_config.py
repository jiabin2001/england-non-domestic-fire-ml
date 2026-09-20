import pytest

from fireml.audit import temporal_year_partitions
from fireml.config import load_yaml


def test_default_temporal_windows_preserve_historical_design():
    cfg = load_yaml("config/analysis.yaml")
    partitions = temporal_year_partitions(cfg)
    assert partitions["temporal_train_years"] == cfg["preferred_main_years"][:10]
    assert partitions["temporal_validation_years"] == ["2020/21", "2021/22"]
    assert partitions["temporal_test_years"] == ["2022/23", "2023/24"]


def test_temporal_windows_follow_nondefault_configuration():
    cfg = {
        "preferred_main_years": ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23"],
        "temporal_validation_years": 1,
        "temporal_test_years": 3,
    }
    assert temporal_year_partitions(cfg) == {
        "temporal_train_years": ["2018/19"],
        "temporal_validation_years": ["2019/20"],
        "temporal_test_years": ["2020/21", "2021/22", "2022/23"],
    }


@pytest.mark.parametrize("setting", ["temporal_validation_years", "temporal_test_years"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_temporal_windows_reject_invalid_counts(setting, value):
    cfg = load_yaml("config/analysis.yaml")
    cfg[setting] = value
    with pytest.raises(ValueError, match=f"{setting} must be a positive integer"):
        temporal_year_partitions(cfg)


def test_temporal_windows_require_training_years():
    cfg = load_yaml("config/analysis.yaml")
    cfg["temporal_test_years"] = len(cfg["preferred_main_years"]) - cfg["temporal_validation_years"]
    with pytest.raises(ValueError, match="at least one training year"):
        temporal_year_partitions(cfg)


@pytest.mark.parametrize("years", [["2020/21", "2019/20"], ["2020/21", "2020/21"]])
def test_temporal_windows_reject_unordered_or_duplicate_years(years):
    cfg = load_yaml("config/analysis.yaml")
    cfg["preferred_main_years"] = years
    with pytest.raises(ValueError, match="unique years in chronological order"):
        temporal_year_partitions(cfg)
