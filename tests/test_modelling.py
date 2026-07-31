import pytest

import fireml.modelling as modelling


def test_explicit_xgboost_cpu_does_not_depend_on_detected_hardware(monkeypatch):
    def fail_if_called():
        raise AssertionError("explicit CPU selection should not inspect GPU hardware")

    monkeypatch.setattr(modelling, "detect_xgb_device", fail_if_called)
    assert modelling.resolve_xgb_device("cpu") == "cpu"


def test_explicit_xgboost_cuda_fails_fast_when_unavailable(monkeypatch):
    monkeypatch.setattr(modelling, "detect_xgb_device", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        modelling.resolve_xgb_device("cuda")


def test_auto_xgboost_device_is_visible(monkeypatch):
    monkeypatch.setattr(modelling, "detect_xgb_device", lambda: "cpu")
    with pytest.warns(RuntimeWarning, match="hardware can change fitted results"):
        assert modelling.resolve_xgb_device("auto") == "cpu"


def test_invalid_xgboost_device_is_rejected():
    with pytest.raises(ValueError, match="cpu, cuda, auto"):
        modelling.resolve_xgb_device("gpu-maybe")
