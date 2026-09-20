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


def test_training_environment_records_versions_without_requiring_test_extra(monkeypatch):
    def version(package):
        if package == "pytest":
            raise modelling.importlib.metadata.PackageNotFoundError(package)
        return "training-version"

    monkeypatch.setattr(modelling.importlib.metadata, "version", version)
    receipt = modelling.training_environment("cpu", 2)
    assert receipt["record_type"] == "model_training_environment"
    assert receipt["xgboost_device"] == "cpu"
    assert receipt["n_jobs"] == 2
    assert receipt["packages"]["scikit-learn"] == "training-version"
    assert receipt["packages"]["pyarrow"] == "training-version"
    assert receipt["packages"]["pytest"] is None
