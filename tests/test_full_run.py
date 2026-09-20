"""Full-run orchestration tests use fake stages and a temporary synthetic source."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


STAGES = (
    ("source import/cache", "fireml.acquire", "acquire_and_cache"),
    ("audit", "fireml.audit", "run_audit"),
    ("cohort", "fireml.cohort", "construct_cohort"),
    ("data checksums", "fireml.reporting", "write_data_archive_manifest"),
    ("core model selection and fitting", "fireml.modelling", "run_core_models"),
    ("core uncertainty", "fireml.uncertainty", "run_uncertainty_analysis"),
    ("grouped permutation", "fireml.interpretability", "run_grouped_permutation_analysis"),
    ("temporal and split robustness", "fireml.robustness", "run_temporal_robustness"),
    ("diagnostic fits", "fireml.review_experiments", "run_review_fits"),
    ("diagnostic uncertainty", "fireml.review_experiments", "run_review_uncertainty"),
    ("reports and figures", "fireml.reporting", "build_report"),
)


@pytest.fixture
def full_run_case(tmp_path, monkeypatch):
    raw = tmp_path / "data/raw/synthetic.ods"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"Synthetic test source; not an ODS research file.")
    metadata = raw.parent / "source_metadata.json"
    metadata.write_text(json.dumps({"sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}))
    sentinel = tmp_path / "outputs/previous-result.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("Previous output must survive rejected preflight.")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/analysis.yaml").write_text("raw_path: data/raw/synthetic.ods\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "synthetic-full-run"\n')
    receipt_path = tmp_path / "outputs/metrics/execution.json"
    calls = []
    failure = {"stage": None}

    def stage_action(name):
        def action():
            calls.append(name)
            assert not sentinel.exists(), "An authorized clean run must remove old output before any stage."
            in_progress = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert in_progress["status"] == "running"
            assert in_progress["stages"][-1]["name"] == name
            assert in_progress["stages"][-1]["status"] == "running"
            if failure["stage"] == name:
                raise RuntimeError("Synthetic stage failure")
        return action

    # Stub imports before loading the script: no modelling/reporting module imports
    # and no research actions are needed to exercise its orchestration semantics.
    modules = {}
    for name, module_name, function_name in STAGES:
        module = modules.setdefault(module_name, ModuleType(module_name))
        setattr(module, function_name, stage_action(name))
    modules["fireml.acquire"].sha256_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = ModuleType("fireml.config")
    config.ROOT = tmp_path
    config.load_yaml = lambda _: {"raw_path": "data/raw/synthetic.ods"}
    config.ensure_output_dirs = lambda: (tmp_path / "outputs/metrics").mkdir(parents=True, exist_ok=True)
    modules["fireml.config"] = config
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    script_path = Path(__file__).resolve().parents[1] / "scripts/06_build_report.py"
    spec = importlib.util.spec_from_file_location("synthetic_full_run_script", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    def git_revision(command, *, cwd, text):
        assert command == ["git", "rev-parse", "HEAD"]
        assert cwd == tmp_path and text is True
        return "synthetic-commit\n"

    monkeypatch.setattr(script.subprocess, "check_output", git_revision)
    monkeypatch.setattr(sys, "argv", [str(script_path), "--clean"])
    return SimpleNamespace(
        script=script, raw=raw, metadata=metadata, sentinel=sentinel,
        receipt_path=receipt_path, calls=calls, failure=failure, root=tmp_path,
    )


def test_missing_source_fails_before_removing_existing_output(full_run_case, capsys):
    case = full_run_case
    before = case.sentinel.read_bytes()
    case.raw.unlink()
    with pytest.raises(SystemExit) as error:
        case.script.main()
    assert error.value.code == 2
    assert "Restore the official ODS" in capsys.readouterr().err
    assert case.sentinel.read_bytes() == before
    assert not case.receipt_path.exists()
    assert case.calls == []


def test_source_checksum_mismatch_fails_before_removing_existing_output(full_run_case, capsys):
    case = full_run_case
    before = case.sentinel.read_bytes()
    case.raw.write_bytes(b"Changed synthetic source version")
    with pytest.raises(SystemExit) as error:
        case.script.main()
    assert error.value.code == 2
    assert "ODS checksum differs" in capsys.readouterr().err
    assert case.sentinel.read_bytes() == before
    assert not case.receipt_path.exists()
    assert case.calls == []


def test_existing_output_requires_explicit_clean_flag(full_run_case, monkeypatch, capsys):
    case = full_run_case
    before = case.sentinel.read_bytes()
    monkeypatch.setattr(sys, "argv", ["06_build_report.py"])
    with pytest.raises(SystemExit) as error:
        case.script.main()
    assert error.value.code == 2
    assert "Use --clean" in capsys.readouterr().err
    assert case.sentinel.read_bytes() == before
    assert not case.receipt_path.exists()
    assert case.calls == []


def test_clean_run_records_all_stages_and_orders_core_diagnostics_then_report(full_run_case):
    case = full_run_case
    case.script.main()
    receipt = json.loads(case.receipt_path.read_text(encoding="utf-8"))
    expected_stages = [name for name, _, _ in STAGES]
    assert not case.sentinel.exists()
    assert case.calls == expected_stages
    assert len(receipt["stages"]) == 11
    assert receipt["status"] == "complete"
    assert receipt["base_git_commit"] == "synthetic-commit"
    assert receipt["started_at_utc"] <= receipt["finished_at_utc"]
    assert [stage["name"] for stage in receipt["stages"]] == expected_stages
    assert all(stage["status"] == "complete" for stage in receipt["stages"])
    assert all(stage["started_at_utc"] <= stage["finished_at_utc"] for stage in receipt["stages"])
    assert case.calls.index("core model selection and fitting") < case.calls.index("diagnostic fits")
    assert case.calls.index("diagnostic uncertainty") < case.calls.index("reports and figures")
    for name in ("config/analysis.yaml", "pyproject.toml"):
        assert receipt["code_and_config_sha256"][name] == hashlib.sha256((case.root / name).read_bytes()).hexdigest()


def test_stage_failure_records_failed_execution_and_stops_later_stages(full_run_case):
    case = full_run_case
    case.failure["stage"] = "core model selection and fitting"
    with pytest.raises(RuntimeError, match="Synthetic stage failure"):
        case.script.main()
    receipt = json.loads(case.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["error"] == "RuntimeError: Synthetic stage failure"
    expected_attempted = [name for name, _, _ in STAGES[:5]]
    assert case.calls == expected_attempted
    assert [stage["name"] for stage in receipt["stages"]] == expected_attempted
    assert all(stage["status"] == "complete" for stage in receipt["stages"][:-1])
    assert "diagnostic fits" not in case.calls
    assert "reports and figures" not in case.calls
