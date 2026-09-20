"""Four bounded diagnostic fits compared with the current core analysis.

The holdouts have already been inspected. These are exploratory sensitivity and
baseline comparisons, not new confirmatory tests or a full study rerun.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

from .config import ROOT
from .evaluation import choose_f1_threshold, classification_metrics
from .modelling import make_model_pipeline, resolve_xgb_device, training_environment


PARTITIONS = ("train", "validation", "test")
EXPERIMENT_IDS = (
    "b_minus_occupied_random", "b_minus_occupied_temporal",
    "building_type_temporal", "arrival_size_temporal",
)
REFERENCE_PREDICTIONS = {
    "random_B": "outputs/metrics/predictions_random_block_B.parquet",
    "temporal_B": "outputs/metrics/predictions_temporal_block_B.parquet",
    "temporal_C": "outputs/metrics/predictions_temporal_block_C.parquet",
}
CODE_INPUTS = (
    "src/fireml/review_experiments.py", "src/fireml/review_uncertainty.py",
    "src/fireml/modelling.py", "src/fireml/preprocessing.py",
    "src/fireml/features.py", "src/fireml/evaluation.py",
    "src/fireml/uncertainty.py", "config/feature_policy.yaml",
    "scripts/08_run_review_experiments.py", "pyproject.toml",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot record Git provenance: git {' '.join(arguments)}")
    return result.stdout.strip()


def _gpu_receipt() -> dict:
    command = [
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("Cannot record the required NVIDIA GPU and driver details.")
    return {"command": command, "output": result.stdout.strip()}


def validate_saved_partitions(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    selection: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """Use saved row order, verifying membership and labels rather than resplitting."""
    frame_columns = {"SOURCE_ROW_ID", "FINANCIAL_YEAR", "LARGER_FIRE"}
    assignment_columns = {
        "design", "partition", "cohort_index", "source_row_id", "financial_year", "larger_fire",
    }
    if not frame_columns.issubset(frame) or not assignment_columns.issubset(assignments):
        raise ValueError("Cohort or saved split assignments lack required identity columns.")
    if not frame.index.is_unique or not frame["SOURCE_ROW_ID"].is_unique:
        raise ValueError("Cohort index and SOURCE_ROW_ID must be unique.")
    if frame[list(frame_columns)].isna().any().any() or not frame.LARGER_FIRE.isin([0, 1]).all():
        raise ValueError("Cohort identity/year/labels must be nonnull and labels binary.")
    if assignments[list(assignment_columns)].isna().any().any():
        raise ValueError("Saved split assignments contain null values.")
    if set(assignments.design) != {"random", "temporal"}:
        raise ValueError("Expected exactly the core random and temporal designs.")
    if set(references) != set(REFERENCE_PREDICTIONS):
        raise ValueError("All three core B/C prediction references are required.")
    splits: dict[str, dict[str, np.ndarray]] = {}
    for design in ("random", "temporal"):
        design_rows = assignments[assignments.design.eq(design)]
        if set(design_rows.partition) != set(PARTITIONS):
            raise ValueError(f"Missing or unexpected partitions for {design}.")
        if (
            len(design_rows) != len(frame)
            or not design_rows.cohort_index.is_unique
            or not design_rows.source_row_id.is_unique
            or set(design_rows.cohort_index) != set(frame.index)
        ):
            raise ValueError(f"{design} assignments do not partition the exact cohort once.")
        splits[design] = {}
        for partition in PARTITIONS:
            rows = design_rows[design_rows.partition.eq(partition)]
            indices = rows.cohort_index.to_numpy()
            actual = frame.loc[indices]
            for saved_column, cohort_column in (
                ("source_row_id", "SOURCE_ROW_ID"),
                ("financial_year", "FINANCIAL_YEAR"),
                ("larger_fire", "LARGER_FIRE"),
            ):
                if not np.array_equal(rows[saved_column].to_numpy(), actual[cohort_column].to_numpy()):
                    raise ValueError(f"{design}/{partition} {cohort_column} differs from saved assignments.")
            if set(actual.LARGER_FIRE) != {0, 1}:
                raise ValueError(f"{design}/{partition} requires both target classes.")
            if design == "temporal":
                expected_years = selection["split_years"][f"temporal_{partition}"]
                if set(actual.FINANCIAL_YEAR) != set(expected_years):
                    raise ValueError(f"Temporal {partition} years differ from core selection.")
            splits[design][partition] = indices
    if any(len(splits["random"][part]) != len(splits["temporal"][part]) for part in PARTITIONS):
        raise ValueError("Saved random and temporal partition sizes must match.")
    for name, saved in references.items():
        design = name.split("_", 1)[0]
        actual = frame.loc[splits[design]["test"]].set_index("SOURCE_ROW_ID")
        if (
            not frame_columns.issubset(saved)
            or saved[list(frame_columns)].isna().any().any()
            or not saved.SOURCE_ROW_ID.is_unique
            or set(saved.SOURCE_ROW_ID) != set(actual.index)
        ):
            raise ValueError(f"Core reference {name} has different test identities.")
        aligned = saved.set_index("SOURCE_ROW_ID").loc[actual.index]
        for column in ("FINANCIAL_YEAR", "LARGER_FIRE"):
            if not np.array_equal(aligned[column].to_numpy(), actual[column].to_numpy()):
                raise ValueError(f"Core reference {name} has different {column} values.")
        if "probability" not in aligned or not np.isfinite(aligned.probability).all():
            raise ValueError(f"Core reference {name} lacks finite probabilities.")
        if not aligned.probability.between(0, 1).all():
            raise ValueError(f"Core reference {name} has invalid probabilities.")
    return splits


def experiment_plan(audit: dict, selection: dict, cfg: dict) -> list[dict]:
    """Freeze exactly the requested four fits without any test-driven selection."""
    if cfg["baseline_logistic_c"] != 1.0:
        raise ValueError("This bounded protocol fixes the single-field baseline C at 1.0.")
    if cfg["xgboost_device"] != "cuda" or selection["xgboost_device"] != "cuda":
        raise ValueError("The review protocol preserves the core CUDA setting.")
    original_columns = audit["feature_blocks"]["B"]
    if "OCCUPIED_TIME" not in original_columns or len(set(original_columns)) != len(original_columns):
        raise ValueError("Core Block B must contain OCCUPIED_TIME and unique fields.")
    columns = [column for column in original_columns if column != "OCCUPIED_TIME"]
    plan = []
    for design in ("random", "temporal"):
        if selection["selected_family_by_block"][design]["B"] != "xgboost":
            raise ValueError("The core selected Block B family must be XGBoost.")
        plan.append({
            "experiment_id": f"b_minus_occupied_{design}", "design": design,
            "model": "xgboost", "columns": columns,
            "parameters": selection["selected_hyperparameters"][design]["xgboost"],
            "reference_key": f"{design}_B", "reference_block": "B",
            "purpose": "Fixed-configuration removal sensitivity for OCCUPIED_TIME",
        })
    for experiment_id, column, block in (
        ("building_type_temporal", "BUILDING_TYPE", "B"),
        ("arrival_size_temporal", "FIRE_SIZE_ON_ARRIVAL", "C"),
    ):
        plan.append({
            "experiment_id": experiment_id, "design": "temporal",
            "model": "logistic_regression", "columns": [column],
            "parameters": {"C": 1.0}, "reference_key": f"temporal_{block}",
            "reference_block": block,
            "purpose": (
                "Single-field categorical baseline with fixed C=1.0 and no search; "
                "comparison changes information and model family, not a pure feature increment"
            ),
        })
    return plan


def _output_directory(cfg: dict) -> Path:
    expected = (ROOT / "outputs/diagnostics").resolve()
    target = (ROOT / cfg["output_dir"]).resolve()
    if target != expected:
        raise ValueError("Review experiments may write only to outputs/diagnostics.")
    return target


def _prepare_inputs(cfg: dict) -> tuple[pd.DataFrame, dict, list[dict], dict, dict]:
    analysis = yaml.safe_load((ROOT / "config/analysis.yaml").read_text(encoding="utf-8"))
    selection = _read_json(ROOT / "outputs/metrics/model_selection.json")
    audit = _read_json(ROOT / "outputs/metrics/audit_receipt.json")
    archive = _read_json(ROOT / "outputs/metrics/data_archive_manifest.json")
    source_path = ROOT / analysis["raw_path"]
    core_source = next(item for item in archive["files"] if item["path"] == analysis["raw_path"])
    metadata = _read_json(ROOT / "data/raw/source_metadata.json")
    source_hash = file_sha256(source_path)
    if source_hash != core_source["sha256"] or source_hash != metadata["sha256"]:
        raise ValueError("Source ODS differs from the current core manifest/metadata; refusing fits.")
    frame = pd.read_parquet(ROOT / analysis["cohort_path"])
    assignments = pd.read_csv(ROOT / "outputs/tables/split_assignments.csv")
    references = {key: pd.read_parquet(ROOT / path) for key, path in REFERENCE_PREDICTIONS.items()}
    splits = validate_saved_partitions(frame, assignments, references, selection)
    reference_checks = {}
    for key, predictions in references.items():
        design, block = key.split("_", 1)
        performance = pd.read_csv(ROOT / f"outputs/tables/{design}_validation_performance.csv")
        family = selection["selected_family_by_block"][design][block]
        core_row = performance[performance.block.eq(block) & performance.model.eq(family)]
        if len(core_row) != 1:
            raise ValueError(f"Expected one core performance row for {key}/{family}.")
        observed_ap = float(average_precision_score(predictions.LARGER_FIRE, predictions.probability))
        expected = core_row.iloc[0]
        if (
            int(expected.n) != len(predictions)
            or int(expected.positive_count) != int(predictions.LARGER_FIRE.sum())
            or not np.isclose(observed_ap, float(expected.pr_auc), rtol=1e-10, atol=1e-12)
        ):
            raise ValueError(f"Core predictions disagree with saved counts/AP for {key}.")
        reference_checks[key] = {"model": family, "n": len(predictions), "average_precision": observed_ap}
    plan = experiment_plan(audit, selection, cfg)
    required_features = {column for experiment in plan for column in experiment["columns"]}
    if not required_features.issubset(frame.columns):
        raise ValueError(f"Cohort lacks required features: {sorted(required_features - set(frame.columns))}")
    input_paths = (
        "config/review_experiments.yaml", "config/analysis.yaml",
        "data/raw/source_metadata.json", analysis["raw_path"], analysis["cohort_path"],
        "outputs/tables/split_assignments.csv", "outputs/metrics/model_selection.json",
        "outputs/metrics/audit_receipt.json", "outputs/metrics/data_archive_manifest.json",
        "outputs/tables/random_validation_performance.csv",
        "outputs/tables/temporal_validation_performance.csv",
        *REFERENCE_PREDICTIONS.values(),
    )
    input_hashes = {path: file_sha256(ROOT / path) for path in input_paths}
    code_hashes = {path: file_sha256(ROOT / path) for path in CODE_INPUTS}
    identity = {
        "input_sha256": input_hashes, "code_sha256": code_hashes,
        "config": cfg, "experiment_plan": plan, "estimator_seed": int(selection["seed"]),
        "n_jobs": int(analysis["n_jobs"]), "reference_prediction_checks": reference_checks,
    }
    return frame, splits, plan, identity, selection


def _log(output: Path, event: str, experiment_id: str, **extra: Any) -> None:
    record = {"timestamp_utc": _utc_now(), "event": event, "experiment_id": experiment_id, **extra}
    with (output / "fit_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
    print(f"{record['timestamp_utc']} {experiment_id}: {event}", flush=True)


def _verify_completed_experiment(output: Path, entry: dict) -> None:
    for relative, expected in entry["artifact_sha256"].items():
        if file_sha256(output / relative) != expected:
            raise ValueError(f"Completed experiment artifact was changed: {relative}")


def _fit_one(
    frame: pd.DataFrame, split: dict, experiment: dict, *,
    output: Path, seed: int, n_jobs: int, device: str,
) -> dict:
    experiment_id = experiment["experiment_id"]
    columns = experiment["columns"]
    pipeline = make_model_pipeline(
        columns, experiment["model"], experiment["parameters"], seed, n_jobs, device,
    )
    _log(output, "fit_started", experiment_id, training_rows=len(split["train"]))
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(frame.loc[split["train"], columns], frame.loc[split["train"], "LARGER_FIRE"])
    seconds = time.perf_counter() - start
    actual_device = "cpu (LogisticRegression)"
    booster_config_sha256 = None
    if experiment["model"] == "xgboost":
        booster_config_text = pipeline.named_steps["model"].get_booster().save_config()
        actual_device = json.loads(booster_config_text)["learner"]["generic_param"]["device"]
        if not actual_device.startswith("cuda"):
            raise RuntimeError(f"XGBoost actually fitted on {actual_device}; this protocol requires CUDA.")
        booster_config_sha256 = hashlib.sha256(booster_config_text.encode("utf-8")).hexdigest()
        _log(output, "cuda_fit_verified", experiment_id, actual_device=actual_device)
    validation_probability = pipeline.predict_proba(frame.loc[split["validation"], columns])[:, 1]
    threshold = choose_f1_threshold(
        frame.loc[split["validation"], "LARGER_FIRE"].to_numpy(), validation_probability,
    )
    _log(output, "validation_threshold_frozen", experiment_id, threshold=threshold)
    model_relative = f"models/{experiment_id}.joblib"
    joblib.dump(pipeline, output / model_relative, compress=3)
    artifacts = [model_relative]
    rows = []
    for partition in ("validation", "test"):
        subset = frame.loc[split[partition]]
        probability = validation_probability if partition == "validation" else pipeline.predict_proba(subset[columns])[:, 1]
        metrics = classification_metrics(subset.LARGER_FIRE.to_numpy(), probability, threshold)
        rows.append({
            "experiment_id": experiment_id, "design": experiment["design"],
            "split_role": partition, "model": experiment["model"],
            "reference_block": experiment["reference_block"],
            "feature_count": len(columns), "fit_seconds": seconds,
            "parameters": json.dumps(experiment["parameters"], sort_keys=True), **metrics,
        })
        prediction = subset[["SOURCE_ROW_ID", "FINANCIAL_YEAR", "BUILDING_TYPE", "LARGER_FIRE"]].copy()
        prediction.insert(0, "cohort_index", subset.index)
        prediction["probability"] = probability
        prediction["prediction"] = (probability >= threshold).astype("int8")
        relative = f"predictions/{experiment_id}_{partition}.parquet"
        prediction.to_parquet(output / relative, index=False)
        artifacts.append(relative)
    _log(output, "fit_completed", experiment_id, fit_seconds=seconds)
    return {
        "status": "completed", "completed_at_utc": _utc_now(), "threshold": threshold,
        "fit_seconds": seconds, "metrics": rows,
        "actual_model_device": actual_device, "booster_config_sha256": booster_config_sha256,
        "artifact_sha256": {relative: file_sha256(output / relative) for relative in artifacts},
    }


def run_review_fits(*, resume: bool = False) -> Path:
    """Run exactly four fits, or resume verified incomplete work without refitting it."""
    cfg = yaml.safe_load((ROOT / "config/review_experiments.yaml").read_text(encoding="utf-8"))
    output = _output_directory(cfg)
    receipt_path = output / "experiment_receipt.json"
    existing = _read_json(receipt_path) if receipt_path.exists() else None
    if existing and existing.get("status") == "fits_completed":
        raise FileExistsError("Four fits are already complete. Use --uncertainty-only; completed fits are never overwritten.")
    preexisting = {path.name for path in output.iterdir()} if output.exists() else set()
    if preexisting and not (resume and existing):
        raise FileExistsError("Review output is nonempty. Use --resume only for an interrupted run with its receipt.")
    frame, splits, plan, identity, selection = _prepare_inputs(cfg)
    if existing and existing["identity"] != identity:
        raise ValueError("Resume inputs, protocol or implementation differ from the recorded run.")
    device = resolve_xgb_device(cfg["xgboost_device"])
    if existing:
        current_runtime = training_environment(device, identity["n_jobs"])
        if existing["training_environment"] != current_runtime:
            raise ValueError("Resume environment differs from the original fit environment.")
        if existing["gpu"] != _gpu_receipt():
            raise ValueError("Resume GPU/driver differs from the original fit environment.")
        receipt = existing
        for entry in receipt["experiments"].values():
            if entry.get("status") == "completed":
                _verify_completed_experiment(output, entry)
    else:
        for relative in ("models", "predictions"):
            (output / relative).mkdir(parents=True, exist_ok=True)
        receipt = {
            "record_type": "post_review_exploratory_experiments",
            "status": "fitting", "started_at_utc": _utc_now(),
            "interpretation": cfg["status"], "identity": identity,
            "git_head_at_start": _git_value("rev-parse", "HEAD"),
            "git_tracked_worktree_status_at_start": _git_value("status", "--porcelain", "--untracked-files=no"),
            "training_environment": training_environment(device, identity["n_jobs"]),
            "gpu": _gpu_receipt(),
            "reference_predictions": {
                key: {"path": path, "sha256": identity["input_sha256"][path]}
                for key, path in REFERENCE_PREDICTIONS.items()
            },
            "partition_sizes": {
                design: {part: len(indices) for part, indices in split.items()}
                for design, split in splits.items()
            },
            "threshold_rule": "Maximum validation F1; test data do not select settings or thresholds",
            "experiments": {},
        }
        _write_json(output / "config_snapshot.json", cfg)
        _write_json(receipt_path, receipt)
    for experiment in plan:
        experiment_id = experiment["experiment_id"]
        if receipt["experiments"].get(experiment_id, {}).get("status") == "completed":
            _log(output, "verified_completed_fit_reused", experiment_id)
            continue
        receipt["experiments"][experiment_id] = {"status": "fitting", "started_at_utc": _utc_now()}
        _write_json(receipt_path, receipt)
        try:
            entry = _fit_one(
                frame, splits[experiment["design"]], experiment, output=output,
                seed=int(selection["seed"]), n_jobs=identity["n_jobs"], device=device,
            )
        except Exception as error:
            receipt["experiments"][experiment_id] = {
                "status": "failed", "failed_at_utc": _utc_now(), "error": str(error),
            }
            _write_json(receipt_path, receipt)
            _log(output, "fit_failed", experiment_id, error=str(error))
            raise
        receipt["experiments"][experiment_id] = entry
        _write_json(receipt_path, receipt)
        rows = [row for result in receipt["experiments"].values() for row in result.get("metrics", [])]
        pd.DataFrame(rows).to_csv(output / "performance.csv", index=False)
    # Rebuild this summary even when a crash happened after the last per-fit receipt.
    rows = [row for experiment in plan for row in receipt["experiments"][experiment["experiment_id"]]["metrics"]]
    pd.DataFrame(rows).to_csv(output / "performance.csv", index=False)
    # Recheck every input to avoid silently accepting concurrent artifact changes.
    for relative, expected in identity["input_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"An input changed while fitting: {relative}")
    receipt["status"] = "fits_completed"
    receipt["completed_at_utc"] = _utc_now()
    receipt["performance_sha256"] = file_sha256(output / "performance.csv")
    _write_json(receipt_path, receipt)
    return output


def run_review_uncertainty(*, overwrite: bool = False) -> Path:
    """Compute intervals from recorded predictions, without requiring data or CUDA."""
    from .review_uncertainty import METHOD_NOTES, paired_model_ap_intervals, split_ap_intervals

    cfg = yaml.safe_load((ROOT / "config/review_experiments.yaml").read_text(encoding="utf-8"))
    output = _output_directory(cfg)
    receipt = _read_json(output / "experiment_receipt.json")
    if receipt["status"] != "fits_completed":
        raise ValueError("Finish all four fits before computing review uncertainty.")
    if receipt["identity"]["config"] != cfg:
        raise ValueError("Current review configuration differs from the frozen fit protocol.")
    if (output / "uncertainty_receipt.json").exists() and not overwrite:
        raise FileExistsError("Uncertainty is already recorded. Use --overwrite-uncertainty to explicitly recompute it.")
    if file_sha256(output / "performance.csv") != receipt["performance_sha256"]:
        raise ValueError("Review performance table differs from its completion receipt.")
    plan = receipt["identity"]["experiment_plan"]
    inputs = {}
    predictions = {}
    for experiment in plan:
        experiment_id = experiment["experiment_id"]
        _verify_completed_experiment(output, receipt["experiments"][experiment_id])
        relative = f"predictions/{experiment_id}_test.parquet"
        inputs[relative] = file_sha256(output / relative)
        predictions[experiment_id] = pd.read_parquet(output / relative)
    references = {}
    for key, saved in receipt["reference_predictions"].items():
        if file_sha256(ROOT / saved["path"]) != saved["sha256"]:
            raise ValueError(f"Core reference changed: {key}")
        references[key] = pd.read_parquet(ROOT / saved["path"])
        inputs[saved["path"]] = saved["sha256"]
    repeats, seed = int(cfg["bootstrap_repeats"]), int(cfg["bootstrap_seed"])
    start = time.perf_counter()
    tables = []
    for experiment in plan:
        experiment_id = experiment["experiment_id"]
        _log(output, "paired_bootstrap_started", experiment_id, repeats=repeats)
        table = paired_model_ap_intervals(
            predictions[experiment_id], references[experiment["reference_key"]],
            design=experiment["design"], model=experiment_id,
            reference_model=f"core_{experiment['reference_key']}", repeats=repeats, seed=seed,
        )
        table.insert(0, "experiment_id", experiment_id)
        tables.append(table)
        _log(output, "paired_bootstrap_completed", experiment_id)
    _log(output, "partially_paired_bootstrap_started", "b_minus_occupied_random_minus_temporal", repeats=repeats)
    table = split_ap_intervals(
        predictions["b_minus_occupied_random"], predictions["b_minus_occupied_temporal"],
        model="b_minus_occupied", repeats=repeats, seed=seed,
    )
    table.insert(0, "experiment_id", "b_minus_occupied_random_minus_temporal")
    tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(output / "uncertainty.csv", index=False)
    runtime = training_environment("not_used", 1)
    runtime["record_type"] = "fixed_model_uncertainty_environment"
    _write_json(output / "uncertainty_receipt.json", {
        "record_type": "post_review_fixed_model_uncertainty", "completed_at_utc": _utc_now(),
        "bootstrap_repeats": repeats, "bootstrap_seed": seed,
        "elapsed_seconds": time.perf_counter() - start, "method_notes": METHOD_NOTES,
        "interpretation": cfg["status"], "input_sha256": inputs,
        "code_sha256": file_sha256(ROOT / "src/fireml/review_uncertainty.py"),
        "output_sha256": file_sha256(output / "uncertainty.csv"),
        "runtime_environment": runtime,
    })
    _log(output, "all_uncertainty_completed", "diagnostics")
    return output
