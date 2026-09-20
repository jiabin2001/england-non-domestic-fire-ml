"""Rebuild the complete study and diagnostics; --clean removes previous outputs."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

from fireml.acquire import acquire_and_cache, sha256_file
from fireml.audit import run_audit
from fireml.cohort import construct_cohort
from fireml.config import ROOT, ensure_output_dirs, load_yaml
from fireml.interpretability import run_grouped_permutation_analysis
from fireml.modelling import run_core_models
from fireml.reporting import build_report, write_data_archive_manifest
from fireml.review_experiments import run_review_fits, run_review_uncertainty
from fireml.robustness import run_temporal_robustness
from fireml.uncertainty import run_uncertainty_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Replace outputs/ with a fresh complete run.")
    args = parser.parse_args()
    cfg = load_yaml("config/analysis.yaml")
    if not (ROOT / cfg["raw_path"]).is_file():
        parser.error("Restore the official ODS before running the complete study; diagnostics verify its checksum.")
    metadata_path = ROOT / "data/raw/source_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(ROOT / cfg["raw_path"]) != metadata["sha256"]:
            parser.error("ODS checksum differs from recorded source metadata. Resolve the data version before replacing outputs.")
    output = ROOT / "outputs"
    if output.exists() and any(output.iterdir()) and not args.clean:
        parser.error("Existing outputs found. Use --clean to replace them, or 07_render_report.py to render only.")
    # Resolve the final target before recursive removal; reject links outside ROOT.
    if output.exists() and args.clean:
        if output.is_symlink() or output.resolve() != ROOT.resolve() / "outputs":
            raise ValueError("Refusing to clear an output directory outside the project.")
        shutil.rmtree(output)
    ensure_output_dirs()
    receipt_path = ROOT / "outputs/metrics/execution.json"
    inputs = sorted([
        *ROOT.glob("src/fireml/*.py"), *ROOT.glob("config/*.yaml"),
        *ROOT.glob("scripts/*.py"), ROOT / "pyproject.toml",
    ])
    receipt = {
        "record_type": "complete_study_execution",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_and_config_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "stages": [],
        "interpretation": "A fresh computation on previously examined periods; not new prospective validation.",
    }

    def save() -> None:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    save()
    stages = (
        ("source import/cache", acquire_and_cache), ("audit", run_audit),
        ("cohort", construct_cohort), ("data checksums", write_data_archive_manifest),
        ("core model selection and fitting", run_core_models),
        ("core uncertainty", run_uncertainty_analysis),
        ("grouped permutation", run_grouped_permutation_analysis),
        ("temporal and split robustness", run_temporal_robustness),
        ("diagnostic fits", run_review_fits),
        ("diagnostic uncertainty", run_review_uncertainty),
        ("reports and figures", build_report),
    )
    try:
        for name, action in stages:
            stage = {"name": name, "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
            receipt["stages"].append(stage)
            save()
            print(f"Starting: {name}", flush=True)
            action()
            stage.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
            save()
            print(f"Completed: {name}", flush=True)
    except Exception as error:
        stage.update(status="failed", finished_at_utc=datetime.now(timezone.utc).isoformat())
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    receipt.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print("Complete study and diagnostics rebuilt. See reports/final_analysis_report.md.", flush=True)


if __name__ == "__main__":
    main()
