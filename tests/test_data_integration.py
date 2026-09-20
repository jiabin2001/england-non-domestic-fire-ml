"""Optional checks of externally archived data; these never rebuild the cohort."""

import hashlib
import json

import pandas as pd
import pytest

from fireml.config import ROOT, load_yaml
from fireml.features import resolve_blocks
from fireml.splits import make_temporal_split


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def archive():
    manifest = json.loads(
        (ROOT / "outputs/metrics/data_archive_manifest.json").read_text(encoding="utf-8")
    )
    missing = [item["path"] for item in manifest["files"] if not (ROOT / item["path"]).is_file()]
    if missing:
        pytest.fail(
            "Integration checks require the archived research data. Restore the exact "
            "files described in README.md before using -m integration. Missing: "
            + ", ".join(missing),
            pytrace=False,
        )
    return manifest


def test_data_archive_manifest_matches_local_recovery_files(archive):
    assert archive["external_archive_required"] is True
    assert len(archive["files"]) == 3
    for item in archive["files"]:
        local = ROOT / item["path"]
        assert item["exists"] is True
        assert item["size_bytes"] == local.stat().st_size
        assert item["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()


def test_archived_cohort_matches_saved_temporal_assignments(archive):
    cfg = load_yaml("config/analysis.yaml")
    frame = pd.read_parquet(ROOT / cfg["cohort_path"])
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    assert resolve_blocks(frame.columns) == audit["feature_blocks"]
    split = make_temporal_split(
        frame, audit["temporal_train_years"], audit["temporal_validation_years"],
        audit["temporal_test_years"],
    )
    assignments = pd.read_csv(ROOT / "outputs/tables/split_assignments.csv")
    for partition, indices in split.items():
        saved = assignments[
            (assignments["design"] == "temporal") & (assignments["partition"] == partition)
        ]
        assert saved["cohort_index"].tolist() == indices.tolist()
        assert saved["source_row_id"].tolist() == frame.loc[indices, "SOURCE_ROW_ID"].tolist()
