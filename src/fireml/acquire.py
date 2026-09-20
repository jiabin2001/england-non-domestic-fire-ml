from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT, ensure_output_dirs, load_yaml


REQUIRED_COLUMNS = {"FINANCIAL_YEAR", "SPREAD_OF_FIRE"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_column_names(columns: pd.Index) -> list[str]:
    return [str(value).strip().upper() for value in columns]


def clean_import_artifacts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Resolve non-record cells produced by the current ODS export layout.

    The July 2026 ODS has two trailing, unnamed columns populated below the
    incident table. Rows without FINANCIAL_YEAR are not incident records. The
    rule is structural and is recorded rather than silently applied.
    """
    import_shape = [int(frame.shape[0]), int(frame.shape[1])]
    unnamed = [column for column in frame if column.startswith("UNNAMED:")]
    non_record_rows = int(frame["FINANCIAL_YEAR"].isna().sum())
    clean = frame.loc[frame["FINANCIAL_YEAR"].notna()].drop(columns=unnamed).copy()
    artifact = {
        "ods_import_shape": import_shape,
        "logical_incident_shape": [int(clean.shape[0]), int(clean.shape[1])],
        "non_record_rows_removed": non_record_rows,
        "unnamed_columns_removed": unnamed,
        "rule": "Retain rows with FINANCIAL_YEAR and drop unnamed trailing columns.",
    }
    return clean, artifact


def acquire_and_cache(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Read the immutable ODS once, identify its incident sheet, and cache Parquet."""
    ensure_output_dirs()
    cfg = load_yaml("config/analysis.yaml")
    raw_path = ROOT / cfg["raw_path"]
    parquet_path = ROOT / cfg["parquet_path"]
    metadata_path = ROOT / "data/raw/source_metadata.json"

    if parquet_path.exists() and not force:
        frame = pd.read_parquet(parquet_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Cached data and its recorded provenance can be used without the ODS.
        # Do not evaluate raw_path.stat() unless a timestamp actually needs filling,
        # and do not invent a historical download time when the source is absent.
        if "download_timestamp_utc" not in metadata and raw_path.exists():
            metadata["download_timestamp_utc"] = datetime.fromtimestamp(
                raw_path.stat().st_mtime, timezone.utc
            ).isoformat()
        if any(column.startswith("UNNAMED:") for column in frame.columns):
            frame, artifact = clean_import_artifacts(frame)
            frame.to_parquet(parquet_path, index=False)
            metadata["import_artifact_resolution"] = artifact
            metadata["data_shape"] = [int(frame.shape[0]), int(frame.shape[1])]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if "all_sheets" in metadata:
            pd.DataFrame(metadata["all_sheets"]).to_csv(
                ROOT / "outputs/tables/ods_sheet_inventory.csv", index=False
            )
        return frame, metadata
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Official ODS missing at {raw_path}. Download it from {cfg['source_url']}."
        )

    # A single pandas call imports every sheet once. Only the identified data sheet
    # is retained; every downstream step reads Parquet, never the ODS.
    sheets = pd.read_excel(raw_path, sheet_name=None, engine="odf", dtype=str)
    sheet_inventory: list[dict] = []
    candidates: list[tuple[str, pd.DataFrame]] = []
    for sheet_name, sheet in sheets.items():
        sheet.columns = normalise_column_names(sheet.columns)
        sheet_inventory.append(
            {"sheet_name": sheet_name, "rows": len(sheet), "columns": len(sheet.columns)}
        )
        if REQUIRED_COLUMNS.issubset(sheet.columns):
            candidates.append((sheet_name, sheet))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one incident sheet containing {sorted(REQUIRED_COLUMNS)}; "
            f"found {[name for name, _ in candidates]}."
        )
    data_sheet, frame = candidates[0]
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    frame, artifact = clean_import_artifacts(frame)
    frame.to_parquet(parquet_path, index=False)
    pd.DataFrame(sheet_inventory).to_csv(
        ROOT / "outputs/tables/ods_sheet_inventory.csv", index=False
    )

    stat = raw_path.stat()
    metadata = {
        "official_dataset_name": "Other Building Fires Dataset",
        "source_page_url": cfg["source_page_url"],
        "guidance_url": cfg["guidance_url"],
        "download_url": cfg["source_url"],
        "download_timestamp_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "download_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_page_updated": cfg["official_page_updated"],
        "file_name": raw_path.name,
        "file_size_bytes": stat.st_size,
        "sha256": sha256_file(raw_path),
        "all_sheets": sheet_inventory,
        "data_sheet": data_sheet,
        "data_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "import_artifact_resolution": artifact,
        "python": platform.python_version(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame, metadata
