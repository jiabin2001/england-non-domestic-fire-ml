import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from fireml import acquire


@pytest.mark.parametrize(
    ("recorded_timestamp", "source_present"),
    [("2026-07-22T00:00:00+00:00", False), (None, False), (None, True)],
)
def test_cached_import_uses_available_provenance_without_requiring_ods(
    tmp_path, monkeypatch, recorded_timestamp, source_present
):
    raw = tmp_path / "data/raw/source.ods"
    cache = tmp_path / "data/interim/source.parquet"
    raw.parent.mkdir(parents=True)
    cache.parent.mkdir(parents=True)
    cache.touch()
    if source_present:
        raw.touch()
    metadata_path = raw.parent / "source_metadata.json"
    metadata = {"sha256": "historical-source-checksum"}
    if recorded_timestamp is not None:
        metadata["download_timestamp_utc"] = recorded_timestamp
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    cached_frame = pd.DataFrame({"FINANCIAL_YEAR": ["2020/21"]})

    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "ensure_output_dirs", lambda: None)
    monkeypatch.setattr(acquire, "load_yaml", lambda _: {
        "raw_path": "data/raw/source.ods", "parquet_path": "data/interim/source.parquet"
    })
    monkeypatch.setattr(acquire.pd, "read_parquet", lambda path: cached_frame.copy())

    def unexpected_ods_read(*args, **kwargs):
        pytest.fail("A valid Parquet cache must not trigger an ODS import.")

    monkeypatch.setattr(acquire.pd, "read_excel", unexpected_ods_read)
    frame, result = acquire.acquire_and_cache()

    pd.testing.assert_frame_equal(frame, cached_frame)
    assert result["sha256"] == "historical-source-checksum"
    if recorded_timestamp is not None:
        assert result["download_timestamp_utc"] == recorded_timestamp
    elif source_present:
        assert result["download_timestamp_utc"] == datetime.fromtimestamp(
            raw.stat().st_mtime, timezone.utc
        ).isoformat()
    else:
        assert "download_timestamp_utc" not in result
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == result
