from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ROOT, load_yaml
from .target import map_target


def _add_flow(rows: list[dict], step: str, before: int, after: int, reason: str) -> None:
    rows.append({
        "step": len(rows) + 1,
        "stage": step,
        "records_before": int(before),
        "records_excluded": int(before - after),
        "records_remaining": int(after),
        "reason": reason,
    })


def construct_cohort(
    roofs_positive: bool = False,
    include_late_calls: bool = False,
    include_2024_excluding_suffolk: bool = False,
    save_main: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_yaml("config/analysis.yaml")
    frame = pd.read_parquet(ROOT / cfg["parquet_path"]).copy()
    frame.insert(0, "SOURCE_ROW_ID", np.arange(len(frame), dtype=np.int64))
    flow: list[dict] = []
    _add_flow(flow, "Official incident dataset", len(frame), len(frame), "All disclosed primary other-building fire records.")

    years = list(cfg["preferred_main_years"])
    if include_2024_excluding_suffolk:
        years.append("2024/25")
    before = len(frame)
    frame = frame[frame["FINANCIAL_YEAR"].isin(years)].copy()
    _add_flow(flow, "Restrict analysis years", before, len(frame), f"Retain {years[0]} to {years[-1]} only.")

    if include_2024_excluding_suffolk:
        before = len(frame)
        suffolk = frame["FRS_TERRITORY"].astype(str).str.contains("Suffolk", case=False, na=False)
        frame = frame[~(frame["FINANCIAL_YEAR"].eq("2024/25") & suffolk)].copy()
        _add_flow(flow, "Exclude Suffolk in 2024/25", before, len(frame), "Official guidance reports incomplete Suffolk data September 2024–March 2025.")

    before = len(frame)
    duplicate = frame.drop(columns="SOURCE_ROW_ID").duplicated(keep="first")
    frame = frame.loc[~duplicate].copy()
    _add_flow(flow, "Remove exact duplicates", before, len(frame), "Prespecified complete-row deduplication.")

    if not include_late_calls:
        before = len(frame)
        late = frame["LATE_CALL"].astype(str).str.strip().str.casefold().eq("yes")
        frame = frame.loc[~late].copy()
        _add_flow(flow, "Exclude late calls", before, len(frame), "Fire was known extinguished when the call was made; different prediction scenario.")

    before = len(frame)
    target, mapping = map_target(frame["SPREAD_OF_FIRE"], roofs_positive=roofs_positive)
    frame["LARGER_FIRE"] = target
    frame = frame.loc[frame["LARGER_FIRE"].notna()].copy()
    frame["LARGER_FIRE"] = frame["LARGER_FIRE"].astype("int8")
    action = "Map roofs positive; exclude only unmappable targets." if roofs_positive else "Exclude roofs/roof spaces and other unmappable targets."
    _add_flow(flow, "Apply binary target mapping", before, len(frame), action)

    frame = frame.reset_index(drop=True)
    flow_frame = pd.DataFrame(flow)
    if save_main and not roofs_positive and not include_late_calls and not include_2024_excluding_suffolk:
        frame.to_parquet(ROOT / cfg["cohort_path"], index=False)
        flow_frame.to_csv(ROOT / "outputs/tables/cohort_flow.csv", index=False)
        receipt = {
            "rows": len(frame),
            "positive_count": int(frame["LARGER_FIRE"].sum()),
            "positive_prevalence": float(frame["LARGER_FIRE"].mean()),
            "years": sorted(frame["FINANCIAL_YEAR"].unique()),
            "target_mapping": mapping,
        }
        (ROOT / "outputs/metrics/cohort_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return frame, flow_frame

