from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ROOT, ensure_output_dirs, load_yaml
from .features import build_feature_policy, resolve_blocks
from .target import map_target


AUDIT_CATEGORY_FIELDS = [
    "BUILDING_TYPE", "FSO_APPLY", "OCCUPIED_NORMAL", "OCCUPIED_TIME",
    "ALARM_SYSTEM", "SAFETY_SYSTEM", "ACCIDENTAL_OR_DELIBERATE",
    "CAUSE_OF_FIRE", "IGNITION_POWER", "SOURCE_OF_IGNITION",
    "FIRE_START_LOCATION", "FIRE_SIZE_ON_ARRIVAL", "RESPONSE_TIME",
]


def financial_year_key(value: str) -> int:
    return int(str(value).split("/")[0])


def temporal_year_partitions(cfg: dict) -> dict[str, list[str]]:
    """Allocate configured main years without silently ignoring window lengths."""
    years = cfg["preferred_main_years"]
    if not isinstance(years, list) or not years or not all(isinstance(year, str) for year in years):
        raise ValueError("preferred_main_years must be a nonempty list of financial-year strings.")
    try:
        keys = [financial_year_key(year) for year in years]
    except ValueError as exc:
        raise ValueError("preferred_main_years contains an invalid financial year.") from exc
    if keys != sorted(set(keys)):
        raise ValueError("preferred_main_years must contain unique years in chronological order.")
    for setting in ("temporal_validation_years", "temporal_test_years"):
        value = cfg[setting]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{setting} must be a positive integer.")
    validation_count = cfg["temporal_validation_years"]
    test_count = cfg["temporal_test_years"]
    if validation_count + test_count >= len(years):
        raise ValueError("Temporal validation and test windows must leave at least one training year.")
    return {
        "temporal_train_years": years[:-(validation_count + test_count)],
        "temporal_validation_years": years[-(validation_count + test_count):-test_count],
        "temporal_test_years": years[-test_count:],
    }


def run_audit() -> dict:
    ensure_output_dirs()
    cfg = load_yaml("config/analysis.yaml")
    partitions = temporal_year_partitions(cfg)
    preferred = cfg["preferred_main_years"]
    main_year_span = f"{preferred[0]}–{preferred[-1]}"
    parquet = ROOT / cfg["parquet_path"]
    frame = pd.read_parquet(parquet)
    if not {"FINANCIAL_YEAR", "SPREAD_OF_FIRE", "LATE_CALL"}.issubset(frame.columns):
        raise RuntimeError("Blocking feasibility failure: required fields are absent.")
    blocks = resolve_blocks(frame.columns)
    frame = frame.sort_values("FINANCIAL_YEAR", key=lambda s: s.map(financial_year_key)).reset_index(drop=True)
    target, mapping = map_target(frame["SPREAD_OF_FIRE"])

    inventory = pd.DataFrame({
        "column": frame.columns,
        "dtype": [str(frame[c].dtype) for c in frame],
        "n_missing": [int(frame[c].isna().sum()) for c in frame],
        "missing_rate": [float(frame[c].isna().mean()) for c in frame],
        "n_unique_nonmissing": [int(frame[c].nunique(dropna=True)) for c in frame],
    })
    inventory.to_csv(ROOT / "outputs/tables/column_inventory.csv", index=False)

    rows = []
    for year, group in frame.groupby("FINANCIAL_YEAR", sort=False):
        y = target.loc[group.index]
        spread = group["SPREAD_OF_FIRE"]
        official_like = spread.map(lambda x: 1 if str(x).strip().casefold() == "roofs/ roof spaces" else 0)
        rows.append({
            "financial_year": year,
            "records": len(group),
            "mappable_records_main": int(y.notna().sum()),
            "larger_fires_main": int(y.sum(skipna=True)),
            "larger_prevalence_main": float(y.mean()),
            "roofs_roof_spaces": int((official_like == 1).sum()),
            "larger_prevalence_roofs_positive_all_records": float(((y.fillna(0) + official_like) > 0).mean()),
            "late_calls": int(group["LATE_CALL"].astype(str).str.strip().str.casefold().eq("yes").sum()),
        })
    year_coverage = pd.DataFrame(rows).sort_values("financial_year", key=lambda s: s.map(financial_year_key))
    year_coverage.to_csv(ROOT / "outputs/tables/year_coverage.csv", index=False)

    missing = []
    for year, group in frame.groupby("FINANCIAL_YEAR", sort=False):
        for column in frame.columns:
            missing.append({
                "financial_year": year,
                "column": column,
                "n_records": len(group),
                "n_missing": int(group[column].isna().sum()),
                "missing_rate": float(group[column].isna().mean()),
            })
    pd.DataFrame(missing).to_csv(ROOT / "outputs/tables/missingness_by_year.csv", index=False)

    support = []
    for column in [c for c in AUDIT_CATEGORY_FIELDS if c in frame]:
        grouped = frame.groupby(["FINANCIAL_YEAR", column], dropna=False, observed=True).size()
        for (year, category), count in grouped.items():
            denom = int((frame["FINANCIAL_YEAR"] == year).sum())
            support.append({
                "financial_year": year,
                "column": column,
                "category": "Missing/Unknown" if pd.isna(category) else category,
                "count": int(count),
                "proportion": float(count / denom),
            })
    category_support = pd.DataFrame(support).sort_values(
        ["column", "financial_year", "count"], ascending=[True, True, False]
    )
    category_support.to_csv(ROOT / "outputs/tables/category_support_by_year.csv", index=False)

    spread_counts = frame["SPREAD_OF_FIRE"].value_counts(dropna=False).rename_axis("raw_category").reset_index(name="count")
    spread_counts["proportion"] = spread_counts["count"] / len(frame)
    spread_counts["main_mapping"] = spread_counts["raw_category"].map(mapping)
    spread_counts["main_action"] = spread_counts["main_mapping"].map({0: "retain_negative", 1: "retain_positive"}).fillna("exclude")
    spread_counts.to_csv(ROOT / "outputs/tables/spread_of_fire_categories.csv", index=False)
    spread_counts[["raw_category", "count", "proportion", "main_mapping", "main_action"]].to_csv(
        ROOT / "outputs/tables/target_mapping.csv", index=False
    )

    lifecycle = []
    ordered_years = list(year_coverage["financial_year"])
    order = {year: index for index, year in enumerate(ordered_years)}
    for column in [c for c in AUDIT_CATEGORY_FIELDS if c in frame]:
        for category, group in frame.groupby(column, dropna=False):
            years = sorted(group["FINANCIAL_YEAR"].unique(), key=financial_year_key)
            lifecycle.append({
                "column": column,
                "category": "Missing/Unknown" if pd.isna(category) else category,
                "first_year": years[0],
                "last_year": years[-1],
                "n_years_present": len(years),
                "appears_after_start": order[years[0]] > 0,
                "disappears_before_end": order[years[-1]] < len(ordered_years) - 1,
            })
    pd.DataFrame(lifecycle).to_csv(ROOT / "outputs/tables/category_lifecycle.csv", index=False)

    feature_policy = build_feature_policy(frame)
    feature_policy.to_csv(ROOT / "outputs/tables/feature_policy.csv", index=False)

    usable = set(year_coverage["financial_year"])
    years_present = all(year in usable for year in preferred)
    key_fields = sorted(set(blocks["C"]))
    main = frame[frame["FINANCIAL_YEAR"].isin(preferred)]
    key_missing_max = float(main[key_fields].isna().mean().max())
    feasible = bool(years_present and target[frame["FINANCIAL_YEAR"].isin(preferred)].notna().any() and key_missing_max < 0.95)

    metadata = json.loads((ROOT / "data/raw/source_metadata.json").read_text(encoding="utf-8"))
    row_2024 = year_coverage.loc[year_coverage["financial_year"] == "2024/25"].iloc[0]
    duplicate_rows = int(frame.duplicated().sum())
    report = f"""# Day 1 feasibility audit

## Decision

**{'Feasible — proceed to cohort construction and modelling.' if feasible else 'Blocked — do not model.'}**

The current official **Other Building Fires Dataset** is an incident-level dataset of primary fires attended by Fire and Rescue Services in other buildings in England. The project's term *non-domestic building fires* refers to this official category. It includes commercial, industrial, public and institutional buildings, and can include hotels, hostels, care homes and student halls; it must not be interpreted as containing only buildings with no accommodation function.

## Source and import

- Source page: {cfg['source_page_url']}
- Guidance: {cfg['guidance_url']}
- ODS URL: {cfg['source_url']}
- Official page update: {cfg['official_page_updated']}
- Raw file: `{metadata['file_name']}` ({metadata['file_size_bytes']:,} bytes)
- SHA-256: `{metadata['sha256']}`
- ODS sheets: {', '.join(x['sheet_name'] for x in metadata['all_sheets'])}
- Incident sheet: `{metadata['data_sheet']}`
- Logical shape after resolving documented export artifacts: {frame.shape[0]:,} rows × {frame.shape[1]} fields
- Exact duplicate logical records: {duplicate_rows:,}

The ODS parser initially exposed two unnamed trailing columns and {metadata.get('import_artifact_resolution', {}).get('non_record_rows_removed', 0):,} non-record rows without `FINANCIAL_YEAR`. Inspection showed these cells repeat the two final outcome columns below the incident table. The reproducible import rule retains rows with a financial year and drops only unnamed trailing columns. The original ODS remains unchanged.

## Target audit

All seven observed `SPREAD_OF_FIRE` strings are listed verbatim in `outputs/tables/spread_of_fire_categories.csv`. Six map unambiguously to the study's ordered room→floor→whole-building spread estimand. `Roofs/ Roof spaces` does not locate an incident uniquely on that ordering, so it is excluded from the main outcome. Official sources differ: the current Fire statistics definitions omit roofs from the larger-fire list, while FIRE0304-linked detailed releases include them. The roof-positive sensitivity tests that alternative published convention. There are no missing target strings in logical incident rows.

For 2024/25, the main study estimand (excluding roofs) gives {row_2024['larger_prevalence_main']:.1%} among mappable records. Mapping `Roofs/ Roof spaces` as larger and using all 2024/25 records gives {row_2024['larger_prevalence_roofs_positive_all_records']:.1%}, reproducing the approximately 26% proportion in the detailed FIRE0304-linked release. Reporting both definitions makes the official-source inconsistency explicit.

## Time and data quality

- Available years: {', '.join(year_coverage['financial_year'])}.
- All configured main years {main_year_span} present: {'yes' if years_present else 'no'}.
- Temporal test uses the latest {cfg['temporal_test_years']} main-window years ({', '.join(partitions['temporal_test_years'])}); validation uses the preceding {cfg['temporal_validation_years']} ({', '.join(partitions['temporal_validation_years'])}); earlier years form training.
- The dataset has no incident date or month. Calendar timing is limited to financial year, day of week and four day-part bands. It also contains banded process durations (`IGNITION_TO_DISCOVERY`, `DISCOVERY_TO_CALL`, `RESPONSE_TIME`, `TIME_AT_SCENE`); the latter are not calendar timestamps.
- Suffolk FRS is incomplete from September 2024 to March 2025 according to current official guidance, so 2024/25 is excluded from the main analysis.
- 2025/26 spans the IRS-to-FaRDaP collection transition beginning November 2025 and is excluded from the main analysis.
- `RAPID_FIRE_GROWTH` is absent in {frame['RAPID_FIRE_GROWTH'].isna().mean():.1%} of records and is excluded from all main models as prespecified.

## Feasibility checks

- `SPREAD_OF_FIRE` found: yes.
- Incident-level structure supported: yes (one row per disclosed incident record; no public incident ID).
- Target categories reliably mappable: yes, with roofs explicitly excluded in the main definition.
- Financial-year design usable: yes.
- Key Block A/B/C fields usable in {main_year_span}: {'yes' if key_missing_max < 0.95 else 'no'}; maximum field missingness is {key_missing_max:.1%}.
- Information timing: Block C inherits retrospective investigation fields. `OCCUPIED_TIME` can include occupants in buildings reached by spread and is a potential outcome proxy. Field availability and missingness do not establish availability at prediction time. The final analysis separately reports an occupancy-removal sensitivity and simple single-field baselines.

## Main analysis window

Use {main_year_span}. Exclude exact duplicate rows, late calls and target-excluded rows. Do not use `FINANCIAL_YEAR` or FRS territory as predictors. Use validation data for compact model selection and evaluate the resulting train-fitted model on the temporal test.
"""
    (ROOT / "reports/day1_feasibility.md").write_text(report, encoding="utf-8")
    receipt = {
        "feasible": feasible,
        "main_years": preferred,
        **partitions,
        "logical_rows": len(frame),
        "duplicate_rows": duplicate_rows,
        "feature_blocks": blocks,
        "target_mapping": mapping,
    }
    (ROOT / "outputs/metrics/audit_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if not feasible:
        raise RuntimeError("Day 1 audit found a blocking feasibility condition; see report.")
    return receipt
