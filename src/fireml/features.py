from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import load_yaml


OFFICIAL_MEANINGS = {
    "FRS_TERRITORY": "The FRS territory in which the other-building fire took place",
    "E_CODE_TERRITORY": "The National Statistics E code of the FRS territory",
    "FINANCIAL_YEAR": "The financial year in which the other-building fire took place",
    "DAY_OF_WEEK": "Day of week on which the fire took place",
    "DAY_NIGHT": "Time-of-day band in which the fire took place",
    "BUILDING_TYPE": "Type of building in which the fire occurred",
    "FSO_APPLY": "Whether the Regulatory Reform (Fire Safety) Order 2005 applies",
    "OCCUPIED_NORMAL": "Whether the building was normally occupied",
    "OCCUPIED_TIME": "Whether the building was occupied at the time of the fire, including people in buildings to which the fire spread",
    "ALARM_SYSTEM": "Whether a smoke alarm was present and whether it operated/raised the alarm",
    "SAFETY_SYSTEM": "Whether a safety system was present and whether it operated",
    "IGNITION_TO_DISCOVERY": "Approximate time from ignition to discovery",
    "DISCOVERY_TO_CALL": "Approximate time from discovery to the emergency-services call",
    "LATE_CALL": "Whether the fire was known to be extinguished when the call was made",
    "ACCIDENTAL_OR_DELIBERATE": "Whether the incident was recorded as accidental or deliberate",
    "CAUSE_OF_FIRE": "Main recorded cause of the fire",
    "IGNITION_POWER": "What powered the ignition",
    "SOURCE_OF_IGNITION": "Source of ignition",
    "FIRE_START_LOCATION": "Room or compartment in which the fire started",
    "FIRE_SIZE_ON_ARRIVAL": "Extent of fire when the first FRS crew arrived",
    "OTHER_PROPERTY_AFFECTED_ON_ARRIVAL": "Whether an adjacent property was affected on first arrival",
    "ITEM_IGNITED": "Item or material ignited first",
    "ITEM_CAUSING_SPREAD": "Item or material mainly responsible for fire spread",
    "RAPID_FIRE_GROWTH": "Whether rapid fire growth was recorded",
    "VEHICLES": "Banded number of FRS vehicles attending",
    "VEHICLES_CODE": "Analysis code for VEHICLES",
    "PERSONNEL": "Banded number of FRS personnel attending",
    "PERSONNEL_CODE": "Analysis code for PERSONNEL",
    "RESPONSE_TIME": "Banded time from call to first FRS vehicle arrival",
    "RESPONSE_TIME_CODE": "Analysis code for RESPONSE_TIME",
    "TIME_AT_SCENE": "Time from first vehicle arrival until incident closure",
    "TIME_AT_SCENE_CODE": "Analysis code for TIME_AT_SCENE",
    "FATALITY_CASUALTY": "Whether the incident involved a fatality or casualty",
    "RESCUES": "Number of people rescued from the building",
    "FIRE_DAMAGE_EXTENT": "Horizontal area damaged by flame and heat at fire stop",
    "FIRE_DAMAGE_EXTENT_CODE": "Analysis code for FIRE_DAMAGE_EXTENT",
    "TOTAL_DAMAGE_EXTENT": "Horizontal area damaged by flame, heat, smoke and water",
    "TOTAL_DAMAGE_EXTENT_CODE": "Analysis code for TOTAL_DAMAGE_EXTENT",
    "SPREAD_OF_FIRE": "Extent of fire when the incident was closed",
    "OTHER_PROPERTY_AFFECTED_CLOSE": "Whether an adjacent property was affected by incident closure",
}


TIMING = {
    "FRS_TERRITORY": "incident context",
    "E_CODE_TERRITORY": "incident context",
    "FINANCIAL_YEAR": "call time (derived)",
    "DAY_OF_WEEK": "call time (derived)",
    "DAY_NIGHT": "call time (derived)",
    "BUILDING_TYPE": "incident context; retrospectively recorded",
    "FSO_APPLY": "incident context; retrospectively recorded",
    "OCCUPIED_NORMAL": "incident context; retrospectively recorded",
    "OCCUPIED_TIME": "retrospectively recorded; may reflect occupancy in buildings reached by fire spread",
    "ALARM_SYSTEM": "incident circumstances; retrospectively recorded",
    "SAFETY_SYSTEM": "incident circumstances; retrospectively recorded",
    "IGNITION_TO_DISCOVERY": "estimated incident history",
    "DISCOVERY_TO_CALL": "estimated incident history",
    "LATE_CALL": "known at call/recorded incident status",
    "ACCIDENTAL_OR_DELIBERATE": "investigative/retrospective",
    "CAUSE_OF_FIRE": "investigative/retrospective; may be revised",
    "IGNITION_POWER": "investigative/retrospective; may be revised",
    "SOURCE_OF_IGNITION": "investigative/retrospective; may be revised",
    "FIRE_START_LOCATION": "investigative/retrospective",
    "FIRE_SIZE_ON_ARRIVAL": "first FRS arrival",
    "OTHER_PROPERTY_AFFECTED_ON_ARRIVAL": "first FRS arrival",
    "ITEM_IGNITED": "investigative/retrospective",
    "ITEM_CAUSING_SPREAD": "post-spread/retrospective",
    "RAPID_FIRE_GROWTH": "fire development/retrospective",
    "RESPONSE_TIME": "first FRS arrival",
    "RESPONSE_TIME_CODE": "first FRS arrival",
    "TIME_AT_SCENE": "incident closure",
    "TIME_AT_SCENE_CODE": "incident closure",
    "SPREAD_OF_FIRE": "incident closure/outcome",
    "OTHER_PROPERTY_AFFECTED_CLOSE": "incident closure/outcome",
}


def resolve_blocks(columns: Iterable[str]) -> dict[str, list[str]]:
    """Resolve the complete configured feature schema, failing on missing fields."""
    policy = load_yaml("config/feature_policy.yaml")
    available = set(columns)
    blocks: dict[str, list[str]] = {}
    missing = {
        name: sorted(set(policy["blocks"][name]["candidates"]) - available)
        for name in ("A", "B", "C")
    }
    missing = {name: fields for name, fields in missing.items() if fields}
    if missing:
        raise ValueError(f"Required predictor columns are missing from the configured blocks: {missing}")
    for name in ("A", "B", "C"):
        inherited = list(blocks.get(policy["blocks"][name].get("extends", ""), []))
        own = list(policy["blocks"][name]["candidates"])
        blocks[name] = inherited + own
    assert_no_leakage(blocks, policy["leakage_blacklist"])
    return blocks


def assert_no_leakage(blocks: dict[str, list[str]], blacklist: Iterable[str] | None = None) -> None:
    policy = load_yaml("config/feature_policy.yaml")
    forbidden = set(blacklist or policy["leakage_blacklist"]) | {
        "LARGER_FIRE", "FINANCIAL_YEAR", "FRS_TERRITORY"
    }
    violations = {block: sorted(set(cols) & forbidden) for block, cols in blocks.items()}
    violations = {block: cols for block, cols in violations.items() if cols}
    if violations:
        raise ValueError(f"Leakage/excluded fields entered predictors: {violations}")


def build_feature_policy(frame: pd.DataFrame) -> pd.DataFrame:
    policy = load_yaml("config/feature_policy.yaml")
    blocks = resolve_blocks(frame.columns)
    blacklist = set(policy["leakage_blacklist"])
    rows = []
    for column in frame.columns:
        in_a, in_b, in_c = (column in blocks[x] for x in ("A", "B", "C"))
        blacklisted = column in blacklist
        if blacklisted:
            risk = "high"
            reason = "Excluded: outcome, post-spread, post-closure, or resource-escalation information."
        elif column == "FINANCIAL_YEAR":
            risk = "temporal shortcut"
            reason = "Excluded from predictors; retained only for temporal splitting."
        elif column in {"FRS_TERRITORY", "E_CODE_TERRITORY"}:
            risk = "geographic shortcut"
            reason = "Excluded from main models; retained for quality/subgroup diagnostics."
        elif column == "LATE_CALL":
            risk = "cohort definition"
            reason = "Used only to define the main cohort."
        elif column == "OCCUPIED_TIME":
            risk = "potential outcome proxy (spread-dependent occupancy)"
            reason = (
                "Retained in historical Blocks B/C. Guidance includes people in buildings "
                "to which the fire spread, so this field may encode realised spread. "
                "No removal sensitivity analysis has been run to quantify its effect."
            )
        elif in_c and not in_b:
            risk = "proximal prognostic"
            reason = (
                "Included only in the retrospective incident plus arrival-state model; "
                "Block C also inherits investigative fields from Block B and is not "
                "a model using only information available at first arrival."
            )
        elif in_a or in_b:
            risk = "low/moderate"
            reason = "Included in the specified information block; timing limitations apply."
        else:
            risk = "not assessed for modelling"
            reason = "Outside the prespecified predictor blocks."
        rows.append({
            "raw_field": column,
            "official_meaning": OFFICIAL_MEANINGS.get(column, "Not described in the current guidance"),
            "data_type": str(frame[column].dtype),
            "information_timing": TIMING.get(column, "late incident/outcome or not modelled"),
            "block_a": in_a,
            "block_b": in_b,
            "block_c": in_c,
            "excluded": not (in_a or in_b or in_c),
            "leakage_risk": risk,
            "reason": reason,
        })
    return pd.DataFrame(rows)

