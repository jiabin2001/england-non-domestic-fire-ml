from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .cohort import construct_cohort
from .config import ROOT, load_yaml
from .drift import build_drift_tables, subgroup_performance
from .evaluation import choose_f1_threshold, classification_metrics
from .features import resolve_blocks
from .modelling import make_model_pipeline
from .splits import make_random_split_like, make_temporal_split


def _fit_validation_then_test(
    frame: pd.DataFrame,
    split: dict[str, np.ndarray],
    columns: list[str],
    family: str,
    parameters: dict[str, Any],
    seed: int,
    n_jobs: int,
    device: str,
) -> tuple[dict, float]:
    development_model = make_model_pipeline(columns, family, parameters, seed, n_jobs, device)
    development_model.fit(frame.loc[split["train"], columns], frame.loc[split["train"], "LARGER_FIRE"])
    validation_probability = development_model.predict_proba(frame.loc[split["validation"], columns])[:, 1]
    threshold = choose_f1_threshold(frame.loc[split["validation"], "LARGER_FIRE"].to_numpy(), validation_probability)
    dev = np.concatenate([split["train"], split["validation"]])
    final_model = make_model_pipeline(columns, family, parameters, seed, n_jobs, device)
    final_model.fit(frame.loc[dev, columns], frame.loc[dev, "LARGER_FIRE"])
    probability = final_model.predict_proba(frame.loc[split["test"], columns])[:, 1]
    return classification_metrics(frame.loc[split["test"], "LARGER_FIRE"].to_numpy(), probability, threshold), threshold


def run_temporal_robustness() -> dict[str, pd.DataFrame]:
    cfg = load_yaml("config/analysis.yaml")
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "outputs/metrics/locked_model_config.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / cfg["cohort_path"])
    blocks = resolve_blocks(frame.columns)
    columns = blocks["B"]
    family = lock["selected_family_by_block"]["temporal"]["B"]
    parameters = lock["selected_hyperparameters"]["temporal"][family]
    seed, n_jobs, device = int(cfg["random_seed"]), int(cfg["n_jobs"]), lock["xgboost_device"]
    temporal = make_temporal_split(frame, audit["temporal_train_years"], audit["temporal_validation_years"], audit["temporal_test_years"])

    # Natural annual expanding windows, with locked hyperparameters and a fixed
    # 0.5 descriptive threshold (primary annual comparison is threshold-free PR-AUC).
    expanding_rows = []
    years = audit["main_years"]
    for test_year in years[len(audit["temporal_train_years"]):]:
        earlier = [year for year in years if int(year.split("/")[0]) < int(test_year.split("/")[0])]
        train_idx = frame.index[frame["FINANCIAL_YEAR"].isin(earlier)].to_numpy()
        test_idx = frame.index[frame["FINANCIAL_YEAR"].eq(test_year)].to_numpy()
        model = make_model_pipeline(columns, family, parameters, seed, n_jobs, device)
        model.fit(frame.loc[train_idx, columns], frame.loc[train_idx, "LARGER_FIRE"])
        probability = model.predict_proba(frame.loc[test_idx, columns])[:, 1]
        metrics = classification_metrics(frame.loc[test_idx, "LARGER_FIRE"].to_numpy(), probability, 0.5)
        expanding_rows.append({
            "train_start": earlier[0], "train_end": earlier[-1], "test_year": test_year,
            "model": family, "block": "B", "threshold_note": "fixed 0.5; PR-AUC is primary", **metrics,
        })
    expanding = pd.DataFrame(expanding_rows)
    expanding.to_csv(ROOT / "outputs/tables/expanding_window_performance.csv", index=False)

    main_row = pd.read_csv(ROOT / "outputs/tables/temporal_validation_performance.csv")
    main_row = main_row[(main_row["block"] == "B") & (main_row["model"] == family)].iloc[0].to_dict()
    sensitivity_rows = [{"analysis": "main_temporal_definition", "test_period": "2022/23-2023/24", **main_row}]

    for name, kwargs in (
        ("roofs_roof_spaces_positive", {"roofs_positive": True}),
        ("include_late_calls", {"include_late_calls": True}),
    ):
        sensitivity_frame, _ = construct_cohort(save_main=False, **kwargs)
        sensitivity_split = make_temporal_split(
            sensitivity_frame, audit["temporal_train_years"], audit["temporal_validation_years"], audit["temporal_test_years"]
        )
        metrics, threshold = _fit_validation_then_test(
            sensitivity_frame, sensitivity_split, resolve_blocks(sensitivity_frame.columns)["B"],
            family, parameters, seed, n_jobs, device,
        )
        sensitivity_rows.append({
            "analysis": name, "test_period": "2022/23-2023/24", "design": "temporal",
            "split_role": "test", "block": "B", "model": family,
            "parameters": json.dumps(parameters), **metrics,
        })

    # Optional new-year check: exclude incomplete Suffolk, train through 2023/24,
    # reuse the primary locked threshold and do not retune on 2024/25.
    extended, _ = construct_cohort(include_2024_excluding_suffolk=True, save_main=False)
    train_idx = extended.index[extended["FINANCIAL_YEAR"].isin(audit["main_years"])].to_numpy()
    test_idx = extended.index[extended["FINANCIAL_YEAR"].eq("2024/25")].to_numpy()
    model = make_model_pipeline(resolve_blocks(extended.columns)["B"], family, parameters, seed, n_jobs, device)
    model.fit(extended.loc[train_idx, columns], extended.loc[train_idx, "LARGER_FIRE"])
    probability = model.predict_proba(extended.loc[test_idx, columns])[:, 1]
    locked_threshold = float(lock["thresholds"]["temporal"]["B"][family])
    metrics = classification_metrics(extended.loc[test_idx, "LARGER_FIRE"].to_numpy(), probability, locked_threshold)
    sensitivity_rows.append({
        "analysis": "include_2024_25_exclude_suffolk", "test_period": "2024/25",
        "design": "temporal_new_year", "split_role": "test", "block": "B", "model": family,
        "parameters": json.dumps(parameters), **metrics,
    })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(ROOT / "outputs/tables/sensitivity_analysis_results.csv", index=False)

    # Prespecified small random-seed stability check. Seed 1 is recomputed here by
    # the same locked procedure so all rows have identical provenance.
    seed_rows = []
    for stability_seed in cfg["random_stability_seeds"]:
        random_split = make_random_split_like(frame, temporal, int(stability_seed))
        metrics, threshold = _fit_validation_then_test(
            frame, random_split, columns, family,
            lock["selected_hyperparameters"]["random"][family], int(stability_seed), n_jobs, device,
        )
        seed_rows.append({
            "seed": int(stability_seed), "design": "random", "block": "B", "model": family, **metrics,
        })
    random_stability = pd.DataFrame(seed_rows)
    random_stability.to_csv(ROOT / "outputs/tables/random_seed_stability.csv", index=False)

    build_drift_tables(frame, temporal)
    prediction = pd.read_parquet(ROOT / "outputs/metrics/predictions_temporal_block_B.parquet")
    dev_indices = np.concatenate([temporal["train"], temporal["validation"]])
    subgroup = subgroup_performance(frame, prediction, dev_indices, locked_threshold)
    return {
        "expanding": expanding,
        "sensitivity": sensitivity,
        "random_stability": random_stability,
        "subgroup": subgroup,
    }

