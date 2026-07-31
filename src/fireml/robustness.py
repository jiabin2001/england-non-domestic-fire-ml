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
from .uncertainty import add_pr_auc_prevalence_context


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
    probability = development_model.predict_proba(frame.loc[split["test"], columns])[:, 1]
    return classification_metrics(frame.loc[split["test"], "LARGER_FIRE"].to_numpy(), probability, threshold), threshold


def _build_random_split_stability(
    frame: pd.DataFrame,
    temporal: dict[str, np.ndarray],
    columns: list[str],
    family: str,
    parameters: dict[str, Any],
    estimator_seed: int,
    split_seeds: list[int],
    n_jobs: int,
    device: str,
) -> pd.DataFrame:
    configured_seeds = [int(value) for value in split_seeds]
    if len(configured_seeds) != len(set(configured_seeds)):
        raise ValueError("random_stability_seeds must be unique.")

    output_path = ROOT / "outputs/tables/random_seed_stability.csv"
    seed_rows = []
    for split_seed in configured_seeds:
        random_split = make_random_split_like(frame, temporal, split_seed)
        metrics, _ = _fit_validation_then_test(
            frame, random_split, columns, family, parameters,
            estimator_seed, n_jobs, device,
        )
        seed_rows.append({
            "split_seed": split_seed, "estimator_seed": estimator_seed,
            "design": "random", "block": "B", "model": family, **metrics,
        })

    random_stability = pd.DataFrame(seed_rows)
    random_stability.to_csv(output_path, index=False)
    return random_stability


def run_random_split_stability() -> pd.DataFrame:
    """Run only the configured Block B split-assignment sensitivity analysis."""
    cfg = load_yaml("config/analysis.yaml")
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / cfg["cohort_path"])
    columns = resolve_blocks(frame.columns)["B"]
    family = selection["selected_family_by_block"]["temporal"]["B"]
    estimator_seed = int(cfg["random_seed"])
    temporal = make_temporal_split(
        frame,
        audit["temporal_train_years"],
        audit["temporal_validation_years"],
        audit["temporal_test_years"],
    )
    return _build_random_split_stability(
        frame,
        temporal,
        columns,
        family,
        selection["selected_hyperparameters"]["random"][family],
        estimator_seed,
        cfg["random_stability_seeds"],
        int(cfg["n_jobs"]),
        selection["xgboost_device"],
    )


def run_temporal_robustness() -> dict[str, pd.DataFrame]:
    cfg = load_yaml("config/analysis.yaml")
    audit = json.loads((ROOT / "outputs/metrics/audit_receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "outputs/metrics/model_selection.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / cfg["cohort_path"])
    blocks = resolve_blocks(frame.columns)
    columns = blocks["B"]
    family = selection["selected_family_by_block"]["temporal"]["B"]
    parameters = selection["selected_hyperparameters"]["temporal"][family]
    seed, n_jobs, device = int(cfg["random_seed"]), int(cfg["n_jobs"]), selection["xgboost_device"]
    primary_threshold = float(selection["validation_thresholds"]["temporal"]["B"][family])
    temporal = make_temporal_split(frame, audit["temporal_train_years"], audit["temporal_validation_years"], audit["temporal_test_years"])

    # Natural annual expanding windows, with selected hyperparameters and a fixed
    # 0.5 descriptive threshold (primary annual comparison is threshold-free AP).
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
            "model": family, "block": "B", "threshold_note": "fixed 0.5; average precision is primary", **metrics,
        })
    expanding = add_pr_auc_prevalence_context(pd.DataFrame(expanding_rows))
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
        metrics, _ = _fit_validation_then_test(
            sensitivity_frame, sensitivity_split, resolve_blocks(sensitivity_frame.columns)["B"],
            family, parameters, seed, n_jobs, device,
        )
        sensitivity_rows.append({
            "analysis": name, "test_period": "2022/23-2023/24", "design": "temporal",
            "split_role": "test", "block": "B", "model": family,
            "parameters": json.dumps(parameters), **metrics,
        })

    # Optional new-year check: exclude incomplete Suffolk, train through 2023/24,
    # reuse the primary validation-derived threshold and do not retune on 2024/25.
    extended, _ = construct_cohort(include_2024_excluding_suffolk=True, save_main=False)
    extended_columns = resolve_blocks(extended.columns)["B"]
    train_idx = extended.index[extended["FINANCIAL_YEAR"].isin(audit["main_years"])].to_numpy()
    test_idx = extended.index[extended["FINANCIAL_YEAR"].eq("2024/25")].to_numpy()
    model = make_model_pipeline(extended_columns, family, parameters, seed, n_jobs, device)
    model.fit(extended.loc[train_idx, extended_columns], extended.loc[train_idx, "LARGER_FIRE"])
    probability = model.predict_proba(extended.loc[test_idx, extended_columns])[:, 1]
    metrics = classification_metrics(
        extended.loc[test_idx, "LARGER_FIRE"].to_numpy(), probability, primary_threshold
    )
    sensitivity_rows.append({
        "analysis": "include_2024_25_exclude_suffolk", "test_period": "2024/25",
        "design": "temporal_new_year", "split_role": "test", "block": "B", "model": family,
        "parameters": json.dumps(parameters), **metrics,
    })
    sensitivity = add_pr_auc_prevalence_context(pd.DataFrame(sensitivity_rows))
    sensitivity.to_csv(ROOT / "outputs/tables/sensitivity_analysis_results.csv", index=False)

    # Random-split stability check: vary only the split assignment while holding
    # the estimator seed and selected hyperparameters fixed.
    random_stability = _build_random_split_stability(
        frame,
        temporal,
        columns,
        family,
        selection["selected_hyperparameters"]["random"][family],
        seed,
        cfg["random_stability_seeds"],
        n_jobs,
        device,
    )

    build_drift_tables(frame, temporal)
    prediction = pd.read_parquet(ROOT / "outputs/metrics/predictions_temporal_block_B.parquet")
    dev_indices = np.concatenate([temporal["train"], temporal["validation"]])
    subgroup = subgroup_performance(frame, prediction, dev_indices, primary_threshold)
    return {
        "expanding": expanding,
        "sensitivity": sensitivity,
        "random_stability": random_stability,
        "subgroup": subgroup,
    }
