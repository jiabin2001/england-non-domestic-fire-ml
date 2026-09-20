from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .config import ROOT


DRIFT_FIELDS = ["BUILDING_TYPE", "ALARM_SYSTEM", "SAFETY_SYSTEM", "FIRE_SIZE_ON_ARRIVAL"]


def _distribution(series: pd.Series, categories: list[str]) -> np.ndarray:
    values = series.fillna("Missing/Unknown").astype(str).value_counts(normalize=True)
    return np.array([values.get(category, 0.0) for category in categories], dtype=float)


def build_drift_tables(frame: pd.DataFrame, split: dict[str, np.ndarray]) -> None:
    annual = frame.groupby("FINANCIAL_YEAR", observed=True)["LARGER_FIRE"].agg(
        incident_count="size", larger_fire_count="sum", larger_fire_prevalence="mean"
    ).reset_index()
    annual.to_csv(ROOT / "outputs/tables/annual_incident_prevalence.csv", index=False)

    support_rows = []
    for partition, indices in split.items():
        subset = frame.loc[indices]
        for column in DRIFT_FIELDS:
            counts = subset[column].fillna("Missing/Unknown").astype(str).value_counts()
            for category, count in counts.items():
                support_rows.append({
                    "partition": partition, "column": column, "category": category,
                    "count": int(count), "proportion": float(count / len(subset)),
                })
    pd.DataFrame(support_rows).to_csv(ROOT / "outputs/tables/category_support_by_split.csv", index=False)

    comparison = []
    train, test = frame.loc[split["train"]], frame.loc[split["test"]]
    for column in DRIFT_FIELDS:
        categories = sorted(set(train[column].fillna("Missing/Unknown").astype(str)) | set(test[column].fillna("Missing/Unknown").astype(str)))
        p, q = _distribution(train[column], categories), _distribution(test[column], categories)
        comparison.append({
            "column": column,
            "train_categories": int((p > 0).sum()),
            "test_categories": int((q > 0).sum()),
            "unseen_in_test_count": int(((p == 0) & (q > 0)).sum()),
            "absent_from_test_count": int(((p > 0) & (q == 0)).sum()),
            "jensen_shannon_divergence_base2": float(jensenshannon(p, q, base=2.0) ** 2),
        })
    pd.DataFrame(comparison).to_csv(ROOT / "outputs/tables/drift_summary.csv", index=False)

    major = frame.loc[split["train"], "BUILDING_TYPE"].value_counts().head(10).index
    proportions = (
        frame[frame["BUILDING_TYPE"].isin(major)]
        .groupby(["FINANCIAL_YEAR", "BUILDING_TYPE"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = frame.groupby("FINANCIAL_YEAR", observed=True).size().rename("year_total")
    proportions = proportions.join(totals, on="FINANCIAL_YEAR")
    proportions["proportion"] = proportions["count"] / proportions["year_total"]
    proportions.to_csv(ROOT / "outputs/tables/annual_major_building_type_proportions.csv", index=False)


def subgroup_performance(frame: pd.DataFrame, predictions: pd.DataFrame, development_indices: np.ndarray, threshold: float) -> pd.DataFrame:
    from .evaluation import classification_metrics

    major = frame.loc[development_indices, "BUILDING_TYPE"].value_counts().head(10).index.tolist()
    merged = predictions.copy()
    rows = []
    for category in major:
        group = merged[merged["BUILDING_TYPE"] == category]
        if len(group) == 0:
            continue
        try:
            metrics = classification_metrics(group["LARGER_FIRE"].to_numpy(), group["probability"].to_numpy(), threshold)
        except ValueError:
            # ROC-AUC is undefined in a one-class subgroup; retain other descriptive fields.
            y = group["LARGER_FIRE"].to_numpy()
            prob = group["probability"].to_numpy()
            pred = (prob >= threshold).astype(int)
            from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, balanced_accuracy_score, brier_score_loss
            metrics = {
                "n": len(group), "positive_count": int(y.sum()), "positive_prevalence": float(y.mean()),
                "pr_auc": float(average_precision_score(y, prob)), "roc_auc": np.nan,
                "recall": float(recall_score(y, pred, zero_division=0)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "brier_score": float(brier_score_loss(y, prob)), "threshold": threshold,
            }
        rows.append({"building_type": category, **metrics})
    result = pd.DataFrame(rows).sort_values("n", ascending=False)
    result.to_csv(ROOT / "outputs/tables/building_type_subgroup_performance.csv", index=False)
    return result

