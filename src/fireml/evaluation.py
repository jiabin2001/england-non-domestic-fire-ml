from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "n": int(y_true.size),
        "positive_count": int(y_true.sum()),
        "positive_prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    validate_metric_ranges(metrics)
    return metrics


def validate_metric_ranges(metrics: dict[str, float | int]) -> None:
    for key in ("positive_prevalence", "pr_auc", "roc_auc", "recall", "precision", "f1", "balanced_accuracy", "brier_score", "threshold"):
        value = float(metrics[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Metric {key} outside [0, 1]: {value}")


def calibration_points(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    observed, predicted = calibration_curve(y_true, probability, n_bins=bins, strategy="quantile")
    return predicted, observed

