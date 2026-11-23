"""Metrics.

AUC is the primary metric because it does not depend on a threshold and is not
distorted by the prevalence difference between the sets. Alongside it we report
sensitivity and specificity at a threshold **fixed on the training data**, which
is the realistic deployment situation — nobody can re-tune a threshold on a new
hospital's labels — plus specificity at 90% sensitivity as a screening operating
point.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")  # AUC undefined with one class present
    return float(roc_auc_score(y_true, y_score))


def threshold_from_training(y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.90) -> float:
    """Pick the threshold on training scores that reaches a target sensitivity.

    This is the deployable operating point: it is chosen once, on training data,
    and then applied unchanged to the test set.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target_sensitivity)[0]
    if ok.size == 0:
        return float(thr[np.argmax(tpr)])
    return float(thr[ok[0]])


def sensitivity_specificity_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def specificity_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target: float = 0.90) -> float:
    """Specificity at the operating point that first reaches target sensitivity."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target)[0]
    if ok.size == 0:
        return float("nan")
    return float(1 - fpr[ok[0]])


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
    }


def summarise(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    screening_sensitivity: float = 0.90,
) -> dict[str, float]:
    """Point estimates for one test set (CIs are added by the bootstrap module)."""
    sens, spec = sensitivity_specificity_at_threshold(y_true, y_score, threshold)
    return {
        "auc": auc_score(y_true, y_score),
        "sensitivity": sens,
        "specificity": spec,
        "specificity_at_90_sens": specificity_at_sensitivity(y_true, y_score, screening_sensitivity),
        "threshold": threshold,
        **confusion_at_threshold(y_true, y_score, threshold),
    }
