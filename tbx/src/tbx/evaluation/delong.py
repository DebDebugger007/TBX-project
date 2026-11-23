"""DeLong's test for two correlated ROC AUCs.

Used to decide whether two models or two conditions genuinely differ on the
*same* test set — for example whether the lung-masked model really scores below
the whole-image model, rather than looking lower by chance. This is the fast
DeLong algorithm (Sun and Xu, 2014), which computes the AUC covariance from
midrank statistics in O(n log n).

Returns the two AUCs, the z statistic and a two-sided p-value. For the
correction-strategy comparisons in Phase 6 the caller applies a Holm correction
across the family of tests so multiple testing does not manufacture significance.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    midrank = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        midrank[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midrank
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Two-sided DeLong test comparing AUC(score_a) and AUC(score_b).

    Both score vectors must be over the *same* samples ``y_true``.
    """
    y_true = np.asarray(y_true).astype(int)
    order = (-y_true).argsort(kind="mergesort")  # positives first
    label_1_count = int(y_true.sum())

    preds = np.vstack((np.asarray(score_a), np.asarray(score_b)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = float((aucs[0] - aucs[1]) / np.sqrt(var))
        p = float(2 * stats.norm.sf(abs(z)))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(aucs[0] - aucs[1]),
        "z": z,
        "p_value": p,
    }
