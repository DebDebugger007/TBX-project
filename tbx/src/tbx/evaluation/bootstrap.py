"""Stratified bootstrap confidence intervals.

Every metric in the project carries a 95% CI from 2,000 stratified bootstrap
resamples. On 138 Montgomery images the intervals are wide — in a 28-image test
fold a single misclassified image moves accuracy by about 3.6 points — and
reporting that width honestly is one of the project's research questions, not an
afterthought.

Stratified means we resample the TB and normal cases separately so every
resample keeps the original class balance; otherwise a resample could end up with
one class absent and an undefined AUC.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .metrics import auc_score


def _stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_true = np.asarray(y_true)
    idx = []
    for cls in np.unique(y_true):
        pool = np.where(y_true == cls)[0]
        idx.append(rng.choice(pool, size=pool.size, replace=True))
    return np.concatenate(idx)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile CI for any metric_fn(y_true, y_score) -> float."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)

    point = metric_fn(y_true, y_score)
    samples = np.empty(n, dtype=float)
    for i in range(n):
        bi = _stratified_indices(y_true, rng)
        samples[i] = metric_fn(y_true[bi], y_score[bi])

    samples = samples[~np.isnan(samples)]
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    lo, hi = np.quantile(samples, [lo_q, hi_q])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_resamples": int(samples.size)}


def bootstrap_auc_ci(y_true, y_score, n: int = 2000, ci: float = 0.95, seed: int = 42) -> dict[str, float]:
    return bootstrap_metric_ci(y_true, y_score, auc_score, n=n, ci=ci, seed=seed)
