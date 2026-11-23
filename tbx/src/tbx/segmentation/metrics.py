"""Segmentation metrics."""
from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Dice overlap between two binary masks.

    Used to judge the U-Net on held-out Montgomery masks before trusting its
    Shenzhen output. A Dice in the low-to-mid 0.9s is the usual bar for lung
    fields; below that, fall back to bounding-box crops.
    """
    p = (pred > 0).astype(np.float32).ravel()
    t = (target > 0).astype(np.float32).ravel()
    inter = float((p * t).sum())
    return (2 * inter + eps) / (p.sum() + t.sum() + eps)
