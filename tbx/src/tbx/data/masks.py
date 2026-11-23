"""Lung masks and the image-region switch.

Masks are binary uint8 arrays with values {0, 1}. They go through the same
pad-to-square + resize as the image (nearest-neighbour), so a mask prepared at
size N lines up pixel-for-pixel with ``preprocess_image(..., N)``.

Regions:

* ``whole``         the image unchanged
* ``lung_cropped``  cropped to the lung bounding box (plus a small margin),
                    padded to square; the caller resizes back to the input size
* ``lung_masked``   the lungs blacked out, everything else kept. This is the
                    control for "is the model even using the lungs?": if a model
                    still scores well with the lungs removed, it is relying on
                    cues outside the lung field.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .preprocess import resize_with_pad


def _read_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"could not read mask {path}")
    return m


def merge_montgomery_masks(left_path: str | Path, right_path: str | Path) -> np.ndarray:
    """Union of the left and right lung masks, as uint8 {0, 255} at native size."""
    left = _read_mask(left_path)
    right = _read_mask(right_path)
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (((left > 0) | (right > 0)).astype(np.uint8)) * 255


def prepare_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Binarise and pad-resize a mask to ``size``. Returns uint8 {0, 1}."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 0).astype(np.uint8)
    out = resize_with_pad(binary, size, interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


def _bbox(mask: np.ndarray, margin_frac: float) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my = int(round((y1 - y0) * margin_frac))
    mx = int(round((x1 - x0) * margin_frac))
    return max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)


def _pad_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    if h == w:
        return img
    out = np.zeros((side, side), dtype=img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top:top + h, left:left + w] = img
    return out


def apply_region(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    region: str,
    margin_frac: float = 0.05,
) -> np.ndarray:
    """Apply the region setting to a preprocessed image and its aligned mask."""
    if region == "whole":
        return img
    if mask is None:
        raise ValueError(
            f"region={region!r} needs a lung mask but none was found. "
            "Run scripts/03_train_unet_masks.py first so Shenzhen has masks."
        )
    if mask.shape != img.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {img.shape}")
    if not mask.any():
        raise ValueError("lung mask is empty; cannot apply region")

    if region == "lung_cropped":
        y0, y1, x0, x1 = _bbox(mask, margin_frac)
        return _pad_square(img[y0:y1, x0:x1])
    if region == "lung_masked":
        out = img.copy()
        out[mask > 0] = 0.0
        return out
    raise ValueError(f"unknown region {region!r}")
