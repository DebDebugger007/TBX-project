"""Image loading and resizing.

Every image goes through the same path: read at native bit depth, scale to
[0, 1] float32, pad to a square (aspect ratio preserved, never stretched), then
resize. Montgomery is stored at 12 bits inside 16-bit PNGs on some copies, so the
scale factor is chosen from the actual dtype and value range, not assumed.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def load_image01(path: str | Path, dataset: str | None = None, montgomery_bits: int = 12) -> np.ndarray:
    """Read a CXR as a single-channel float32 image in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype == np.uint8:
        out = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        # 12-bit data in a 16-bit container: scale by 4095 unless values exceed it
        max_12 = float(2 ** montgomery_bits - 1)
        denom = max_12 if img.max() <= max_12 else 65535.0
        out = img.astype(np.float32) / denom
    else:
        out = img.astype(np.float32)
        lo, hi = float(out.min()), float(out.max())
        out = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
    return np.clip(out, 0.0, 1.0)


def resize_with_pad(img: np.ndarray, size: int, interpolation: int | None = None) -> np.ndarray:
    """Zero-pad to a centred square, then resize to ``size`` x ``size``.

    Works for float images and binary masks; pass ``cv2.INTER_NEAREST`` for masks.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    if h != w:
        top = (side - h) // 2
        left = (side - w) // 2
        padded = np.zeros((side, side), dtype=img.dtype)
        padded[top:top + h, left:left + w] = img
        img = padded
    if side == size:
        return img
    if interpolation is None:
        interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interpolation)


CORRECTIONS = ("none", "histeq", "clahe", "zscore")


def apply_correction(img01: np.ndarray, correction: str = "none") -> np.ndarray:
    """Per-image intensity correction (Phase 6). Input and output in [0, 1].

    Each image is corrected using only its own pixels, so no information from
    the test hospital leaks into training:

    * ``histeq``  histogram equalisation
    * ``clahe``   contrast-limited adaptive histogram equalisation (clip 2.0, 8x8 tiles)
    * ``zscore``  subtract the mean, divide by the SD, map +-3 SD onto [0, 1]

    ``histeq`` and ``zscore`` take their statistics from the image content only,
    not the pure-black border: Montgomery films carry large black borders (up to
    a third of the frame) that Shenzhen films lack, and letting them into the
    statistics would squeeze or shift the anatomy differently per hospital, i.e.
    amplify the very acquisition difference the correction is meant to remove.
    Border pixels stay black.
    """
    if correction == "none":
        return img01
    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)
    fg = img01 > (1.0 / 255.0)
    if fg.sum() < 1000:  # essentially blank image: use everything
        fg = np.ones_like(fg)

    if correction == "clahe":
        u8 = (img01 * 255).round().astype(np.uint8)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255.0
    if correction == "histeq":
        u8 = (img01 * 255).round().astype(np.uint8)
        hist = np.bincount(u8[fg], minlength=256).astype(np.float64)
        cdf = hist.cumsum()
        cdf_min = cdf[hist > 0][0]
        lut = np.clip((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1.0), 0.0, 1.0).astype(np.float32)
        out = lut[u8]
        out[~fg] = 0.0
        return out
    if correction == "zscore":
        vals = img01[fg]
        m, s = float(vals.mean()), float(vals.std())
        out = np.clip(((img01 - m) / (s + 1e-6) + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)
        out[~fg] = 0.0
        return out
    raise ValueError(f"unknown correction {correction!r}; choose from {CORRECTIONS}")


def _cache_path(path: str | Path, size: int, correction: str = "none") -> Path | None:
    """Where the resized copy of ``path`` lives, or None if caching is off.

    The NLM originals are ~4000 px PNGs that take far longer to decode than a
    GPU takes to train on the result, so each image is decoded and resized once
    and kept as a small .npy under ``<raw_root>/../processed/cache/<size>/``.
    Set TBX_NO_CACHE=1 to disable.
    """
    if os.environ.get("TBX_NO_CACHE"):
        return None
    p = Path(path).resolve()
    # .../data/raw/<Set>/CXR_png/x.png -> .../data/processed/cache/<size>/<Set>/x.npy
    for parent in p.parents:
        if parent.name == "raw":
            rel = p.relative_to(parent)
            folder = str(size) if correction == "none" else f"{size}_{correction}"
            return parent.parent / "processed" / "cache" / folder / rel.parts[0] / f"{p.stem}.npy"
    return None


def preprocess_image(path: str | Path, dataset: str, size: int = 224, correction: str = "none") -> np.ndarray:
    """Load, scale to [0, 1], optionally intensity-correct, and pad-resize to
    ``size``. Returns float32 (size, size).

    The correction runs on the original image before padding, so the black
    padding never enters its statistics, and the geometry (and so mask
    alignment) is identical for every correction.
    """
    cache = _cache_path(path, size, correction)
    if cache is not None and cache.exists():
        try:
            return np.load(cache)
        except Exception:
            pass  # corrupt cache entry: rebuild below
    img = apply_correction(load_image01(path, dataset), correction)
    img = resize_with_pad(img, size).astype(np.float32)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(tmp, img)
            os.replace(tmp, cache)
        except OSError:
            pass
    return img
