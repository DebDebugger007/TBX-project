"""Lung-field texture and intensity features for the classical baseline.

Following Jaeger et al. (2014b): grey-level co-occurrence matrix (GLCM)
statistics, local binary pattern (LBP) histograms and an intensity histogram,
all computed **inside the lung mask only**. Restricting the features to the
lungs is the point — it gives one model that cannot use borders, text or
equipment, as a contrast against the whole-image CNN.
"""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..config import FeatureConfig

_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _quantise(img01: np.ndarray, levels: int) -> np.ndarray:
    """Map a [0,1] float image to integer grey levels for the GLCM."""
    q = np.clip((img01 * (levels - 1)).round().astype(np.int32), 0, levels - 1)
    return q


def _glcm_features(img01: np.ndarray, mask: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    """GLCM props averaged over distances and angles, restricted to the lungs.

    Pixels outside the mask are set to a sentinel level and excluded by masking
    the image to its lung bounding box first, which keeps the co-occurrence
    statistics local to the lung field without a full masked-GLCM implementation.
    """
    q = _quantise(img01, cfg.glcm_levels)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return {f"glcm_{p}": 0.0 for p in _GLCM_PROPS}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = q[y0:y1, x0:x1]

    angles = [np.deg2rad(a) for a in cfg.glcm_angles_deg]
    glcm = graycomatrix(
        patch,
        distances=cfg.glcm_distances,
        angles=angles,
        levels=cfg.glcm_levels,
        symmetric=True,
        normed=True,
    )
    feats = {}
    for prop in _GLCM_PROPS:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    return feats


def _lbp_features(img01: np.ndarray, mask: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    """Uniform-LBP histogram over lung pixels only."""
    img_u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_u8, cfg.lbp_points, cfg.lbp_radius, method="uniform")
    n_bins = cfg.lbp_points + 2
    vals = lbp[mask > 0]
    if vals.size == 0:
        return {f"lbp_{i}": 0.0 for i in range(n_bins)}
    hist, _ = np.histogram(vals, bins=n_bins, range=(0, n_bins), density=True)
    return {f"lbp_{i}": float(hist[i]) for i in range(n_bins)}


def _intensity_features(img01: np.ndarray, mask: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    vals = img01[mask > 0]
    if vals.size == 0:
        base = {f"int_bin_{i}": 0.0 for i in range(cfg.intensity_bins)}
        base.update({"int_mean": 0.0, "int_std": 0.0, "int_skew": 0.0})
        return base
    hist, _ = np.histogram(vals, bins=cfg.intensity_bins, range=(0, 1), density=True)
    feats = {f"int_bin_{i}": float(hist[i]) for i in range(cfg.intensity_bins)}
    m, s = float(vals.mean()), float(vals.std())
    skew = float(((vals - m) ** 3).mean() / (s ** 3 + 1e-8))
    feats.update({"int_mean": m, "int_std": s, "int_skew": skew})
    return feats


def lung_field_features(img01: np.ndarray, mask: np.ndarray, cfg: FeatureConfig | None = None) -> dict[str, float]:
    """All lung-field features for one image as an ordered dict."""
    cfg = cfg or FeatureConfig()
    feats: dict[str, float] = {}
    feats.update(_glcm_features(img01, mask, cfg))
    feats.update(_lbp_features(img01, mask, cfg))
    feats.update(_intensity_features(img01, mask, cfg))
    return feats


def feature_names(cfg: FeatureConfig | None = None) -> list[str]:
    cfg = cfg or FeatureConfig()
    names = [f"glcm_{p}" for p in _GLCM_PROPS]
    names += [f"lbp_{i}" for i in range(cfg.lbp_points + 2)]
    names += [f"int_bin_{i}" for i in range(cfg.intensity_bins)]
    names += ["int_mean", "int_std", "int_skew"]
    return names
