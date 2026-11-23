"""Frozen stratified K-fold splits.

Splits are made once per dataset and written to
``<splits_dir>/<dataset>_folds.json``; every experiment (classical and CNN)
reads the same file, so models are always compared on identical folds. Image
IDs are sorted before splitting, so the folds depend only on the set of images
and the seed, not on row order.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..utils.io import ensure_dir, load_json, save_json


def _folds_path(splits_dir: str | Path, dataset: str) -> Path:
    return Path(splits_dir) / f"{dataset}_folds.json"


def make_stratified_folds(
    df: pd.DataFrame,
    dataset: str,
    splits_dir: str | Path,
    n_folds: int = 5,
    seed: int = 42,
) -> Path | None:
    """Write stratified folds for one dataset. Returns the file path, or None if
    the dataset has no images."""
    sub = df[df["dataset"] == dataset].sort_values("image_id").reset_index(drop=True)
    if sub.empty:
        print(f"[splits] no {dataset} images; skipping")
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[list[str]] = []
    for _, test_idx in skf.split(sub["image_id"], sub["label"]):
        folds.append(sorted(sub.loc[test_idx, "image_id"].tolist()))

    ensure_dir(splits_dir)
    path = _folds_path(splits_dir, dataset)
    save_json({"dataset": dataset, "n_folds": n_folds, "seed": seed, "folds": folds}, path)

    sizes = ", ".join(str(len(f)) for f in folds)
    print(f"[splits] wrote {path} ({n_folds} folds: {sizes})")
    return path


def load_folds(splits_dir: str | Path, dataset: str) -> list[list[str]]:
    """Return the folds as a list of image-id lists (fold index = position)."""
    path = _folds_path(splits_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/01_prepare_data.py first")
    data = load_json(path)
    return data["folds"] if isinstance(data, dict) else data


def fold_assignment(folds: list[list[str]]) -> dict[str, int]:
    """Map image_id -> fold index."""
    return {image_id: i for i, ids in enumerate(folds) for image_id in ids}
