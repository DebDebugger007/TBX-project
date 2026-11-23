"""Phase 4 — compute lung-field texture features for the classical baseline.

Usage:
    python scripts/04_build_features.py --config configs/default.yaml

Produces:
    data/processed/features.csv    one row per image, feature columns + label

Montgomery uses its merged manual mask; Shenzhen uses the U-Net mask from
Phase 3 (run 03 first). Images without any mask are skipped with a warning.
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tbx.config import load_config
from tbx.data.masks import merge_montgomery_masks, prepare_mask
from tbx.data.preprocess import preprocess_image
from tbx.features import lung_field_features
from tbx.utils import seed_everything


def _mask_for(row, size, masks_dir):
    if row["dataset"] == "montgomery" and isinstance(row.get("left_mask_path"), str) and row["left_mask_path"]:
        m = merge_montgomery_masks(row["left_mask_path"], row["right_mask_path"])
        return prepare_mask(m, size)
    cand = Path(masks_dir) / f"{row['image_id']}.png"
    if cand.exists():
        raw = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
        return prepare_mask(raw, size)
    return None


def features_path(processed_root, correction: str = "none") -> Path:
    """features.csv for uncorrected images, features_<correction>.csv otherwise."""
    name = "features.csv" if correction == "none" else f"features_{correction}.csv"
    return Path(processed_root) / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--corrections", nargs="+", default=["none"],
                    help="intensity corrections to build feature tables for (Phase 6), "
                         "e.g. --corrections none histeq clahe zscore")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)
    df = pd.read_csv(cfg.paths.metadata_csv)

    masks = {row["image_id"]: _mask_for(row, args.size, cfg.paths.masks_dir) for _, row in df.iterrows()}
    for correction in args.corrections:
        rows, skipped = [], 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"features[{correction}]"):
            mask = masks[row["image_id"]]
            if mask is None:
                skipped += 1
                continue
            img = preprocess_image(row["image_path"], row["dataset"], args.size, correction)
            feats = lung_field_features(img, mask, cfg.features)
            feats.update({"image_id": row["image_id"], "label": int(row["label"]), "dataset": row["dataset"]})
            rows.append(feats)

        out = pd.DataFrame(rows)
        out_path = features_path(cfg.paths.processed_root, correction)
        out.to_csv(out_path, index=False)
        print(f"[features] wrote {out_path} ({len(out)} rows, {skipped} skipped for missing mask)")


if __name__ == "__main__":
    main()
