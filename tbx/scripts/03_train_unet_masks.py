"""Phase 3 — train the lung U-Net on Montgomery and generate Shenzhen masks.

Usage:
    python scripts/03_train_unet_masks.py --config configs/default.yaml --epochs 60

Produces:
    outputs/segmentation/unet.pt          trained weights
    outputs/segmentation/unet_report.json held-out Montgomery Dice
    data/processed/masks/CHNCXR_*.png      generated Shenzhen masks

If the held-out Dice is poor, the region experiments fall back to bounding-box
lung crops rather than trusting these masks (see tbx.data.masks.apply_region).
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import pandas as pd

from tbx.config import load_config
from tbx.segmentation.train import generate_masks, train_unet
from tbx.utils import seed_everything
from tbx.utils.io import ensure_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)
    df = pd.read_csv(cfg.paths.metadata_csv)

    seg_out = ensure_dir(Path(cfg.paths.outputs_root) / "segmentation")
    report = train_unet(
        df, size=args.size, epochs=args.epochs, seed=cfg.data.seed,
        out_path=seg_out / "unet.pt",
    )
    save_json(report, seg_out / "unet_report.json")

    generate_masks(df, report["model_path"], cfg.paths.masks_dir, dataset="shenzhen", size=args.size)
    print("[unet] done — Shenzhen masks written to", cfg.paths.masks_dir)


if __name__ == "__main__":
    main()
