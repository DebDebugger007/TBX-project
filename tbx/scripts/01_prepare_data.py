"""Phase 2 — build the metadata table and the frozen splits.

Usage:
    python scripts/01_prepare_data.py --raw data/raw --config configs/default.yaml

Produces:
    data/processed/metadata.csv
    data/processed/splits/montgomery_folds.json
    data/processed/splits/shenzhen_folds.json
"""
import argparse

import _bootstrap_path  # noqa: F401

from tbx.config import load_config
from tbx.data.metadata import build_metadata
from tbx.data.splits import make_stratified_folds
from tbx.utils import seed_everything
from tbx.utils.io import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="override paths.raw_root")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)
    raw_root = args.raw or cfg.paths.raw_root

    df = build_metadata(raw_root)
    ensure_dir(cfg.paths.processed_root)
    df.to_csv(cfg.paths.metadata_csv, index=False)
    print(f"[prepare] wrote {cfg.paths.metadata_csv} ({len(df)} rows)")

    for dataset in ("montgomery", "shenzhen"):
        make_stratified_folds(df, dataset, cfg.paths.splits_dir, cfg.data.n_folds, cfg.data.seed)


if __name__ == "__main__":
    main()
