"""Phase 4–5 — run one experiment from its config.

Usage:
    python scripts/05_run_experiment.py --config configs/E1_montgomery_within.yaml

For the classical experiment (E0) run 04_build_features.py first. Results land in
outputs/<name>/results.json with pooled AUC and 95% bootstrap CIs.
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import pandas as pd

from tbx.config import load_config
from tbx.runner import run_experiment
from tbx.utils import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)

    df = pd.read_csv(cfg.paths.metadata_csv)
    features_df = None
    if cfg.model == "classical":
        name = "features.csv" if cfg.correction == "none" else f"features_{cfg.correction}.csv"
        fpath = Path(cfg.paths.processed_root) / name
        if not fpath.exists():
            raise SystemExit(
                f"{fpath} not found; build it with "
                f"python scripts/04_build_features.py --corrections {cfg.correction}"
            )
        features_df = pd.read_csv(fpath)

    run_experiment(cfg, df, features_df)


if __name__ == "__main__":
    main()
