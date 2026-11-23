"""Phase 3 — exploratory data analysis and the shortcut audit.

Usage:
    python scripts/02_run_eda.py --config configs/default.yaml

Produces, under outputs/eda/:
    visual_audit.png            side-by-side grid (two sets x two classes)
    class_balance.png
    age_distribution.png
    sex_ratio.png
    intensity_histogram.png
    metadata_summary.csv        counts, age and sex summary (feeds Table 2.2)
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import pandas as pd

from tbx.config import load_config
from tbx.eda import (
    plot_age_distributions,
    plot_class_balance,
    plot_intensity_histograms,
    plot_sex_ratios,
    save_audit_grid,
    summarise_metadata,
)
from tbx.utils import seed_everything
from tbx.utils.io import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)
    df = pd.read_csv(cfg.paths.metadata_csv)

    out = ensure_dir(Path(cfg.paths.outputs_root) / "eda")
    summarise_metadata(df).to_csv(out / "metadata_summary.csv", index=False)
    print(f"[eda] wrote {out/'metadata_summary.csv'}")

    save_audit_grid(df, out / "visual_audit.png", seed=cfg.data.seed)
    plot_class_balance(df, out / "class_balance.png")
    plot_age_distributions(df, out / "age_distribution.png")
    plot_sex_ratios(df, out / "sex_ratio.png")
    plot_intensity_histograms(df, out / "intensity_histogram.png", seed=cfg.data.seed)


if __name__ == "__main__":
    main()
