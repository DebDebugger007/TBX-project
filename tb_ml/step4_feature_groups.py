"""Step 4 - which kind of feature matters most?

Trains both models on each feature group on its own - intensity (6 features),
texture (5), edges (2) - and on all 13 together, using the same repeated
5-fold cross-validation as step 2. If, say, texture alone scores almost as well
as all 13 features, texture carries most of the useful information.

Saves to results/:
  feature_groups_metrics.csv   mean and std of every metric per dataset, model and group
  feature_groups_auc.png       bar chart of AUC per group

Usage:
    python step4_feature_groups.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from features import FEATURE_GROUPS
from models import get_xy, make_models, mean_std_table, repeated_cv, summarise_folds

RESULTS = Path(__file__).parent / "results"


def main():
    df = pd.read_csv(RESULTS / "features.csv")
    rows = []

    for dataset in ["montgomery", "shenzhen"]:
        data = df[df.dataset == dataset]
        for group, cols in FEATURE_GROUPS.items():
            X, y = get_xy(data, cols)
            for name, (model, grid) in make_models().items():
                folds, _, _ = repeated_cv(model, grid, X, y)
                s = summarise_folds(folds)
                print(f"{dataset:10s} {group:9s} {name:12s} AUC {s['auc_mean']:.3f} ± {s['auc_std']:.3f}")
                rows.append({"dataset": dataset, "group": group, "n_features": len(cols),
                             "model": name, **s})

    out = pd.DataFrame(rows).round(3)
    out.to_csv(RESULTS / "feature_groups_metrics.csv", index=False)
    print("\nMean ± standard deviation over 25 test folds:")
    print(mean_std_table(out, ["dataset", "group", "model"]).to_string(index=False))

    # Bar chart: AUC per group, one panel per dataset
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dataset in zip(axes, ["montgomery", "shenzhen"]):
        sub = out[out.dataset == dataset]
        mean = sub.pivot(index="group", columns="model", values="auc_mean").loc[list(FEATURE_GROUPS)]
        std = sub.pivot(index="group", columns="model", values="auc_std").loc[list(FEATURE_GROUPS)]
        mean.plot.bar(yerr=std, ax=ax, capsize=3, rot=0)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("feature group")
        ax.set_ylabel("AUC (mean ± std, 25 folds)")
        ax.set_ylim(0.4, 1.0)
    fig.suptitle("Which feature group carries the TB signal?")
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_groups_auc.png", dpi=120)
    print(f"Saved {RESULTS / 'feature_groups_auc.png'}")


if __name__ == "__main__":
    main()
