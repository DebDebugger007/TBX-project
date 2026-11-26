"""Step 2 - evaluate a Decision Tree and an SVM on each dataset.

For Montgomery and for Shenzhen separately, using repeated stratified 5-fold
cross-validation (5 folds x 5 repeats = 25 test results per model):
  1. split the images into 5 parts with the same TB/normal ratio in each
  2. for each part: choose the model's settings by 5-fold cross-validation on
     the other 4 parts, train, then test on this part
  3. repeat with 5 different shuffles; report mean ± standard deviation

Saves to results/:
  within_dataset_metrics.csv      mean and std of every metric
  within_dataset_folds.csv        the metrics of all 25 test folds
  confusion_<dataset>.png         confusion matrices (first repeat: every image tested once)
  roc_<dataset>.png               ROC curves (first repeat)
  tree_<dataset>.png              a decision tree trained on the whole dataset (top levels)
  tree_importance_<dataset>.png   which features that tree relied on most

Usage:
    python step2_train_models.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.tree import plot_tree

from features import FEATURE_NAMES
from models import (get_xy, make_models, mean_std_table, repeated_cv,
                    summarise_folds, tune_and_fit)

RESULTS = Path(__file__).parent / "results"


def main():
    df = pd.read_csv(RESULTS / "features.csv")
    summary_rows, fold_tables = [], []

    for dataset in ["montgomery", "shenzhen"]:
        X, y = get_xy(df[df.dataset == dataset], FEATURE_NAMES)
        print(f"\n=== {dataset}: {len(y)} images, repeated 5-fold CV ===")

        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(10, 4))
        fig_roc, ax_roc = plt.subplots(figsize=(5, 5))

        for i, (name, (model, grid)) in enumerate(make_models().items()):
            folds, oof_scores, oof_pred = repeated_cv(model, grid, X, y)
            summary = summarise_folds(folds)
            print(f"{name}: AUC {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}, "
                  f"accuracy {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}")
            summary_rows.append({"dataset": dataset, "model": name, **summary})
            fold_tables.append(folds.assign(dataset=dataset, model=name))

            ConfusionMatrixDisplay.from_predictions(
                y, oof_pred, display_labels=["Normal", "TB"], ax=axes_cm[i], colorbar=False)
            axes_cm[i].set_title(name)
            RocCurveDisplay.from_predictions(y, oof_scores, name=name, ax=ax_roc)

        fig_cm.suptitle(f"Confusion matrices - {dataset} (each image tested once)")
        fig_cm.tight_layout()
        fig_cm.savefig(RESULTS / f"confusion_{dataset}.png", dpi=120)
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax_roc.set_title(f"ROC curves - {dataset}")
        ax_roc.legend()
        fig_roc.tight_layout()
        fig_roc.savefig(RESULTS / f"roc_{dataset}.png", dpi=120)
        plt.close("all")

        # One tree trained on the whole dataset, only to look at what it learned.
        tree, grid = make_models()["DecisionTree"]
        best_tree, _, _ = tune_and_fit(tree, grid, X, y)
        save_tree_plots(best_tree, dataset)

    summary = pd.DataFrame(summary_rows).round(3)
    summary.to_csv(RESULTS / "within_dataset_metrics.csv", index=False)
    pd.concat(fold_tables).round(3).to_csv(RESULTS / "within_dataset_folds.csv", index=False)
    print("\nMean ± standard deviation over 25 test folds:")
    print(mean_std_table(summary, ["dataset", "model"]).to_string(index=False))


def save_tree_plots(tree, dataset):
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(tree, feature_names=FEATURE_NAMES, class_names=["Normal", "TB"],
              filled=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title(f"Decision tree - {dataset} (top 3 levels)")
    fig.savefig(RESULTS / f"tree_{dataset}.png", dpi=110, bbox_inches="tight")

    imp = pd.Series(tree.feature_importances_, index=FEATURE_NAMES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax)
    ax.set_xlabel("Importance (share of impurity reduction)")
    ax.set_title(f"Decision tree feature importance - {dataset}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"tree_importance_{dataset}.png", dpi=120)
    plt.close("all")


if __name__ == "__main__":
    main()
