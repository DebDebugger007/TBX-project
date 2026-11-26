"""Step 3 - cross-dataset test: train on one hospital, test on the other.

  Shenzhen   -> Montgomery : train on all Shenzhen images, test on all Montgomery images
  Montgomery -> Shenzhen   : train on all Montgomery images, test on all Shenzhen images

Settings are again chosen by 5-fold cross-validation on the training hospital
only. Comparing these scores with step 2 shows how much performance drops when
a model meets X-rays from a different hospital (different machines, patients
and image quality).

Saves results/cross_dataset_metrics.csv

Usage:
    python step3_cross_dataset.py
"""
from pathlib import Path

import pandas as pd

from features import FEATURE_NAMES
from models import evaluate, get_xy, make_models, rounded, tune_and_fit

RESULTS = Path(__file__).parent / "results"


def main():
    df = pd.read_csv(RESULTS / "features.csv")
    rows = []

    for train_set, test_set in [("shenzhen", "montgomery"), ("montgomery", "shenzhen")]:
        X_train, y_train = get_xy(df[df.dataset == train_set], FEATURE_NAMES)
        X_test, y_test = get_xy(df[df.dataset == test_set], FEATURE_NAMES)
        print(f"\n=== train {train_set} ({len(y_train)}) -> test {test_set} ({len(y_test)}) ===")

        for name, (model, grid) in make_models().items():
            best, params, cv_acc = tune_and_fit(model, grid, X_train, y_train)
            metrics, _ = evaluate(best, X_test, y_test)
            print(f"{name}: {rounded(metrics)}")
            rows.append({"train": train_set, "test": test_set, "model": name,
                         "best_settings": str(params), "cv_accuracy": cv_acc, **metrics})

    out = pd.DataFrame(rows).round(3)
    out.to_csv(RESULTS / "cross_dataset_metrics.csv", index=False)
    print("\n", out[["train", "test", "model", "accuracy", "sensitivity", "specificity", "f1", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
