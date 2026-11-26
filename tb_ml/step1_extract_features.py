"""Step 1 - extract numerical features from every chest X-ray.

Reads all images from the Montgomery and Shenzhen sets, computes the 13
features in features.py for each one, and saves a table:

    results/features.csv   one row per image: image_id, dataset, label, 13 features

It also saves a box plot of every feature for normal vs TB images, which is a
quick way to see which features differ between the two groups.

Usage:
    python step1_extract_features.py
    python step1_extract_features.py --raw D:/tb_data/raw
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from features import FEATURE_NAMES, extract_features

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# folder name of each dataset inside the raw data folder
DATASETS = {"montgomery": "MontgomerySet", "shenzhen": "ChinaSet_AllFiles"}


def label_from_filename(path):
    """NLM naming: ..._0.png = normal, ..._1.png = TB."""
    return int(path.stem.rsplit("_", 1)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "data" / "raw"),
                    help="folder containing MontgomerySet/ and ChinaSet_AllFiles/")
    args = ap.parse_args()
    raw = Path(args.raw)

    rows = []
    for dataset, folder in DATASETS.items():
        images = sorted((raw / folder / "CXR_png").glob("*.png"))
        if not images:
            raise SystemExit(f"No images found in {raw / folder / 'CXR_png'}")
        print(f"{dataset}: {len(images)} images")
        for i, path in enumerate(images, 1):
            row = {"image_id": path.stem, "dataset": dataset,
                   "label": label_from_filename(path)}
            row.update(extract_features(path))
            rows.append(row)
            if i % 100 == 0 or i == len(images):
                print(f"  {i}/{len(images)} done")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "features.csv", index=False)
    print(f"Saved {len(df)} rows to {RESULTS / 'features.csv'}")

    # Count of images per dataset and class
    print(df.groupby(["dataset", "label"]).size().unstack().rename(columns={0: "normal", 1: "TB"}))

    save_sample_grid(raw)

    # Box plot: each feature, normal vs TB (both datasets together)
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for ax, name in zip(axes.ravel(), FEATURE_NAMES):
        ax.boxplot([df.loc[df.label == 0, name], df.loc[df.label == 1, name]])
        ax.set_xticks([1, 2], ["Normal", "TB"])
        ax.set_title(name)
    for ax in axes.ravel()[len(FEATURE_NAMES):]:
        ax.axis("off")
    fig.suptitle("Feature values: normal vs TB")
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_boxplots.png", dpi=120)
    print(f"Saved {RESULTS / 'feature_boxplots.png'}")


def save_sample_grid(raw):
    """Two normal and two TB images from each dataset, as the model sees them (256 x 256)."""
    from features import load_image
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.5))
    for row, (dataset, folder) in enumerate(DATASETS.items()):
        images = sorted((raw / folder / "CXR_png").glob("*.png"))
        normal = [p for p in images if label_from_filename(p) == 0][:2]
        tb = [p for p in images if label_from_filename(p) == 1][:2]
        for ax, path in zip(axes[row], normal + tb):
            ax.imshow(load_image(path), cmap="gray")
            ax.set_title(f"{dataset} - {'TB' if label_from_filename(path) else 'Normal'}\n{path.stem}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Sample images after resizing to 256 x 256 (top: Montgomery, bottom: Shenzhen)")
    fig.tight_layout(h_pad=2.5, rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS / "sample_images.png", dpi=110)
    print(f"Saved {RESULTS / 'sample_images.png'}")


if __name__ == "__main__":
    main()
