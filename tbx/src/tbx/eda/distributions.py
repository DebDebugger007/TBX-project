"""Distribution plots and a numeric summary of the metadata.

Class balance, age distributions and sex ratios per set and per class, plus
intensity histograms per set. Anything that differs systematically between the
sets, or between TB and normal within one set, is a candidate shortcut and goes
on the list the lung-masking experiment later tests.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.preprocess import preprocess_image
from ..utils.io import ensure_dir

_LABELS = {0: "normal", 1: "TB"}


def summarise_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Per-set, per-class counts and age/sex summary for Table 2.2 / the EDA."""
    rows = []
    for ds in df["dataset"].unique():
        for label in (0, 1):
            sub = df[(df["dataset"] == ds) & (df["label"] == label)]
            ages = sub["age"].dropna()
            rows.append(
                {
                    "dataset": ds,
                    "class": _LABELS[label],
                    "n": len(sub),
                    "age_mean": round(ages.mean(), 1) if len(ages) else np.nan,
                    "age_sd": round(ages.std(), 1) if len(ages) else np.nan,
                    "pct_female": round(100 * (sub["sex"] == "F").mean(), 1) if len(sub) else np.nan,
                    "pct_missing_age": round(100 * sub["age"].isna().mean(), 1) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_class_balance(df: pd.DataFrame, out_path: str | Path) -> Path:
    counts = df.groupby(["dataset", "label"]).size().unstack(fill_value=0)
    counts = counts.rename(columns=_LABELS)
    ax = counts.plot(kind="bar", figsize=(5, 3.5))
    ax.set_ylabel("images")
    ax.set_xlabel("")
    ax.set_title("Class balance per dataset")
    ax.legend(title="")
    return _save(ax.figure, out_path)


def plot_age_distributions(df: pd.DataFrame, out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, ds in zip(axes, ["montgomery", "shenzhen"]):
        for label in (0, 1):
            ages = df[(df["dataset"] == ds) & (df["label"] == label)]["age"].dropna()
            if len(ages):
                ax.hist(ages, bins=15, alpha=0.55, label=_LABELS[label])
        ax.set_title(ds)
        ax.set_xlabel("age (years)")
        ax.legend()
    axes[0].set_ylabel("images")
    fig.suptitle("Age distribution by class")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_sex_ratios(df: pd.DataFrame, out_path: str | Path) -> Path:
    tab = (
        df.assign(female=(df["sex"] == "F").astype(int))
        .groupby(["dataset", "label"])["female"]
        .mean()
        .mul(100)
        .unstack()
        .rename(columns=_LABELS)
    )
    ax = tab.plot(kind="bar", figsize=(5, 3.5))
    ax.set_ylabel("% female")
    ax.set_xlabel("")
    ax.set_title("Sex ratio per dataset and class")
    ax.legend(title="")
    return _save(ax.figure, out_path)


def plot_intensity_histograms(
    df: pd.DataFrame, out_path: str | Path, size: int = 256, per_set: int = 40, seed: int = 42
) -> Path:
    """Mean pixel-intensity histogram per set on a sample of images.

    A systematic difference in the intensity profile between the two sets is one
    of the acquisition-shift cues the cross-dataset model could latch onto.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for ds in ["montgomery", "shenzhen"]:
        sub = df[df["dataset"] == ds].sample(n=min(per_set, (df["dataset"] == ds).sum()), random_state=seed)
        acc = np.zeros(50)
        for _, row in sub.iterrows():
            img = preprocess_image(row["image_path"], ds, size)
            hist, edges = np.histogram(img, bins=50, range=(0, 1), density=True)
            acc += hist
        acc /= max(len(sub), 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centres, acc, label=f"{ds} (n={len(sub)})")
    ax.set_xlabel("pixel intensity (normalised)")
    ax.set_ylabel("density")
    ax.set_title("Mean intensity histogram per set")
    ax.legend()
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] wrote {out_path}")
    return out_path
