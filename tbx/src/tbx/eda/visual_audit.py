"""Side-by-side visual audit.

The brief asks for this before any modelling: look at a grid of images from both
sets and write down everything a model could use *other than* disease — borders
and padding, text and laterality markers, collimation edges, contrast, how much
neck and abdomen is in frame, rotation, cables and devices. The checklist lives
in Appendix B of the IPR; this module just builds the grids to work through.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.preprocess import preprocess_image
from ..utils.io import ensure_dir


def sample_grid(df: pd.DataFrame, dataset: str, label: int, n: int, seed: int = 42) -> pd.DataFrame:
    sub = df[(df["dataset"] == dataset) & (df["label"] == label)]
    n = min(n, len(sub))
    return sub.sample(n=n, random_state=seed)


def save_audit_grid(
    df: pd.DataFrame,
    out_path: str | Path,
    size: int = 256,
    per_cell: int = 3,
    seed: int = 42,
) -> Path:
    """A 2x2 block grid: {Montgomery, Shenzhen} x {normal, TB}.

    Each cell shows ``per_cell`` randomly drawn images at a working resolution,
    so the two sets sit next to each other and systematic differences (framing,
    contrast, markers) are visible at a glance.
    """
    blocks = [
        ("montgomery", 0, "Montgomery — normal"),
        ("montgomery", 1, "Montgomery — TB"),
        ("shenzhen", 0, "Shenzhen — normal"),
        ("shenzhen", 1, "Shenzhen — TB"),
    ]
    fig, axes = plt.subplots(len(blocks), per_cell, figsize=(per_cell * 2.6, len(blocks) * 2.6))
    if per_cell == 1:
        axes = axes[:, None]

    for r, (ds, label, title) in enumerate(blocks):
        chosen = sample_grid(df, ds, label, per_cell, seed=seed)
        for c in range(per_cell):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(chosen):
                row = chosen.iloc[c]
                img = preprocess_image(row["image_path"], ds, size)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                if c == 0:
                    ax.set_title(title, fontsize=9, loc="left")
    fig.suptitle("Visual audit — two sets, two classes", fontsize=11)
    fig.tight_layout()

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] wrote {out_path}")
    return out_path
