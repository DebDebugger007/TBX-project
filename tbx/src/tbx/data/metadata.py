"""Build the one metadata table every later stage reads.

One row per image with: image_id, dataset, label, image_path, mask paths
(Montgomery only; empty for Shenzhen), reading_path, age, sex, finding, and the
original width/height (read from the PNG header only, so this stays fast).

The TB label comes from the last digit of the filename (``_0`` normal, ``_1``
TB), which is the NLM convention.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .readings import parse_reading

DATASET_DIRS = {
    "montgomery": "MontgomerySet",
    "shenzhen": "ChinaSet_AllFiles",
}


def _label_from_stem(stem: str) -> int:
    last = stem.rsplit("_", 1)[-1]
    if last not in {"0", "1"}:
        raise ValueError(f"cannot read TB label from filename {stem!r}")
    return int(last)


def _find_dataset_root(raw_root: Path, folder: str) -> Path | None:
    """Return raw_root/folder, also tolerating one extra nesting level
    (e.g. ChinaSet_AllFiles/ChinaSet_AllFiles/ after some unzip tools)."""
    direct = raw_root / folder
    if (direct / "CXR_png").is_dir():
        return direct
    nested = direct / folder
    if (nested / "CXR_png").is_dir():
        return nested
    return None


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return (0, 0)


def _rows_for(dataset: str, root: Path) -> list[dict]:
    rows = []
    for img_path in sorted((root / "CXR_png").glob("*.png")):
        stem = img_path.stem
        try:
            label = _label_from_stem(stem)
        except ValueError:
            print(f"[metadata] skipping {img_path.name}: no _0/_1 label suffix")
            continue

        left = right = ""
        if dataset == "montgomery":
            lp = root / "ManualMask" / "leftMask" / img_path.name
            rp = root / "ManualMask" / "rightMask" / img_path.name
            if lp.exists() and rp.exists():
                left, right = str(lp), str(rp)

        reading_path = root / "ClinicalReadings" / f"{stem}.txt"
        if reading_path.exists():
            info = parse_reading(reading_path, dataset)
        else:
            info = {"age": None, "sex": None, "finding": ""}

        w, h = _image_size(img_path)
        rows.append(
            {
                "image_id": stem,
                "dataset": dataset,
                "label": label,
                "image_path": str(img_path),
                "left_mask_path": left,
                "right_mask_path": right,
                "reading_path": str(reading_path) if reading_path.exists() else "",
                "age": info["age"],
                "sex": info["sex"],
                "finding": info["finding"],
                "width": w,
                "height": h,
            }
        )
    return rows


def build_metadata(raw_root: str | Path) -> pd.DataFrame:
    """Scan the NLM layout under ``raw_root`` and return the metadata table."""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"raw data folder not found: {raw_root.resolve()}\n"
            "Download the Montgomery and Shenzhen sets and unzip them there "
            "(see 'Data layout' in the README)."
        )

    rows: list[dict] = []
    for dataset, folder in DATASET_DIRS.items():
        root = _find_dataset_root(raw_root, folder)
        if root is None:
            print(f"[metadata] WARNING: {raw_root / folder / 'CXR_png'} not found; no {dataset} images")
            continue
        ds_rows = _rows_for(dataset, root)
        n_tb = sum(r["label"] for r in ds_rows)
        print(f"[metadata] {dataset}: {len(ds_rows)} images ({n_tb} TB, {len(ds_rows) - n_tb} normal)")
        rows.extend(ds_rows)

    if not rows:
        raise RuntimeError(f"no images found under {raw_root.resolve()}")

    df = pd.DataFrame(rows)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df
