"""Torch dataset for the CNN, with the image region as a setting.

The region ("whole", "lung_cropped", "lung_masked") is applied here, so the same
dataset class feeds every experiment and the experiment matrix is driven entirely
by config. Masks come from the Montgomery manual masks or the U-Net-generated
Shenzhen masks, resolved by image_id.

Augmentation is mild and, deliberately, has **no horizontal flip** — flipping a
chest X-ray moves the heart and any laterality marker to the wrong side, which
never happens in a real image and could teach the model something false.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import ModelConfig
from ..data.masks import apply_region, merge_montgomery_masks, prepare_mask
from ..data.preprocess import preprocess_image, resize_with_pad

# ImageNet statistics — the backbone is pretrained, so we match its expected
# input normalisation after expanding the single X-ray channel to three.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_mask_for_row(row: pd.Series, size: int, masks_dir: Optional[str]) -> Optional[np.ndarray]:
    if row["dataset"] == "montgomery" and row.get("left_mask_path"):
        m = merge_montgomery_masks(row["left_mask_path"], row["right_mask_path"])
        return prepare_mask(m, size)
    # Shenzhen (or Montgomery without manual mask): use a generated mask if present
    gen = row.get("gen_mask_path", "")
    if not gen and masks_dir:
        candidate = Path(masks_dir) / f"{row['image_id']}.png"
        gen = str(candidate) if candidate.exists() else ""
    if gen and Path(gen).exists():
        raw = cv2.imread(str(gen), cv2.IMREAD_GRAYSCALE)
        return prepare_mask(raw, size)
    return None


class CXRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        size: int = 224,
        region: str = "whole",
        masks_dir: Optional[str] = None,
        train: bool = False,
        cfg: Optional[ModelConfig] = None,
        seed: int = 42,
        correction: str = "none",
    ):
        self.correction = correction
        self.df = df.reset_index(drop=True)
        self.size = size
        self.region = region
        self.masks_dir = masks_dir
        self.train = train
        self.cfg = cfg or ModelConfig()
        self.rng = np.random.default_rng(seed)
        # Preprocess (and apply the region) once; only augmentation is per-epoch.
        # A few hundred 224 px float images is ~100 MB, and this keeps the GPU
        # fed instead of waiting on PNG decoding every epoch.
        self._base = [self._load_base(r) for _, r in self.df.iterrows()]

    def _load_base(self, row: pd.Series) -> np.ndarray:
        img = preprocess_image(row["image_path"], row["dataset"], self.size, self.correction)
        if self.region != "whole":
            mask = _load_mask_for_row(row, self.size, self.masks_dir)
            img = apply_region(img, mask, self.region)
            if img.shape[0] != self.size:  # cropping changes size
                img = resize_with_pad(img, self.size)
        return img

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg
        # small rotation
        if c.aug_rotation_deg > 0:
            angle = float(self.rng.uniform(-c.aug_rotation_deg, c.aug_rotation_deg))
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
        # random resized crop (scale only, no flip)
        if c.aug_crop_scale < 1.0:
            scale = float(self.rng.uniform(c.aug_crop_scale, 1.0))
            h, w = img.shape
            ch, cw = int(h * scale), int(w * scale)
            y0 = int(self.rng.integers(0, h - ch + 1))
            x0 = int(self.rng.integers(0, w - cw + 1))
            img = resize_with_pad(img[y0:y0 + ch, x0:x0 + cw], self.size)
        # brightness
        if c.aug_brightness > 0:
            delta = float(self.rng.uniform(-c.aug_brightness, c.aug_brightness))
            img = np.clip(img + delta, 0.0, 1.0)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._base[idx]

        if self.train:
            img = self._augment(img)

        # 1->3 channels, ImageNet normalisation
        img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
        img3 = (img3 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(img3.transpose(2, 0, 1))
        label = int(row["label"])
        return tensor, label, row["image_id"]


def build_transforms(*args, **kwargs):
    """Kept for API symmetry; augmentation is handled inside CXRDataset so that
    the region switch and the mask lookup stay in one place."""
    return None
