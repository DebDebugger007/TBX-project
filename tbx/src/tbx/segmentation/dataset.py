"""Torch dataset for lung-mask training (Montgomery image/mask pairs)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data.masks import merge_montgomery_masks, prepare_mask
from ..data.preprocess import preprocess_image


class LungMaskDataset(Dataset):
    """Yield (image[1,H,W], mask[1,H,W]) tensors for Montgomery.

    Only rows with both left and right mask paths are usable, so the caller
    should filter to Montgomery before constructing this.
    """

    def __init__(self, df: pd.DataFrame, size: int = 256, augment: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.augment = augment
        # Decode once up front: the full-size PNGs take far longer to decode than
        # the GPU takes to train on them, and ~140 images at 256 px fit easily
        # in memory (~70 MB).
        self._items = [
            (
                preprocess_image(r["image_path"], r["dataset"], size),
                prepare_mask(merge_montgomery_masks(r["left_mask_path"], r["right_mask_path"]), size),
            )
            for _, r in self.df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img, mask = self._items[idx]

        if self.augment is not None:
            img, mask = self.augment(img, mask)

        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, mask_t
