"""Train the lung U-Net on Montgomery and generate Shenzhen masks.

Both routines are deliberately plain PyTorch so they run the same on a laptop
RTX GPU under WSL2 or on Colab. The training split holds out a fraction of
Montgomery to report Dice honestly before the model is pointed at Shenzhen.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.masks import prepare_mask
from ..data.preprocess import preprocess_image
from ..utils.io import ensure_dir
from .dataset import LungMaskDataset
from .metrics import dice_score
from .unet import UNet


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - ((num + eps) / den).mean()


def train_unet(
    df: pd.DataFrame,
    size: int = 256,
    epochs: int = 60,
    batch_size: int = 4,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    out_path: str | Path = "outputs/segmentation/unet.pt",
) -> dict:
    """Train on Montgomery pairs; return {model_path, val_dice}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mc = df[(df["dataset"] == "montgomery") & (df["left_mask_path"] != "")].reset_index(drop=True)
    if mc.empty:
        raise ValueError("no Montgomery image/mask pairs found")

    val = mc.sample(frac=val_frac, random_state=seed)
    train = mc.drop(val.index)
    train_ds = LungMaskDataset(train, size=size)
    val_ds = LungMaskDataset(val, size=size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    for epoch in range(epochs):
        model.train()
        for img, mask in train_dl:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = bce(logits, mask) + _dice_loss(logits, mask)
            loss.backward()
            opt.step()

        val_dice = _evaluate(model, val_dl, device)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), out_path)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[unet] epoch {epoch:3d}  val Dice {val_dice:.4f}  (best {best_dice:.4f})")

    print(f"[unet] best held-out Montgomery Dice {best_dice:.4f} -> {out_path}")
    return {"model_path": str(out_path), "val_dice": best_dice}


@torch.no_grad()
def _evaluate(model, loader, device) -> float:
    model.eval()
    scores = []
    for img, mask in loader:
        img = img.to(device)
        prob = torch.sigmoid(model(img)).cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8)
        for p, m in zip(pred, mask.numpy()):
            scores.append(dice_score(p[0], m[0]))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def generate_masks(
    df: pd.DataFrame,
    model_path: str | Path,
    masks_dir: str | Path,
    dataset: str = "shenzhen",
    size: int = 256,
    device: str | None = None,
) -> pd.DataFrame:
    """Run the trained U-Net over a dataset and write binary masks to disk.

    Returns the subframe with a filled ``gen_mask_path`` column. Writes one
    cleaned-up mask per image (largest two connected components kept, to drop
    stray specks) so the crop/mask region logic gets a sensible lung field.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    out_dir = ensure_dir(masks_dir)
    sub = df[df["dataset"] == dataset].reset_index(drop=True).copy()
    paths = []
    for _, row in sub.iterrows():
        img = preprocess_image(row["image_path"], dataset, size)
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(t)).cpu().numpy()[0, 0]
        mask = _postprocess((prob > 0.5).astype(np.uint8))
        p = out_dir / f"{row['image_id']}.png"
        cv2.imwrite(str(p), (mask * 255).astype(np.uint8))
        paths.append(str(p))
    sub["gen_mask_path"] = paths
    print(f"[unet] wrote {len(sub)} {dataset} masks to {out_dir}")
    return sub


def _postprocess(mask: np.ndarray) -> np.ndarray:
    """Keep the two largest components (the two lungs), fill small holes."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = (np.argsort(areas)[::-1] + 1)[:2]
    out = np.isin(labels, keep).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return out
