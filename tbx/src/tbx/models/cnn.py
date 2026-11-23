"""CNN training.

The main model is an ImageNet-pretrained ResNet-18 fine-tuned at 224 pixels
(512 later if GPU time allows), with DenseNet-121 as an architecture check. The
backbone is frozen for a few warm-up epochs so the fresh head settles before the
whole network moves, which matters on a few hundred images.

Within-set evaluation uses the frozen stratified folds: for each held-out fold
the model trains on the other four (a slice of which is carved off as validation
for early stopping), and the out-of-fold scores are pooled. The operating-point
threshold is fixed on the training scores, never on the held-out fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.splits import fold_assignment, load_folds
from ..evaluation.metrics import summarise, threshold_from_training
from ..utils.io import ensure_dir
from .datasets import CXRDataset


def build_cnn(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    import timm

    model = timm.create_model(name, pretrained=pretrained, num_classes=1, in_chans=3)
    return model


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    classifier = model.get_classifier()
    class_params = set(id(p) for p in classifier.parameters())
    for p in model.parameters():
        if id(p) not in class_params:
            p.requires_grad = trainable


@torch.no_grad()
def _infer(model, loader, device) -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    scores, labels, ids = [], [], []
    for x, y, image_id in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        scores.append(p)
        labels.append(y.numpy())
        ids.extend(image_id)
    return np.concatenate(scores), np.concatenate(labels), ids


def _train_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: ExperimentConfig,
    device: str,
) -> nn.Module:
    mc = cfg.model_cfg
    model = build_cnn(cfg.model, mc.pretrained).to(device)

    train_ds = CXRDataset(train_df, cfg.data.image_size, cfg.region, cfg.paths.masks_dir, correction=cfg.correction, train=True, cfg=mc, seed=cfg.data.seed)
    val_ds = CXRDataset(val_df, cfg.data.image_size, cfg.region, cfg.paths.masks_dir, correction=cfg.correction, train=False, cfg=mc, seed=cfg.data.seed)
    train_dl = DataLoader(train_ds, batch_size=mc.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=mc.batch_size, shuffle=False, num_workers=0)

    # class-weighted loss for Montgomery's 42% / Shenzhen's 51% prevalence
    pos = float(train_df["label"].sum())
    neg = float(len(train_df) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=mc.lr, weight_decay=mc.weight_decay)

    best_auc, best_state = -1.0, None
    from ..evaluation.metrics import auc_score

    for epoch in range(mc.epochs):
        _set_backbone_trainable(model, trainable=epoch >= mc.freeze_backbone_epochs)
        model.train()
        for x, y, _ in train_dl:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        vs, vy, _ = _infer(model, val_dl, device)
        va = auc_score(vy, vs)
        if va > best_auc:
            best_auc = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_cnn_cv(df: pd.DataFrame, cfg: ExperimentConfig, dataset: str, device: str | None = None) -> dict:
    """Within-set 5-fold CV for the CNN on one dataset; pooled OOF scores."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    folds = load_folds(cfg.paths.splits_dir, dataset)
    assign = fold_assignment(folds)

    sub = df[df["dataset"] == dataset].copy()
    sub["fold"] = sub["image_id"].map(assign)
    sub = sub.dropna(subset=["fold"])
    sub["fold"] = sub["fold"].astype(int)
    sub = sub.reset_index(drop=True)

    rng = np.random.default_rng(cfg.data.seed)
    oof_scores = np.full(len(sub), np.nan)
    per_fold = []
    ckpt_dir = ensure_dir(Path(cfg.paths.outputs_root) / cfg.name / "checkpoints")

    for fold_idx in range(cfg.data.n_folds):
        te_mask = (sub["fold"] == fold_idx).to_numpy()
        tr_pool = sub[~te_mask]
        if te_mask.sum() == 0 or tr_pool.empty:
            continue
        # carve a validation slice (one of the remaining folds) for early stopping
        val_fold = (fold_idx + 1) % cfg.data.n_folds
        val_df = tr_pool[tr_pool["fold"] == val_fold]
        train_df = tr_pool[tr_pool["fold"] != val_fold]
        if val_df.empty:  # tiny-dataset fallback
            val_df = tr_pool.sample(frac=0.2, random_state=cfg.data.seed)
            train_df = tr_pool.drop(val_df.index)

        model = _train_one(train_df, val_df, cfg, device)

        # threshold on training scores, applied to the held-out fold
        tr_ds = CXRDataset(train_df, cfg.data.image_size, cfg.region, cfg.paths.masks_dir, correction=cfg.correction, train=False, cfg=cfg.model_cfg)
        te_df = sub[te_mask]
        te_ds = CXRDataset(te_df, cfg.data.image_size, cfg.region, cfg.paths.masks_dir, correction=cfg.correction, train=False, cfg=cfg.model_cfg)
        tr_dl = DataLoader(tr_ds, batch_size=cfg.model_cfg.batch_size)
        te_dl = DataLoader(te_ds, batch_size=cfg.model_cfg.batch_size)

        tr_scores, tr_true, _ = _infer(model, tr_dl, device)
        thr = threshold_from_training(tr_true, tr_scores, cfg.eval.screening_sensitivity)
        te_scores, te_true, te_ids = _infer(model, te_dl, device)
        oof_scores[np.where(te_mask)[0]] = te_scores

        m = summarise(te_true, te_scores, thr, cfg.eval.screening_sensitivity)
        m["fold"] = fold_idx
        per_fold.append(m)
        torch.save(model.state_dict(), ckpt_dir / f"fold{fold_idx}.pt")
        print(f"[cnn:{dataset}] fold {fold_idx} AUC {m['auc']:.3f}")

    return {
        "dataset": dataset,
        "model": cfg.model,
        "region": cfg.region,
        "oof_true": sub["label"].tolist(),
        "oof_scores": oof_scores.tolist(),
        "image_ids": sub["image_id"].tolist(),
        "per_fold": per_fold,
    }
