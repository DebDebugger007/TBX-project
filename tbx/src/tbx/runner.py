"""One entry point that runs any experiment from its config.

This is where the "direction is a setting" design pays off: ``run_experiment``
reads a config, dispatches on ``model`` and ``eval_mode``, and writes a results
bundle (point metrics, per-fold table, pooled scores, bootstrap CIs) to
``outputs/<name>/``. Within-set (Phase 4) is fully wired; cross-dataset (Phase 5,
opening week 7) uses the same models trained on one whole set and tested once on
the other.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .evaluation.bootstrap import bootstrap_auc_ci, bootstrap_metric_ci
from .evaluation.metrics import (
    auc_score,
    specificity_at_sensitivity,
    summarise,
    threshold_from_training,
)
from .utils.io import ensure_dir, save_json


def _ci_block(y_true, y_score, cfg: ExperimentConfig) -> dict:
    """Bootstrap CIs on the pooled out-of-fold (or cross-test) scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    ok = ~np.isnan(y_score)
    y_true, y_score = y_true[ok], y_score[ok]

    auc_ci = bootstrap_auc_ci(y_true, y_score, cfg.eval.bootstrap_n, cfg.eval.ci, cfg.data.seed)
    spec90_ci = bootstrap_metric_ci(
        y_true, y_score,
        lambda t, s: specificity_at_sensitivity(t, s, cfg.eval.screening_sensitivity),
        cfg.eval.bootstrap_n, cfg.eval.ci, cfg.data.seed,
    )
    return {"auc": auc_ci, "specificity_at_90_sens": spec90_ci}


def _run_within(df: pd.DataFrame, cfg: ExperimentConfig, features_df: pd.DataFrame | None) -> dict:
    dataset = cfg.train  # within-set: train == test
    if cfg.model == "classical":
        from .models.classical import train_classical_cv

        if features_df is None:
            raise ValueError("classical model needs a features table (run 04_build_features.py)")
        res = train_classical_cv(features_df, cfg, dataset)
    else:
        from .models.cnn import train_cnn_cv

        res = train_cnn_cv(df, cfg, dataset)

    res["ci"] = _ci_block(res["oof_true"], res["oof_scores"], cfg)
    res["pooled_auc"] = auc_score(np.asarray(res["oof_true"])[~np.isnan(res["oof_scores"])],
                                  np.asarray(res["oof_scores"])[~np.isnan(res["oof_scores"])])
    return res


def _run_cross(df: pd.DataFrame, cfg: ExperimentConfig, features_df: pd.DataFrame | None = None) -> dict:
    """Train on the whole train set (minus a validation slice for the CNN), test
    once on the whole test set. The classical model uses the features table; the
    CNN's weights are saved so Grad-CAM (07_gradcam.py) can explain its errors."""
    if cfg.model == "classical":
        from .models.classical import train_classical_cross

        if features_df is None:
            raise ValueError("classical model needs a features table (run 04_build_features.py)")
        res = train_classical_cross(features_df, cfg)
        res["ci"] = _ci_block(res["oof_true"], res["oof_scores"], cfg)
        res["pooled_auc"] = res["point"]["auc"]
        return res

    import torch
    from torch.utils.data import DataLoader

    from .models.cnn import _infer, _train_one
    from .models.datasets import CXRDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_all = df[df["dataset"] == cfg.train].reset_index(drop=True)
    test_all = df[df["dataset"] == cfg.test].reset_index(drop=True)

    val = train_all.sample(frac=0.15, random_state=cfg.data.seed)
    train = train_all.drop(val.index)
    model = _train_one(train, val, cfg, device)
    ckpt_dir = ensure_dir(Path(cfg.paths.outputs_root) / cfg.name / "checkpoints")
    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    tr_ds = CXRDataset(train_all, cfg.data.image_size, cfg.region, cfg.paths.masks_dir, correction=cfg.correction, train=False, cfg=cfg.model_cfg)
    te_ds = CXRDataset(test_all, cfg.data.image_size, cfg.region, cfg.paths.masks_dir, correction=cfg.correction, train=False, cfg=cfg.model_cfg)
    tr_scores, tr_true, _ = _infer(model, DataLoader(tr_ds, batch_size=cfg.model_cfg.batch_size), device)
    thr = threshold_from_training(tr_true, tr_scores, cfg.eval.screening_sensitivity)
    te_scores, te_true, te_ids = _infer(model, DataLoader(te_ds, batch_size=cfg.model_cfg.batch_size), device)

    point = summarise(te_true, te_scores, thr, cfg.eval.screening_sensitivity)
    res = {
        "train": cfg.train,
        "test": cfg.test,
        "model": cfg.model,
        "region": cfg.region,
        "oof_true": te_true.tolist(),
        "oof_scores": te_scores.tolist(),
        "image_ids": list(te_ids),
        "point": point,
    }
    res["ci"] = _ci_block(te_true, te_scores, cfg)
    res["pooled_auc"] = point["auc"]
    return res


def run_experiment(cfg: ExperimentConfig, df: pd.DataFrame, features_df: pd.DataFrame | None = None) -> dict:
    cfg.validate()
    if cfg.eval_mode == "within":
        res = _run_within(df, cfg, features_df)
    else:
        res = _run_cross(df, cfg, features_df)

    res["config"] = cfg.as_dict()
    out_dir = ensure_dir(Path(cfg.paths.outputs_root) / cfg.name)
    save_json(res, out_dir / "results.json")

    auc = res["ci"]["auc"]
    print(
        f"[{cfg.name}] pooled AUC {auc['point']:.3f} "
        f"(95% CI {auc['lo']:.3f}-{auc['hi']:.3f})  -> {out_dir/'results.json'}"
    )
    return res
