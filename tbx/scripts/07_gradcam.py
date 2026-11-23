"""Phase 5 — Grad-CAM: where does the CNN look?

Usage:
    python scripts/07_gradcam.py --config configs/E3_mc_to_sz_cross.yaml
    python scripts/07_gradcam.py --config configs/E2_shenzhen_within.yaml --per-group 6

Run it after the experiment itself (05_run_experiment.py), which saves the
weights. Each test image is explained by the model that actually scored it: the
single cross-dataset model, or, for within-set runs, the fold model for which
that image was held out.

Produces, under outputs/<experiment>/gradcam/:
    gradcam_grid.png      heatmaps for four groups of test images:
                          confident TB, missed TB, false alarms, confident normal
    lung_attention.csv    per test image: share of Grad-CAM mass inside the lung
                          mask, the lung's share of the image, and their ratio
    summary.json          the same, averaged (whole-image runs only)

The lung-attention ratio is the quantitative half of the shortcut audit: a ratio
near 1 means attention is spread over the image no more than the lungs' size
would predict; well above 1 means it concentrates on the lungs; below 1 means it
is mostly elsewhere (borders, text, shoulders, devices).
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch.utils.data import DataLoader

from tbx.config import load_config
from tbx.data.splits import fold_assignment, load_folds
from tbx.models.cnn import build_cnn
from tbx.models.datasets import CXRDataset, _load_mask_for_row
from tbx.utils import seed_everything
from tbx.utils.io import ensure_dir, load_json, save_json

GROUPS = [
    ("confident TB", 1, False),      # highest-scoring TB images
    ("missed TB", 1, True),          # lowest-scoring TB images
    ("false alarm", 0, False),       # highest-scoring normal images
    ("confident normal", 0, True),   # lowest-scoring normal images
]


def _target_layer(model, name: str):
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("densenet"):
        return model.features.denseblock4
    raise ValueError(f"no Grad-CAM target layer defined for {name!r}")


def _checkpoint_map(cfg, out_dir: Path, image_ids: list[str]) -> dict[str, Path]:
    ckpt_dir = out_dir / "checkpoints"
    if cfg.eval_mode == "cross":
        p = ckpt_dir / "model.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; re-run 05_run_experiment.py for {cfg.name} (it now saves weights)")
        return {i: p for i in image_ids}
    assign = fold_assignment(load_folds(cfg.paths.splits_dir, cfg.train))
    out = {}
    for i in image_ids:
        p = ckpt_dir / f"fold{assign[i]}.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; re-run 05_run_experiment.py for {cfg.name}")
        out[i] = p
    return out


def _select(scores: pd.DataFrame, n: int) -> dict[str, list[str]]:
    sel = {}
    for name, label, ascending in GROUPS:
        sub = scores[scores["label"] == label].sort_values("score", ascending=ascending)
        sel[name] = sub["image_id"].head(n).tolist()
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the CNN experiment to explain")
    ap.add_argument("--per-group", type=int, default=4, help="images per group in the grid")
    ap.add_argument("--no-lung-attention", action="store_true", help="skip the all-image lung-attention pass")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.model == "classical":
        raise SystemExit("Grad-CAM applies to the CNN experiments only")
    seed_everything(cfg.data.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.paths.outputs_root) / cfg.name
    res_path = out_dir / "results.json"
    if not res_path.exists():
        raise SystemExit(f"{res_path} not found; run 05_run_experiment.py --config {args.config} first")
    res = load_json(res_path)

    scores = pd.DataFrame({"image_id": res["image_ids"], "label": res["oof_true"], "score": res["oof_scores"]})
    scores = scores.dropna(subset=["score"])
    meta = pd.read_csv(cfg.paths.metadata_csv).set_index("image_id", drop=False)

    selected = _select(scores, args.per_group)
    grid_ids = {i for ids in selected.values() for i in ids}
    do_lung = cfg.region == "whole" and not args.no_lung_attention
    todo = scores["image_id"].tolist() if do_lung else sorted(grid_ids)

    ckpts = _checkpoint_map(cfg, out_dir, todo)
    by_ckpt: dict[Path, list[str]] = {}
    for i in todo:
        by_ckpt.setdefault(ckpts[i], []).append(i)

    cams: dict[str, np.ndarray] = {}
    bases: dict[str, np.ndarray] = {}
    lung_rows = []
    for ckpt, ids in sorted(by_ckpt.items()):
        model = build_cnn(cfg.model, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()

        rows = meta.loc[ids].reset_index(drop=True)
        ds = CXRDataset(rows, cfg.data.image_size, cfg.region, cfg.paths.masks_dir, correction=cfg.correction, train=False, cfg=cfg.model_cfg)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        print(f"[gradcam] {ckpt.parent.parent.name}/{ckpt.name}: {len(ids)} images")

        with GradCAM(model=model, target_layers=[_target_layer(model, cfg.model)]) as cam:
            k = 0
            for x, _, batch_ids in dl:
                maps = cam(input_tensor=x.to(device), targets=[BinaryClassifierOutputTarget(1)] * len(x))
                for j, image_id in enumerate(batch_ids):
                    m = maps[j]
                    if image_id in grid_ids:
                        cams[image_id] = m
                        bases[image_id] = ds._base[k + j]
                    if do_lung:
                        mask = _load_mask_for_row(rows.iloc[k + j], cfg.data.image_size, cfg.paths.masks_dir)
                        if mask is not None and m.sum() > 0:
                            inside = float(m[mask > 0].sum() / m.sum())
                            area = float(mask.mean())
                            lung_rows.append({"image_id": image_id, "cam_in_lung": inside,
                                              "lung_area": area, "ratio": inside / area if area else np.nan})
                k += len(batch_ids)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    gc_dir = ensure_dir(out_dir / "gradcam")

    # --- the grid -----------------------------------------------------------
    n = args.per_group
    fig, axes = plt.subplots(len(GROUPS), n, figsize=(n * 2.6, len(GROUPS) * 2.8), squeeze=False)
    sc = scores.set_index("image_id")
    for r, (gname, _, _) in enumerate(GROUPS):
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            ids = selected[gname]
            if c >= len(ids):
                continue
            i = ids[c]
            ax.imshow(bases[i], cmap="gray", vmin=0, vmax=1)
            ax.imshow(cams[i], cmap="jet", alpha=0.35, vmin=0, vmax=1)
            mask = (_load_mask_for_row(meta.loc[i], cfg.data.image_size, cfg.paths.masks_dir)
                    if cfg.region == "whole" else None)
            if mask is not None:
                ax.contour(mask, levels=[0.5], colors="white", linewidths=0.6)
            ax.set_title(f"{i}\nlabel {int(sc.loc[i, 'label'])}  score {sc.loc[i, 'score']:.2f}", fontsize=7)
        axes[r, 0].text(-0.08, 0.5, gname, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10)
    fig.suptitle(f"Grad-CAM — {cfg.name} (train {cfg.train}, test {cfg.test}, region {cfg.region})", fontsize=11)
    fig.tight_layout()
    grid_path = gc_dir / "gradcam_grid.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam] wrote {grid_path}")

    # --- lung attention -----------------------------------------------------
    if do_lung and lung_rows:
        la = pd.DataFrame(lung_rows).merge(scores, on="image_id")
        la.to_csv(gc_dir / "lung_attention.csv", index=False)
        summary = {
            "experiment": cfg.name,
            "n_images": int(len(la)),
            "n_without_mask": int(len(todo) - len(la)),
            "mean_cam_in_lung": float(la["cam_in_lung"].mean()),
            "mean_lung_area": float(la["lung_area"].mean()),
            "median_ratio": float(la["ratio"].median()),
            "by_label": {
                ("TB" if lab == 1 else "normal"): {
                    "mean_cam_in_lung": float(g["cam_in_lung"].mean()),
                    "median_ratio": float(g["ratio"].median()),
                }
                for lab, g in la.groupby("label")
            },
        }
        save_json(summary, gc_dir / "summary.json")
        print(f"[gradcam] {summary['mean_cam_in_lung']:.1%} of attention inside the lungs, which cover "
              f"{summary['mean_lung_area']:.1%} of the image (median ratio {summary['median_ratio']:.2f}) "
              f"-> {gc_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
