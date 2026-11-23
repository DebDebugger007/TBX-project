"""End-to-end smoke test on a tiny synthetic NLM-layout dataset.

Builds a handful of fake Montgomery/Shenzhen images, masks and reading files,
then runs the non-deep-learning pipeline: metadata -> splits -> EDA -> features
-> classical within-set experiment with bootstrap CIs. Also unit-checks the
bootstrap and DeLong routines. Run with:

    python tests/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from tbx.config import load_config
from tbx.data.metadata import build_metadata
from tbx.data.splits import make_stratified_folds
from tbx.eda import save_audit_grid, summarise_metadata, plot_class_balance
from tbx.evaluation.bootstrap import bootstrap_auc_ci
from tbx.evaluation.delong import delong_roc_test


def _make_cxr(tb: bool, rng) -> np.ndarray:
    """A 256x256 fake X-ray: oval 'lungs' on a lighter torso, TB adds specks."""
    img = np.full((256, 256), 40, dtype=np.uint8)
    cv2.ellipse(img, (90, 130), (45, 80), 0, 0, 360, 110, -1)
    cv2.ellipse(img, (166, 130), (45, 80), 0, 0, 360, 110, -1)
    if tb:
        for _ in range(rng.integers(8, 16)):
            cx = int(rng.integers(50, 206)); cy = int(rng.integers(60, 200))
            cv2.circle(img, (cx, cy), int(rng.integers(2, 6)), int(rng.integers(150, 220)), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _lung_masks():
    left = np.zeros((256, 256), np.uint8)
    right = np.zeros((256, 256), np.uint8)
    cv2.ellipse(left, (90, 130), (45, 80), 0, 0, 360, 255, -1)
    cv2.ellipse(right, (166, 130), (45, 80), 0, 0, 360, 255, -1)
    return left, right


def build_fake_raw(root: Path, n_mc=20, n_sz=40, seed=0):
    rng = np.random.default_rng(seed)
    left, right = _lung_masks()

    mc = root / "MontgomerySet"
    (mc / "CXR_png").mkdir(parents=True)
    (mc / "ManualMask" / "leftMask").mkdir(parents=True)
    (mc / "ManualMask" / "rightMask").mkdir(parents=True)
    (mc / "ClinicalReadings").mkdir(parents=True)
    for i in range(n_mc):
        tb = i % 2 == 0
        stem = f"MCUCXR_{i:04d}_{int(tb)}"
        cv2.imwrite(str(mc / "CXR_png" / f"{stem}.png"), _make_cxr(tb, rng))
        cv2.imwrite(str(mc / "ManualMask" / "leftMask" / f"{stem}.png"), left)
        cv2.imwrite(str(mc / "ManualMask" / "rightMask" / f"{stem}.png"), right)
        sex = "M" if i % 3 else "F"
        (mc / "ClinicalReadings" / f"{stem}.txt").write_text(
            f"Patient's Sex: {sex}\nPatient's Age: {40 + i % 30:03d}Y\n"
            + ("tuberculosis, upper lobe" if tb else "normal")
        )

    sz = root / "ChinaSet_AllFiles"
    (sz / "CXR_png").mkdir(parents=True)
    (sz / "ClinicalReadings").mkdir(parents=True)
    for i in range(n_sz):
        tb = i % 2 == 0
        stem = f"CHNCXR_{i:04d}_{int(tb)}"
        cv2.imwrite(str(sz / "CXR_png" / f"{stem}.png"), _make_cxr(tb, rng))
        sex = "female" if i % 2 else "male"
        (sz / "ClinicalReadings" / f"{stem}.txt").write_text(
            f"{sex} {35 + i % 40}yrs\n" + ("PTB" if tb else "normal")
        )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tbx_smoke_"))
    raw = tmp / "data" / "raw"
    build_fake_raw(raw)
    print(f"[smoke] fake raw data at {raw}")

    # Point a config at the temp tree
    import yaml
    base = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text())
    base["paths"]["raw_root"] = str(raw)
    base["paths"]["processed_root"] = str(tmp / "data" / "processed")
    base["paths"]["metadata_csv"] = str(tmp / "data" / "processed" / "metadata.csv")
    base["paths"]["splits_dir"] = str(tmp / "data" / "processed" / "splits")
    base["paths"]["masks_dir"] = str(tmp / "data" / "processed" / "masks")
    base["paths"]["outputs_root"] = str(tmp / "outputs")
    base["eval"]["bootstrap_n"] = 200  # keep the smoke test quick
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(base))
    cfg = load_config(cfg_path)

    # 1. metadata + splits
    import pandas as pd
    df = build_metadata(cfg.paths.raw_root)
    Path(cfg.paths.processed_root).mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.paths.metadata_csv, index=False)
    for ds in ("montgomery", "shenzhen"):
        make_stratified_folds(df, ds, cfg.paths.splits_dir, cfg.data.n_folds, cfg.data.seed)
    assert df["age"].notna().all(), "age parsing failed"
    assert set(df["sex"].dropna()) <= {"M", "F"}, "sex parsing failed"
    print("[smoke] metadata + splits OK")

    # 2. EDA (just confirm the plots write)
    out_eda = tmp / "outputs" / "eda"
    summarise_metadata(df).to_csv(tmp / "summary.csv", index=False)
    save_audit_grid(df, out_eda / "visual_audit.png", seed=cfg.data.seed)
    plot_class_balance(df, out_eda / "class_balance.png")
    print("[smoke] EDA OK")

    # 3. features (Montgomery has masks; Shenzhen has none -> skipped here, which
    #    is the real behaviour before the U-Net runs)
    from tbx.data.masks import merge_montgomery_masks, prepare_mask
    from tbx.data.preprocess import preprocess_image
    from tbx.features import lung_field_features
    rows = []
    for _, r in df[df.dataset == "montgomery"].iterrows():
        m = prepare_mask(merge_montgomery_masks(r.left_mask_path, r.right_mask_path), 256)
        img = preprocess_image(r.image_path, r.dataset, 256)
        f = lung_field_features(img, m, cfg.features)
        f.update({"image_id": r.image_id, "label": int(r.label), "dataset": r.dataset})
        rows.append(f)
    feats = pd.DataFrame(rows)
    feats.to_csv(Path(cfg.paths.processed_root) / "features.csv", index=False)
    print(f"[smoke] features OK ({feats.shape[0]} rows, {feats.shape[1]-3} features)")

    # 4. classical within-set experiment through the runner
    from tbx.config import load_config as lc
    cfg.name = "SMOKE_classical"
    cfg.model = "classical"
    cfg.train = cfg.test = "montgomery"
    cfg.eval_mode = "within"
    cfg.region = "whole"
    from tbx.runner import run_experiment
    res = run_experiment(cfg, df, feats)
    auc = res["ci"]["auc"]
    assert 0.0 <= auc["lo"] <= auc["point"] <= auc["hi"] <= 1.0, "CI ordering wrong"
    print("[smoke] classical within-set experiment OK")

    # 5. unit-check bootstrap + DeLong on a clean separable signal
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(60), np.ones(60)].astype(int)
    s_good = np.r_[rng.normal(0, 1, 60), rng.normal(2.5, 1, 60)]
    s_bad = np.r_[rng.normal(0, 1, 60), rng.normal(0.3, 1, 60)]
    ci = bootstrap_auc_ci(y, s_good, n=200)
    dl = delong_roc_test(y, s_good, s_bad)
    assert ci["point"] > 0.8, "bootstrap AUC unexpectedly low"
    assert dl["p_value"] < 0.05 and dl["auc_a"] > dl["auc_b"], "DeLong did not separate the models"
    print(f"[smoke] bootstrap AUC {ci['point']:.3f} CI ({ci['lo']:.3f}-{ci['hi']:.3f}); "
          f"DeLong p={dl['p_value']:.1e}")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
