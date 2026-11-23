"""Phase 5 — paired AUC comparisons (DeLong's test, Holm-corrected).

Usage:
    python scripts/08_compare_auc.py                       # all standard pairs that have results
    python scripts/08_compare_auc.py --pairs E3_mc_to_sz_cross:E3_mc_to_sz_cross_lungmask

DeLong's test needs both models scored on the *same* test images, so it answers
questions like:

* whole image vs lung-masked / lung-cropped, same direction   (is the model using the lungs?)
* CNN vs classical lung-field baseline, same direction        (does the CNN beat lungs-only?)

It cannot compare a within-set AUC against a cross-dataset AUC (different test
sets); for that drop, compare the bootstrap CIs in the results table.

Images are matched by image_id. p-values are Holm-corrected across every
comparison *within its family*: "phase5" (region controls and CNN vs
classical) and "phase6" (each intensity correction vs none) are corrected
separately, and --pairs runs form their own "custom" family.

Produces:
    outputs/auc_comparisons.csv
    outputs/auc_comparisons.md
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np
import pandas as pd

from tbx.config import load_config
from tbx.evaluation.delong import delong_roc_test
from tbx.utils.io import load_json

# CNN whole-image run -> the classical run on the same test set
CLASSICAL_PARTNER = {
    "E1_montgomery_within": "E0_classical_within",
    "E2_shenzhen_within": "E0_classical_within_shenzhen",
    "E3_mc_to_sz_cross": "E3_classical_mc_to_sz_cross",
    "E4_sz_to_mc_cross": "E4_classical_sz_to_mc_cross",
}
REGION_SUFFIXES = {"_lungcrop": "lung-cropped", "_lungmask": "lung-masked"}
CORRECTION_SUFFIXES = {"_histeq": "histeq", "_clahe": "CLAHE", "_zscore": "z-score"}
# Phase 6 compares each correction against the uncorrected cross-dataset run
CORRECTION_BASES = [
    "E3_mc_to_sz_cross", "E4_sz_to_mc_cross",
    "E3_classical_mc_to_sz_cross", "E4_classical_sz_to_mc_cross",
]


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def _scores(res: dict) -> pd.DataFrame:
    df = pd.DataFrame({"image_id": res["image_ids"], "label": res["oof_true"], "score": res["oof_scores"]})
    return df.dropna(subset=["score"]).drop_duplicates("image_id").set_index("image_id")


def default_pairs(available: set[str]) -> list[tuple[str, str, str, str]]:
    """(A, B, comparison, family). Holm runs separately within each family."""
    pairs = []
    for base, classical in CLASSICAL_PARTNER.items():
        for suffix, label in REGION_SUFFIXES.items():
            if base in available and base + suffix in available:
                pairs.append((base, base + suffix, f"whole vs {label}", "phase5"))
        if base in available and classical in available:
            pairs.append((base, classical, "CNN vs classical", "phase5"))
    for base in CORRECTION_BASES:
        for suffix, label in CORRECTION_SUFFIXES.items():
            if base in available and base + suffix in available:
                pairs.append((base, base + suffix, f"none vs {label}", "phase6"))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pairs", nargs="*", help="explicit A:B experiment-name pairs (default: all standard pairs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.paths.outputs_root)
    results = {p.parent.name: load_json(p) for p in root.glob("*/results.json")}

    if args.pairs:
        pairs = []
        for spec in args.pairs:
            a, _, b = spec.partition(":")
            if not b:
                raise SystemExit(f"bad pair {spec!r}; use A:B")
            pairs.append((a, b, "custom", "custom"))
    else:
        pairs = default_pairs(set(results))

    rows = []
    for a, b, kind, family in pairs:
        missing = [x for x in (a, b) if x not in results]
        if missing:
            print(f"[compare] skip {a} vs {b}: no results for {', '.join(missing)}")
            continue
        sa, sb = _scores(results[a]), _scores(results[b])
        common = sa.index.intersection(sb.index)
        if len(common) == 0:
            print(f"[compare] skip {a} vs {b}: no test images in common (different test sets)")
            continue
        sa, sb = sa.loc[common], sb.loc[common]
        if not (sa["label"].values == sb["label"].values).all():
            raise SystemExit(f"label mismatch between {a} and {b}; check the inputs")
        d = delong_roc_test(sa["label"].values, sa["score"].values, sb["score"].values)
        rows.append({
            "family": family, "comparison": kind, "A": a, "B": b, "n": len(common),
            "auc_A": d["auc_a"], "auc_B": d["auc_b"], "diff": d["auc_diff"],
            "z": d["z"], "p": d["p_value"],
        })

    if not rows:
        print("[compare] nothing to compare yet — run the paired experiments first")
        return

    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, idx in table.groupby("family").groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p"].tolist())
    table["significant"] = table["p_holm"] < 0.05
    table.to_csv(root / "auc_comparisons.csv", index=False)

    md = table.copy()
    for c in ("auc_A", "auc_B", "diff", "z"):
        md[c] = md[c].map(lambda v: f"{v:.3f}")
    for c in ("p", "p_holm"):
        md[c] = md[c].map(lambda v: f"{v:.2g}")
    md["significant"] = md["significant"].map({True: "yes", False: "no"})
    (root / "auc_comparisons.md").write_text(md.to_markdown(index=False), encoding="utf-8")

    print(md.to_string(index=False))
    print(f"\n[compare] {len(table)} comparisons, Holm-corrected within each family "
          f"({', '.join(sorted(table['family'].unique()))}) -> {root / 'auc_comparisons.csv'}")


if __name__ == "__main__":
    main()
