"""Collate every experiment's results.json into one table.

Usage:
    python scripts/06_make_results_table.py --config configs/default.yaml

Produces:
    outputs/results_table.csv
    outputs/results_table.md    (paste-ready for Chapter II — Preliminary Results)

Each row is one experiment with pooled AUC and its 95% CI, plus sensitivity and
specificity at the fixed operating point. This is the table the IPR fills in as
each phase completes (Appendix B, "AUC (95% CI)" column).
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import pandas as pd

from tbx.config import load_config
from tbx.utils.io import load_json


def _mean_point(per_fold, key):
    vals = [m[key] for m in per_fold if m.get(key) == m.get(key)]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.paths.outputs_root)
    rows = []
    for res_path in sorted(root.glob("*/results.json")):
        res = load_json(res_path)
        c = res.get("config", {})
        auc = res["ci"]["auc"]
        per_fold = res.get("per_fold", [])
        sens = _mean_point(per_fold, "sensitivity") if per_fold else res.get("point", {}).get("sensitivity", float("nan"))
        spec = _mean_point(per_fold, "specificity") if per_fold else res.get("point", {}).get("specificity", float("nan"))
        rows.append({
            "experiment": c.get("name", res_path.parent.name),
            "train": c.get("train"),
            "test": c.get("test"),
            "region": c.get("region"),
            "model": c.get("model"),
            "eval": c.get("eval_mode"),
            "auc": round(auc["point"], 3),
            "auc_lo": round(auc["lo"], 3),
            "auc_hi": round(auc["hi"], 3),
            "sensitivity": round(sens, 3) if sens == sens else None,
            "specificity": round(spec, 3) if spec == spec else None,
        })

    if not rows:
        print("[results] no results.json files found under", root)
        return

    table = pd.DataFrame(rows)
    table.to_csv(root / "results_table.csv", index=False)

    # Markdown with the AUC + CI folded into one readable column
    md = table.copy()
    md["AUC (95% CI)"] = md.apply(lambda r: f"{r.auc:.3f} ({r.auc_lo:.3f}–{r.auc_hi:.3f})", axis=1)
    cols = ["experiment", "train", "test", "region", "model", "eval", "AUC (95% CI)", "sensitivity", "specificity"]
    (root / "results_table.md").write_text(md[cols].to_markdown(index=False), encoding="utf-8")
    print(f"[results] wrote {root/'results_table.csv'} and {root/'results_table.md'} ({len(table)} experiments)")


if __name__ == "__main__":
    main()
