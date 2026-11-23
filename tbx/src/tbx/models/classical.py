"""Classical lung-field baseline: LR / RBF-SVM on texture features.

Hyperparameters are chosen by an inner cross-validation inside each training
fold only, so no test-fold information leaks into model selection. The estimator
is wrapped in a standardiser; the whole pipeline is refit per fold.

The baseline is expected to score below the CNN within a dataset. Its value is in
the cross-dataset test: because it only ever sees the lungs, a *smaller*
cross-dataset drop for this model than for the CNN is itself evidence that the
CNN relies on something outside the lungs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import ExperimentConfig
from ..data.splits import fold_assignment, load_folds
from ..evaluation.metrics import summarise, threshold_from_training


def _make_estimator(kind: str):
    if kind == "logreg":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    elif kind == "svm":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))])
        grid = {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]}
    else:
        raise ValueError(f"unknown classical estimator {kind!r}")
    return pipe, grid


def _predict_scores(est, X) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def train_classical_cv(
    features_df: pd.DataFrame,
    cfg: ExperimentConfig,
    dataset: str,
) -> dict:
    """Within-set 5-fold CV for the classical baseline on one dataset.

    ``features_df`` must have the feature columns, plus image_id, label and
    dataset. Returns pooled out-of-fold scores and per-fold point metrics; the
    caller attaches bootstrap CIs on the pooled scores.
    """
    folds = load_folds(cfg.paths.splits_dir, dataset)
    assign = fold_assignment(folds)

    sub = features_df[features_df["dataset"] == dataset].copy()
    sub["fold"] = sub["image_id"].map(assign)
    sub = sub.dropna(subset=["fold"])
    sub["fold"] = sub["fold"].astype(int)

    feat_cols = [c for c in sub.columns if c not in {"image_id", "label", "dataset", "fold"}]
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.data.seed)

    oof_scores = np.full(len(sub), np.nan)
    oof_true = sub["label"].to_numpy()
    sub = sub.reset_index(drop=True)
    per_fold = []

    for fold_idx in range(cfg.data.n_folds):
        te = sub["fold"] == fold_idx
        tr = ~te
        if te.sum() == 0:
            continue
        Xtr, ytr = sub.loc[tr, feat_cols].to_numpy(), sub.loc[tr, "label"].to_numpy()
        Xte, yte = sub.loc[te, feat_cols].to_numpy(), sub.loc[te, "label"].to_numpy()

        pipe, grid = _make_estimator(cfg.model_cfg.classical_estimator)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=1)  # in-process: worker pools crash loading torch DLLs on Windows/Jupyter
        search.fit(Xtr, ytr)
        best = search.best_estimator_

        train_scores = _predict_scores(best, Xtr)
        thr = threshold_from_training(ytr, train_scores, cfg.eval.screening_sensitivity)
        test_scores = _predict_scores(best, Xte)
        oof_scores[np.where(te)[0]] = test_scores

        m = summarise(yte, test_scores, thr, cfg.eval.screening_sensitivity)
        m["fold"] = fold_idx
        m["best_params"] = search.best_params_
        per_fold.append(m)
        print(f"[classical:{dataset}] fold {fold_idx} AUC {m['auc']:.3f}")

    return {
        "dataset": dataset,
        "model": f"classical_{cfg.model_cfg.classical_estimator}",
        "oof_true": oof_true.tolist(),
        "oof_scores": oof_scores.tolist(),
        "image_ids": sub["image_id"].tolist(),
        "per_fold": per_fold,
    }


def train_classical_cross(features_df: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """Cross-dataset run for the classical baseline: fit on all of ``cfg.train``,
    test once on all of ``cfg.test``.

    Hyperparameters are picked by inner CV on the training set only, and the
    operating-point threshold is fixed on the training scores, exactly as in the
    within-set runs, so the cross drop is comparable with the CNN's.
    """
    feat_cols = [c for c in features_df.columns if c not in {"image_id", "label", "dataset", "fold"}]
    tr = features_df[features_df["dataset"] == cfg.train].reset_index(drop=True)
    te = features_df[features_df["dataset"] == cfg.test].reset_index(drop=True)
    if tr.empty or te.empty:
        raise ValueError(
            f"features table has {len(tr)} {cfg.train} rows and {len(te)} {cfg.test} rows; "
            "run 03 (masks) and 04 (features) so both sets are present"
        )

    Xtr, ytr = tr[feat_cols].to_numpy(), tr["label"].to_numpy()
    Xte, yte = te[feat_cols].to_numpy(), te["label"].to_numpy()

    pipe, grid = _make_estimator(cfg.model_cfg.classical_estimator)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.data.seed)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=1)  # in-process: worker pools crash loading torch DLLs on Windows/Jupyter
    search.fit(Xtr, ytr)
    best = search.best_estimator_

    thr = threshold_from_training(ytr, _predict_scores(best, Xtr), cfg.eval.screening_sensitivity)
    te_scores = _predict_scores(best, Xte)
    point = summarise(yte, te_scores, thr, cfg.eval.screening_sensitivity)
    point["best_params"] = search.best_params_
    print(f"[classical:{cfg.train}->{cfg.test}] AUC {point['auc']:.3f}")

    return {
        "train": cfg.train,
        "test": cfg.test,
        "model": f"classical_{cfg.model_cfg.classical_estimator}",
        "oof_true": yte.tolist(),
        "oof_scores": te_scores.tolist(),
        "image_ids": te["image_id"].tolist(),
        "point": point,
    }
