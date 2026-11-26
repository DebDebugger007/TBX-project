"""The two machine learning models and how they are evaluated.

Decision Tree
    Asks a series of yes/no questions about the features
    (e.g. "is glcm_contrast > 12.3?") and follows the answers down to a leaf
    that says Normal or TB. Each question is chosen to best separate the two
    classes (lowest Gini impurity). max_depth limits how many questions in a
    row it can ask, which stops it memorising the training images.

Support Vector Machine (SVM)
    Finds the boundary between Normal and TB that leaves the widest possible
    gap (margin) to the nearest images of each class. The RBF kernel lets that
    boundary be curved. C controls how much it is allowed to misclassify
    training images to get a wider margin; gamma controls how curved the
    boundary can be. SVMs measure distances, so the features are first scaled
    to mean 0 and standard deviation 1 (StandardScaler).

Settings (hyperparameters) are chosen with 5-fold cross-validation on the
training data only: the training data is split into 5 parts, the model is
trained on 4 and checked on the 5th, five times, and the setting with the best
average accuracy is kept. The test data is never used for this choice.

Evaluation uses repeated stratified 5-fold cross-validation (see repeated_cv):
the dataset is split into 5 parts, each part is the test set once, and the
whole thing is repeated 5 times with different shuffles - 25 test results in
total. We report their mean and standard deviation. This is much more reliable
than a single train/test split, especially for Montgomery (138 images).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def make_models():
    """Return {name: (model, settings to try)}."""
    tree = DecisionTreeClassifier(random_state=SEED)
    tree_grid = {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 5, 10]}

    svm = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=SEED)),
    ])
    svm_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1]}

    return {"DecisionTree": (tree, tree_grid), "SVM": (svm, svm_grid)}


def tune_and_fit(model, grid, X_train, y_train):
    """Pick the best settings by 5-fold cross-validation, then refit on all training data."""
    search = GridSearchCV(model, grid, cv=5, scoring="accuracy")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate(model, X_test, y_test):
    """Standard classification metrics on the test set. TB (label 1) is the positive class."""
    y_pred = model.predict(X_test)
    # A TB score for each image, used for the ROC curve / AUC (higher = more likely TB).
    # Decision Tree: share of TB training images in the leaf the image lands in.
    # SVM: signed distance from the boundary (positive = TB side).
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity": float(tp / (tp + fn)),   # of the TB patients, how many were found
        "specificity": float(tn / (tn + fp)),   # of the healthy patients, how many were cleared
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }, y_prob


N_FOLDS = 5
N_REPEATS = 5
METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]


def repeated_cv(model, grid, X, y):
    """Repeated stratified 5-fold cross-validation (5 repeats = 25 test folds).

    In every fold the settings are chosen again by tune_and_fit on that fold's
    training part only, so the test part is never seen before it is scored.

    Returns:
      folds      DataFrame, one row of metrics per test fold (25 rows)
      oof_scores TB score for every image from the first repeat, in which each
                 image is in the test part exactly once (used for ROC / confusion plots)
      oof_pred   predicted class for every image from the first repeat
    """
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    oof_scores = np.zeros(len(y))
    oof_pred = np.zeros(len(y), dtype=int)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        best, params, _ = tune_and_fit(model, grid, X[train_idx], y[train_idx])
        metrics, scores = evaluate(best, X[test_idx], y[test_idx])
        rows.append({"repeat": i // N_FOLDS, "fold": i % N_FOLDS, "best_settings": str(params), **metrics})
        if i < N_FOLDS:  # first repeat
            oof_scores[test_idx] = scores
            oof_pred[test_idx] = best.predict(X[test_idx])
    return pd.DataFrame(rows), oof_scores, oof_pred


def summarise_folds(folds):
    """Mean and standard deviation of each metric over the 25 test folds."""
    out = {}
    for m in METRICS:
        out[f"{m}_mean"] = folds[m].mean()
        out[f"{m}_std"] = folds[m].std()
    return out


def mean_std_table(df, index_cols):
    """Format 'mean ± std' columns for printing."""
    table = df[index_cols].copy()
    for m in METRICS:
        table[m] = df[f"{m}_mean"].map("{:.3f}".format) + " ± " + df[f"{m}_std"].map("{:.3f}".format)
    return table


def get_xy(df, feature_names):
    return df[feature_names].to_numpy(), df["label"].to_numpy()


def rounded(d, n=3):
    return {k: (round(v, n) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}
