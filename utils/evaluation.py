"""Shared evaluation helpers so every classifier is measured the same way.

Protocol (identical for every supervised model):
  1. Stratified 80/20 train/test split with a fixed seed.
  2. Hyperparameters tuned with 5-fold stratified cross-validation on the
     training set only (GridSearchCV). The test set is never seen while tuning.
  3. The tuned model is scored once on the held-out test set.

Both numbers are reported. The CV score is the more reliable estimate on small
datasets; a single test split of ~30-100 samples can swing by a few percent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED = 42
TARGET = 0.99


def evaluate_classifier(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    param_grid: dict,
    out_dir: Path,
    dataset: str,
    class_names=None,
    plot: bool = True,
) -> dict:
    """Tune, test, print a report, and save metrics.json and a confusion matrix."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    t0 = time.perf_counter()
    search = GridSearchCV(estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X_tr, y_tr)
    train_secs = time.perf_counter() - t0

    model = search.best_estimator_
    y_pred = model.predict(X_te)
    binary = len(np.unique(y)) == 2

    result = {
        "model": name,
        "dataset": dataset,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "cv_accuracy": float(search.best_score_),
        "cv_accuracy_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "test_accuracy": float(accuracy_score(y_te, y_pred)),
        "test_f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "test_errors": int((y_pred != y_te).sum()),
        "n_test": int(len(y_te)),
        "tuning_seconds": round(train_secs, 1),
    }
    if binary and hasattr(model, "predict_proba"):
        result["test_roc_auc"] = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    _print_report(result, y_te, y_pred, class_names)
    out_dir = Path(out_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if plot:
        _plot_confusion(confusion_matrix(y_te, y_pred), class_names, name, result, out_dir)
    return {"result": result, "model": model, "split": (X_tr, X_te, y_tr, y_te)}


def _print_report(r, y_te, y_pred, class_names):
    print(f"\n{r['model']} on {r['dataset']} ({r['n_samples']} samples, {r['n_features']} features)")
    print(f"  Best params      : {r['best_params']}")
    print(f"  CV accuracy      : {r['cv_accuracy']:.4f} ± {r['cv_accuracy_std']:.4f}  (5-fold, train set)")
    print(f"  Test accuracy    : {r['test_accuracy']:.4f}  ({r['test_errors']} wrong out of {r['n_test']})")
    print(f"  Test F1 (macro)  : {r['test_f1_macro']:.4f}")
    if "test_roc_auc" in r:
        print(f"  Test ROC-AUC     : {r['test_roc_auc']:.4f}")
    hit = r["cv_accuracy"] >= TARGET and r["test_accuracy"] >= TARGET
    print(f"  ≥99% on both CV and test: {'YES' if hit else 'NO'}")
    names = None if class_names is None else [str(c) for c in class_names]
    print(classification_report(y_te, y_pred, target_names=names, digits=3))


def _plot_confusion(cm, class_names, name, r, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        return
    size = 4 + 0.35 * len(cm)
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\ntest acc {r['test_accuracy']:.3f} · CV acc {r['cv_accuracy']:.3f}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=110)
    plt.close(fig)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return repr(v)
