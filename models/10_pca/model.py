"""Principal Component Analysis on handwritten Digits (64 pixel features).

Run: python models/10_pca/model.py

PCA is not a classifier, so it has no accuracy of its own. We measure:
  - how many components keep 90 / 95 / 99% of the variance, and
  - how accurate a classifier (SVM) is on the reduced data vs all 64 features,
    using 5-fold CV with PCA fitted inside each fold (no leakage).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

SEED = 42
HERE = Path(__file__).parent

data = load_digits()
X, y = data.data, data.target

pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
needed = {t: int(np.searchsorted(cum, t) + 1) for t in (0.90, 0.95, 0.99)}
for t, n in needed.items():
    print(f"{n:>2} of 64 components keep {t:.0%} of the variance")

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
svm = lambda: SVC(kernel="rbf", C=10, gamma="scale")
print("\nSVM accuracy (5-fold CV):")
results = {}
for label, n in [("all 64 features", None), *[(f"PCA {n} comp ({t:.0%} var)", n) for t, n in needed.items()]]:
    model = svm() if n is None else make_pipeline(PCA(n, random_state=SEED), svm())
    s = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    results[label] = {"components": n or 64, "cv_accuracy": float(s.mean()), "cv_std": float(s.std())}
    print(f"  {label:<26} {s.mean():.4f} ± {s.std():.4f}")

out = HERE / "results"
out.mkdir(exist_ok=True)
(out / "metrics.json").write_text(json.dumps({
    "model": "PCA", "dataset": "Digits", "n_samples": len(X), "n_features": X.shape[1],
    "components_for_variance": {f"{t:.0%}": n for t, n in needed.items()},
    "svm_accuracy": results,
}, indent=2) + "\n")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(range(1, 65), cum, "o-", ms=3)
    for t, n in needed.items():
        ax[0].axhline(t, ls="--", lw=0.8, color="gray"); ax[0].annotate(f"{t:.0%}: {n} comps", (n, t), (n + 2, t - 0.06))
    ax[0].set(title="Cumulative explained variance", xlabel="Components", ylabel="Variance kept")
    Z = PCA(2).fit_transform(X)
    sc = ax[1].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8)
    ax[1].legend(*sc.legend_elements(), title="digit", fontsize=7, loc="best")
    ax[1].set(title="Digits projected onto 2 principal components", xlabel="PC1", ylabel="PC2")
    fig.tight_layout(); fig.savefig(out / "pca_results.png", dpi=110)
except ImportError:
    pass
