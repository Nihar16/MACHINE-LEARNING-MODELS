"""K-Means clustering on Iris.

Run: python models/09_kmeans/model.py

K-Means never sees the labels. We pick k with the elbow method and silhouette
score, then, only for evaluation, compare clusters to the true species:
  - Adjusted Rand Index (ARI): 1.0 = clusters match species exactly, 0 = random.
  - Matched accuracy: map each cluster to one species (Hungarian algorithm),
    then count correctly grouped flowers. Comparable to classifier accuracy.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
HERE = Path(__file__).parent


def matched_accuracy(y, labels):
    cm = confusion_matrix(y, labels)
    rows, cols = linear_sum_assignment(-cm)
    return cm[rows, cols].sum() / cm.sum()


data = load_iris()
X, y = data.data, data.target

# 1. Choose k without labels.
ks = range(2, 9)
inertia, silhouette = [], []
for k in ks:
    km = KMeans(k, n_init=20, random_state=SEED).fit(X)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X, km.labels_))
print("k  inertia   silhouette")
for k, i, s in zip(ks, inertia, silhouette):
    print(f"{k}  {i:8.1f}  {s:.3f}")
print("Silhouette peaks at k=2 (setosa vs the rest); the elbow is at k=3, matching the 3 species.\n")

# 2. Fit k=3 on raw and on standardized features and compare to the species.
variants = {"raw features": X, "standardized features": StandardScaler().fit_transform(X)}
rows = {}
for name, Xv in variants.items():
    km = KMeans(3, n_init=20, random_state=SEED).fit(Xv)
    rows[name] = {
        "silhouette": float(silhouette_score(Xv, km.labels_)),
        "ari": float(adjusted_rand_score(y, km.labels_)),
        "matched_accuracy": float(matched_accuracy(y, km.labels_)),
        "labels": km.labels_,
    }
    r = rows[name]
    print(f"k=3, {name:<22} silhouette {r['silhouette']:.3f}  ARI {r['ari']:.3f}  matched accuracy {r['matched_accuracy']:.4f}")

best_name = max(rows, key=lambda n: rows[n]["matched_accuracy"])
best = rows[best_name]
print(f"\nBest: {best_name}. ≥99% matched accuracy: {'YES' if best['matched_accuracy'] >= 0.99 else 'NO'}")
print("Versicolor and virginica overlap in feature space, so an unsupervised method mixes some of them.")

out = HERE / "results"
out.mkdir(exist_ok=True)
(out / "metrics.json").write_text(json.dumps({
    "model": "K-Means", "dataset": "Iris", "n_samples": len(X), "k": 3, "features": best_name,
    "silhouette": best["silhouette"], "adjusted_rand_index": best["ari"],
    "matched_accuracy": best["matched_accuracy"],
}, indent=2) + "\n")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(list(ks), inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
    ax[1].plot(list(ks), silhouette, "o-"); ax[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
    ax[2].scatter(X[:, 2], X[:, 3], c=best["labels"], cmap="viridis", s=25, edgecolor="k", lw=0.3)
    ax[2].set(title=f"K-Means clusters (matched acc {best['matched_accuracy']:.3f})",
              xlabel=data.feature_names[2], ylabel=data.feature_names[3])
    fig.tight_layout(); fig.savefig(out / "kmeans_results.png", dpi=110)
except ImportError:
    pass
