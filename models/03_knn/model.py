"""K-Nearest Neighbors on handwritten Digits (8x8 images, 10 classes).

Run: python models/03_knn/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

from utils.evaluation import evaluate_classifier

# Pixels already share one scale (0-16), so no scaler: scaling would give
# near-empty border pixels as much weight as the informative centre pixels.
data = load_digits()
grid = {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"], "p": [1, 2]}

evaluate_classifier("KNN", data.data, data.target, KNeighborsClassifier(), grid,
                    Path(__file__).parent, "Digits", data.target_names)
