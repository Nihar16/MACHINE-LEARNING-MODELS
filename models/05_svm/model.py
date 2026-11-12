"""Support Vector Machine (RBF kernel) on handwritten Digits.

Run: python models/05_svm/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import load_digits
from sklearn.svm import SVC

from utils.evaluation import evaluate_classifier

data = load_digits()
grid = [
    {"kernel": ["rbf"], "C": [1, 3, 10, 30], "gamma": [0.0005, 0.001, 0.002, "scale"]},
    {"kernel": ["poly"], "C": [0.1, 1], "degree": [2, 3], "gamma": ["scale"]},
]

evaluate_classifier("SVM", data.data, data.target, SVC(), grid,
                    Path(__file__).parent, "Digits", data.target_names)
