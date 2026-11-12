"""Logistic Regression on Breast Cancer Wisconsin (binary: malignant vs benign).

Run: python models/02_logistic_regression/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.evaluation import evaluate_classifier

data = load_breast_cancer()
pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=10_000))])
grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "clf__penalty": ["l1", "l2"], "clf__solver": ["liblinear"]}

out = evaluate_classifier("Logistic Regression", data.data, data.target, pipe, grid,
                          Path(__file__).parent, "Breast Cancer Wisconsin", data.target_names)

# Interpretability: the largest standardized coefficients drive the prediction.
coefs = out["model"].named_steps["clf"].coef_[0]
top = sorted(zip(data.feature_names, coefs), key=lambda t: -abs(t[1]))[:5]
print("Top features (positive -> benign):")
for f, c in top:
    print(f"  {f:<25}{c:+.3f}")
