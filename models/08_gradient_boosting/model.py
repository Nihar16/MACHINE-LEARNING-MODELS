"""Gradient Boosting (XGBoost and LightGBM) on Breast Cancer Wisconsin.

Run: python models/08_gradient_boosting/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import json

from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier

from utils.evaluation import evaluate_classifier

data = load_breast_cancer()
here = Path(__file__).parent
grid = {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4]}

xgb = evaluate_classifier("XGBoost", data.data, data.target,
                          XGBClassifier(subsample=0.8, colsample_bytree=0.8, n_jobs=1, verbosity=0),
                          grid, here / "xgboost", "Breast Cancer Wisconsin", data.target_names)
lgbm = evaluate_classifier("LightGBM", data.data, data.target,
                           LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                          min_child_samples=10, n_jobs=1, verbose=-1),
                           grid, here / "lightgbm", "Breast Cancer Wisconsin", data.target_names)

# Report the stronger of the two (by CV, never by test score) as this folder's headline.
best = max((xgb, lgbm), key=lambda o: o["result"]["cv_accuracy"])["result"]
(here / "results").mkdir(exist_ok=True)
(here / "results" / "metrics.json").write_text(json.dumps(best, indent=2) + "\n")
print(f"Best by CV: {best['model']}")
