"""Gaussian Naive Bayes on Wine (13 chemical measurements, 3 cultivars).

Run: python models/04_naive_bayes/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from utils.evaluation import evaluate_classifier

# GaussianNB assumes each feature is normally distributed within a class.
# A Yeo-Johnson power transform makes skewed features more Gaussian; the grid
# lets cross-validation decide whether that helps.
data = load_wine()
pipe = Pipeline([("transform", PowerTransformer()), ("clf", GaussianNB())])
grid = {"transform": [PowerTransformer(), "passthrough"], "clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}

evaluate_classifier("Gaussian Naive Bayes", data.data, data.target, pipe, grid,
                    Path(__file__).parent, "Wine", data.target_names)
