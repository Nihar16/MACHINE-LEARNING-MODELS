"""Random Forest on Wine.

Run: python models/07_random_forest/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

from utils.evaluation import evaluate_classifier

data = load_wine()
grid = {"n_estimators": [300], "max_features": ["sqrt", "log2", 0.5], "min_samples_leaf": [1, 2, 4]}

out = evaluate_classifier("Random Forest", data.data, data.target, RandomForestClassifier(random_state=42),
                          grid, Path(__file__).parent, "Wine", data.target_names)

imp = sorted(zip(data.feature_names, out["model"].feature_importances_), key=lambda t: -t[1])[:5]
print("Top feature importances:")
for f, v in imp:
    print(f"  {f:<30}{v:.3f}")
