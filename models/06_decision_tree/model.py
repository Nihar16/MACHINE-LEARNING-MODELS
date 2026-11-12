"""Decision Tree on Breast Cancer Wisconsin, with pruning to limit overfitting.

Run: python models/06_decision_tree/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.evaluation import evaluate_classifier

data = load_breast_cancer()
grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [1, 3, 5, 10],
    "ccp_alpha": [0.0, 0.005, 0.01],
}

out = evaluate_classifier("Decision Tree", data.data, data.target, DecisionTreeClassifier(random_state=42),
                          grid, Path(__file__).parent, "Breast Cancer Wisconsin", data.target_names)

tree = out["model"]
print(f"Tree depth {tree.get_depth()}, {tree.get_n_leaves()} leaves. Rules:")
print(export_text(tree, feature_names=list(data.feature_names), max_depth=3))
