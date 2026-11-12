"""Neural Network (Multi-Layer Perceptron) on handwritten Digits.

Run: python models/11_neural_network/model.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils/

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.evaluation import evaluate_classifier

data = load_digits()
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=2000, early_stopping=False, random_state=42)),
])
grid = {
    "mlp__hidden_layer_sizes": [(256,), (256, 128)],
    "mlp__alpha": [1e-4, 1e-2, 1.0],
}

evaluate_classifier("Neural Network (MLP)", data.data, data.target, pipe, grid,
                    Path(__file__).parent, "Digits", data.target_names)
