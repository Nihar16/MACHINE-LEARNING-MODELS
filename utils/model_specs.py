"""Registry of every model in the repo, with a search space for each task.

Parameter keys:
  - a bare name (e.g. "C") is a hyperparameter of the final estimator;
  - "estimator" swaps in a different final estimator (e.g. XGBoost vs LightGBM);
  - a key that already contains "__" (e.g. "pca__n_components") is used as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

@dataclass
class TaskSpec:
    make: Callable[[], object]
    params: list[dict] | dict

@dataclass
class ModelSpec:
    key: str
    name: str
    folder: str
    tasks: dict[str, TaskSpec]
    extra_steps: Callable[[], list] = field(default=lambda: [])  # steps between preprocessing and the model
    slow: bool = False  # tune on fewer rows for large datasets

def _xgb_clf():
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=300, n_jobs=1, verbosity=0)

def _xgb_reg():
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=300, n_jobs=1, verbosity=0)

def _lgbm_clf():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, n_jobs=1, verbose=-1)

def _lgbm_reg():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=300, n_jobs=1, verbose=-1)

_BOOST_PARAMS = {
    "n_estimators": [200, 400, 800],
    "learning_rate": loguniform(0.01, 0.3),
    "max_depth": [3, 4, 6, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
_TREE_PARAMS = {"max_depth": [2, 3, 4, 6, 8, 12, 20, None], "min_samples_leaf": [1, 2, 5, 10, 20], "ccp_alpha": [0.0, 0.001, 0.01]}
_FOREST_PARAMS = {
    "n_estimators": [200, 400],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
    "min_samples_leaf": [1, 2, 4],
    "max_depth": [None, 10, 20],
}
_KNN_PARAMS = {"n_neighbors": [1, 3, 5, 7, 9, 15, 25], "weights": ["uniform", "distance"], "p": [1, 2]}
_MLP_PARAMS = {
    "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128)],
    "alpha": loguniform(1e-5, 1.0),
    "learning_rate_init": loguniform(1e-4, 1e-2),
}

SPECS: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec("linear_regression", "Linear Regression (OLS / Ridge / Lasso)", "01_linear_regression", {
        "regression": TaskSpec(LinearRegression, [
            {"estimator": [LinearRegression()]},
            {"estimator": [Ridge()], "alpha": loguniform(1e-3, 1e3)},
            {"estimator": [Lasso(max_iter=20_000)], "alpha": loguniform(1e-4, 10)},
        ]),
    }),
    ModelSpec("logistic_regression", "Logistic Regression", "02_logistic_regression", {
        "classification": TaskSpec(lambda: LogisticRegression(max_iter=5_000),
                                   {"C": loguniform(1e-3, 1e2), "class_weight": [None, "balanced"]}),
    }),
    ModelSpec("knn", "K-Nearest Neighbors", "03_knn", {
        "classification": TaskSpec(KNeighborsClassifier, _KNN_PARAMS),
        "regression": TaskSpec(KNeighborsRegressor, _KNN_PARAMS),
    }, slow=True),
    ModelSpec("naive_bayes", "Gaussian Naive Bayes", "04_naive_bayes", {
        "classification": TaskSpec(GaussianNB, {"var_smoothing": loguniform(1e-11, 1e-2)}),
    }),
    ModelSpec("svm", "Support Vector Machine", "05_svm", {
        "classification": TaskSpec(SVC, [
            {"kernel": ["rbf"], "C": loguniform(1e-2, 1e3), "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1],
             "class_weight": [None, "balanced"]},
            {"kernel": ["poly"], "degree": [2, 3], "C": loguniform(1e-2, 1e2)},
        ]),
        "regression": TaskSpec(SVR, {"C": loguniform(1e-2, 1e3), "epsilon": loguniform(1e-3, 1),
                                     "gamma": ["scale", 1e-3, 1e-2, 1e-1]}),
    }, slow=True),
    ModelSpec("decision_tree", "Decision Tree", "06_decision_tree", {
        "classification": TaskSpec(lambda: DecisionTreeClassifier(random_state=42),
                                   {**_TREE_PARAMS, "criterion": ["gini", "entropy"]}),
        "regression": TaskSpec(lambda: DecisionTreeRegressor(random_state=42), _TREE_PARAMS),
    }),
    ModelSpec("random_forest", "Random Forest", "07_random_forest", {
        "classification": TaskSpec(lambda: RandomForestClassifier(random_state=42, n_jobs=1),
                                   {**_FOREST_PARAMS, "class_weight": [None, "balanced"]}),
        "regression": TaskSpec(lambda: RandomForestRegressor(random_state=42, n_jobs=1), _FOREST_PARAMS),
    }),
    ModelSpec("gradient_boosting", "Gradient Boosting (XGBoost / LightGBM)", "08_gradient_boosting", {
        "classification": TaskSpec(_xgb_clf, [
            {"estimator": [_xgb_clf()], **_BOOST_PARAMS},
            {"estimator": [_lgbm_clf()], **_BOOST_PARAMS, "num_leaves": [15, 31, 63]},
        ]),
        "regression": TaskSpec(_xgb_reg, [
            {"estimator": [_xgb_reg()], **_BOOST_PARAMS},
            {"estimator": [_lgbm_reg()], **_BOOST_PARAMS, "num_leaves": [15, 31, 63]},
        ]),
    }),
    ModelSpec("pca", "PCA + linear model", "10_pca", {
        "classification": TaskSpec(lambda: LogisticRegression(max_iter=5_000),
                                   {"pca__n_components": [0.8, 0.9, 0.95, 0.99], "C": loguniform(1e-3, 1e2)}),
        "regression": TaskSpec(Ridge, {"pca__n_components": [0.8, 0.9, 0.95, 0.99], "alpha": loguniform(1e-3, 1e3)}),
    }, extra_steps=lambda: [("pca", PCA(random_state=42))]),
    ModelSpec("neural_network", "Neural Network (MLP)", "11_neural_network", {
        "classification": TaskSpec(lambda: MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
                                   _MLP_PARAMS),
        "regression": TaskSpec(lambda: MLPRegressor(max_iter=500, early_stopping=True, random_state=42),
                               _MLP_PARAMS),
    }, slow=True),
]}

