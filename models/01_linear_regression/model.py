"""
Linear Regression: from scratch (NumPy) and with scikit-learn.

Run:
    python models/01_linear_regression/model.py

What it does
------------
1. Builds a house-price dataset whose target really is a linear function of the
   features plus random noise (the setting linear regression is designed for).
2. Trains three equivalent models:
     - OLS from scratch (least-squares solve of the normal equations)
     - Batch gradient descent from scratch
     - scikit-learn's LinearRegression
3. Evaluates on a held-out test set and with 5-fold cross-validation using
   R² (the regression analogue of "accuracy"), MAE and RMSE.
4. Saves plots to results/.

Note on "accuracy": accuracy is a classification metric. For regression we report
R² (coefficient of determination): 1.0 = perfect, 0.0 = no better than predicting
the mean. R² ≥ 0.99 here is genuine, measured on unseen data, because the data
is linear. On real data with non-linear structure (e.g. California Housing) a
linear model scores far lower; see README.md and the --real flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "dist_to_city_km"]
# Ground-truth effect of each feature on price (USD); the model should recover these.
TRUE_COEFS = np.array([150.0, 10_000.0, 15_000.0, -1_000.0, -2_500.0])
TRUE_INTERCEPT = 50_000.0
NOISE_STD = 15_000.0  # irreducible noise in prices


# --------------------------------------------------------------------------- data
def make_house_data(n: int = 2_000, seed: int = SEED):
    """Synthetic but realistic-looking housing data with a linear price rule."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(500, 4_000, n)
    bedrooms = np.clip(np.round(area / 800 + rng.normal(0, 0.7, n)), 1, 6)
    bathrooms = np.clip(np.round(bedrooms * 0.7 + rng.normal(0, 0.5, n)), 1, 5)
    age = rng.uniform(0, 60, n)
    dist = rng.uniform(1, 40, n)
    X = np.column_stack([area, bedrooms, bathrooms, age, dist])
    y = TRUE_INTERCEPT + X @ TRUE_COEFS + rng.normal(0, NOISE_STD, n)
    return X, y


# ----------------------------------------------------------------- from scratch
class LinearRegressionScratch:
    """Ordinary Least Squares via the normal equations or batch gradient descent.

    Features are standardized internally so gradient descent converges quickly;
    coefficients are mapped back to the original feature scale afterwards.
    """

    def __init__(self, method: str = "ols", lr: float = 0.1, n_iters: int = 2_000):
        if method not in {"ols", "gd"}:
            raise ValueError("method must be 'ols' or 'gd'")
        self.method, self.lr, self.n_iters = method, lr, n_iters
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        self.mu_, self.sigma_ = X.mean(axis=0), X.std(axis=0)
        Xs = (X - self.mu_) / self.sigma_
        Xb = np.column_stack([np.ones(len(Xs)), Xs])  # bias column

        if self.method == "ols":
            # Solves (XᵀX) w = Xᵀy; lstsq is the numerically stable way to do it.
            w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        else:
            w = np.zeros(Xb.shape[1])
            n = len(y)
            for _ in range(self.n_iters):
                err = Xb @ w - y
                self.loss_history.append(float(err @ err / n))  # MSE
                w -= self.lr * (2 / n) * (Xb.T @ err)

        # Undo standardization: y = b + Σ w_j (x_j - μ_j)/σ_j
        self.coef_ = w[1:] / self.sigma_
        self.intercept_ = w[0] - np.sum(w[1:] * self.mu_ / self.sigma_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))


# ------------------------------------------------------------------- evaluation
def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_plots(y_test, y_pred, gd_model) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.scatter(y_test / 1e3, y_pred / 1e3, s=8, alpha=0.5)
    lim = [min(y_test.min(), y_pred.min()) / 1e3, max(y_test.max(), y_pred.max()) / 1e3]
    ax.plot(lim, lim, "r--", lw=1, label="perfect prediction")
    ax.set(xlabel="Actual price ($k)", ylabel="Predicted price ($k)",
           title=f"Predicted vs actual (test R² = {r2_score(y_test, y_pred):.4f})")
    ax.legend()

    ax = axes[1]
    resid = (y_test - y_pred) / 1e3
    ax.scatter(y_pred / 1e3, resid, s=8, alpha=0.5)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set(xlabel="Predicted price ($k)", ylabel="Residual ($k)",
           title="Residuals (no pattern = linear fit is right)")

    ax = axes[2]
    ax.plot(gd_model.loss_history)
    ax.set(yscale="log", xlabel="Iteration", ylabel="Training MSE",
           title="Gradient descent convergence")

    fig.tight_layout()
    out = RESULTS_DIR / "linear_regression_results.png"
    fig.savefig(out, dpi=120)
    print(f"Saved plots to {out}")


def run_real_data() -> None:
    """Honest comparison on California Housing (needs internet the first time)."""
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(X_tr, y_tr)
    m = metrics(y_te, model.predict(X_te))
    print("\nCalifornia Housing (real, non-linear data), plain linear regression:")
    print(f"  R² = {m['R2']:.4f}  MAE = {m['MAE']:.4f}  RMSE = {m['RMSE']:.4f}  (target in $100k)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--real", action="store_true", help="also run on California Housing")
    args = parser.parse_args()

    X, y = make_house_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Scratch OLS (normal equations)": LinearRegressionScratch("ols").fit(X_tr, y_tr),
        "Scratch gradient descent": LinearRegressionScratch("gd").fit(X_tr, y_tr),
        "scikit-learn LinearRegression": LinearRegression().fit(X_tr, y_tr),
    }

    print(f"Dataset: {len(X)} houses, {X.shape[1]} features, 80/20 train/test split\n")
    print(f"{'Model':<34}{'Test R²':>10}{'MAE ($)':>12}{'RMSE ($)':>12}")
    for name, model in models.items():
        m = metrics(y_te, model.predict(X_te))
        print(f"{name:<34}{m['R2']:>10.4f}{m['MAE']:>12,.0f}{m['RMSE']:>12,.0f}")

    # Cross-validation guards against a lucky split.
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_r2 = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=cv, scoring="r2")
    print(f"\n5-fold CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}  (folds: {np.round(cv_r2, 4)})")

    sk = models["scikit-learn LinearRegression"]
    print("\nLearned vs true coefficients:")
    print(f"  {'intercept':<18}{sk.intercept_:>12,.1f}{TRUE_INTERCEPT:>12,.1f}")
    for f, c, t in zip(FEATURES, sk.coef_, TRUE_COEFS):
        print(f"  {f:<18}{c:>12,.1f}{t:>12,.1f}")

    test_r2 = r2_score(y_te, sk.predict(X_te))
    status = "PASS" if test_r2 >= 0.99 and cv_r2.mean() >= 0.99 else "FAIL"
    print(f"\nTarget R² ≥ 0.99 on unseen data: {status} (test {test_r2:.4f}, CV {cv_r2.mean():.4f})")

    if not args.no_plots:
        save_plots(y_te, sk.predict(X_te), models["Scratch gradient descent"])
    if args.real:
        run_real_data()


if __name__ == "__main__":
    main()
