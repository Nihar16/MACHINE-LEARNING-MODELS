"""Train any model in this repo on your own tabular data.

Works on any CSV / Excel / Parquet file: numeric, text-category, boolean and
date columns, with missing values. The same engine backs every
models/*/train.py script, automl.py and predict.py.

Python API (e.g. in Colab):
    from utils.generalized import train_model, run_automl
    result = train_model(df, target="price", model="random_forest")
    board = run_automl(df, target="price")

Honesty rules baked in:
  - hyperparameters are tuned with cross-validation on the training split only;
  - the reported score comes from a held-out test split the model never saw;
  - the model is chosen by CV score, never by test score;
  - scores are compared with a dumb baseline, and suspiciously perfect
    single-column predictors are flagged as possible leakage.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.model_specs import SPECS, ModelSpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
TASKS = ("classification", "regression")


# ============================================================== data cleaning
def _is_text(s: pd.Series) -> bool:
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype))


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Learns, on training data only, which columns to drop or convert.

    - drops constant columns and ID-like columns (unique per row);
    - drops free-text columns (almost every value unique);
    - turns date columns into year / month / day / day-of-week numbers;
    - turns booleans into 0/1 and text categories into plain strings.
    """

    def __init__(self, unique_ratio: float = 0.95):
        self.unique_ratio = unique_ratio

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        n = len(X)
        self.columns_ = list(X.columns)
        self.drop_, self.dates_, self.notes_ = [], [], []
        for c in X.columns:
            s = X[c]
            nunique = s.nunique(dropna=True)
            if nunique <= 1:
                self._drop(c, "constant")
            elif pd.api.types.is_datetime64_any_dtype(s):
                self.dates_.append(c)
            elif pd.api.types.is_bool_dtype(s):
                continue
            elif _is_text(s):
                if self._looks_like_date(s):
                    self.dates_.append(c)
                elif n >= 20 and nunique >= self.unique_ratio * s.notna().sum():
                    self._drop(c, "ID or free text (almost every value unique)")
            elif (pd.api.types.is_integer_dtype(s) and n >= 20 and nunique == n
                  and re.search(r"(^|[_\s])id$|^id", str(c), re.I)):
                self._drop(c, "ID column")
        for c in self.dates_:
            self.notes_.append(f"'{c}': date, expanded to year/month/day/day-of-week")
        return self

    def _drop(self, c, why):
        self.drop_.append(c)
        self.notes_.append(f"'{c}': dropped ({why})")

    @staticmethod
    def _looks_like_date(s: pd.Series) -> bool:
        sample = s.dropna().astype(str).head(200)
        if sample.empty or not sample.str.contains(r"\d{1,4}[-/.]\d{1,2}").mean() > 0.9:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.9

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:  # columns missing at prediction time become all-missing
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.columns_].drop(columns=self.drop_)
        for c in self.dates_:
            d = pd.to_datetime(X.pop(c), errors="coerce", format="mixed")
            for part in ("year", "month", "day", "dayofweek"):
                X[f"{c}_{part}"] = getattr(d.dt, part).astype(float)
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_bool_dtype(s):
                X[c] = s.astype(float)
            elif _is_text(s):
                X[c] = s.astype(object).where(s.notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))
            else:
                X[c] = pd.to_numeric(s, errors="coerce").astype(float)
        return X


def build_preprocessor() -> list:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                 max_categories=30, sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    return [("clean", FeatureCleaner()), ("prep", prep)]


# ======================================================= loading and checking
def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")  # sniffs , ; | separators


def detect_task(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y) or _is_text(y):
        return "classification"
    yv = pd.to_numeric(y, errors="coerce").dropna()
    if (yv == yv.round()).all() and yv.nunique() <= 20:
        return "classification"
    return "regression"


def prepare_xy(df: pd.DataFrame, target: str, task: str | None, drop: list[str] | None = None):
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    df = df.drop(columns=[c for c in (drop or []) if c in df.columns])
    notes = []
    n_missing = int(df[target].isna().sum())
    if n_missing:
        df = df[df[target].notna()]
        notes.append(f"dropped {n_missing} rows with no target value")
    y_raw, X = df[target], df.drop(columns=[target])
    task = task or detect_task(y_raw)
    encoder = None
    if task == "classification":
        counts = y_raw.astype(str).value_counts()
        rare = counts[counts < 3].index
        if len(rare):
            keep = ~y_raw.astype(str).isin(rare)
            X, y_raw = X[keep], y_raw[keep]
            notes.append(f"dropped classes with fewer than 3 rows: {list(rare)}")
        encoder = LabelEncoder().fit(y_raw.astype(str))
        y = encoder.transform(y_raw.astype(str))
        if len(encoder.classes_) < 2:
            raise SystemExit("The target needs at least two classes with 3+ rows each.")
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise SystemExit(f"Target '{target}' has non-numeric values; use --task classification.")
    if len(X) < 30:
        raise SystemExit(f"Only {len(X)} usable rows; at least 30 are needed for a train/test split.")
    return X.reset_index(drop=True), y, task, encoder, notes


def leakage_check(X: pd.DataFrame, y, task: str, max_rows: int = 5_000) -> list[str]:
    """Flag columns that predict the target almost perfectly on their own."""
    if len(X) > max_rows:
        idx = np.random.default_rng(SEED).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    Xc = FeatureCleaner().fit_transform(X)
    if task == "classification":
        baseline = np.bincount(y).max() / len(y)
        tree, cv, cut = DecisionTreeClassifier(max_depth=4, random_state=SEED), StratifiedKFold(3, shuffle=True, random_state=SEED), 0.99
    else:
        baseline, tree, cv, cut = 0.0, DecisionTreeRegressor(max_depth=6, random_state=SEED), KFold(3, shuffle=True, random_state=SEED), 0.99
    warnings_ = []
    for c in list(Xc.columns)[:200]:
        col = Xc[[c]]
        if _is_text(col[c]):
            col = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(col.fillna("__missing__"))
        else:
            col = col.fillna(col.median())
        try:
            score = cross_val_score(tree, col, y, cv=cv).mean()
        except ValueError:
            continue
        if score >= cut and score > baseline + 0.01:
            warnings_.append(f"'{c}' alone predicts the target with score {score:.3f}. If this column is "
                             "derived from the target or only known after the outcome, drop it (--drop).")
    return warnings_


# ===================================================================== search
def _make_pipeline(spec: ModelSpec, task: str) -> Pipeline:
    est = spec.tasks[task].make()
    if task == "regression":  # standardize the target too: helps SVR, KNN, MLP, linear models
        est = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
    return Pipeline([*build_preprocessor(), *spec.extra_steps(), ("model", est)])


def _param_space(spec: ModelSpec, task: str):
    base = "model" if task == "classification" else "model__regressor"
    params = spec.tasks[task].params
    out = []
    for grid in params if isinstance(params, list) else [params]:
        g = {}
        for k, v in grid.items():
            if k == "estimator":
                g[base] = v
            elif "__" in k:
                g[k] = v
            else:
                g[f"{base}__{k}"] = v
        out.append(g)
    return out


def _cv(task, y, folds):
    if task == "classification":
        return StratifiedKFold(min(folds, int(np.bincount(y).min())), shuffle=True, random_state=SEED)
    return KFold(folds, shuffle=True, random_state=SEED)


def tune(spec: ModelSpec, X_tr, y_tr, task: str, n_iter: int, folds: int, tune_rows: int, scoring: str):
    """Randomized search with CV on the training data; returns the search and the refit model."""
    limit = tune_rows // 4 if spec.slow else tune_rows
    Xs, ys = X_tr, y_tr
    if len(X_tr) > limit:  # tune on a subsample, then refit the winner on all training rows
        Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=limit, random_state=SEED,
                                        stratify=y_tr if task == "classification" else None)
    search = RandomizedSearchCV(_make_pipeline(spec, task), _param_space(spec, task), n_iter=n_iter,
                                scoring=scoring, cv=_cv(task, ys, folds), n_jobs=-1, random_state=SEED,
                                error_score=np.nan)
    search.fit(Xs, ys)
    model = search.best_estimator_
    if len(Xs) < len(X_tr):
        model = clone(model).fit(X_tr, y_tr)
    return search, model


def evaluate(model, X_te, y_te, task: str) -> dict:
    pred = model.predict(X_te)
    if task == "regression":
        return {"r2": r2_score(y_te, pred), "mae": mean_absolute_error(y_te, pred),
                "rmse": float(np.sqrt(mean_squared_error(y_te, pred)))}
    m = {"accuracy": accuracy_score(y_te, pred), "balanced_accuracy": balanced_accuracy_score(y_te, pred),
         "f1_macro": f1_score(y_te, pred, average="macro")}
    if len(np.unique(y_te)) == 2:
        try:
            score = (model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba")
                     else model.decision_function(X_te))
            m["roc_auc"] = roc_auc_score(y_te, score)
        except (AttributeError, ValueError):
            pass
    return {k: float(v) for k, v in m.items()}


def _baseline(y_tr, y_te, task):
    if task == "classification":
        majority = np.bincount(y_tr).argmax()
        return {"accuracy": float((y_te == majority).mean())}
    return {"r2": float(r2_score(y_te, np.full(len(y_te), np.mean(y_tr))))}


def _best_params(search) -> dict:
    def clean(v):
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float):
            return float(f"{v:.4g}")
        if hasattr(v, "fit"):
            return type(v).__name__
        return v if isinstance(v, (int, str, bool, type(None))) else str(v)

    return {k.replace("model__regressor__", "").replace("model__", ""): clean(v)
            for k, v in search.best_params_.items()}


# ================================================================= public API
def train_model(df: pd.DataFrame, target: str, model: str, task: str | None = None, *, drop=None,
                test_size: float = 0.2, n_iter: int = 20, folds: int = 5, tune_rows: int = 20_000,
                metric: str | None = None, target_score: float = 0.99, out_dir: str | Path | None = None,
                verbose: bool = True) -> dict:
    """Tune `model` (a key of utils.model_specs.SPECS) on df and evaluate it on a held-out split."""
    return run_automl(df, target, models=[model], task=task, drop=drop, test_size=test_size, n_iter=n_iter,
                      folds=folds, tune_rows=tune_rows, metric=metric, target_score=target_score,
                      out_dir=out_dir, verbose=verbose)


def run_automl(df: pd.DataFrame, target: str, models: list[str] | None = None, task: str | None = None, *,
               drop=None, test_size: float = 0.2, n_iter: int = 20, folds: int = 5, tune_rows: int = 20_000,
               metric: str | None = None, target_score: float = 0.99, out_dir: str | Path | None = None,
               verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    X, y, task, encoder, notes = prepare_xy(df, target, task, drop)
    keys = models or [k for k, s in SPECS.items() if task in s.tasks]
    for k in keys:
        if k not in SPECS:
            raise SystemExit(f"Unknown model '{k}'. Choose from: {', '.join(SPECS)}")
        if task not in SPECS[k].tasks:
            raise SystemExit(f"{SPECS[k].name} does not support {task}. Try a model that supports it, "
                             f"or override detection with --task.")
    scoring = metric or ("accuracy" if task == "classification" else "r2")
    headline = "accuracy" if task == "classification" else "r2"

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=SEED,
                                              stratify=y if task == "classification" else None)
    cleaner = FeatureCleaner().fit(X_tr)
    say(f"\nData: {len(X)} rows, {X.shape[1]} feature columns, target '{target}' -> {task}"
        + (f" ({len(encoder.classes_)} classes)" if encoder is not None else ""))
    for n in notes + cleaner.notes_:
        say(f"  - {n}")
    say(f"Split: {len(X_tr)} train / {len(X_te)} test. Tuning: {n_iter} settings x {folds}-fold CV per model.")

    leaks = leakage_check(X_tr, y_tr, task)
    dup = int(X_te.merge(X_tr.drop_duplicates(), how="inner").shape[0]) if X.shape[1] else 0
    data_warnings = list(leaks)
    if dup > 0.05 * len(X_te):
        data_warnings.append(f"{dup} of {len(X_te)} test rows also appear in the training data; "
                             "the test score may be optimistic.")
    baseline = _baseline(y_tr, y_te, task)
    if task == "classification" and baseline["accuracy"] >= 0.9:
        data_warnings.append(f"Classes are imbalanced: always predicting the most common class already scores "
                             f"{baseline['accuracy']:.1%} accuracy. Judge the model by balanced accuracy / F1.")

    board = []
    for k in keys:
        spec = SPECS[k]
        t0 = time.perf_counter()
        say(f"  tuning {spec.name} ...", end="", flush=True)
        try:
            search, model = tune(spec, X_tr, y_tr, task, n_iter, folds, tune_rows, scoring)
        except Exception as e:  # one failing model shouldn't stop the comparison
            say(f" failed: {e}")
            continue
        test = evaluate(model, X_te, y_te, task)
        row = {"key": k, "model": spec.name, "cv_score": float(search.best_score_),
               "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]), **{f"test_{m}": v for m, v in test.items()},
               "best_params": _best_params(search), "seconds": round(time.perf_counter() - t0, 1), "_model": model}
        board.append(row)
        say(f" CV {scoring} {row['cv_score']:.4f}, test {headline} {test[headline]:.4f} ({row['seconds']}s)")
    if not board:
        raise SystemExit("Every model failed; see the errors above.")

    board.sort(key=lambda r: -r["cv_score"])
    best = board[0]
    base_score = list(baseline.values())[0]
    if best[f"test_{headline}"] <= base_score + 0.01:
        data_warnings.append(f"The best model ({best[f'test_{headline}']:.4f}) does not clearly beat the baseline "
                             f"({base_score:.4f}): the features carry little signal about '{target}'.")
    reached = best["cv_score"] >= target_score and best[f"test_{headline}"] >= target_score
    report = {
        "target": target, "task": task, "metric": scoring, "n_rows": len(X), "n_train": len(X_tr), "n_test": len(X_te),
        "classes": list(map(str, encoder.classes_)) if encoder is not None else None,
        "baseline_test": baseline, "data_notes": notes + cleaner.notes_, "warnings": data_warnings,
        "best_model": best["key"], "target_score": target_score, "target_reached": bool(reached),
        "leaderboard": [{k: v for k, v in r.items() if k != "_model"} for r in board],
    }
    _print_summary(report, best, headline, say)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": best["_model"], "label_encoder": encoder, "task": task, "target": target,
                     "feature_columns": list(X.columns), "model": best["key"], "report": report},
                    out / "model.joblib")
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        pd.DataFrame(report["leaderboard"]).drop(columns="best_params").to_csv(out / "leaderboard.csv", index=False)
        say(f"\nSaved {out / 'model.joblib'}, report.json and leaderboard.csv")
        say(f"Predict on new rows:  python predict.py --model {out / 'model.joblib'} --data new_rows.csv")
    report["model_object"] = best["_model"]
    report["label_encoder"] = encoder
    return report


def _print_summary(report, best, headline, say):
    task = report["task"]
    say("\nLeaderboard (ranked by CV score; the test score is the honest estimate on unseen data):")
    extra = "test_balanced_accuracy" if task == "classification" else "test_mae"
    say(f"  {'model':<42}{'CV':>16}{'test ' + headline:>16}{extra.replace('test_', ''):>20}")
    for r in report["leaderboard"]:
        say(f"  {r['model']:<42}{r['cv_score']:>10.4f} ±{r['cv_std']:.3f}{r['test_' + headline]:>16.4f}{r[extra]:>20.4f}")
    say(f"\nBest: {best['model']}  params {best['best_params']}")
    b = report["baseline_test"]
    say(f"Baseline ({'always predict the most common class' if task == 'classification' else 'always predict the mean'}): "
        f"{headline} {list(b.values())[0]:.4f}")
    for w in report["warnings"]:
        say(f"WARNING: {w}")
    goal = report["target_score"]
    if report["target_reached"]:
        say(f"Target {goal:.0%} {headline}: REACHED on both CV and the held-out test set.")
        if report["warnings"]:
            say("  ...but check the warnings above before trusting it.")
    else:
        say(f"Target {goal:.0%} {headline}: NOT reached (best test {best['test_' + headline]:.4f}). The ceiling is set "
            "by how much signal the features carry about the target. More informative features or more rows "
            "help; more tuning rarely closes a large gap.")


def predict_file(model_path, data_path, out_path=None) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    df = load_table(data_path)
    X = df.drop(columns=[bundle["target"]], errors="ignore")
    missing = [c for c in bundle["feature_columns"] if c not in X.columns]
    if missing:
        print(f"Note: {len(missing)} training columns missing from the new data, treated as empty: {missing[:10]}")
    pipe = bundle["pipeline"]
    pred = pipe.predict(X)
    out = df.copy()
    enc = bundle["label_encoder"]
    col = f"predicted_{bundle['target']}"
    if enc is not None:
        out[col] = enc.inverse_transform(np.asarray(pred, dtype=int))
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X)
                for i, cls in enumerate(enc.classes_):
                    out[f"prob_{cls}"] = proba[:, i]
            except AttributeError:
                pass
    else:
        out[col] = pred
    if out_path:
        out.to_csv(out_path, index=False)
        print(f"Wrote {len(out)} predictions to {out_path}")
    return out


# ======================================================================== CLIs
def _common_args(p: argparse.ArgumentParser):
    p.add_argument("--data", required=True, help="CSV / Excel / Parquet file")
    p.add_argument("--target", required=True, help="column to predict")
    p.add_argument("--task", choices=TASKS, help="override automatic task detection")
    p.add_argument("--drop", nargs="*", default=[], help="columns to ignore (e.g. leaky ones)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-iter", type=int, default=20, help="hyperparameter settings tried per model")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--tune-rows", type=int, default=20_000, help="tune on at most this many rows")
    p.add_argument("--metric", help="sklearn scoring name, e.g. balanced_accuracy, f1_macro, roc_auc, neg_mean_absolute_error")
    p.add_argument("--target-score", type=float, default=0.99)
    p.add_argument("--out", help="output folder (default: ml_output/<model>)")


def _kwargs(a):
    return dict(task=a.task, drop=a.drop, test_size=a.test_size, n_iter=a.n_iter, folds=a.folds,
                tune_rows=a.tune_rows, metric=a.metric, target_score=a.target_score)


def train_cli(key: str, argv=None):
    spec = SPECS[key]
    p = argparse.ArgumentParser(description=f"Train {spec.name} on your own data "
                                f"({', '.join(spec.tasks)}).")
    _common_args(p)
    a = p.parse_args(argv)
    return train_model(load_table(a.data), a.target, key, out_dir=a.out or f"ml_output/{key}", **_kwargs(a))


def automl_cli(argv=None):
    p = argparse.ArgumentParser(description="Try every model in this repo on your data and keep the best.")
    _common_args(p)
    p.add_argument("--models", nargs="*", help=f"subset to try: {' '.join(SPECS)}")
    a = p.parse_args(argv)
    return run_automl(load_table(a.data), a.target, models=a.models, out_dir=a.out or "ml_output/best", **_kwargs(a))


def predict_cli(argv=None):
    p = argparse.ArgumentParser(description="Predict with a model saved by train.py or automl.py.")
    p.add_argument("--model", required=True, help="path to model.joblib")
    p.add_argument("--data", required=True, help="CSV / Excel / Parquet with the same feature columns")
    p.add_argument("--out", default="predictions.csv")
    a = p.parse_args(argv)
    predict_file(a.model, a.data, a.out)


# ================================================================== clustering
def run_kmeans(df: pd.DataFrame, k: int | None = None, k_max: int = 10, target: str | None = None, *,
               drop=None, out_dir: str | Path | None = None, verbose: bool = True) -> dict:
    """Cluster any table with K-Means. If k is None, pick it by silhouette score.

    `target` is optional and used only to score the clusters afterwards (ARI and
    matched accuracy); K-Means itself never sees it.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score

    say = print if verbose else (lambda *a, **k: None)
    df = df.drop(columns=[c for c in (drop or []) if c in df.columns])
    y = df.pop(target) if target else None
    prep = Pipeline(build_preprocessor())
    Z = prep.fit_transform(df)
    sample = np.random.default_rng(SEED).choice(len(Z), min(len(Z), 5_000), replace=False)
    say(f"\nData: {len(df)} rows, {df.shape[1]} columns -> {Z.shape[1]} numeric features after cleaning")
    for n in prep.named_steps["clean"].notes_:
        say(f"  - {n}")

    scores = {}
    if k is None:
        say("k  silhouette (higher = better separated clusters)")
        for kk in range(2, min(k_max, len(Z) - 1) + 1):
            labels = KMeans(kk, n_init=10, random_state=SEED).fit_predict(Z)
            scores[kk] = float(silhouette_score(Z[sample], labels[sample]))
            say(f"{kk:<3}{scores[kk]:.3f}")
        k = max(scores, key=scores.get)
        say(f"Chosen k = {k} (highest silhouette). Pass --k to force a different number.")
    model = Pipeline([*build_preprocessor(), ("model", KMeans(k, n_init=20, random_state=SEED))]).fit(df)
    labels = model.predict(df)
    report = {"task": "clustering", "k": int(k), "silhouette_by_k": scores,
              "silhouette": float(silhouette_score(Z[sample], labels[sample])),
              "cluster_sizes": {int(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}}
    say(f"k={k}: silhouette {report['silhouette']:.3f}, cluster sizes {report['cluster_sizes']}")
    if y is not None:
        cm = confusion_matrix(pd.factorize(y.astype(str))[0], labels)
        r, c = linear_sum_assignment(-cm)
        report["adjusted_rand_index"] = float(adjusted_rand_score(y.astype(str), labels))
        report["matched_accuracy"] = float(cm[r, c].sum() / cm.sum())
        say(f"Against '{target}': ARI {report['adjusted_rand_index']:.3f}, matched accuracy "
            f"{report['matched_accuracy']:.4f} (clusters only match labels when the labels form separate groups)")
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": model, "label_encoder": None, "task": "clustering", "target": "cluster",
                     "feature_columns": list(df.columns), "model": "kmeans", "report": report}, out / "model.joblib")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        assigned = df.assign(cluster=labels)
        if y is not None:
            assigned[target] = y.values
        assigned.to_csv(out / "clusters.csv", index=False)
        say(f"Saved {out / 'model.joblib'}, report.json and clusters.csv")
    report["model_object"] = model
    report["labels"] = labels
    return report


def kmeans_cli(argv=None):
    p = argparse.ArgumentParser(description="Cluster your own data with K-Means.")
    p.add_argument("--data", required=True)
    p.add_argument("--k", type=int, help="number of clusters (default: chosen by silhouette)")
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--target", help="optional label column, used only to score the clusters")
    p.add_argument("--drop", nargs="*", default=[])
    p.add_argument("--out", default="ml_output/kmeans")
    a = p.parse_args(argv)
    return run_kmeans(load_table(a.data), a.k, a.k_max, a.target, drop=a.drop, out_dir=a.out)
