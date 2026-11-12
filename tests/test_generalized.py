"""Tests for the generalized training engine. Run: python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, make_blobs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.generalized import FeatureCleaner, detect_task, predict_file, run_automl, run_kmeans, train_model  # noqa: E402
from utils.model_specs import SPECS  # noqa: E402

FAST = dict(n_iter=2, folds=3, verbose=False)


@pytest.fixture(scope="module")
def clf_df():
    d = load_breast_cancer(as_frame=True).frame.sample(300, random_state=0)
    return d.assign(target=d.target.map({0: "malignant", 1: "benign"}))


@pytest.fixture(scope="module")
def reg_df():
    return load_diabetes(as_frame=True).frame


def messy_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n),
        "joined": pd.date_range("2021-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "x": rng.normal(size=n),
        "color": rng.choice(["red", "green", "blue"], n),
        "flag": rng.random(n) < 0.5,
        "constant": 1,
    })
    df["label"] = np.where(df.x + (df.color == "red") * 1.5 + rng.normal(0, 0.3, n) > 0.5, "A", "B")
    df.loc[::10, "x"] = np.nan
    df.loc[::7, "color"] = np.nan
    return df


@pytest.mark.parametrize("key", [k for k, s in SPECS.items() if "classification" in s.tasks])
def test_every_classifier_trains(key, clf_df):
    r = train_model(clf_df, "target", key, **FAST)
    assert r["task"] == "classification"
    assert r["leaderboard"][0]["test_accuracy"] > 0.85


@pytest.mark.parametrize("key", [k for k, s in SPECS.items() if "regression" in s.tasks])
def test_every_regressor_trains(key, reg_df):
    r = train_model(reg_df, "target", key, **FAST)
    assert r["task"] == "regression"
    assert r["leaderboard"][0]["test_r2"] > 0.1


def test_detect_task():
    assert detect_task(pd.Series(["a", "b"] * 10)) == "classification"
    assert detect_task(pd.Series([0, 1, 2] * 10)) == "classification"
    assert detect_task(pd.Series(np.linspace(0, 1, 30))) == "regression"


def test_cleaner_handles_ids_dates_constants_bools():
    X = messy_df().drop(columns="label")
    out = FeatureCleaner().fit(X).transform(X)
    assert "id" not in out and "constant" not in out and "joined" not in out
    assert {"joined_year", "joined_month", "joined_day", "joined_dayofweek"} <= set(out.columns)
    assert out["flag"].isin([0.0, 1.0]).all()


def test_messy_data_train_save_and_predict(tmp_path):
    df = messy_df()
    r = train_model(df, "label", "random_forest", out_dir=tmp_path, **FAST)
    assert r["leaderboard"][0]["test_accuracy"] > 0.8
    new = df.drop(columns=["label", "flag"]).head(4).copy()  # a column is missing at prediction time
    new.loc[0, "color"] = "purple"  # unseen category
    new.to_csv(tmp_path / "new.csv", index=False)
    preds = predict_file(tmp_path / "model.joblib", tmp_path / "new.csv")
    assert set(preds["predicted_label"]) <= {"A", "B"}
    assert np.allclose(preds[["prob_A", "prob_B"]].sum(axis=1), 1)


def test_leakage_is_flagged():
    df = messy_df()
    df["leak"] = (df.label == "A").astype(int)
    r = run_automl(df, "label", models=["logistic_regression"], **FAST)
    assert any("'leak'" in w for w in r["warnings"])


def test_no_signal_is_reported_honestly():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "y": rng.choice(["p", "q"], 300)})
    r = run_automl(df, "y", models=["logistic_regression"], **FAST)
    assert not r["target_reached"]
    assert any("does not clearly beat the baseline" in w for w in r["warnings"])


def test_kmeans_picks_k():
    X, y = make_blobs(400, centers=4, cluster_std=0.6, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b"]).assign(label=y)
    r = run_kmeans(df, target="label", verbose=False)
    assert r["k"] == 4 and r["matched_accuracy"] > 0.98


def test_wrong_task_is_rejected(reg_df):
    with pytest.raises(SystemExit):
        train_model(reg_df, "target", "logistic_regression", **FAST)
