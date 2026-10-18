import numpy as np
import pandas as pd
import pytest

from greenqueue.carbon.features import build_dataset
from greenqueue.carbon.forecasters import (
    LGBQuantileForecaster,
    RidgeForecaster,
    persistence,
    seasonal_naive,
)
from greenqueue.carbon.tuning import (
    assert_not_test,
    rolling_origin_folds,
    tune_lightgbm,
    tune_ridge,
)
from greenqueue.eval.metrics import mae, pinball


def _data(n_days: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=24 * n_days, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    t = np.arange(len(idx))
    ci = (
        150
        + 60 * np.sin(2 * np.pi * t / 24)
        + 20 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0, 8, len(idx))
    )
    return build_dataset(pd.Series(ci, idx))


def test_baselines_read_the_right_columns() -> None:
    df = _data(30)
    assert (persistence(df) == df["lag_1"].to_numpy()).all()
    assert (seasonal_naive(df, 168) == df["snaive168"].to_numpy()).all()


def test_seasonal_naive_beats_persistence_on_periodic_signal() -> None:
    df = _data(60)
    far = df[df["h"] >= 12]
    assert mae(far["y"], seasonal_naive(far, 24)) < mae(far["y"], persistence(far))


def test_ridge_and_lgb_beat_persistence_and_quantiles_are_ordered() -> None:
    df = _data()
    cut = df["issue_time"].quantile(0.8)
    tr, te = df[df["issue_time"] < cut], df[df["issue_time"] >= cut + pd.Timedelta(days=2)]
    ridge = RidgeForecaster(1.0).fit(tr)
    lgbm = LGBQuantileForecaster({"n_estimators": 60}).fit(tr)
    q = lgbm.predict(te)
    base = mae(te["y"], persistence(te))
    assert mae(te["y"], ridge.predict(te)) < base
    assert mae(te["y"], q[0.5]) < base
    assert (q[0.05] <= q[0.5]).all() and (q[0.5] <= q[0.95]).all()
    # a quantile model is better at its own level than the median is at that level
    assert pinball(te["y"], q[0.95], 0.95) < pinball(te["y"], q[0.5], 0.95)


def test_folds_are_chronological_and_purged() -> None:
    df = _data()
    folds = rolling_origin_folds(df, n_folds=3, val_months=2)
    assert len(folds) == 3
    prev_val_end = None
    for fit, val in folds:
        assert df["issue_time"].iloc[fit].max() < df["issue_time"].iloc[val].min() - pd.Timedelta(
            days=1
        )
        if prev_val_end is not None:
            assert df["issue_time"].iloc[val].max() > prev_val_end
        prev_val_end = df["issue_time"].iloc[val].max()


def test_assert_not_test_guard() -> None:
    df = _data(60)
    with pytest.raises(AssertionError):
        assert_not_test(df, df["issue_time"].min() + pd.Timedelta(days=5))
    assert_not_test(df, df["issue_time"].max() + pd.Timedelta(days=1))


def test_tuning_smoke() -> None:
    df = _data(200)
    ts = df["issue_time"].max() + pd.Timedelta(days=1)
    assert tune_ridge(df, ts) in set(np.logspace(-3, 3, 13))
    best = tune_lightgbm(df, ts, n_trials=2, max_estimators=40)
    assert {"num_leaves", "learning_rate", "n_estimators"} <= set(best)
