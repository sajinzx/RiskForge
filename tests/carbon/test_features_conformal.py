import numpy as np
import pandas as pd
import pytest

from greenqueue.carbon.conformal import CQR, SplitConformal, conformal_quantile
from greenqueue.carbon.features import HOUR, build_dataset, feature_columns, horizon_bucket
from greenqueue.eval.metrics import coverage, pinball


def _series(n_days: int = 40, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=24 * n_days, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(
        100 + 30 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 5, len(idx)), idx
    )


def test_dataset_shape_and_alignment() -> None:
    ci = _series()
    df = build_dataset(ci)
    assert set(df["issue_time"].dt.hour) == {0, 12}
    assert df["h"].min() == 1 and df["h"].max() == 48
    # target of horizon h starts (h-1) hours after the issue time and y matches the series
    assert ((df["target_time"] - df["issue_time"]) == (df["h"] - 1) * HOUR).all()
    assert (ci.reindex(df["target_time"]).to_numpy() == df["y"].to_numpy()).all()
    row = df.iloc[0]
    assert row["lag_1"] == ci[row["issue_time"] - HOUR]
    assert row["lag_24"] == ci[row["issue_time"] - 24 * HOUR]


def test_no_leakage_features_ignore_data_from_issue_time_on() -> None:
    """Overwriting all data from an issue time on must not change that issue's features."""
    ci = _series()
    full = build_dataset(ci, require_target=False)
    cols = feature_columns()
    rng = np.random.default_rng(1)
    issues = full["issue_time"].drop_duplicates().to_numpy()
    for t in rng.choice(issues, size=10, replace=False):
        t = pd.Timestamp(t)
        broken = ci.copy()
        broken[broken.index >= t] = 9999.0
        # snaive features for h<=24 legitimately read t-1.. only; broken data starts at t
        other = build_dataset(broken, require_target=False)
        a = full[full["issue_time"] == t][cols].reset_index(drop=True)
        b = other[other["issue_time"] == t][cols].reset_index(drop=True)
        if len(b) == 0:  # issue time fell outside the shortened target range
            continue
        pd.testing.assert_frame_equal(a, b)


def test_horizon_bucket() -> None:
    assert list(horizon_bucket(np.array([1, 6, 7, 12, 13, 24, 25, 48]))) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_conformal_quantile_finite_sample_rule() -> None:
    s = np.arange(1, 10, dtype=float)  # n=9, alpha=0.2 -> k=ceil(10*0.8)=8
    assert conformal_quantile(s, 0.2) == 8.0
    assert conformal_quantile(s, 0.05) == float("inf")


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_split_conformal_coverage_on_iid(alpha: float) -> None:
    rng = np.random.default_rng(0)
    n = 4000
    h = rng.integers(1, 49, 2 * n)
    y = rng.normal(0, 1 + h / 24)  # heteroscedastic in horizon
    point = np.zeros(2 * n)
    m = SplitConformal(alpha).fit(h[:n], point[:n], y[:n])
    lo, hi = m.predict(h[n:], point[n:])
    assert abs(coverage(y[n:], lo, hi) - (1 - alpha)) < 0.02


def test_cqr_corrects_overconfident_quantiles() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    h = rng.integers(1, 49, 2 * n)
    y = rng.normal(0, 3, 2 * n)
    lo, hi = np.full(2 * n, -1.0), np.full(2 * n, 1.0)  # far too narrow
    assert coverage(y[n:], lo[n:], hi[n:]) < 0.4
    m = CQR(0.1).fit(h[:n], lo[:n], hi[:n], y[:n])
    clo, chi = m.predict(h[n:], lo[n:], hi[n:])
    assert abs(coverage(y[n:], clo, chi) - 0.9) < 0.02


def test_pinball_known_value() -> None:
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball(np.array([8.0]), np.array([10.0]), 0.9) == pytest.approx(0.2)
