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


def _shifted_stream(n_issues: int = 600, seed: int = 0):  # type: ignore[no-untyped-def]
    """Hourly-issue stream whose noise doubles halfway (a regime shift)."""
    from greenqueue.carbon.features import HOUR

    rng = np.random.default_rng(seed)
    issues = pd.date_range("2024-01-01", periods=n_issues, freq="12h", tz="UTC")
    rows = []
    for k, t in enumerate(issues):
        sd = 1.0 if k < n_issues // 2 else 2.5
        for hh in range(1, 49):
            rows.append((t, hh, t + (hh - 1) * HOUR, rng.normal(0, sd)))
    df = pd.DataFrame(rows, columns=["issue_time", "h", "target_time", "y"])
    df["lo"], df["hi"] = -1.645, 1.645  # correct for sd=1 only
    return df


def test_aci_restores_coverage_after_a_shift_where_static_cqr_fails() -> None:
    from greenqueue.carbon.conformal import aci_intervals

    df = _shifted_stream()
    first = df[df["issue_time"] < df["issue_time"].iloc[len(df) // 2]]
    cal = {
        int(b): np.maximum(g["lo"] - g["y"], g["y"] - g["hi"]).to_numpy()
        for b, g in first.groupby(horizon_bucket(first["h"].to_numpy()))
    }
    second = df[df["issue_time"] >= df["issue_time"].iloc[len(df) // 2]].reset_index(drop=True)
    y = second["y"].to_numpy()
    static = float(np.mean((y >= second["lo"]) & (y <= second["hi"])))
    lo, hi, used = aci_intervals(
        second["issue_time"].to_numpy(),
        second["h"].to_numpy(),
        second["target_time"].to_numpy(),
        y,
        second["lo"].to_numpy(),
        second["hi"].to_numpy(),
        cal,
        alpha=0.1,
        gamma=0.02,
    )
    adaptive = float(np.mean((y >= lo) & (y <= hi)))
    assert static < 0.75 and adaptive > 0.86
    assert used.min() < 0.1  # the level really adapted downwards (wider intervals)


def test_aci_never_uses_outcomes_that_are_not_yet_observed() -> None:
    """Corrupting outcomes not yet observed at an issue time must not change that interval."""
    from greenqueue.carbon.conformal import aci_intervals

    df = _shifted_stream(n_issues=60)
    cal = {b: np.linspace(-1, 3, 200) for b in range(4)}
    args = (df["issue_time"].to_numpy(), df["h"].to_numpy(), df["target_time"].to_numpy())
    base = aci_intervals(
        *args,
        df["y"].to_numpy(),
        df["lo"].to_numpy(),
        df["hi"].to_numpy(),
        cal,
        alpha=0.1,
        gamma=0.05,
    )
    t_cut = df["issue_time"].iloc[48 * 20]
    y2 = df["y"].to_numpy().copy()
    y2[(df["target_time"] + HOUR > t_cut).to_numpy()] = 1e6  # outcomes unknown at t_cut
    alt = aci_intervals(
        *args, y2, df["lo"].to_numpy(), df["hi"].to_numpy(), cal, alpha=0.1, gamma=0.05
    )
    before = (df["issue_time"] <= t_cut).to_numpy()
    np.testing.assert_array_equal(base[0][before], alt[0][before])
    np.testing.assert_array_equal(base[1][before], alt[1][before])
