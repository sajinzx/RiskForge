from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenqueue.carbon.entsoe_api import download_zone, month_windows
from greenqueue.carbon.intensity import compute_intensity, load_emission_factors
from greenqueue.carbon.preprocess import clean_series

EF_PATH = Path(__file__).parents[2] / "configs" / "emission_factors.yaml"


def _idx(n: int, freq: str = "60min") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


def test_intensity_hand_computed() -> None:
    factors, excl = load_emission_factors(EF_PATH)
    gen = pd.DataFrame(
        {"Nuclear": [100.0, 0.0], "Fossil Gas": [100.0, 0.0], "Solar": [0.0, 0.0]}, _idx(2)
    )
    ci = compute_intensity(gen, factors, excl)
    assert ci.iloc[0] == pytest.approx((12 + 490) / 2)
    assert np.isnan(ci.iloc[1])  # zero total generation


def test_intensity_drops_pumped_storage_consumption_and_clips_negative() -> None:
    factors, excl = load_emission_factors(EF_PATH)
    cols = pd.MultiIndex.from_tuples(
        [
            ("Wind Onshore", "Actual Aggregated"),
            ("Hydro Pumped Storage", "Actual Consumption"),
            ("Fossil Gas", "Actual Aggregated"),
        ]
    )
    gen = pd.DataFrame([[100.0, 500.0, -50.0]], _idx(1), columns=cols)
    assert compute_intensity(gen, factors, excl).iloc[0] == pytest.approx(11.0)


def test_intensity_unknown_type_raises() -> None:
    factors, _ = load_emission_factors(EF_PATH)
    with pytest.raises(KeyError):
        compute_intensity(pd.DataFrame({"Mystery": [1.0]}, _idx(1)), factors)


def test_clean_series_rules() -> None:
    n = 24 * 25
    base = pd.Series(np.arange(n, dtype=float) % 100 + 100, _idx(n))
    s = base.copy()
    s.iloc[300:302] = np.nan  # 2h gap -> linear
    s.iloc[400:410] = np.nan  # 10h gap -> seasonal
    s.iloc[450:490] = np.nan  # 40h gap -> stays missing
    s.iloc[200] = 5000.0  # outlier -> missing -> linear (1h gap)
    out, flag, rep = clean_series(s)
    assert (flag.iloc[300:302] == 1).all() and (flag.iloc[400:410] == 2).all()
    assert out.iloc[400] == base.iloc[400 - 168]
    assert flag.iloc[200] == 1 and rep["outliers"] == 1
    assert out.iloc[450:490].isna().all() and rep["still_missing"] == 40


def test_clean_series_rejects_duplicates() -> None:
    idx = _idx(3).append(_idx(1))
    with pytest.raises(ValueError):
        clean_series(pd.Series([1.0, 2.0, 3.0, 4.0], idx))


def test_month_windows_and_download_dedupe() -> None:
    a, b = pd.Timestamp("2024-01-15", tz="UTC"), pd.Timestamp("2024-03-10", tz="UTC")
    wins = month_windows(a, b)
    assert wins[0][0] == a and wins[-1][1] == b and len(wins) == 3
    assert all(wins[i][1] == wins[i + 1][0] for i in range(2))

    def q(zone: str, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        # overlapping boundary row, as real API responses can have
        idx = pd.date_range(s, e, freq="D", tz="UTC")
        return pd.DataFrame({"Nuclear": 1.0}, idx)

    df = download_zone(q, "FR", a, b)
    assert not df.index.has_duplicates and df.index.is_monotonic_increasing
