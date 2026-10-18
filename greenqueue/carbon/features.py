"""Direct multi-horizon forecasting dataset (implementation.md §7.3).

Conventions (all UTC, hourly, labels are interval *starts*):
- A forecast is issued at ``T`` (00:00 or 12:00). The last hour fully observed at ``T`` has label
  ``T - 1h``, so every history feature uses labels ``<= T - 1h``.
- Horizon ``h`` (1..48) targets the hour with label ``T + (h - 1)h``, i.e. h=1 is the hour that
  starts at the issue time. This is the hour a scheduler deciding at ``T`` is about to run in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = tuple(range(1, 49))
LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
ROLL_WINDOWS = (24, 168)
HOUR = pd.Timedelta(hours=1)

BUCKET_EDGES = (6, 12, 24, 48)


def horizon_bucket(h: np.ndarray | int) -> np.ndarray:
    """Map horizon(s) to bucket ids 0..3 for {1-6, 7-12, 13-24, 25-48}."""
    return np.searchsorted(np.array(BUCKET_EDGES), np.asarray(h), side="left")


def feature_columns(use_neso: bool = False) -> list[str]:
    cols = [f"lag_{k}" for k in LAGS]
    for w in ROLL_WINDOWS:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    cols += ["snaive24", "snaive168", "h", "hour_sin", "hour_cos", "dow", "month_sin", "month_cos"]
    cols += ["holiday"]
    if use_neso:
        cols.append("neso_fc")
    return cols


def _holiday_flags(dates: pd.DatetimeIndex, country: str) -> np.ndarray:
    import holidays

    years = sorted(set(dates.year))
    cal = holidays.country_holidays(country, years=years)
    return np.array([d.date() in cal for d in dates], dtype="int8")


def build_dataset(
    ci: pd.Series,
    *,
    issue_hours: tuple[int, ...] = (0, 12),
    horizons: tuple[int, ...] = HORIZONS,
    country: str = "GB",
    neso: pd.Series | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Build one row per (issue time, horizon). ``ci`` is an hourly UTC series (NaN = missing)."""
    ci = ci.asfreq("h")
    idx = pd.DatetimeIndex(ci.index)
    issues = idx[idx.hour.isin(issue_hours)]
    issues = issues[
        (issues - 168 * HOUR >= idx[0]) & (issues + max(horizons) * HOUR <= idx[-1] + HOUR)
    ]

    def at(times: pd.DatetimeIndex) -> np.ndarray:
        return ci.reindex(times).to_numpy(dtype=float)

    base: dict[str, np.ndarray] = {}
    for k in LAGS:
        base[f"lag_{k}"] = at(issues - k * HOUR)
    for w in ROLL_WINDOWS:
        r = ci.rolling(w, min_periods=w // 2)
        base[f"roll_mean_{w}"] = r.mean().reindex(issues - HOUR).to_numpy()
        base[f"roll_std_{w}"] = r.std().reindex(issues - HOUR).to_numpy()

    n_i, n_h = len(issues), len(horizons)
    h_arr = np.tile(np.array(horizons), n_i)
    issue_rep = issues.repeat(n_h)
    target = issue_rep + (h_arr - 1) * HOUR
    data: dict[str, np.ndarray] = {k: np.repeat(v, n_h) for k, v in base.items()}
    k24 = np.ceil(h_arr / 24).astype(int)
    data["snaive24"] = at(target - pd.to_timedelta(24 * k24, unit="h"))
    data["snaive168"] = at(target - 168 * HOUR)
    data["h"] = h_arr
    hour = target.hour.to_numpy()
    month = target.month.to_numpy()
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["dow"] = target.dayofweek.to_numpy()
    data["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    data["holiday"] = _holiday_flags(target.normalize(), country)
    if neso is not None:
        data["neso_fc"] = neso.reindex(target).to_numpy(dtype=float)

    df = pd.DataFrame(data)
    df.insert(0, "target_time", target)
    df.insert(0, "issue_time", issue_rep)
    df["y"] = at(target)
    if require_target:
        df = df[df["y"].notna()]
    return df.reset_index(drop=True)
