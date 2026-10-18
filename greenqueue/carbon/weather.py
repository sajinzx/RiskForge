"""Open-Meteo archived forecasts (previous-runs API) for weather covariates (§6.3, §7.3).

Only the ``previous_day2`` vintage is used: the forecast issued two days before the target hour.
For a forecast issued at ``T`` and a horizon up to 48 h, such a value was published at or before
``T``, so it is legitimate input. Reanalysis and the stitched "historical forecast" product are not
used because they contain information from after the issue time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VINTAGE = "previous_day2"


def _chunks(start: date, end: date, days: int = 180) -> list[tuple[date, date]]:
    out, a = [], start
    while a <= end:
        b = min(a + timedelta(days=days - 1), end)
        out.append((a, b))
        a = b + timedelta(days=1)
    return out


def fetch_point(
    get_json: Callable[[str, dict[str, Any]], dict[str, Any]],
    lat: float,
    lon: float,
    start: date,
    end: date,
    variables: Sequence[str],
) -> pd.DataFrame:
    """Hourly UTC frame (one column per variable) for one location, fetched in 180-day chunks."""
    names = [f"{v}_{VINTAGE}" for v in variables]
    parts = []
    for a, b in _chunks(start, end):
        body = get_json(
            URL,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": a.isoformat(),
                "end_date": b.isoformat(),
                "hourly": ",".join(names),
                "timezone": "UTC",
            },
        )
        if "hourly" not in body:
            raise RuntimeError(f"Open-Meteo error: {body.get('reason', body)}")
        h = body["hourly"]
        idx = pd.DatetimeIndex(pd.to_datetime(h["time"]), tz="UTC")
        parts.append(
            pd.DataFrame(
                {
                    v: np.array([np.nan if x is None else x for x in h[n]], dtype=float)
                    for v, n in zip(variables, names, strict=True)
                },
                index=idx,
            )
        )
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="first")].sort_index()


def combine_points(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide frame ``<variable>__<point>`` plus the cross-point mean ``<variable>__mean``."""
    wide = pd.concat({p: f for p, f in frames.items()}, axis=1)
    wide.columns = [f"{col[1]}__{col[0]}" for col in wide.columns]
    variables = sorted({c.split("__")[0] for c in wide.columns})
    for v in variables:
        cols = [c for c in wide.columns if c.startswith(f"{v}__") and not c.endswith("__mean")]
        wide[f"{v}__mean"] = wide[cols].mean(axis=1)
    return wide


def weather_feature_names(variables: Sequence[str]) -> tuple[str, ...]:
    """Features used by F4: the mean over points of every variable and the two wind regions."""
    names = [f"w_{v}__mean" for v in variables]
    names += ["w_wind_speed_100m__north_sea_offshore", "w_wind_speed_100m__scotland_onshore"]
    return tuple(names)
