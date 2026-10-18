"""ENTSO-E generation download in monthly chunks (needs ENTSOE_API_KEY in the environment)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def month_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Consecutive ``[a, b)`` windows split at month boundaries covering ``[start, end)``."""
    edges = [start]
    cur = (start + pd.offsets.MonthBegin(1)).normalize()
    while cur < end:
        edges.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    edges.append(end)
    return list(zip(edges[:-1], edges[1:], strict=True))


def download_zone(
    query: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame],
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch generation for ``zone`` month by month; ``query`` wraps ``query_generation``."""
    parts = [query(zone, a, b) for a, b in month_windows(start, end)]
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df
