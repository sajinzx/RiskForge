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
    *,
    skip: tuple[type[Exception], ...] = (),
    missing: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Fetch generation for ``zone`` month by month; ``query`` wraps ``query_generation``.

    Months whose query raises one of the ``skip`` exceptions (for example "no matching data")
    are left out and appended to ``missing`` so the caller can report the gaps.
    """
    parts = []
    for a, b in month_windows(start, end):
        try:
            parts.append(query(zone, a, b))
        except skip:
            if missing is not None:
                missing.append((a, b))
    if not parts:
        raise ValueError(f"no data for {zone} between {start} and {end}")
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df
