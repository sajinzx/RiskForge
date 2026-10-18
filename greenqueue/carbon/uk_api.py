"""Client for the NESO Carbon Intensity API (Great Britain, half-hourly).

The API returns, for a requested ``from``/``to``, the half-hour slot that *contains* ``from``, so
the first row of a window can start before ``from``. We keep only slots whose start lies in
``[from, to)`` and de-duplicate on start time, so adjacent windows never double count a slot.
Timestamps are interval-start labels in UTC (implementation.md §7.1).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BASE_URL = "https://api.carbonintensity.org.uk"
MAX_WINDOW_DAYS = 14
SLOTS_PER_DAY = 48
_API_TIME_FORMAT = "%Y-%m-%dT%H:%MZ"


class _Response(Protocol):
    status_code: int

    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class _Session(Protocol):
    def get(self, url: str, *, timeout: float) -> _Response: ...


class TransientApiError(Exception):
    """Raised for HTTP statuses worth retrying (429 and 5xx)."""


def windows(
    start: datetime, end: datetime, days: int = MAX_WINDOW_DAYS
) -> Iterator[tuple[datetime, datetime]]:
    """Yield consecutive half-open windows ``[a, b)`` of at most ``days`` days."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware (UTC)")
    if end <= start:
        raise ValueError("end must be after start")
    step = timedelta(days=days)
    a = start
    while a < end:
        b = min(a + step, end)
        yield a, b
        a = b


def _fmt(t: datetime) -> str:
    return t.astimezone(UTC).strftime(_API_TIME_FORMAT)


def fetch_json(
    session: _Session,
    url: str,
    *,
    attempts: int = 5,
    wait_min: float = 1.0,
    wait_max: float = 60.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """GET ``url`` and return JSON, retrying transient errors with exponential backoff."""
    retrying = Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(min=wait_min, max=wait_max),
        retry=retry_if_exception_type(
            (TransientApiError, requests.ConnectionError, requests.Timeout)
        ),
        reraise=True,
    )
    try:
        for attempt in retrying:
            with attempt:
                resp = session.get(url, timeout=timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise TransientApiError(f"HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                body: dict[str, Any] = resp.json()
                return body
    except RetryError as exc:  # pragma: no cover - reraise=True makes this unreachable
        raise exc
    raise AssertionError("unreachable")  # pragma: no cover


def _slot_start(raw: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(raw, _API_TIME_FORMAT).replace(tzinfo=UTC))


def parse_intensity(payload: dict[str, Any], lo: datetime, hi: datetime) -> pd.DataFrame:
    """Turn an ``/intensity`` payload into rows indexed by UTC slot start within ``[lo, hi)``.

    Columns: ``forecast`` (NESO's own forecast, kept as a baseline), ``actual`` (may be
    NaN), ``index``.
    """
    rows = []
    for item in payload.get("data", []):
        intensity = item["intensity"]
        rows.append(
            {
                "time": _slot_start(item["from"]),
                "forecast": intensity.get("forecast"),
                "actual": intensity.get("actual"),
                "index": intensity.get("index"),
            }
        )
    return _finish(pd.DataFrame(rows, columns=["time", "forecast", "actual", "index"]), lo, hi)


def parse_generation(payload: dict[str, Any], lo: datetime, hi: datetime) -> pd.DataFrame:
    """Turn a ``/generation`` payload into one column per fuel holding its percentage share."""
    rows = []
    for item in payload.get("data", []):
        row: dict[str, Any] = {"time": _slot_start(item["from"])}
        for entry in item["generationmix"]:
            row[entry["fuel"]] = entry["perc"]
        rows.append(row)
    return _finish(pd.DataFrame(rows), lo, hi)


def _finish(df: pd.DataFrame, lo: datetime, hi: datetime) -> pd.DataFrame:
    if df.empty:
        return df.set_index(pd.DatetimeIndex([], tz="UTC", name="time"))
    df = df[(df["time"] >= pd.Timestamp(lo)) & (df["time"] < pd.Timestamp(hi))]
    df = df.drop_duplicates(subset="time").set_index("time").sort_index()
    df.index = pd.DatetimeIndex(df.index, name="time").tz_convert("UTC")
    return df


def download_range(
    start: datetime,
    end: datetime,
    *,
    session: _Session | None = None,
    pause_s: float = 0.5,
    base_url: str = BASE_URL,
    wait_min: float = 1.0,
    wait_max: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download intensity and generation mix for ``[start, end)``; returns ``(intensity, mix)``."""
    sess: _Session = session if session is not None else requests.Session()
    intensity_parts: list[pd.DataFrame] = []
    mix_parts: list[pd.DataFrame] = []
    for lo, hi in windows(start, end):
        span = f"{_fmt(lo)}/{_fmt(hi)}"
        intensity_parts.append(
            parse_intensity(
                fetch_json(
                    sess, f"{base_url}/intensity/{span}", wait_min=wait_min, wait_max=wait_max
                ),
                lo,
                hi,
            )
        )
        mix_parts.append(
            parse_generation(
                fetch_json(
                    sess, f"{base_url}/generation/{span}", wait_min=wait_min, wait_max=wait_max
                ),
                lo,
                hi,
            )
        )
        time.sleep(pause_s)
    return _concat(intensity_parts), _concat(mix_parts)


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="first")].sort_index()


def row_count_report(df: pd.DataFrame, start: datetime, end: datetime) -> dict[str, float]:
    """Compare rows with the expected 48 slots per day so the owner can spot gaps."""
    expected = int((end - start) / timedelta(minutes=30))
    return {
        "rows": float(len(df)),
        "expected": float(expected),
        "missing": float(expected - len(df)),
        "missing_pct": 100.0 * (expected - len(df)) / expected,
    }


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
