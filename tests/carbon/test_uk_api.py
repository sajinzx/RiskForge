from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import requests

from greenqueue.carbon import uk_api


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)


class FakeResponse:
    def __init__(self, status_code: int = 200, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> Any:
        return self._body


class FakeSession:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.urls.append(url)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, FakeResponse)
        return item


def _intensity_payload(starts: list[str], actual: int | None = 100) -> dict[str, Any]:
    return {
        "data": [
            {
                "from": s,
                "to": s,
                "intensity": {"forecast": 90, "actual": actual, "index": "low"},
            }
            for s in starts
        ]
    }


def test_windows_cover_range_without_gaps_or_overlap() -> None:
    start, end = _t("2024-01-01T00:00Z"), _t("2024-02-02T00:00Z")  # 32 days
    wins = list(uk_api.windows(start, end))
    assert wins[0][0] == start and wins[-1][1] == end
    assert all(b - a <= timedelta(days=14) for a, b in wins)
    assert all(wins[i][1] == wins[i + 1][0] for i in range(len(wins) - 1))
    assert len(wins) == 3


def test_windows_reject_naive_and_empty() -> None:
    with pytest.raises(ValueError):
        list(uk_api.windows(datetime(2024, 1, 1), datetime(2024, 1, 2)))
    with pytest.raises(ValueError):
        list(uk_api.windows(_t("2024-01-02T00:00Z"), _t("2024-01-01T00:00Z")))


def test_parse_intensity_trims_leading_slot_and_dedupes() -> None:
    payload = _intensity_payload(
        ["2023-12-31T23:30Z", "2024-01-01T00:00Z", "2024-01-01T00:30Z", "2024-01-01T00:30Z"]
    )
    df = uk_api.parse_intensity(payload, _t("2024-01-01T00:00Z"), _t("2024-01-01T01:00Z"))
    assert list(df.index) == [_t("2024-01-01T00:00Z"), _t("2024-01-01T00:30Z")]
    assert str(df.index.tz) == "UTC"
    assert df["forecast"].iloc[0] == 90


def test_parse_intensity_keeps_missing_actual_as_nan() -> None:
    payload = _intensity_payload(["2024-01-01T00:00Z"], actual=None)
    df = uk_api.parse_intensity(payload, _t("2024-01-01T00:00Z"), _t("2024-01-01T01:00Z"))
    assert df["actual"].isna().all()


def test_parse_generation_one_column_per_fuel() -> None:
    payload = {
        "data": [
            {
                "from": "2024-01-01T00:00Z",
                "to": "2024-01-01T00:30Z",
                "generationmix": [{"fuel": "wind", "perc": 54.6}, {"fuel": "gas", "perc": 12}],
            }
        ]
    }
    df = uk_api.parse_generation(payload, _t("2024-01-01T00:00Z"), _t("2024-01-01T01:00Z"))
    assert df.loc[_t("2024-01-01T00:00Z"), "wind"] == 54.6
    assert set(df.columns) == {"wind", "gas"}


def test_fetch_json_retries_transient_then_succeeds() -> None:
    session = FakeSession(
        [FakeResponse(503), requests.ConnectionError("boom"), FakeResponse(200, {"ok": 1})]
    )
    out = uk_api.fetch_json(session, "u", wait_min=0, wait_max=0)
    assert out == {"ok": 1}
    assert len(session.urls) == 3


def test_fetch_json_gives_up_after_attempts() -> None:
    session = FakeSession([FakeResponse(500)] * 3)
    with pytest.raises(uk_api.TransientApiError):
        uk_api.fetch_json(session, "u", attempts=3, wait_min=0, wait_max=0)


def test_fetch_json_does_not_retry_client_errors() -> None:
    session = FakeSession([FakeResponse(404)])
    with pytest.raises(requests.HTTPError):
        uk_api.fetch_json(session, "u", wait_min=0, wait_max=0)
    assert len(session.urls) == 1


def test_download_range_stitches_windows_without_duplicates() -> None:
    start, end = _t("2024-01-01T00:00Z"), _t("2024-01-16T00:00Z")  # 15 days -> 2 windows
    w1_end = "2024-01-15T00:00Z"
    gen = {
        "data": [
            {"from": "2024-01-01T00:00Z", "to": "x", "generationmix": [{"fuel": "gas", "perc": 1}]}
        ]
    }
    gen2 = {"data": [{"from": w1_end, "to": "x", "generationmix": [{"fuel": "gas", "perc": 2}]}]}
    session = FakeSession(
        [
            FakeResponse(200, _intensity_payload(["2024-01-01T00:00Z", "2024-01-14T23:30Z"])),
            FakeResponse(200, gen),
            # second window's API response repeats the slot before its start (14 Jan 23:30)
            FakeResponse(200, _intensity_payload(["2024-01-14T23:30Z", w1_end])),
            FakeResponse(200, gen2),
        ]
    )
    intensity, mix = uk_api.download_range(
        start, end, session=session, pause_s=0, wait_min=0, wait_max=0
    )
    assert list(intensity.index) == [
        _t("2024-01-01T00:00Z"),
        _t("2024-01-14T23:30Z"),
        _t(w1_end),
    ]
    assert len(mix) == 2
    assert session.urls[0].endswith("/intensity/2024-01-01T00:00Z/2024-01-15T00:00Z")


def test_row_count_report() -> None:
    start, end = _t("2024-01-01T00:00Z"), _t("2024-01-02T00:00Z")
    df = uk_api.parse_intensity(
        _intensity_payload(["2024-01-01T00:00Z", "2024-01-01T00:30Z"]), start, end
    )
    report = uk_api.row_count_report(df, start, end)
    assert report["expected"] == uk_api.SLOTS_PER_DAY
    assert report["missing"] == uk_api.SLOTS_PER_DAY - 2
