import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from greenqueue.carbon.weather import combine_points, fetch_point, weather_feature_names
from greenqueue.experiments.ew import run_ew


def test_fetch_point_chunks_and_handles_missing_values() -> None:
    calls: list[dict] = []  # type: ignore[type-arg]

    def get_json(url: str, params: dict) -> dict:  # type: ignore[type-arg]
        calls.append(params)
        times = pd.date_range(params["start_date"], params["end_date"] + "T23:00", freq="h")
        n = len(times)
        return {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                "wind_speed_100m_previous_day2": [None] + [5.0] * (n - 1),
                "temperature_2m_previous_day2": [1.0] * n,
            }
        }

    df = fetch_point(
        get_json,
        53.0,
        -1.0,
        date(2024, 3, 8),
        date(2024, 12, 31),
        ["wind_speed_100m", "temperature_2m"],
    )
    assert len(calls) == 2 and df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert np.isnan(df["wind_speed_100m"].iloc[0]) and df["wind_speed_100m"].iloc[1] == 5.0
    wide = combine_points({"a": df, "b": df + 2.0})
    assert "temperature_2m__mean" in wide.columns
    assert wide["temperature_2m__mean"].iloc[5] == 2.0  # mean of 1 and 3
    assert "w_wind_speed_100m__north_sea_offshore" in weather_feature_names(["wind_speed_100m"])


def test_weather_experiment_smoke_and_helpful_when_signal_is_in_weather(
    smoke_root: tuple[Path, dict],  # type: ignore[type-arg]
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    shutil.copytree(smoke_root[0], root)  # do not disturb the shared session fixture
    # rebuild the processed series from the synthetic raw file and add a weather signal
    from greenqueue.carbon.gb import build_gb_hourly

    gb = build_gb_hourly(
        root / "data" / "raw" / "uk_ci",
        root / "data" / "processed",
        root / "configs" / "preprocess.yaml",
    )
    rng = np.random.default_rng(3)
    idx = gb.index
    wind = (
        pd.Series(np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
        .rolling(48, min_periods=1)
        .mean()
    )
    # make CI depend on the weather 'forecast' so that a model with weather must win
    gb2 = gb.copy()
    gb2["ci"] = gb["ci"] - 8.0 * wind
    gb2.to_parquet(root / "data" / "processed" / "gb_hourly.parquet")
    names = {
        f"{v}__mean": wind * (1.0 if v == "wind_speed_100m" else 0.0)
        for v in ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover"]
    }
    names["wind_speed_100m__north_sea_offshore"] = wind
    names["wind_speed_100m__scotland_onshore"] = wind
    wdir = root / "data" / "raw" / "weather"
    wdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(names, index=idx).to_parquet(wdir / "GB_previous_day2.parquet")

    s = run_ew(root, smoke=True)
    out = root / "results" / "ew_smoke"
    assert (out / "weather_ablation.csv").exists() and (out / "mae_vs_horizon_weather.png").exists()
    t = pd.read_csv(out / "weather_ablation.csv").set_index("model")
    assert t.loc["F4_weather", "mae"] < t.loc["F3_no_weather", "mae"]
    assert s["mae_improvement_from_weather"]["mean"] > 0
