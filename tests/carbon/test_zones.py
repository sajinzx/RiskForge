from pathlib import Path

import numpy as np
import pandas as pd

from greenqueue.carbon.zones import build_zone_hourly, restore_columns

ROOT = Path(__file__).parents[2]


def test_restore_columns_rebuilds_multiindex_tuples() -> None:
    df = pd.DataFrame(columns=["Wind Onshore | Actual Aggregated", "Solar"])
    cols = list(restore_columns(df).columns)
    assert cols == [("Wind Onshore", "Actual Aggregated"), "Solar"]


def test_build_zone_hourly_weights_by_emission_factor_and_ignores_pumping(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=96 * 14, freq="15min", tz="UTC")
    gen = pd.DataFrame(
        {
            "Fossil Gas | Actual Aggregated": 100.0,
            "Wind Onshore | Actual Aggregated": 100.0,
            "Hydro Pumped Storage | Actual Consumption": 500.0,
        },
        index=idx,
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    gen.to_parquet(raw / "FR.parquet")
    out = build_zone_hourly(
        "FR",
        raw,
        tmp_path / "proc",
        ROOT / "configs" / "emission_factors.yaml",
        ROOT / "configs" / "preprocess.yaml",
    )
    assert len(out) >= 24 * 14 - 1
    assert np.allclose(out["ci"].dropna(), (490 + 11) / 2)  # gas 490 and onshore wind 11
    assert (tmp_path / "proc" / "FR_hourly.parquet").exists()
