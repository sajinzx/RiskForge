"""ENTSO-E zones: raw generation parquet to a clean hourly carbon-intensity series (§7.1, §7.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from greenqueue.carbon.intensity import compute_intensity, load_emission_factors
from greenqueue.carbon.preprocess import clean_series

HOLIDAY_COUNTRY = {"DE_LU": "DE", "FR": "FR", "PL": "PL", "ES": "ES", "GB": "GB"}


def restore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the ' | ' joining done when saving (type | kind) MultiIndex columns to parquet."""
    out = df.copy()
    out.columns = [tuple(c.split(" | ")) if " | " in str(c) else str(c) for c in df.columns]
    return out


def build_zone_hourly(
    zone: str,
    raw_dir: Path,
    out_dir: Path,
    emission_factors: Path,
    preprocess_cfg: Path,
) -> pd.DataFrame:
    """Create ``<zone>_hourly.parquet`` (columns ci, imputed) and its quality report."""
    cfg: dict[str, Any] = yaml.safe_load(preprocess_cfg.read_text(encoding="utf8"))
    gen = restore_columns(pd.read_parquet(raw_dir / f"{zone}.parquet"))
    gen.index = pd.DatetimeIndex(gen.index)
    factors, excluded = load_emission_factors(emission_factors)
    # hourly mean of the generation (15-minute data for DE) before weighting by emission factors
    hourly_gen = gen.groupby(gen.index.floor("h")).mean()
    ci_raw = compute_intensity(hourly_gen, factors, excluded).asfreq("h")
    ci, flag, report = clean_series(
        ci_raw,
        bounds=(cfg["ci_bounds"][0], cfg["ci_bounds"][1]),
        interp_max_hours=cfg["interp_max_hours"],
        seasonal_max_hours=cfg["seasonal_max_hours"],
        seasonal_lag_hours=cfg["seasonal_lag_hours"],
    )
    out = pd.DataFrame({"ci": ci, "imputed": flag})
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"{zone}_hourly.parquet")
    (out_dir / f"quality_report_{zone}.json").write_text(json.dumps(report, indent=2))
    return out
