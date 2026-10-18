"""Great Britain: raw NESO half-hours to a clean hourly series (implementation.md §7.1, §7.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from greenqueue.carbon.preprocess import clean_series


def hourly_from_half_hours(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean of the two half-hour slots (interval-start labels). Needs >=1 slot per hour."""
    idx = pd.DatetimeIndex(raw.index)
    hourly = raw[["actual", "forecast"]].astype(float).groupby(idx.floor("h")).mean()
    return hourly.asfreq("h")


def build_gb_hourly(raw_dir: Path, out_dir: Path, preprocess_cfg: Path) -> pd.DataFrame:
    """Create ``data/processed/gb_hourly.parquet`` with columns ci, imputed, neso_fc."""
    cfg: dict[str, Any] = yaml.safe_load(preprocess_cfg.read_text(encoding="utf8"))
    raw = pd.read_parquet(raw_dir / "intensity.parquet")
    hourly = hourly_from_half_hours(raw)
    ci, flag, report = clean_series(
        hourly["actual"],
        bounds=(cfg["ci_bounds"][0], cfg["ci_bounds"][1]),
        interp_max_hours=cfg["interp_max_hours"],
        seasonal_max_hours=cfg["seasonal_max_hours"],
        seasonal_lag_hours=cfg["seasonal_lag_hours"],
    )
    lo, hi = cfg["ci_bounds"]
    neso = hourly["forecast"].where(hourly["forecast"].between(lo, hi))  # out-of-range -> missing
    out = pd.DataFrame({"ci": ci, "imputed": flag, "neso_fc": neso})
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "gb_hourly.parquet")
    (out_dir / "quality_report_GB.json").write_text(json.dumps(report, indent=2))
    return out
