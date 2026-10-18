"""Gap filling and quality reporting for carbon series (implementation.md §7.2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def clean_series(
    s: pd.Series,
    *,
    bounds: tuple[float, float] = (0.0, 1200.0),
    interp_max_hours: float = 2.0,
    seasonal_max_hours: float = 24.0,
    seasonal_lag_hours: float = 168.0,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Clean a regular-frequency UTC series.

    Out-of-bounds values become missing; gaps up to ``interp_max_hours`` are linearly
    interpolated (flag 1); gaps up to ``seasonal_max_hours`` are filled from the same time
    ``seasonal_lag_hours`` earlier (flag 2); longer gaps stay NaN (flag 0) so they are excluded
    downstream. Returns ``(clean, imputed_flag, quality_report)``.
    """
    if s.index.has_duplicates:
        raise ValueError("duplicate timestamps")
    if not s.index.is_monotonic_increasing:
        raise ValueError("index must be sorted")
    if len(s) < 2:
        raise ValueError("need at least two rows")
    diffs = pd.Series(s.index).diff().dropna()
    if diffs.nunique() != 1:
        raise ValueError("series must have a regular frequency")
    step_h = diffs.iloc[0].total_seconds() / 3600
    lo, hi = bounds
    x = s.astype(float).copy()
    bad = (x < lo) | (x > hi)
    outliers = int(bad.sum())
    x[bad] = np.nan
    n = len(x)
    flag = np.zeros(n, dtype="int8")
    missing_before = int(x.isna().sum())

    orig = x.to_numpy().copy()
    vals = orig.copy()
    isna = np.isnan(orig)
    lag = int(round(seasonal_lag_hours / step_h))
    i = 0
    while i < n:
        if not isna[i]:
            i += 1
            continue
        j = i
        while j < n and isna[j]:
            j += 1
        length_h = (j - i) * step_h
        if length_h <= interp_max_hours and i > 0 and j < n:
            vals[i:j] = np.interp(np.arange(i, j), [i - 1, j], [orig[i - 1], orig[j]])
            flag[i:j] = 1
        elif length_h <= seasonal_max_hours:
            for k in range(i, j):
                if k - lag >= 0 and not np.isnan(orig[k - lag]):
                    vals[k] = orig[k - lag]
                    flag[k] = 2
        i = j
    out = pd.Series(vals, index=x.index, name=s.name)
    pct = (lambda c: 100.0 * c / n) if n else (lambda c: 0.0)
    report = {
        "rows": int(n),
        "missing_pct": pct(missing_before),
        "imputed_linear_pct": pct(int((flag == 1).sum())),
        "imputed_seasonal_pct": pct(int((flag == 2).sum())),
        "still_missing": int(out.isna().sum()),
        "outliers": outliers,
    }
    return out, pd.Series(flag, index=x.index, name="imputed"), report
