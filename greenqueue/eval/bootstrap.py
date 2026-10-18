"""Block bootstrap for dependent data (implementation.md §12): 7-day blocks, 1000 resamples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_ci(
    values: np.ndarray,
    times: pd.Series | pd.DatetimeIndex,
    *,
    block_days: int = 7,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile CI from resampling whole ``block_days`` blocks.

    Returns ``(mean, lo, hi)``. Rows in the same block stay together so serial correlation
    is preserved; the statistic is the ratio of resampled sums to resampled counts.
    """
    v = np.asarray(values, dtype=float)
    t = pd.DatetimeIndex(times)
    if len(v) != len(t):
        raise ValueError("values and times must have the same length")
    block = np.asarray((t - t.min()) // pd.Timedelta(days=block_days), dtype=np.int64)
    n_blocks = int(np.max(block)) + 1
    sums = np.bincount(block, weights=v, minlength=n_blocks)
    counts = np.bincount(block, minlength=n_blocks).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    est = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


def paired_saving_ci(
    base: np.ndarray,
    other: np.ndarray,
    blocks: np.ndarray,
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Saving ``1 - sum(other)/sum(base)`` with a CI from resampling whole blocks (paired).

    ``blocks`` labels each row (for example seed and week of arrival), so rows from the same
    week stay together and both schedulers are resampled identically.
    """
    base, other = np.asarray(base, dtype=float), np.asarray(other, dtype=float)
    _, inv = np.unique(blocks, return_inverse=True)
    n_blocks = int(np.max(inv)) + 1
    sb = np.bincount(inv, weights=base, minlength=n_blocks)
    so = np.bincount(inv, weights=other, minlength=n_blocks)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    est = 1.0 - so[pick].sum(axis=1) / sb[pick].sum(axis=1)
    a = (1 - level) / 2
    point = 1.0 - float(other.sum()) / float(base.sum())
    return float(point), float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


def rate_ci(
    events: np.ndarray,
    blocks: np.ndarray,
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Share of True in ``events`` with a block-bootstrap CI (blocks resampled whole)."""
    e = np.asarray(events, dtype=float)
    _, inv = np.unique(blocks, return_inverse=True)
    n_blocks = int(np.max(inv)) + 1
    s = np.bincount(inv, weights=e, minlength=n_blocks)
    c = np.bincount(inv, minlength=n_blocks).astype(float)
    pick = np.random.default_rng(seed).integers(0, n_blocks, size=(n_resamples, n_blocks))
    est = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    a = (1 - level) / 2
    return float(e.mean()), float(np.quantile(est, a)), float(np.quantile(est, 1 - a))
