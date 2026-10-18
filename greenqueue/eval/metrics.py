"""Forecast metrics: pinball loss, MAE, coverage, interval width (implementation.md §8.6)."""

from __future__ import annotations

import numpy as np


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mean pinball loss of quantile prediction ``q`` at level ``tau``."""
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of observations inside ``[lo, hi]``."""
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)))


def interval_score_pinball(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Mean of the two pinball losses of the interval ends (levels alpha/2 and 1-alpha/2)."""
    return 0.5 * (pinball(y, lo, alpha / 2) + pinball(y, hi, 1 - alpha / 2))
