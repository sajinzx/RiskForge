"""Hand-written split conformal and CQR, calibrated per horizon bucket (§8.3).

Finite-sample rule: with ``n`` calibration scores the interval uses the
``ceil((n + 1)(1 - alpha))``-th smallest score, which gives marginal coverage >= 1 - alpha under
exchangeability. If that rank exceeds ``n`` the correction is infinite (not enough data).
"""

from __future__ import annotations

import math

import numpy as np

from greenqueue.carbon.features import horizon_bucket


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n+1)(1-alpha))``-th smallest score; ``inf`` if there are too few scores."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n or n == 0 else float(s[k - 1])


class _PerBucket:
    """Common bookkeeping: one conformal quantile per horizon bucket, pooled fallback."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.q: dict[int, float] = {}
        self.q_pooled = float("inf")

    def _fit_scores(self, h: np.ndarray, scores: np.ndarray) -> None:
        b = horizon_bucket(h)
        self.q_pooled = conformal_quantile(scores, self.alpha)
        self.q = {int(k): conformal_quantile(scores[b == k], self.alpha) for k in np.unique(b)}

    def _corr(self, h: np.ndarray) -> np.ndarray:
        b = horizon_bucket(h)
        return np.array([self.q.get(int(k), self.q_pooled) for k in b])


class SplitConformal(_PerBucket):
    """U1: ``point +- q`` with ``q`` the conformal quantile of absolute residuals."""

    def fit(self, h: np.ndarray, point: np.ndarray, y: np.ndarray) -> SplitConformal:
        self._fit_scores(h, np.abs(np.asarray(y) - np.asarray(point)))
        return self

    def predict(self, h: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self._corr(h)
        return point - c, point + c


class CQR(_PerBucket):
    """U2: conformalized quantile regression (Romano et al. 2019).

    Score ``s = max(lo - y, y - hi)`` is positive when ``y`` is outside the raw interval and
    negative when inside; the interval is widened (or shrunk) by the conformal quantile of ``s``.
    """

    def fit(self, h: np.ndarray, lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> CQR:
        self._fit_scores(h, np.maximum(np.asarray(lo) - y, np.asarray(y) - np.asarray(hi)))
        return self

    def predict(
        self, h: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self._corr(h)
        return lo - c, hi + c
