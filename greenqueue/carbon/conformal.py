"""Hand-written split conformal and CQR, calibrated per horizon bucket (§8.3).

Finite-sample rule: with ``n`` calibration scores the interval uses the
``ceil((n + 1)(1 - alpha))``-th smallest score, which gives marginal coverage >= 1 - alpha under
exchangeability. If that rank exceeds ``n`` the correction is infinite (not enough data).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

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


def score_quantile_clipped(sorted_scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile at miscoverage ``alpha`` from sorted scores, clipped to their range.

    ACI can drive ``alpha`` outside (0, 1). Gibbs and Candes return an infinite or empty interval
    there; for use in a scheduler we clip to the largest or smallest score instead (a documented
    weakening of the guarantee at the extremes).
    """
    n = len(sorted_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(sorted_scores[min(max(k, 1), n) - 1])


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    """Nanoseconds since the epoch (UTC), independent of the index unit or time zone."""
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return np.asarray(idx.as_unit("ns").to_numpy().astype("int64"))


def aci_intervals(
    issue_time: pd.DatetimeIndex | pd.Series | np.ndarray,
    h: np.ndarray,
    target_time: pd.DatetimeIndex | pd.Series | np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    cal_scores: dict[int, np.ndarray],
    *,
    alpha: float,
    gamma: float,
    obs_delay_h: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U3: adaptive conformal inference on top of CQR scores, with delayed feedback.

    One miscoverage level ``alpha_b`` per horizon bucket, updated as
    ``alpha_b += gamma*(alpha - err)``. The outcome of a row is only known once its target hour has
    ended, so a forecast issued at ``T`` may only use rows with ``target_time + obs_delay <= T``.
    Rows must be sorted by issue time.
    Returns ``(lo_adj, hi_adj, alpha_b_used)`` where the last is the level in force for each row.
    """
    bucket = horizon_bucket(h)
    sorted_scores = {b: np.sort(s) for b, s in cal_scores.items()}
    n = len(y)
    obs = _ns(pd.DatetimeIndex(target_time) + pd.Timedelta(hours=obs_delay_h))
    order = np.argsort(obs, kind="stable")
    a = dict.fromkeys(sorted_scores, alpha)
    err = np.zeros(n)
    out_lo, out_hi, used = np.empty(n), np.empty(n), np.empty(n)
    issues = _ns(pd.DatetimeIndex(issue_time))
    starts = np.flatnonzero(np.r_[True, issues[1:] != issues[:-1]])
    ends = np.r_[starts[1:], n]
    ptr = 0
    for s, e in zip(starts, ends, strict=True):
        t_issue = issues[s]
        while ptr < n and obs[order[ptr]] <= t_issue:
            r = order[ptr]
            b = int(bucket[r])
            a[b] += gamma * (alpha - err[r])
            ptr += 1
        for b in np.unique(bucket[s:e]):
            sel = np.arange(s, e)[bucket[s:e] == b]
            c = score_quantile_clipped(sorted_scores[int(b)], a[int(b)])
            out_lo[sel], out_hi[sel], used[sel] = lo[sel] - c, hi[sel] + c, a[int(b)]
        err[s:e] = ((y[s:e] < out_lo[s:e]) | (y[s:e] > out_hi[s:e])).astype(float)
    return out_lo, out_hi, used
