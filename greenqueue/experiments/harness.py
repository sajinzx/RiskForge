"""Glue between forecasts, a synthetic energy estimator and the simulator (E4 and later)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from greenqueue.carbon.conformal import conformal_quantile
from greenqueue.core.types import CarbonForecast, EnergyEstimate, ForecastBundle, Job, JobView


class ForecastReplay:
    """Serves the forecast most recently issued (00:00 or 12:00 UTC) at or before each step.

    A forecast issued at ``T`` only used information up to ``T - 1h``; at a later step it is
    sliced to start at the current hour, so a scheduler never sees a forecast for the past.
    """

    def __init__(self, fc: pd.DataFrame, start: pd.Timestamp, dt_h: float = 1.0) -> None:
        if dt_h != 1.0:
            raise NotImplementedError("forecasts are hourly; dt_h must be 1")
        self.start = start
        self.by_issue: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for issue, g in fc.groupby("issue_time"):
            g = g.sort_values("h")
            if len(g) == 48 and g["h"].iloc[0] == 1:
                self.by_issue[pd.Timestamp(str(issue))] = (
                    g["point"].to_numpy(),
                    g["lo"].to_numpy(),
                    g["hi"].to_numpy(),
                )

    def at(self, step: int) -> CarbonForecast:
        ts = self.start + pd.Timedelta(hours=step)
        issue = ts.floor("12h")
        if issue not in self.by_issue:
            raise KeyError(f"no complete forecast issued at {issue}")
        k = int((ts - issue) / pd.Timedelta(hours=1))
        p, lo, hi = self.by_issue[issue]
        return CarbonForecast(step, p[k:], lo[k:], hi[k:])


class SyntheticEstimator:
    """Duration and energy estimates with a known lognormal error, plus a conformal upper bound.

    STAND-IN until real GPU measurements exist: the error model is an assumption (``sigma``), not
    a fitted energy model. ``duration_hi_h`` is the one-sided split-conformal upper bound at level
    ``1 - alpha`` computed from a *separate* calibration draw of the same noise.
    """

    def __init__(
        self, jobs: Sequence[Job], sigma: float, alpha: float, dt_h: float, seed: int
    ) -> None:
        rng = np.random.default_rng(seed + 10_000)
        self.noise = {j.id: float(rng.normal(0.0, sigma)) for j in jobs}
        cal = rng.normal(0.0, sigma, size=500)
        # score = log(true / estimate) = -noise; upper bound multiplies by exp(q)
        self.q_log = conformal_quantile(-cal, alpha) if sigma > 0 else 0.0
        self.by_id = {j.id: j for j in jobs}
        self.dt_h = dt_h

    def estimate(self, jobs: list[JobView]) -> dict[str, EnergyEstimate]:
        out = {}
        for v in jobs:
            j = self.by_id[v.id]
            true_h = j.duration_steps * self.dt_h
            e_true = j.true_energy_kwh(self.dt_h)
            f = math.exp(self.noise[v.id])
            hi = math.exp(self.q_log)
            out[v.id] = EnergyEstimate(
                v.id, e_true * f, e_true * f / hi, e_true * f * hi, true_h * f, true_h * f * hi
            )
        return out


def make_provider(replay: ForecastReplay, estimator: SyntheticEstimator):  # type: ignore[no-untyped-def]
    def provide(t: int, views: list[JobView]) -> ForecastBundle:
        return ForecastBundle(replay.at(t), estimator.estimate(views))

    return provide
