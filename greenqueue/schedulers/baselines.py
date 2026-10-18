"""Online schedulers S0 (ASAP), S1 (greedy window), S2 (threshold) and simple S4.

They see only :class:`JobView` and a :class:`ForecastBundle`. Capacity is respected by
construction: running rigid jobs are committed first, then jobs are added only if they fit.
"""

from __future__ import annotations

import math

import numpy as np

from greenqueue.core.types import Decision, ForecastBundle, JobView


def _dur_steps(v: JobView, bundle: ForecastBundle, dt_h: float, upper: bool) -> int:
    est = bundle.energy[v.id]
    hours = est.duration_hi_h if upper else est.duration_point_h
    return max(1, math.ceil(hours / dt_h - 1e-9))


def _remaining_steps(v: JobView, bundle: ForecastBundle, dt_h: float, upper: bool) -> int:
    est = bundle.energy[v.id]
    hours = est.duration_hi_h if upper else est.duration_point_h
    left = max(hours - v.elapsed_run_h, 0.0)
    if v.started and not v.running:
        left += v.checkpoint_overhead_h  # resuming costs extra time
    return max(1, math.ceil(left / dt_h - 1e-9))


class Asap:
    """S0: start every job as soon as GPUs are free, first come first served; never pause."""

    name = "S0_asap"

    def decide(
        self,
        t: int,
        queue: list[JobView],
        running: list[JobView],
        bundle: ForecastBundle,
        free_gpus: int,
    ) -> list[Decision]:
        out = [Decision(j.id, "run") for j in running]
        for q in sorted(queue, key=lambda v: (v.arrival, v.id)):
            if q.gpus <= free_gpus:
                out.append(Decision(q.id, "start"))
                free_gpus -= q.gpus
        return out


class GreedyWindow:
    """S1: start each job at the step in its slack whose forecast window has the lowest mean CI.

    With ``use_upper_duration`` the window length and the latest start use the duration *upper*
    bound, which is the simple S4 variant (conformal duration bound, point carbon forecast).
    Replanned every step; once started a job runs to completion.
    """

    def __init__(self, dt_h: float = 1.0, use_upper_duration: bool = False) -> None:
        self.dt_h = dt_h
        self.upper = use_upper_duration
        self.name = "S4_simple" if use_upper_duration else "S1_greedy"

    def _wants_now(self, v: JobView, bundle: ForecastBundle, t: int) -> tuple[bool, bool]:
        """Return ``(start_now, urgent)`` for a not-yet-started job."""
        dur = _dur_steps(v, bundle, self.dt_h, self.upper)
        latest = math.floor(v.deadline - dur + 1e-9)
        if t >= latest:
            return True, True
        fc = bundle.carbon.point
        last = min(latest, t + len(fc) - 1)
        best_s, best_cost = t, math.inf
        for s in range(t, last + 1):
            window = fc[s - t : s - t + dur]
            cost = float(np.mean(window))
            if cost < best_cost - 1e-12:
                best_s, best_cost = s, cost
        return best_s == t, False

    def decide(
        self,
        t: int,
        queue: list[JobView],
        running: list[JobView],
        bundle: ForecastBundle,
        free_gpus: int,
    ) -> list[Decision]:
        out = [Decision(j.id, "run") for j in running]
        wants = []
        for q in queue:
            now, urgent = self._wants_now(q, bundle, t)
            if now:
                wants.append((0 if urgent else 1, q.deadline, q.id, q))
        for _, _, _, q in sorted(wants, key=lambda w: w[:3]):
            if q.gpus <= free_gpus:
                out.append(Decision(q.id, "start"))
                free_gpus -= q.gpus
        return out


class Threshold:
    """S2: run while the forecast CI is at or below its ``q`` quantile, or when slack runs out."""

    name = "S2_threshold"

    def __init__(self, q: float = 0.3, dt_h: float = 1.0) -> None:
        self.q = q
        self.dt_h = dt_h

    def decide(
        self,
        t: int,
        queue: list[JobView],
        running: list[JobView],
        bundle: ForecastBundle,
        free_gpus: int,
    ) -> list[Decision]:
        fc = bundle.carbon.point
        cheap = bool(fc[0] <= np.quantile(fc, self.q))
        capacity = free_gpus + sum(j.gpus for j in running)
        chosen: list[JobView] = []
        used = 0
        # rigid running jobs cannot be paused
        for j in running:
            if not j.interruptible:
                chosen.append(j)
                used += j.gpus
        cands = []
        for v in [*running, *queue]:
            if v in chosen:
                continue
            urgent = t + _remaining_steps(v, bundle, self.dt_h, False) >= v.deadline - 1e-9
            if urgent or cheap:
                cands.append((0 if urgent else 1, 0 if v.running else 1, v.deadline, v.id, v))
        for *_, v in sorted(cands, key=lambda c: c[:4]):
            if used + v.gpus <= capacity:
                chosen.append(v)
                used += v.gpus
        return [Decision(v.id, "run" if v.started else "start") for v in chosen]
