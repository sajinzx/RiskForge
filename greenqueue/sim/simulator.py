"""Discrete-time GPU cluster simulator with exact accounting (implementation.md §10).

The simulator is the only component that sees true durations, energies and carbon intensity.
Per step it (1) builds the scheduler's view, (2) asks for decisions, (3) validates them against the
rules of the world, and (4) executes them with true values. A job that misses its deadline still
runs to completion and is counted as a violation.

A decision list names the jobs to run this step (actions ``start`` / ``run``); any other job does
not run. A running job that is left out is *paused*; a later resume adds ``checkpoint_overhead_h``
of extra running time (time and energy), exactly once per pause.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from greenqueue.core.types import (
    CarbonForecast,
    Decision,
    ForecastBundle,
    Job,
    JobView,
    Scheduler,
)

EPS = 1e-9

BundleProvider = Callable[[int, list[JobView]], ForecastBundle]


class InvalidDecision(Exception):
    """The scheduler broke a rule of the world (capacity, pausing a rigid job, ...)."""


@dataclass
class _State:
    job: Job
    remaining_h: float
    elapsed_h: float = 0.0
    started: bool = False
    running: bool = False
    finished: bool = False
    first_start: int | None = None
    finish_step: float | None = None
    energy_kwh: float = 0.0
    emissions_g: float = 0.0
    pauses: int = 0
    profile: list[tuple[int, float]] = field(default_factory=list)


@dataclass(frozen=True)
class JobResult:
    id: str
    arrival: int
    first_start: int | None
    finish_step: float | None
    deadline: float
    energy_kwh: float
    emissions_g: float
    pauses: int
    violated: bool
    wait_h: float
    profile: tuple[tuple[int, float], ...] = ()  # (step, kWh) for re-scoring


@dataclass
class SimResult:
    jobs: list[JobResult]
    dt_h: float
    gpu_busy_steps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    solver_fallbacks: int = 0

    @property
    def total_emissions_g(self) -> float:
        return float(sum(j.emissions_g for j in self.jobs))

    @property
    def total_energy_kwh(self) -> float:
        return float(sum(j.energy_kwh for j in self.jobs))

    @property
    def unfinished(self) -> int:
        return sum(j.finish_step is None for j in self.jobs)

    @property
    def violation_rate(self) -> float:
        return float(np.mean([j.violated for j in self.jobs])) if self.jobs else 0.0

    @property
    def mean_wait_h(self) -> float:
        return float(np.mean([j.wait_h for j in self.jobs])) if self.jobs else 0.0

    @property
    def p95_wait_h(self) -> float:
        return float(np.percentile([j.wait_h for j in self.jobs], 95)) if self.jobs else 0.0

    def cvar_job_carbon(self, level: float = 0.9) -> float:
        """Mean per-job emissions (g) over the worst ``1 - level`` share of jobs."""
        v = np.sort([j.emissions_g for j in self.jobs])
        k = max(1, math.ceil((1 - level) * len(v)))
        return float(v[-k:].mean()) if len(v) else 0.0


def perfect_bundle_provider(true_ci: np.ndarray, horizon: int = 48) -> BundleProvider:
    """Test helper: a bundle with the true CI and no energy estimates (never for evaluation)."""

    def provide(t: int, queue: list[JobView]) -> ForecastBundle:
        seg = true_ci[t : t + horizon].astype(float)
        return ForecastBundle(CarbonForecast(t, seg, seg, seg), {})

    return provide


def simulate(
    jobs: Sequence[Job],
    scheduler: Scheduler,
    true_ci: np.ndarray,
    *,
    capacity_gpus: int,
    dt_h: float = 1.0,
    provider: BundleProvider,
) -> SimResult:
    """Run ``scheduler`` over ``true_ci`` (gCO2/kWh per step) and return the exact accounting."""
    if len({j.id for j in jobs}) != len(jobs):
        raise ValueError("duplicate job ids")
    for j in jobs:
        if j.gpus > capacity_gpus:
            raise ValueError(f"job {j.id} needs {j.gpus} GPUs > capacity {capacity_gpus}")
    states = {j.id: _State(j, j.duration_steps * dt_h) for j in jobs}
    order = sorted(states.values(), key=lambda s: (s.job.arrival, s.job.id))
    busy = np.zeros(len(true_ci))

    for t in range(len(true_ci)):
        if all(s.finished for s in order):
            break
        live = [s for s in order if s.job.arrival <= t and not s.finished]
        running_v = [_view(s) for s in live if s.running]
        queue_v = [_view(s) for s in live if not s.running]
        free = capacity_gpus - sum(v.gpus for v in running_v)
        bundle = provider(t, queue_v + running_v)
        decisions = scheduler.decide(t, queue_v, running_v, bundle, free)
        run_ids = _validate(decisions, states, t, capacity_gpus)

        for s in live:
            if s.job.id in run_ids:
                if s.started and not s.running:
                    s.remaining_h += s.job.checkpoint_overhead_h  # resume after a pause
                if not s.started:
                    s.started, s.first_start = True, t
                run_h = min(dt_h, s.remaining_h)
                kwh = s.job.gpus * s.job.power_kw_per_gpu * run_h
                s.energy_kwh += kwh
                s.profile.append((t, kwh))
                s.emissions_g += kwh * float(true_ci[t])
                s.remaining_h -= run_h
                s.elapsed_h += run_h
                s.running = True
                busy[t] += s.job.gpus * run_h / dt_h
                if s.remaining_h <= EPS:
                    s.finished, s.running = True, False
                    s.finish_step = t + run_h / dt_h
            elif s.running:
                s.running = False
                s.pauses += 1

    results = []
    for s in order:
        j = s.job
        done = s.finish_step is not None
        results.append(
            JobResult(
                id=j.id,
                arrival=j.arrival,
                first_start=s.first_start,
                finish_step=s.finish_step,
                deadline=j.deadline,
                energy_kwh=s.energy_kwh,
                emissions_g=s.emissions_g,
                pauses=s.pauses,
                violated=(not done) or s.finish_step > j.deadline + EPS,  # type: ignore[operator]
                wait_h=((s.first_start - j.arrival) * dt_h)
                if s.first_start is not None
                else math.inf,
                profile=tuple(s.profile),
            )
        )
    return SimResult(results, dt_h, busy)


def _view(s: _State) -> JobView:
    j = s.job
    return JobView(
        id=j.id,
        arrival=j.arrival,
        gpus=j.gpus,
        deadline=j.deadline,
        interruptible=j.interruptible,
        checkpoint_overhead_h=j.checkpoint_overhead_h,
        elapsed_run_h=s.elapsed_h,
        started=s.started,
        running=s.running,
        features=j.features,
    )


def _validate(
    decisions: list[Decision], states: dict[str, _State], t: int, capacity: int
) -> set[str]:
    run_ids: set[str] = set()
    for d in decisions:
        if d.job_id not in states:
            raise InvalidDecision(f"unknown job {d.job_id}")
        if d.action not in ("start", "run"):
            continue
        s = states[d.job_id]
        if s.job.arrival > t or s.finished:
            raise InvalidDecision(f"job {d.job_id} cannot run at step {t}")
        if d.job_id in run_ids:
            raise InvalidDecision(f"job {d.job_id} listed twice")
        run_ids.add(d.job_id)
    if sum(states[i].job.gpus for i in run_ids) > capacity:
        raise InvalidDecision(f"capacity exceeded at step {t}")
    for s in states.values():
        if s.running and not s.job.interruptible and s.job.id not in run_ids:
            raise InvalidDecision(f"rigid job {s.job.id} paused at step {t}")
    return run_ids
