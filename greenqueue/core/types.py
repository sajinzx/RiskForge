"""Core types shared by the simulator and schedulers (implementation.md §4.3).

Time is a step index ``t`` of length ``dt_h`` hours. The *truth* lives in :class:`Job`, which only
the simulator and the offline oracle may see. Online schedulers get :class:`JobView` plus a
:class:`ForecastBundle`, so they cannot read true durations, energies or carbon intensity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class Job:
    """A job with its true attributes (simulator-private)."""

    id: str
    arrival: int  # first step at which the job may run
    duration_steps: int  # true duration, in whole steps
    gpus: int
    power_kw_per_gpu: float  # true mean power per GPU while running
    deadline: float  # step by which the job must have finished (finish time <= deadline)
    interruptible: bool = False
    checkpoint_overhead_h: float = 0.0  # extra running time added at every resume
    features: dict[str, float] = field(default_factory=dict)

    def true_energy_kwh(self, dt_h: float) -> float:
        return self.gpus * self.power_kw_per_gpu * self.duration_steps * dt_h


@dataclass(frozen=True)
class JobView:
    """What an online scheduler may know about a job (no truth)."""

    id: str
    arrival: int
    gpus: int
    deadline: float
    interruptible: bool
    checkpoint_overhead_h: float
    elapsed_run_h: float  # hours executed so far (including overhead)
    started: bool
    running: bool  # ran in the previous step
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CarbonForecast:
    """Forecast of carbon intensity (gCO2/kWh) for steps ``start_step .. start_step+len-1``."""

    start_step: int
    point: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    def __len__(self) -> int:
        return len(self.point)


@dataclass(frozen=True)
class EnergyEstimate:
    job_id: str
    energy_point: float  # kWh
    energy_lo: float
    energy_hi: float
    duration_point_h: float
    duration_hi_h: float


@dataclass(frozen=True)
class ForecastBundle:
    carbon: CarbonForecast
    energy: dict[str, EnergyEstimate]


Action = Literal["start", "run", "pause", "wait"]


@dataclass(frozen=True)
class Decision:
    job_id: str
    action: Action


class CarbonForecaster(Protocol):
    def predict_at(self, step: int) -> CarbonForecast: ...


class EnergyEstimator(Protocol):
    def estimate(self, jobs: list[JobView]) -> dict[str, EnergyEstimate]: ...


class Scheduler(Protocol):
    def decide(
        self,
        t: int,
        queue: list[JobView],
        running: list[JobView],
        bundle: ForecastBundle,
        free_gpus: int,
    ) -> list[Decision]: ...
