"""GPU power logging and energy integration (implementation.md §7.5, §9.1).

The sampler is injected so everything here is testable without a GPU. ``NvmlSampler`` is the real
one and imports ``pynvml`` lazily, so this module imports on any machine.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Sampler = Callable[[], tuple[float, float]]  # () -> (power in watts, utilisation in percent)


class NvmlSampler:
    """Power (W) and utilisation (%) of one GPU through NVML."""

    def __init__(self, device_index: int = 0) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def __call__(self) -> tuple[float, float]:
        power_w = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
        return power_w, util

    def info(self) -> dict[str, str | float]:
        n = self._nvml
        name = n.nvmlDeviceGetName(self._h)
        return {
            "gpu_name": name.decode() if isinstance(name, bytes) else str(name),
            "driver": str(n.nvmlSystemGetDriverVersion()),
            "tdp_w": n.nvmlDeviceGetPowerManagementLimit(self._h) / 1000.0,
        }


@dataclass
class PowerTrace:
    t_s: np.ndarray  # seconds since the start of logging
    power_w: np.ndarray
    util_pct: np.ndarray


class PowerLogger:
    """Background thread sampling ``sampler`` every ``interval_s`` seconds."""

    def __init__(self, sampler: Sampler, interval_s: float = 0.1) -> None:
        self.sampler, self.interval_s = sampler, interval_s
        self._t: list[float] = []
        self._p: list[float] = []
        self._u: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            p, u = self.sampler()
            self._t.append(time.perf_counter() - t0)
            self._p.append(p)
            self._u.append(u)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PowerLogger:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def trace(self) -> PowerTrace:
        return PowerTrace(np.array(self._t), np.array(self._p), np.array(self._u))


def measure_idle_w(sampler: Sampler, seconds: float, interval_s: float = 0.1) -> float:
    """Mean power with no workload running."""
    with PowerLogger(sampler, interval_s) as log:
        time.sleep(seconds)
    tr = log.trace()
    if len(tr.power_w) == 0:
        raise RuntimeError("no power samples collected")
    return float(tr.power_w.mean())


def integrate_energy_kwh(trace: PowerTrace, idle_w: float, warmup_s: float = 0.0) -> float:
    """Trapezoidal integral of (power - idle) after discarding the first ``warmup_s`` seconds."""
    keep = trace.t_s >= warmup_s
    t, p = trace.t_s[keep], trace.power_w[keep] - idle_w
    if len(t) < 2:
        raise ValueError("not enough samples after warm-up")
    joules = float(np.sum((p[1:] + p[:-1]) / 2.0 * np.diff(t)))
    return joules / 3.6e6
