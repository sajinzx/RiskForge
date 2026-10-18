"""S4: robust conformal model-predictive control (implementation.md §11.1).

Every step it re-solves a MILP over the next ``lookahead`` hours and executes only the decisions
for the current step. Robustness enters through the forecasts it is given:

- carbon: the conformal *upper* bound (or point, or a CVaR-style tail mean) of the forecast;
- duration: the conformal *upper* bound on duration, which is what targets the deadline-miss rate;
- energy: the conformal upper bound on energy (per-step energy = energy bound / duration bound).

Beyond the lookahead window the remaining work of a job is priced at the window's mean carbon
intensity ("neutral terminal price"), so the model only runs a job early when the window is cheaper
than average, and only defers it when its (upper-bound) duration still fits before the deadline.
Deadline and capacity trouble is handled with penalised slack so the MILP is always feasible;
if the solver fails or times out without a solution the step falls back to ``GreedyWindow``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pulp

from greenqueue.core.types import Decision, ForecastBundle, JobView
from greenqueue.schedulers.baselines import GreedyWindow

CiMode = Literal["point", "upper", "cvar"]
LATE_PENALTY = 1.0e6
Z_CVAR_90 = 1.7550  # phi(z_0.9) / 0.1 for a normal distribution


def risk_adjusted_ci(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, mode: CiMode) -> np.ndarray:
    """Carbon intensity used for planning. ``cvar`` assumes a normal error with the 90% interval."""
    if mode == "point":
        return point
    if mode == "upper":
        return hi
    sigma = np.maximum(hi - lo, 0.0) / (2 * 1.645)
    return np.asarray(point + Z_CVAR_90 * sigma)


class RobustMPC:
    name = "S4_robust_mpc"

    def __init__(
        self,
        dt_h: float = 1.0,
        *,
        ci_mode: CiMode = "upper",
        lookahead: int = 48,
        time_limit_s: float = 10.0,
        gap: float = 0.01,
    ) -> None:
        self.dt_h, self.ci_mode, self.lookahead = dt_h, ci_mode, lookahead
        self.time_limit_s, self.gap = time_limit_s, gap
        self.fallback = GreedyWindow(dt_h, use_upper_duration=True)
        self.solves = 0
        self.fallbacks = 0

    # ---- helpers ---------------------------------------------------------------------------
    def _remaining(self, v: JobView, bundle: ForecastBundle) -> int:
        est = bundle.energy[v.id]
        left = max(est.duration_hi_h - v.elapsed_run_h, 0.0)
        if v.started and not v.running:
            left += v.checkpoint_overhead_h
        return max(1, math.ceil(left / self.dt_h - 1e-9))

    def _step_energy(self, v: JobView, bundle: ForecastBundle) -> float:
        est = bundle.energy[v.id]
        return est.energy_hi / max(est.duration_hi_h / self.dt_h, 1e-9)

    # ---- main entry ----------------------------------------------------------------------------
    def decide(
        self,
        t: int,
        queue: list[JobView],
        running: list[JobView],
        bundle: ForecastBundle,
        free_gpus: int,
    ) -> list[Decision]:
        if not queue and not running:
            return []
        capacity = free_gpus + sum(j.gpus for j in running)
        try:
            run_ids = self._solve(t, queue, running, bundle, capacity)
        except _SolverFailure:
            self.fallbacks += 1
            return self.fallback.decide(t, queue, running, bundle, free_gpus)
        return [
            Decision(v.id, "run" if v.started else "start")
            for v in [*running, *queue]
            if v.id in run_ids
        ]

    def _solve(
        self,
        t: int,
        queue: list[JobView],
        running: list[JobView],
        bundle: ForecastBundle,
        capacity: int,
    ) -> set[str]:
        fc = bundle.carbon
        ci = risk_adjusted_ci(fc.point, fc.lo, fc.hi, self.ci_mode)
        horizon = min(self.lookahead, len(ci))
        ci = ci[:horizon]
        pbar = float(np.mean(ci))
        rel = ci - pbar  # carbon relative to the neutral terminal price
        prob = pulp.LpProblem("mpc", pulp.LpMinimize)
        usage: list[list[pulp.LpAffineExpression]] = [[] for _ in range(horizon)]
        cost: list[pulp.LpAffineExpression] = []
        now_vars: dict[str, pulp.LpAffineExpression | pulp.LpVariable | float] = {}

        for v in [*running, *queue]:
            r = self._remaining(v, bundle)
            e = self._step_energy(v, bundle)
            # a job that can no longer finish on time is simply run as soon as possible
            dl = max(math.floor(v.deadline + 1e-9), t + r)
            if not v.interruptible:
                if v.running:  # rigid and already running: committed until done
                    for i in range(min(r, horizon)):
                        usage[i].append(v.gpus)
                    now_vars[v.id] = 1.0
                    continue
                starts = list(range(t, t + horizon))
                options: dict[int, pulp.LpVariable] = {}
                for s in starts:
                    var = pulp.LpVariable(f"y_{v.id}_{s}", cat="Binary")
                    options[s] = var
                    steps = range(s - t, min(s - t + r, horizon))
                    late_steps = max(0, s + r - dl)
                    cost.append(
                        var * (e * float(sum(rel[i] for i in steps)) + LATE_PENALTY * late_steps)
                    )
                    for i in steps:
                        usage[i].append(v.gpus * var)
                can_defer = dl - (t + horizon) >= r
                defer = pulp.LpVariable(f"d_{v.id}", cat="Binary")
                if not can_defer:
                    cost.append(defer * LATE_PENALTY * (horizon + r))  # would break the deadline
                prob += pulp.lpSum(options.values()) + defer == 1
                now_vars[v.id] = options.get(t, 0.0)
            else:
                hi_step = min(horizon, max(dl - t, 0))
                x = {i: pulp.LpVariable(f"x_{v.id}_{i}", cat="Binary") for i in range(hi_step)}
                z = {i: pulp.LpVariable(f"z_{v.id}_{i}", cat="Binary") for i in range(hi_step)}
                prev0 = 1.0 if v.running else 0.0
                o = v.checkpoint_overhead_h / self.dt_h
                for i in range(hi_step):
                    prev = x[i - 1] if i > 0 else prev0
                    prob += z[i] >= x[i] - prev
                    cost.append(x[i] * (e * float(rel[i])))
                    usage[i].append(v.gpus * x[i])
                after = max(0, dl - (t + horizon))
                w_min = max(0, r - after)
                w_min = min(w_min, hi_step)
                free_start = 0.0 if v.started else o
                late = pulp.LpVariable(f"late_{v.id}", lowBound=0)
                cost.append(late * LATE_PENALTY)
                if w_min > 0 and x:
                    prob += (
                        pulp.lpSum(x.values()) + late
                        >= w_min + o * pulp.lpSum(z.values()) - free_start
                    )
                elif w_min > 0:
                    prob += late >= w_min
                if x:
                    prob += pulp.lpSum(x.values()) <= r + o * pulp.lpSum(z.values())
                now_vars[v.id] = x.get(0, 0.0)

        for i in range(horizon):
            if usage[i]:
                prob += pulp.lpSum(usage[i]) <= capacity
        prob += pulp.lpSum(cost) if cost else pulp.lpSum([0])
        self.solves += 1
        prob.solve(pulp.HiGHS(msg=False, timeLimit=self.time_limit_s, gapRel=self.gap))
        status = pulp.LpStatus[prob.status]
        if status in ("Infeasible", "Unbounded", "Undefined"):
            raise _SolverFailure(status)
        out: set[str] = set()
        for jid, expr in now_vars.items():
            val = expr if isinstance(expr, float) else pulp.value(expr)
            if val is None:
                raise _SolverFailure("no value")
            if val > 0.5:
                out.add(jid)
        return out


class _SolverFailure(Exception):
    pass
