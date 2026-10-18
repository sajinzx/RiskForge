"""S3: offline oracle MILP with true carbon, energy and durations (implementation.md §11.1).

Rigid jobs use start variables ``y[j,s]`` (contiguous by construction); interruptible jobs use run
variables ``x[j,t]`` with restart counting and a work constraint that includes checkpoint overhead.
Lateness is allowed only when unavoidable: each block first looks for a plan with no late job-steps,
else finds the minimum lateness, and then minimises carbon with that lateness fixed (lexicographic,
so big-M weights never distort the solver gap). A job finished late is a violation in the simulator.

The horizon is solved in consecutive blocks of ``block_steps`` arrival steps (the guide's rolling
windows); GPUs used by earlier blocks are reserved for later ones.
The plan is replayed through the simulator, which does the exact accounting.

Approximation: when checkpoint overhead is non-zero the MILP charges the partial last step as a
full step, so the plan is near-optimal rather than exactly optimal (error below one overhead
per resume). With zero overhead and whole-step durations the bound is exact.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pulp

from greenqueue.core.types import Decision, ForecastBundle, Job, JobView


@dataclass
class OracleInfo:
    blocks: int = 0
    non_optimal_blocks: int = 0
    heuristic_blocks: int = 0  # blocks where the MILP returned nothing and a greedy plan was used


def _solver(time_limit_s: float, gap: float) -> pulp.LpSolver:
    return pulp.HiGHS(msg=False, timeLimit=time_limit_s, gapRel=gap)


def _build(
    jobs: Sequence[Job],
    ci: np.ndarray,
    spare: np.ndarray,
    dt_h: float,
    late_max: int,
    mode: str,
    late_bound: float,
) -> tuple[
    pulp.LpProblem, dict[str, dict[int, pulp.LpVariable]], dict[str, dict[int, pulp.LpVariable]]
]:
    """Block model. ``mode='late'`` minimises late steps; ``'carbon'`` minimises carbon with the
    number of late job-steps capped at ``late_bound`` (lexicographic: lateness first)."""
    n_steps = len(ci)
    prob = pulp.LpProblem("oracle", pulp.LpMinimize)
    carbon: list[pulp.LpAffineExpression] = []
    late: list[pulp.LpAffineExpression] = []
    usage: dict[int, list[pulp.LpAffineExpression]] = {}
    starts: dict[str, dict[int, pulp.LpVariable]] = {}
    runs: dict[str, dict[int, pulp.LpVariable]] = {}
    csum = np.concatenate([[0.0], np.cumsum(ci)])

    for j in jobs:
        d = j.duration_steps
        dl = math.floor(j.deadline + 1e-9)
        e_step = j.gpus * j.power_kw_per_gpu * dt_h
        if j.interruptible:
            lo, hi = j.arrival, min(n_steps - 1, dl + late_max - 1)
            if hi < lo:
                raise ValueError(f"job {j.id} has no feasible steps inside the horizon")
            x = {t: pulp.LpVariable(f"x_{j.id}_{t}", cat="Binary") for t in range(lo, hi + 1)}
            z = {t: pulp.LpVariable(f"z_{j.id}_{t}", cat="Binary") for t in range(lo, hi + 1)}
            runs[j.id] = x
            o = j.checkpoint_overhead_h / dt_h
            for t in range(lo, hi + 1):
                prob += z[t] >= x[t] - x.get(t - 1, 0)
                carbon.append(x[t] * (e_step * float(ci[t])))
                if t + 1 > dl:
                    late.append(x[t] * 1.0)
                usage.setdefault(t, []).append(j.gpus * x[t])
            prob += pulp.lpSum(x.values()) >= d + o * (pulp.lpSum(z.values()) - 1)
        else:
            last = min(n_steps - d, dl - d + late_max)
            if last < j.arrival:
                raise ValueError(f"job {j.id} has no feasible start inside the horizon")
            y = {
                s: pulp.LpVariable(f"y_{j.id}_{s}", cat="Binary")
                for s in range(j.arrival, last + 1)
            }
            starts[j.id] = y
            prob += pulp.lpSum(y.values()) == 1
            for s, var in y.items():
                carbon.append(var * (e_step * float(csum[s + d] - csum[s])))
                if s + d > dl:
                    late.append(var * float(s + d - dl))
                for t in range(s, s + d):
                    usage.setdefault(t, []).append(j.gpus * var)

    for t, terms in usage.items():
        prob += pulp.lpSum(terms) <= float(spare[t])
    if mode == "late":
        prob += pulp.lpSum(late) if late else pulp.lpSum([0])
    else:
        prob += pulp.lpSum(carbon)
        if late:
            prob += pulp.lpSum(late) <= late_bound
    return prob, starts, runs


def _extract(
    jobs: Sequence[Job],
    starts: dict[str, dict[int, pulp.LpVariable]],
    runs: dict[str, dict[int, pulp.LpVariable]],
) -> dict[str, list[int]]:
    dur = {j.id: j.duration_steps for j in jobs}
    plan: dict[str, list[int]] = {}
    for jid, y in starts.items():
        chosen = [s for s, v in y.items() if (v.value() or 0) > 0.5]
        if not chosen:
            raise RuntimeError(f"solver returned no start for job {jid}")
        plan[jid] = list(range(chosen[0], chosen[0] + dur[jid]))
    for jid, x in runs.items():
        steps = [t for t, v in x.items() if (v.value() or 0) > 0.5]
        if not steps:
            raise RuntimeError(f"solver returned no run steps for job {jid}")
        plan[jid] = steps
    return plan


def _solve_block(
    jobs: Sequence[Job],
    ci: np.ndarray,
    spare: np.ndarray,
    dt_h: float,
    late_max: int,
    time_limit_s: float,
    gap: float,
) -> tuple[dict[str, list[int]], bool]:
    """Lexicographic solve: no lateness if possible, else minimum lateness; then minimum carbon."""
    t0 = time.perf_counter()
    prob, starts, runs = _build(jobs, ci, spare, dt_h, late_max, "carbon", 0.0)
    prob.solve(_solver(time_limit_s, gap))
    status = pulp.LpStatus[prob.status]
    if status == "Infeasible":  # some lateness is unavoidable: find the minimum, then re-solve
        p1, _, _ = _build(jobs, ci, spare, dt_h, late_max, "late", 0.0)
        p1.solve(_solver(time_limit_s / 2, 1e-6))
        if pulp.LpStatus[p1.status] in ("Infeasible", "Unbounded", "Undefined"):
            raise RuntimeError(f"oracle lateness model failed: {pulp.LpStatus[p1.status]}")
        bound = math.ceil(float(pulp.value(p1.objective) or 0.0) - 1e-6)
        prob, starts, runs = _build(jobs, ci, spare, dt_h, late_max, "carbon", float(bound))
        prob.solve(_solver(time_limit_s, gap))
        status = pulp.LpStatus[prob.status]
    if status in ("Infeasible", "Unbounded", "Undefined"):
        raise RuntimeError(f"oracle block failed with status {status}")
    plan = _extract(jobs, starts, runs)
    hit_limit = (time.perf_counter() - t0) >= 0.95 * time_limit_s
    return plan, status == "Optimal" and not hit_limit


def _greedy_block(
    jobs: Sequence[Job], ci: np.ndarray, spare: np.ndarray, dt_h: float
) -> dict[str, list[int]]:
    """Deterministic fallback plan when the MILP returns no solution (not optimal).

    Jobs are placed in deadline order. Rigid jobs take the cheapest contiguous window that
    fits the remaining capacity and meets the deadline (else the earliest window that fits);
    interruptible jobs take the cheapest free steps before the deadline (else the earliest free
    steps), plus extra steps for checkpoint overhead. This is only a feasible incumbent.
    """
    n = len(ci)
    cap = spare.copy()
    plan: dict[str, list[int]] = {}
    for j in sorted(jobs, key=lambda j: (j.deadline, j.arrival, j.id)):
        d, g = j.duration_steps, j.gpus
        dl = math.floor(j.deadline + 1e-9)
        free = cap >= g
        if not j.interruptible:
            best, best_cost, first = None, math.inf, None
            for s in range(j.arrival, n - d + 1):
                if not free[s : s + d].all():
                    continue
                if first is None:
                    first = s
                if s + d <= dl:
                    cost = float(ci[s : s + d].sum())
                    if cost < best_cost:
                        best, best_cost = s, cost
            start = best if best is not None else first
            if start is None:
                raise RuntimeError(f"greedy fallback found no window for job {j.id}")
            steps = list(range(start, start + d))
        else:
            o = j.checkpoint_overhead_h / dt_h
            window = [t for t in range(j.arrival, min(dl, n)) if free[t]]
            window.sort(key=lambda t: (float(ci[t]), t))
            later = [t for t in range(max(j.arrival, dl), n) if free[t]]
            pool = window + later
            need = d
            while True:
                chosen = sorted(pool[:need])
                runs = 1 + sum(1 for a, b in zip(chosen, chosen[1:], strict=False) if b != a + 1)
                want = d + math.ceil(o * (runs - 1) - 1e-9)
                if want <= need or need >= len(pool):
                    break
                need = want
            if len(pool) < need:
                raise RuntimeError(f"greedy fallback found no steps for job {j.id}")
            steps = chosen
        for t in steps:
            cap[t] -= g
        plan[j.id] = steps
    return plan


def solve_oracle(
    jobs: Sequence[Job],
    true_ci: np.ndarray,
    capacity_gpus: int,
    dt_h: float = 1.0,
    *,
    block_steps: int = 168,
    late_max: int = 24,
    time_limit_s: float = 60.0,
    gap: float = 0.01,
) -> tuple[dict[str, list[int]], OracleInfo]:
    """Plan every job with perfect information; returns ``({job id: steps to run}, info)``."""
    info = OracleInfo()
    spare = np.full(len(true_ci), float(capacity_gpus))
    plan: dict[str, list[int]] = {}
    by_block: dict[int, list[Job]] = {}
    for j in sorted(jobs, key=lambda j: (j.arrival, j.id)):
        by_block.setdefault(j.arrival // block_steps, []).append(j)
    for b in sorted(by_block):
        block = by_block[b]
        part: dict[str, list[int]] = {}
        optimal = False
        try:
            for allowance in (late_max, 4 * late_max, 16 * late_max):  # widen if backlog deeper
                try:
                    part, optimal = _solve_block(
                        block, true_ci, spare, dt_h, allowance, time_limit_s, gap
                    )
                    break
                except RuntimeError:
                    if allowance == 16 * late_max:
                        raise
        except RuntimeError:
            part = _greedy_block(block, true_ci, spare, dt_h)
            optimal = False
            info.heuristic_blocks += 1
        info.blocks += 1
        info.non_optimal_blocks += 0 if optimal else 1
        gpus = {j.id: j.gpus for j in block}
        for jid, steps in part.items():
            for t in steps:
                spare[t] -= gpus[jid]
        plan.update(part)
    return plan, info


class PlanScheduler:
    """Replays a precomputed plan (used for the oracle)."""

    name = "S3_oracle"

    def __init__(self, plan: dict[str, list[int]]) -> None:
        self.plan = {k: set(v) for k, v in plan.items()}

    def decide(
        self,
        t: int,
        queue: list[JobView],
        running: list[JobView],
        bundle: ForecastBundle,
        free_gpus: int,
    ) -> list[Decision]:
        return [
            Decision(v.id, "run" if v.started else "start")
            for v in [*running, *queue]
            if t in self.plan[v.id]
        ]
