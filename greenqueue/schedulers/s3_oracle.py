"""S3: offline oracle MILP with true carbon, energy and durations (implementation.md §11.1).

Rigid jobs use start variables ``y[j,s]`` (contiguous by construction); interruptible jobs use run
variables ``x[j,t]`` with restart counting and a work constraint that includes checkpoint overhead.
Lateness is allowed but penalised so the model is always feasible; a job finished late is a
violation in the simulator. The horizon is solved in consecutive blocks of ``block_steps`` arrival
steps (the guide's rolling windows); GPUs used by earlier blocks are reserved for later ones.
The plan is replayed through the simulator, which does the exact accounting.

Approximation: when checkpoint overhead is non-zero the MILP charges the partial last step as a
full step, so the plan is near-optimal rather than exactly optimal (error below one overhead
per resume). With zero overhead and whole-step durations the bound is exact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pulp

from greenqueue.core.types import Decision, ForecastBundle, Job, JobView

LATE_PENALTY_G = 1.0e7  # per late job-step, dwarfs any carbon difference


@dataclass
class OracleInfo:
    blocks: int = 0
    non_optimal_blocks: int = 0


def _solver(time_limit_s: float, gap: float) -> pulp.LpSolver:
    return pulp.HiGHS(msg=False, timeLimit=time_limit_s, gapRel=gap)


def _solve_block(
    jobs: Sequence[Job],
    ci: np.ndarray,
    spare: np.ndarray,
    dt_h: float,
    late_max: int,
    time_limit_s: float,
    gap: float,
) -> tuple[dict[str, list[int]], bool]:
    n_steps = len(ci)
    prob = pulp.LpProblem("oracle", pulp.LpMinimize)
    cost_terms: list[pulp.LpAffineExpression] = []
    usage: dict[int, list[pulp.LpAffineExpression]] = {}
    starts: dict[str, dict[int, pulp.LpVariable]] = {}
    runs: dict[str, dict[int, pulp.LpVariable]] = {}

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
                prev = x.get(t - 1, 0)
                prob += z[t] >= x[t] - prev
                cost_terms.append(
                    x[t] * (e_step * float(ci[t]) + (LATE_PENALTY_G if t + 1 > dl else 0.0))
                )
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
            csum = np.concatenate([[0.0], np.cumsum(ci)])
            for s, var in y.items():
                late = max(0, s + d - dl)
                cost_terms.append(
                    var * (e_step * float(csum[s + d] - csum[s]) + LATE_PENALTY_G * late)
                )
                for t in range(s, s + d):
                    usage.setdefault(t, []).append(j.gpus * var)

    prob += pulp.lpSum(cost_terms)
    for t, terms in usage.items():
        prob += pulp.lpSum(terms) <= float(spare[t])
    prob.solve(_solver(time_limit_s, gap))
    status = pulp.LpStatus[prob.status]
    if status in ("Infeasible", "Unbounded", "Undefined"):
        raise RuntimeError(f"oracle block failed with status {status}")

    plan: dict[str, list[int]] = {}
    for jid, y in starts.items():
        s0 = next(s for s, v in y.items() if (v.value() or 0) > 0.5)
        jd = next(j for j in jobs if j.id == jid).duration_steps
        plan[jid] = list(range(s0, s0 + jd))
    for jid, x in runs.items():
        plan[jid] = [t for t, v in x.items() if (v.value() or 0) > 0.5]
    return plan, status == "Optimal"


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
        for allowance in (late_max, 4 * late_max, 16 * late_max):  # widen if backlog is deeper
            try:
                part, optimal = _solve_block(
                    block, true_ci, spare, dt_h, allowance, time_limit_s, gap
                )
                break
            except RuntimeError:
                if allowance == 16 * late_max:
                    raise
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
