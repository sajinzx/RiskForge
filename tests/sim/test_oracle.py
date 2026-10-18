import numpy as np
import pytest

from greenqueue.core.types import Job
from greenqueue.schedulers.baselines import Asap, GreedyWindow, Threshold
from greenqueue.schedulers.s3_oracle import PlanScheduler, solve_oracle
from greenqueue.sim.simulator import simulate
from tests.sim.test_accounting import job, provider_for


def oracle_run(jobs: list[Job], ci: np.ndarray, cap: int = 1):
    plan, info = solve_oracle(jobs, ci, cap, block_steps=24)
    res = simulate(
        jobs, PlanScheduler(plan), ci, capacity_gpus=cap, provider=provider_for(jobs, ci)
    )
    return res, info


def test_oracle_finds_known_optimum_for_rigid_job() -> None:
    ci = np.array([500.0, 500.0, 500.0, 10.0, 10.0, 500.0, 500.0, 500.0])
    res, _ = oracle_run([job("a", dur=2, deadline=8)], ci)
    assert res.total_emissions_g == pytest.approx(20.0)


def test_oracle_splits_interruptible_job_over_two_clean_windows() -> None:
    ci = np.array([10.0, 500.0, 10.0, 500.0, 500.0, 500.0])
    res, _ = oracle_run([job("a", dur=2, deadline=6, interruptible=True)], ci)
    assert res.total_emissions_g == pytest.approx(20.0)
    assert res.jobs[0].pauses == 1


def test_oracle_respects_capacity_by_sharing_the_clean_window() -> None:
    ci = np.array([500.0, 10.0, 10.0, 500.0, 500.0, 500.0])
    jobs = [job("a", dur=1, deadline=6), job("b", dur=1, deadline=6)]
    res, _ = oracle_run(jobs, ci, cap=1)
    assert res.total_emissions_g == pytest.approx(20.0)  # one job per clean step
    assert res.unfinished == 0 and res.violation_rate == 0.0


def test_oracle_with_overhead_never_worse_than_contiguous_and_meets_deadline() -> None:
    """With overhead the MILP charges the partial last step as a full step, so it is only
    near-optimal (documented); it must still beat the best contiguous placement here."""
    ci = np.array([10.0, 500.0, 10.0, 500.0, 500.0, 500.0])
    j = job("a", dur=2, deadline=6, interruptible=True, checkpoint_overhead_h=0.5)
    res, _ = oracle_run([j], ci)
    assert res.total_emissions_g <= 510.0 + 1e-6
    assert not res.jobs[0].violated


@pytest.mark.parametrize("seed", range(4))
def test_no_online_scheduler_beats_the_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ci = rng.uniform(40, 500, 72)
    jobs = []
    for i in range(8):
        a = int(rng.integers(0, 30))
        d = int(rng.integers(1, 5))
        jobs.append(
            job(
                f"j{i}",
                arrival=a,
                dur=d,
                gpus=int(rng.integers(1, 3)),
                power=float(rng.uniform(0.2, 0.4)),
                deadline=a + d + int(rng.integers(3, 20)),
                interruptible=bool(rng.integers(0, 2)),
            )
        )
    orc, _ = oracle_run(jobs, ci, cap=2)
    assert orc.unfinished == 0
    for sched in (Asap(), GreedyWindow(), Threshold(0.3)):
        res = simulate(jobs, sched, ci, capacity_gpus=2, provider=provider_for(jobs, ci))
        # compare only when the online scheduler also met all deadlines (the oracle may trade
        # lateness for nothing, but it never has to)
        if res.violation_rate == 0.0 and orc.violation_rate == 0.0:
            assert orc.total_emissions_g <= res.total_emissions_g + 1e-6


def test_greedy_fallback_is_feasible_and_completes_in_the_simulator() -> None:
    from greenqueue.schedulers.s3_oracle import _greedy_block

    rng = np.random.default_rng(5)
    ci = rng.uniform(40, 500, 96)
    jobs = []
    for i in range(10):
        a = int(rng.integers(0, 30))
        d = int(rng.integers(1, 5))
        jobs.append(
            job(
                f"g{i}", arrival=a, dur=d, gpus=int(rng.integers(1, 3)),
                power=float(rng.uniform(0.2, 0.4)), deadline=a + d + int(rng.integers(8, 30)),
                interruptible=bool(rng.integers(0, 2)), checkpoint_overhead_h=0.5,
            )
        )  # fmt: skip
    plan = _greedy_block(jobs, ci, np.full(len(ci), 3.0), 1.0)
    use = np.zeros(len(ci))
    for j in jobs:
        for t in plan[j.id]:
            use[t] += j.gpus
    assert use.max() <= 3
    res = simulate(jobs, PlanScheduler(plan), ci, capacity_gpus=3, provider=provider_for(jobs, ci))
    assert res.unfinished == 0
    asap = simulate(jobs, Asap(), ci, capacity_gpus=3, provider=provider_for(jobs, ci))
    assert res.total_emissions_g <= asap.total_emissions_g * 1.001  # cheaper than ASAP here
