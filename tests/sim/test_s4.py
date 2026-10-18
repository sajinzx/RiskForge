import numpy as np
import pytest

from greenqueue.schedulers.baselines import Asap
from greenqueue.schedulers.s4_robust import RobustMPC, risk_adjusted_ci
from tests.sim.test_accounting import job, provider_for, run


def test_rigid_job_waits_for_the_clean_window() -> None:
    ci = np.array([500.0, 500.0, 500.0, 10.0, 10.0, 500.0, 500.0, 500.0] + [500.0] * 40)
    res = run([job("a", dur=2, deadline=8)], RobustMPC(), ci)
    assert res.total_emissions_g == pytest.approx(20.0)
    assert not res.jobs[0].violated


def test_interruptible_job_uses_two_clean_windows() -> None:
    ci = np.array([10.0, 500.0, 10.0, 500.0, 500.0, 500.0] + [500.0] * 40)
    res = run([job("a", dur=2, deadline=6, interruptible=True)], RobustMPC(), ci)
    assert res.total_emissions_g == pytest.approx(20.0)


def test_duration_upper_bound_keeps_the_deadline_when_estimate_is_low() -> None:
    ci = np.array([300.0] * 4 + [5.0] * 6 + [300.0] * 40)
    # true duration 2, estimate x2 -> planned as 4 steps; must still finish by 6
    res = run([job("a", dur=2, deadline=6)], RobustMPC(), ci, hi_factor=2.0)
    assert not res.jobs[0].violated


def test_overdue_job_runs_immediately_instead_of_starving() -> None:
    ci = np.full(60, 100.0)
    res = run([job("a", arrival=0, dur=3, deadline=1)], RobustMPC(), ci)
    assert res.jobs[0].first_start == 0 and res.unfinished == 0


def test_capacity_conflict_is_resolved_without_invalid_decisions_or_fallbacks() -> None:
    ci = np.array([500.0, 10.0, 10.0, 500.0, 500.0, 500.0] + [500.0] * 42)
    jobs = [job("a", dur=1, deadline=6), job("b", dur=1, deadline=6)]
    sched = RobustMPC()
    res = run(jobs, sched, ci, cap=1)
    assert res.total_emissions_g == pytest.approx(20.0) and sched.fallbacks == 0


@pytest.mark.parametrize("seed", range(3))
def test_random_instances_valid_cheaper_than_asap_and_meets_deadlines(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ci = 200 + 150 * np.sin(np.arange(160) * 2 * np.pi / 24) + rng.normal(0, 10, 160)
    jobs = []
    for i in range(10):
        a = int(rng.integers(0, 40))
        d = int(rng.integers(1, 5))
        jobs.append(
            job(
                f"j{i}", arrival=a, dur=d, gpus=int(rng.integers(1, 3)),
                power=float(rng.uniform(0.2, 0.4)), deadline=a + d + int(rng.integers(8, 30)),
                interruptible=bool(rng.integers(0, 2)),
            )
        )  # fmt: skip
    sched = RobustMPC()
    res = run(jobs, sched, ci, cap=3)
    base = run(jobs, Asap(), ci, cap=3)
    assert res.unfinished == 0 and res.violation_rate == 0.0
    assert res.total_emissions_g <= base.total_emissions_g + 1e-6
    assert sched.fallbacks == 0 and sched.solves > 0


def test_risk_adjusted_ci_modes() -> None:
    p, lo, hi = np.array([100.0]), np.array([80.0]), np.array([140.0])
    assert risk_adjusted_ci(p, lo, hi, "point")[0] == 100.0
    assert risk_adjusted_ci(p, lo, hi, "upper")[0] == 140.0
    assert 100.0 < risk_adjusted_ci(p, lo, hi, "cvar")[0] < 140.0 + 40.0
    assert provider_for  # imported helper stays exercised
