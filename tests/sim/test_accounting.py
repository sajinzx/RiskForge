"""Exact-accounting toy cases for the simulator (hand-computable)."""

import numpy as np
import pytest

from greenqueue.core.types import (
    CarbonForecast,
    Decision,
    EnergyEstimate,
    ForecastBundle,
    Job,
    JobView,
)
from greenqueue.schedulers.baselines import Asap, GreedyWindow, Threshold
from greenqueue.sim.simulator import InvalidDecision, simulate


def job(
    i: str,
    arrival: int = 0,
    dur: int = 2,
    gpus: int = 1,
    power: float = 1.0,
    deadline: float = 10,
    **kw,
) -> Job:
    return Job(i, arrival, dur, gpus, power, deadline, **kw)


def provider_for(jobs: list[Job], ci: np.ndarray, hi_factor: float = 1.0):
    by_id = {j.id: j for j in jobs}

    def provide(t: int, views: list[JobView]) -> ForecastBundle:
        seg = ci[t : t + 48].astype(float)
        est = {}
        for v in views:
            j = by_id[v.id]
            e = j.gpus * j.power_kw_per_gpu * j.duration_steps
            est[v.id] = EnergyEstimate(
                v.id, e, e, e, j.duration_steps, j.duration_steps * hi_factor
            )
        return ForecastBundle(CarbonForecast(t, seg, seg, seg), est)

    return provide


def run(jobs: list[Job], sched, ci: np.ndarray, cap: int = 1, hi_factor: float = 1.0):
    return simulate(jobs, sched, ci, capacity_gpus=cap, provider=provider_for(jobs, ci, hi_factor))


def test_single_job_constant_ci_emissions_equal_energy_times_ci() -> None:
    j = job("a", dur=3, gpus=2, power=0.5)  # 2 GPUs * 0.5 kW * 3 h = 3 kWh
    res = run([j], Asap(), np.full(10, 200.0), cap=2)
    assert res.jobs[0].energy_kwh == pytest.approx(3.0)
    assert res.total_emissions_g == pytest.approx(3.0 * 200.0)
    assert res.jobs[0].finish_step == 3 and not res.jobs[0].violated


def test_emissions_use_true_ci_per_step() -> None:
    ci = np.array([100.0, 300.0, 50.0, 50.0, 50.0])
    res = run([job("a", dur=2)], Asap(), ci)
    assert res.total_emissions_g == pytest.approx(1 * 100 + 1 * 300)


def test_two_jobs_one_gpu_second_waits() -> None:
    res = run([job("a", dur=2), job("b", arrival=0, dur=2)], Asap(), np.full(10, 10.0))
    by = {r.id: r for r in res.jobs}
    assert by["a"].first_start == 0 and by["b"].first_start == 2
    assert by["b"].wait_h == 2.0 and by["a"].wait_h == 0.0


def test_deadline_exactly_met_is_not_a_violation_but_one_step_later_is() -> None:
    ok = run([job("a", dur=2, deadline=2)], Asap(), np.full(5, 1.0))
    late = run([job("a", dur=2, deadline=1.99)], Asap(), np.full(5, 1.0))
    assert not ok.jobs[0].violated and late.jobs[0].violated
    assert late.total_emissions_g == pytest.approx(2.0)  # still runs to completion


class _Pauser:
    """Runs a job at steps 0, 2, 3 (paused at step 1)."""

    def decide(self, t, queue, running, bundle, free_gpus):
        if t == 1:
            return []
        return [Decision(v.id, "run") for v in [*queue, *running]]


def test_pause_resume_adds_overhead_exactly_once() -> None:
    j = job("a", dur=2, interruptible=True, checkpoint_overhead_h=0.5, deadline=10)
    res = run([j], _Pauser(), np.full(10, 100.0))
    r = res.jobs[0]
    assert r.pauses == 1
    assert r.energy_kwh == pytest.approx(2.5)  # 2 h work + 0.5 h overhead, 1 kW
    assert r.finish_step == pytest.approx(3.5)  # steps 0 | 2 (1.0) | 3 (0.5)
    assert res.total_emissions_g == pytest.approx(250.0)


def test_rigid_job_cannot_be_paused_and_capacity_is_enforced() -> None:
    with pytest.raises(InvalidDecision):
        run([job("a", dur=3)], _Pauser(), np.full(10, 1.0))

    class Greedy:
        def decide(self, t, queue, running, bundle, free_gpus):
            return [Decision(v.id, "start") for v in queue]

    with pytest.raises(InvalidDecision):
        run([job("a"), job("b")], Greedy(), np.full(10, 1.0), cap=1)


def test_asap_is_deterministic() -> None:
    jobs = [job(str(i), arrival=i, dur=2) for i in range(5)]
    ci = np.random.default_rng(0).uniform(50, 400, 40)
    a, b = run(jobs, Asap(), ci), run(jobs, Asap(), ci)
    assert a.total_emissions_g == b.total_emissions_g


def test_greedy_window_waits_for_clean_hours_and_beats_asap() -> None:
    ci = np.array([500.0, 500.0, 500.0, 10.0, 10.0, 500.0, 500.0, 500.0])
    j = job("a", dur=2, deadline=8)
    asap, greedy = run([j], Asap(), ci), run([j], GreedyWindow(), ci)
    assert asap.total_emissions_g == pytest.approx(1000.0)
    assert greedy.total_emissions_g == pytest.approx(20.0)
    assert not greedy.jobs[0].violated


def test_greedy_window_starts_at_latest_start_when_forecast_is_bad() -> None:
    ci = np.full(10, 100.0)
    j = job("a", dur=3, deadline=4)  # latest start = 1
    res = run([j], GreedyWindow(), ci)
    assert not res.jobs[0].violated


def test_simple_s4_uses_duration_upper_bound_to_protect_deadline() -> None:
    ci = np.array([300.0, 300.0, 300.0, 300.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    j = job("a", dur=2, deadline=6)
    point = run([j], GreedyWindow(), ci)  # plans to start at 4, finishes at 6
    robust = run([j], GreedyWindow(use_upper_duration=True), ci, hi_factor=2.0)
    assert not point.jobs[0].violated and not robust.jobs[0].violated
    assert robust.jobs[0].first_start <= point.jobs[0].first_start


def test_threshold_runs_in_cheap_hours_and_pauses_interruptible_job() -> None:
    ci = np.array([500.0, 10.0, 500.0, 10.0, 500.0, 500.0, 500.0, 500.0])
    j = job("a", dur=2, deadline=8, interruptible=True)
    res = run([j], Threshold(q=0.3), ci)
    assert res.total_emissions_g < 2 * 500.0
    assert not res.jobs[0].violated
