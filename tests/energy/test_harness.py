import json
import time
from pathlib import Path

import numpy as np
import pytest

from greenqueue.energy.harness.power import (
    PowerLogger,
    PowerTrace,
    integrate_energy_kwh,
    measure_idle_w,
)
from greenqueue.energy.harness.sweep import RunConfig, default_grid, done_keys, run_sweep


def test_integration_constant_power_hand_computed() -> None:
    t = np.linspace(0, 10, 101)
    tr = PowerTrace(t, np.full(101, 100.0), np.zeros(101))
    assert integrate_energy_kwh(tr, idle_w=20.0) == pytest.approx(80.0 * 10 / 3.6e6)
    assert integrate_energy_kwh(tr, idle_w=20.0, warmup_s=2.0) == pytest.approx(80.0 * 8 / 3.6e6)


def test_integration_trapezoid_ramp() -> None:
    t = np.array([0.0, 1.0, 2.0])
    tr = PowerTrace(t, np.array([0.0, 100.0, 0.0]), np.zeros(3))
    assert integrate_energy_kwh(tr, 0.0) == pytest.approx(100.0 / 3.6e6)  # triangle area


def test_logger_collects_samples_at_the_requested_rate() -> None:
    with PowerLogger(lambda: (55.0, 10.0), interval_s=0.01) as log:
        time.sleep(0.3)
    tr = log.trace()
    assert 15 < len(tr.power_w) < 40
    assert np.all(tr.power_w == 55.0) and np.all(np.diff(tr.t_s) > 0)
    assert measure_idle_w(lambda: (42.0, 0.0), 0.1, 0.01) == pytest.approx(42.0)


def test_default_grid_matches_the_guide() -> None:
    grid = default_grid()
    by = {}
    for c in grid:
        by.setdefault(c.workload, []).append(c)
    assert len(by["resnet18"]) == 8 and len(by["vit_b16"]) == 3
    assert len(by["bert_base"]) == 9 and len(by["llm_infer"]) == 32
    assert len({c.key() for c in grid}) == len(grid)


def test_sweep_is_resumable_and_writes_expected_fields(tmp_path: Path) -> None:
    grid = [RunConfig("resnet18", "resnet18", "cnn", b, 0, "fp32") for b in (32, 64)]

    def runner(cfg: RunConfig):  # type: ignore[no-untyped-def]
        def step() -> None:
            time.sleep(0.01)

        return step, 1e9 * cfg.batch

    kw = dict(repeats=2, idle_s=0.1, warmup_s=0.05, min_run_s=0.3, interval_s=0.01)
    info = {"gpu_name": "FakeGPU", "driver": "0", "tdp_w": 100.0, "cuda": "x", "torch": "y"}
    sampler = lambda: (60.0, 50.0)  # noqa: E731
    n1 = run_sweep(grid[:1], runner, tmp_path, sampler, info, **kw)  # type: ignore[arg-type]
    assert n1 == 2
    n2 = run_sweep(grid, runner, tmp_path, sampler, info, **kw)  # type: ignore[arg-type]
    assert n2 == 2  # only the second config was left
    assert run_sweep(grid, runner, tmp_path, sampler, info, **kw) == 0  # type: ignore[arg-type]
    rows = [json.loads(x) for x in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and len(done_keys(tmp_path / "runs.jsonl")) == 4
    r = rows[0]
    for f in (
        "energy_per_step_kwh",
        "step_time_s",
        "flops_per_step",
        "idle_w",
        "gpu_name",
        "batch",
    ):
        assert f in r
    assert r["idle_w"] == pytest.approx(60.0)
    assert r["energy_per_step_kwh"] == pytest.approx(
        0.0, abs=1e-9
    )  # power == idle -> no energy above idle
    assert len(list((tmp_path / "traces").glob("*.npz"))) == 4


def test_quick_grid_covers_every_family_and_setting_once() -> None:
    from greenqueue.energy.harness.sweep import default_grid, quick_grid

    g = quick_grid()
    assert {c.family for c in g} == {"cnn", "vit", "bert", "llm"}
    assert len({c.key() for c in g}) == len(g)
    llm = {dict(c.extra)["prompt"] * 10000 + dict(c.extra)["gen"] for c in g if c.family == "llm"}
    assert len(llm) == 4
    assert {c.key() for c in g} <= {c.key() for c in default_grid()} | {
        c.key() for c in g if c.workload != "llm_infer"
    }
