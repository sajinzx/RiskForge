from pathlib import Path

import numpy as np
import pandas as pd

from greenqueue.sim.traces import WEEK_S, load_alibaba_tasks, load_config, synthesize_jobs

CFG = load_config(Path(__file__).parents[2] / "configs" / "sim.yaml")


def _tasks(n_per_week: int = 400, weeks: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(weeks):
        for _ in range(n_per_week):
            rows.append(
                {
                    "job_name": f"j{len(rows)}",
                    "task_name": "t",
                    "gpus": int(rng.integers(1, 5)),
                    "duration_s": float(rng.uniform(900, 20 * 3600)),
                    "wall_s": w * WEEK_S + float(rng.uniform(0, WEEK_S)),
                    "trace_week": w,
                }
            )
    df = pd.DataFrame(rows)
    df["week_s"] = df["wall_s"] % WEEK_S
    return df


def test_loader_filters_and_converts(tmp_path: Path) -> None:
    csv = tmp_path / "t.csv"
    csv.write_text(
        "a,tf,2.0,Terminated,1000.0,8200.0,600,30,50,MISC\n"  # 2 h, 2 inst * 0.5 = 1 GPU
        "b,tf,1.0,Failed,1000.0,8200.0,600,30,100,V100\n"  # failed: dropped
        "c,tf,1.0,Terminated,1000.0,1300.0,600,30,100,V100\n"  # 5 min: dropped
        "d,tf,1.0,Terminated,1000.0,8200.0,600,30,0,V100\n"  # no GPU: dropped
        "e,tf,3.0,Terminated,5000.0,9200.0,600,30,100,V100\n"  # 3 GPUs, 70 min
    )
    df = load_alibaba_tasks(csv, 10 / 60, 72)
    assert list(df["job_name"]) == ["a", "e"]
    assert list(df["gpus"]) == [1, 3]
    assert df["week_s"].between(0, WEEK_S).all()


def test_synthesis_is_deterministic_and_respects_spec() -> None:
    cfg = {**CFG, "jobs_per_week": 100}
    a = synthesize_jobs(_tasks(), cfg, n_weeks=4, seed=3)
    b = synthesize_jobs(_tasks(), cfg, n_weeks=4, seed=3)
    c = synthesize_jobs(_tasks(), cfg, n_weeks=4, seed=4)
    assert [j.deadline for j in a.jobs] == [j.deadline for j in b.jobs]
    assert [j.deadline for j in a.jobs] != [j.deadline for j in c.jobs]
    assert len(a.jobs) == 400 and len({j.id for j in a.jobs}) == 400
    slack = np.array([j.deadline - j.arrival - j.duration_steps for j in a.jobs])
    assert set(np.unique(slack)) <= {6.0, 12.0, 24.0, 48.0}
    assert all(j.gpus <= a.capacity_gpus for j in a.jobs)
    lo, hi = cfg["power_kw_clip"]
    assert all(lo <= j.power_kw_per_gpu <= hi for j in a.jobs)
    assert 0.3 < np.mean([j.interruptible for j in a.jobs]) < 0.7


def test_pool_sized_for_target_utilisation() -> None:
    cfg = {**CFG, "jobs_per_week": 100}
    js = synthesize_jobs(_tasks(), cfg, n_weeks=4, seed=0)
    util = js.meta["offered_gpu_hours"] / (js.capacity_gpus * 4 * 7 * 24)
    assert 0.5 < util <= 0.62
