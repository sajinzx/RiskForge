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


def test_job_gpu_requests_are_capped_so_the_pool_is_not_set_by_outliers() -> None:
    tasks = _tasks()
    tasks.loc[tasks.index[:10], "gpus"] = 400
    cfg = {**CFG, "jobs_per_week": 100, "max_job_gpus": 8}
    js = synthesize_jobs(tasks, cfg, n_weeks=4, seed=0)
    assert max(j.gpus for j in js.jobs) <= 8
    assert js.capacity_gpus < 100
    util = js.meta["offered_gpu_hours"] / (js.capacity_gpus * 4 * 7 * 24)
    assert 0.5 < util <= 0.62


def test_philly_loader_uses_last_attempt_and_filters(tmp_path: Path) -> None:
    import json

    from greenqueue.sim.traces import load_philly_jobs

    def att(s: str, e: str | None, gpus: list[list[str]]) -> dict:
        return {"start_time": s, "end_time": e, "detail": [{"ip": "m", "gpus": g} for g in gpus]}

    jobs = [
        # retried: first attempt ignored; last attempt 2 h on 2 machines x 2 GPUs = 4 GPUs
        {"status": "Pass", "jobid": "a", "vc": "v", "submitted_time": "2017-10-02 00:00:00",
         "attempts": [att("2017-10-02 00:01:00", "2017-10-02 00:30:00", [["gpu0"]]),
                      att("2017-10-02 01:00:00", "2017-10-02 03:00:00", [["g0", "g1"]] * 2)]},
        {"status": "Failed", "jobid": "b", "vc": "v", "submitted_time": "2017-10-02 00:00:00",
         "attempts": [att("2017-10-02 01:00:00", "2017-10-02 03:00:00", [["g0"]])]},
        {"status": "Pass", "jobid": "c", "vc": "v", "submitted_time": "2017-10-02 00:00:00",
         "attempts": [att("2017-10-02 01:00:00", "2017-10-02 01:05:00", [["g0"]])]},  # 5 min
        {"status": "Pass", "jobid": "d", "vc": "v", "submitted_time": "2017-10-02 00:00:00",
         "attempts": [att("2017-10-02 01:00:00", "None", [["g0"]])]},  # never ended
        {"status": "Pass", "jobid": "e", "vc": "v", "submitted_time": "2017-10-10 06:00:00",
         "attempts": [att("2017-10-10 06:00:00", "2017-10-10 07:00:00", [["g0"]])]},
    ]  # fmt: skip
    p = tmp_path / "cluster_job_log"
    p.write_text(json.dumps(jobs))
    df = load_philly_jobs(p, 10 / 60, 72)
    assert list(df["job_name"]) == ["a", "e"]
    assert list(df["gpus"]) == [4, 1]
    assert np.allclose(df["duration_s"], [7200, 3600])
    # 2017-10-02 is a Monday: job a is week 0 at 0 s; job e is Tuesday 06:00 of week 1
    assert list(df["trace_week"]) == [0, 1]
    assert np.allclose(df["week_s"], [0, 24 * 3600 + 6 * 3600])
