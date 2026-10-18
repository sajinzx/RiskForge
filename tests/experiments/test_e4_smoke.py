"""E4 end to end on synthetic carbon, forecasts and trace tasks (no network, no real data)."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from greenqueue.experiments.e4 import run_e4
from greenqueue.sim.traces import WEEK_S

ROOT = Path(__file__).parents[2]


def _tasks(n: int = 300, weeks: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    w = rng.integers(0, weeks, n)
    wall = w * WEEK_S + rng.uniform(0, WEEK_S, n)
    return pd.DataFrame(
        {
            "job_name": [f"j{i}" for i in range(n)],
            "task_name": "t",
            "gpus": rng.integers(1, 4, n),
            "duration_s": rng.uniform(1800, 10 * 3600, n),
            "wall_s": wall,
            "trace_week": w,
            "week_s": wall % WEEK_S,
        }
    )


def test_e4_smoke_end_to_end(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    idx = pd.date_range("2022-06-01", "2023-03-31", freq="h", tz="UTC")
    t = np.arange(len(idx))
    ci = 170 + 70 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(1).normal(0, 10, len(idx))
    ci = np.clip(ci, 20, 400)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    pd.DataFrame({"ci": ci}, index=idx).to_parquet(
        tmp_path / "data" / "processed" / "gb_hourly.parquet"
    )

    series = pd.Series(ci, idx)
    rng = np.random.default_rng(2)
    rows = []
    for split, a, b in (
        ("calib", "2022-07-01", "2022-08-10"),
        ("test", "2023-01-01", "2023-02-10"),
    ):
        for issue in pd.date_range(a, b, freq="12h", tz="UTC"):
            tgt = issue + pd.to_timedelta(np.arange(48), unit="h")
            y = series.reindex(tgt).to_numpy()
            point = y + rng.normal(0, 15, 48)  # forecast with error
            rows.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "issue_time": issue,
                        "h": np.arange(1, 49),
                        "target_time": tgt,
                        "y": y,
                        "point": point,
                        "lo": point - 40,
                        "hi": point + 40,
                    }
                )
            )
    e1 = tmp_path / "results" / "e1_smoke"
    e1.mkdir(parents=True)
    pd.concat(rows).to_parquet(e1 / "forecasts_F3_cqr90.parquet")

    summary = run_e4(tmp_path, smoke=True, tasks=_tasks())

    out = tmp_path / "results" / "e4_smoke"
    assert (out / "scheduling_results.csv").exists() and (out / "savings_vs_s0.png").exists()
    assert all(summary["sanity"].values())
    res = {r["scheduler"]: r for r in summary["results"]}
    # the oracle is the best possible, and any informed scheduler saves carbon on a cyclic grid
    assert (
        res["S3_oracle"]["saving_vs_s0"]
        >= max(res[s]["saving_vs_s0"] for s in ("S1_greedy", "S2_threshold", "S4_simple")) - 1e-6
    )
    assert res["S1_greedy"]["saving_vs_s0"] > 0
    assert res["S3_oracle"]["saving_vs_s0"] > 0
