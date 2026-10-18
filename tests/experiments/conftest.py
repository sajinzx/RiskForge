"""Shared fixture: a synthetic GB-like project root with E1 (smoke) already run once."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenqueue.experiments.e1 import run_e1

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:  # type: ignore[type-arg]
    root = tmp_path_factory.mktemp("smoke_root")
    shutil.copytree(ROOT / "configs", root / "configs")
    idx = pd.date_range("2020-12-20", "2023-07-05", freq="30min", tz="UTC")
    t = np.arange(len(idx))
    rng = np.random.default_rng(0)
    ci = 170 + 60 * np.sin(2 * np.pi * t / 48) + 25 * np.sin(2 * np.pi * t / (48 * 7))
    ci = ci + rng.normal(0, 12, len(idx))
    raw = pd.DataFrame(
        {"forecast": ci + rng.normal(0, 8, len(idx)), "actual": ci, "index": "low"}, index=idx
    )
    raw_dir = root / "data" / "raw" / "uk_ci"
    raw_dir.mkdir(parents=True)
    raw.to_parquet(raw_dir / "intensity.parquet")
    summary = run_e1(root, smoke=True, trials=1)
    return root, summary


@pytest.fixture(scope="session")
def sched_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic carbon, forecasts (E1/E2 style) and trace tasks for scheduling smoke tests."""
    from greenqueue.sim.traces import WEEK_S

    root = tmp_path_factory.mktemp("sched_root")
    shutil.copytree(ROOT / "configs", root / "configs")
    idx = pd.date_range("2022-06-01", "2023-03-31", freq="h", tz="UTC")
    t = np.arange(len(idx))
    ci = np.clip(
        170 + 70 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(1).normal(0, 10, len(idx)),
        20,
        400,
    )
    (root / "data" / "processed").mkdir(parents=True)
    pd.DataFrame({"ci": ci}, index=idx).to_parquet(
        root / "data" / "processed" / "gb_hourly.parquet"
    )

    rng = np.random.default_rng(0)
    n = 300
    w = rng.integers(0, 3, n)
    wall = w * WEEK_S + rng.uniform(0, WEEK_S, n)
    pd.DataFrame(
        {
            "job_name": [f"j{i}" for i in range(n)],
            "task_name": "t",
            "gpus": rng.integers(1, 4, n),
            "duration_s": rng.uniform(1800, 10 * 3600, n),
            "wall_s": wall,
            "trace_week": w,
            "week_s": wall % WEEK_S,
        }
    ).to_parquet(root / "data" / "processed" / "alibaba_tasks.parquet")

    series = pd.Series(ci, idx)
    rows = []
    for split, a, b in (
        ("calib", "2022-07-01", "2022-08-10"),
        ("test", "2023-01-01", "2023-02-10"),
    ):
        for issue in pd.date_range(a, b, freq="12h", tz="UTC"):
            tgt = issue + pd.to_timedelta(np.arange(48), unit="h")
            y = series.reindex(tgt).to_numpy()
            point = y + rng.normal(0, 15, 48)
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
                )  # fmt: skip
            )
    fc = pd.concat(rows)
    for d, name in (
        ("e1_smoke", "forecasts_F3_cqr90.parquet"),
        ("e2_smoke", "forecasts_F3_aci90.parquet"),
    ):
        (root / "results" / d).mkdir(parents=True)
        fc.to_parquet(root / "results" / d / name)
    return root
