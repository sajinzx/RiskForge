"""Job traces to simulator jobs (implementation.md §6.6, §7.6, §10.3).

Traces carry no deadlines and no energy, so both are synthesised from the config. Trace weeks are
re-sampled (with replacement, seeded) onto the target weeks, keeping weekday and time of day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from greenqueue.core.types import Job

ALIBABA_COLUMNS = [
    "job_name",
    "task_name",
    "inst_num",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type",
]
WEEK_S = 7 * 24 * 3600


def load_config(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf8"))
    return cfg


def load_alibaba_tasks(
    csv_path: Path, min_duration_h: float, max_duration_h: float
) -> pd.DataFrame:
    """Terminated GPU tasks with a valid duration; one row per task.

    GPUs per task = ``inst_num * plan_gpu / 100`` rounded up (``plan_gpu`` is a percentage).
    ``week_s`` is the submission time as seconds since Monday 00:00 in the trace clock (UTC+8
    after the documented offset), so it carries weekday and time of day.
    """
    df = pd.read_csv(csv_path, names=ALIBABA_COLUMNS, header=None)
    df = df[(df["status"] == "Terminated") & (df["plan_gpu"] > 0)].dropna(
        subset=["start_time", "end_time", "inst_num"]
    )
    dur_s = df["end_time"] - df["start_time"]
    keep = (dur_s >= min_duration_h * 3600) & (dur_s <= max_duration_h * 3600)
    df = df[keep].copy()
    df["duration_s"] = dur_s[keep]
    df["gpus"] = np.ceil(df["inst_num"] * df["plan_gpu"] / 100.0).astype(int).clip(lower=1)
    local = pd.to_datetime(df["start_time"], unit="s") + pd.Timedelta(hours=8)
    monday = (local.dt.normalize() - pd.to_timedelta(local.dt.dayofweek, unit="D")).min()
    df["wall_s"] = (local - monday).dt.total_seconds()
    df["trace_week"] = (df["wall_s"] // WEEK_S).astype(int)
    df["week_s"] = df["wall_s"] % WEEK_S
    cols = ["job_name", "task_name", "gpus", "duration_s", "wall_s", "trace_week", "week_s"]
    return df[cols].reset_index(drop=True)


@dataclass(frozen=True)
class JobSet:
    jobs: list[Job]
    capacity_gpus: int
    n_steps: int  # replay length in steps (jobs have room to finish inside it)
    meta: dict[str, Any]


def synthesize_jobs(
    tasks: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    n_weeks: int,
    seed: int,
    tail_h: int = 96,
) -> JobSet:
    """Build jobs for ``n_weeks`` target weeks (+ ``tail_h`` for stragglers to finish)."""
    rng = np.random.default_rng(seed)
    dt = float(cfg["dt_h"])
    counts = tasks.groupby("trace_week").size()
    usable = np.array([w for w in counts.index if counts[w] >= cfg["jobs_per_week"]])
    if len(usable) == 0:
        raise ValueError("no trace week has enough tasks for jobs_per_week")
    slack = np.array(cfg["slack_h"], dtype=float)
    probs = np.array(cfg["slack_probs"], dtype=float)
    lo_p, hi_p = cfg["power_kw_clip"]
    steps_per_week = int(round(7 * 24 / dt))
    jobs: list[Job] = []
    for w in range(n_weeks):
        src = int(rng.choice(usable))
        pool = tasks[tasks["trace_week"] == src]
        pick = pool.iloc[rng.choice(len(pool), size=cfg["jobs_per_week"], replace=False)]
        for k, row in enumerate(pick.itertuples(index=False)):
            arrival = w * steps_per_week + int(row.week_s // (dt * 3600))
            dur = max(1, math.ceil(row.duration_s / (dt * 3600)))
            raw_p = cfg["power_kw_median"] * np.exp(rng.normal(0, cfg["power_kw_sigma"]))
            s = float(rng.choice(slack, p=probs))
            jobs.append(
                Job(
                    id=f"w{w}_{k}",
                    arrival=arrival,
                    duration_steps=dur,
                    gpus=int(row.gpus),
                    power_kw_per_gpu=float(np.clip(raw_p, lo_p, hi_p)),
                    deadline=float(arrival + dur + math.ceil(s / dt)),
                    interruptible=bool(rng.random() < cfg["interruptible_share"]),
                    checkpoint_overhead_h=float(cfg["checkpoint_overhead_h"]),
                    features={"gpus": float(row.gpus), "duration_h": dur * dt},
                )
            )
    n_steps = n_weeks * steps_per_week + int(tail_h / dt)
    gpu_hours = sum(j.gpus * j.duration_steps * dt for j in jobs)
    horizon_h = n_weeks * 7 * 24
    need = math.ceil(gpu_hours / (cfg["target_utilization"] * horizon_h))
    cap = max(max(j.gpus for j in jobs), need)
    meta = {"seed": seed, "n_jobs": len(jobs), "offered_gpu_hours": gpu_hours, "pool_gpus": cap}
    return JobSet(jobs, cap, n_steps, meta)
