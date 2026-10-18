"""Parallel runner for scheduling experiments: a *cell* is one parameter setting, a task is a cell
run on one job-mix seed. Workers rebuild their inputs from disk so nothing large is pickled.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from greenqueue.experiments.e4 import (
    _monday_on_or_after,
    evaluate_schedulers,
    true_ci_window,
)
from greenqueue.experiments.harness import ForecastReplay
from greenqueue.sim.traces import load_alibaba_tasks, load_config, synthesize_jobs


@dataclass(frozen=True)
class Cell:
    tag: str
    schedulers: tuple[str, ...]
    alpha: float = 0.05
    sigma: float = 0.3
    cal_sigma: float | None = None  # != sigma simulates a miscalibrated duration bound
    forecast: str = "aci"  # aci | cqr | perfect
    mpc_ci_mode: str = "upper"
    mpc_use_bounds: bool = True
    n_weeks: int = 4
    q: float = 0.3
    oracle_time_limit_s: float = 120.0
    period: str = "test"  # test | calib (calib uses the 2023 year and static CQR forecasts)
    overrides: tuple[tuple[str, Any], ...] = field(default_factory=tuple)  # sim.yaml overrides


def perfect_forecasts(ci: pd.Series, start: pd.Timestamp, n_steps: int) -> pd.DataFrame:
    """Forecasts equal to the truth (for the 'perfect carbon forecast' ablation only)."""
    rows = []
    for issue in pd.date_range(start, periods=n_steps // 12 + 6, freq="12h"):
        tgt = issue + pd.to_timedelta(np.arange(48), unit="h")
        y = ci.reindex(tgt).ffill().bfill().to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "issue_time": issue,
                    "h": np.arange(1, 49),
                    "target_time": tgt,
                    "point": y,
                    "lo": y,
                    "hi": y,
                }
            )
        )
    return pd.concat(rows)


@functools.lru_cache(maxsize=4)
def _inputs(
    root: str, smoke: bool
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], dict[str, Any]]:
    r = Path(root)
    sim_cfg = load_config(r / "configs" / "sim.yaml")
    split_cfg = yaml.safe_load((r / "configs" / "splits.yaml").read_text(encoding="utf8"))
    sp = split_cfg["smoke"] if smoke else split_cfg
    cache = r / "data" / "processed" / "alibaba_tasks.parquet"
    if cache.exists():
        tasks = pd.read_parquet(cache)
    else:
        csv = r / "data" / "raw" / "traces" / "alibaba" / "x" / "pai_task_table.csv"
        tasks = load_alibaba_tasks(csv, sim_cfg["min_duration_h"], sim_cfg["max_duration_h"])
    gb_ci = pd.read_parquet(r / "data" / "processed" / "gb_hourly.parquet")["ci"]
    return tasks, gb_ci, sim_cfg, sp


@functools.lru_cache(maxsize=8)
def _forecast_df(root: str, smoke: bool, source: str, split: str = "test") -> pd.DataFrame | None:
    r = Path(root) / "results"
    if source == "perfect":
        return None
    if split == "calib":  # only the static CQR forecasts exist for the calibration year
        source = "cqr"
    path = (
        r / ("e2_smoke" if smoke else "e2") / "forecasts_F3_aci90.parquet"
        if source == "aci"
        else r / ("e1_smoke" if smoke else "e1") / "forecasts_F3_cqr90.parquet"
    )
    df = pd.read_parquet(path)
    return df[df["split"] == split]


def run_task(cell: Cell, seed: int, root: str, smoke: bool) -> pd.DataFrame:
    tasks, gb_ci, sim_cfg, sp = _inputs(root, smoke)
    cfg = {**sim_cfg, **dict(cell.overrides)}
    if smoke:
        cfg["jobs_per_week"] = min(cfg["jobs_per_week"], 40)
    dt_h = float(cfg["dt_h"])
    js = synthesize_jobs(tasks, cfg, n_weeks=cell.n_weeks, seed=seed)
    start = _monday_on_or_after(
        pd.Timestamp(sp[f"{'calib' if cell.period == 'calib' else 'test'}_start"], tz="UTC")
    )
    ci, _ = true_ci_window(gb_ci, start, js.n_steps)
    fc = _forecast_df(root, smoke, cell.forecast, cell.period)
    if fc is None:
        fc = perfect_forecasts(gb_ci, start, js.n_steps)
    window = fc[
        (fc["target_time"] >= start - pd.Timedelta(days=1))
        & (fc["target_time"] <= start + pd.Timedelta(hours=js.n_steps + 48))
    ]
    replay = ForecastReplay(window, start, dt_h)
    res, info = evaluate_schedulers(
        js, ci, replay, sigma=cell.sigma, alpha=cell.alpha, q=cell.q, dt_h=dt_h, seed=seed,
        only=cell.schedulers, cal_sigma=cell.cal_sigma, mpc_ci_mode=cell.mpc_ci_mode,
        mpc_use_bounds=cell.mpc_use_bounds, oracle_time_limit_s=cell.oracle_time_limit_s,
    )  # fmt: skip
    frames = []
    spw = int(round(168 / dt_h))
    arrival = {j.id: j.arrival for j in js.jobs}
    for name, r in res.items():
        frames.append(
            pd.DataFrame(
                {
                    "tag": cell.tag,
                    "seed": seed,
                    "scheduler": name,
                    "id": [x.id for x in r.jobs],
                    "week": [arrival[x.id] // spw for x in r.jobs],
                    "emissions_g": [x.emissions_g for x in r.jobs],
                    "violated": [x.violated for x in r.jobs],
                    "wait_h": [x.wait_h for x in r.jobs],
                    "finished": [x.finish_step is not None for x in r.jobs],
                    "fallbacks": info.get("mpc_fallbacks", 0) if name == "S4_mpc" else 0,
                    "solves": info.get("mpc_solves", 0) if name == "S4_mpc" else 0,
                    "pool_gpus": js.capacity_gpus,
                    "oracle_blocks": info.get("oracle_blocks", 0) if name == "S3_oracle" else 0,
                    "oracle_heuristic_blocks": (
                        info.get("oracle_heuristic_blocks", 0) if name == "S3_oracle" else 0
                    ),
                    "oracle_time_limited_blocks": (
                        info.get("oracle_non_optimal_blocks", 0) if name == "S3_oracle" else 0
                    ),
                }
            )
        )
    return pd.concat(frames)


def run_cells(
    cells: list[Cell], seeds: list[int], root: Path, *, smoke: bool, n_jobs: int = 6
) -> pd.DataFrame:
    """Run every (cell, seed) pair in parallel and stack the per-job results."""
    jobs = [(c, s) for c in cells for s in seeds]
    if n_jobs == 1 or len(jobs) == 1:
        parts = [run_task(c, s, str(root), smoke) for c, s in jobs]
    else:
        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_task)(c, s, str(root), smoke) for c, s in jobs
        )
    return pd.concat(parts, ignore_index=True)


def with_overrides(cell: Cell, **kw: Any) -> Cell:
    return replace(cell, overrides=tuple(sorted({**dict(cell.overrides), **kw}.items())))


def summarize_cells(df: pd.DataFrame, oracle_tag: str | None = None) -> pd.DataFrame:
    """Per (tag, scheduler): violation rate, saving vs ASAP and share of the oracle saving.

    Savings are paired with the ASAP run of the same tag and seeds (same jobs), with
    block-bootstrap CIs over (seed, week). The oracle comes from ``oracle_tag`` if given, else
    from the same tag.
    """
    from greenqueue.eval.bootstrap import paired_saving_ci, rate_ci

    df = df.copy()
    df["block"] = df["seed"].astype(str) + "_" + df["week"].astype(str)
    rows = []
    for tag, tg in df.groupby("tag"):
        s0 = tg[tg["scheduler"] == "S0_asap"].set_index(["seed", "id"])
        orc_src = df[df["tag"] == (oracle_tag or tag)]
        orc = orc_src[orc_src["scheduler"] == "S3_oracle"].set_index(["seed", "id"])
        for sched, g in tg.groupby("scheduler"):
            rate, lo, hi = rate_ci(g["violated"].to_numpy(), g["block"].to_numpy())
            gi = g.set_index(["seed", "id"])
            base = s0.loc[gi.index, "emissions_g"].to_numpy()
            sav, slo, shi = paired_saving_ci(
                base, gi["emissions_g"].to_numpy(), gi["block"].to_numpy()
            )
            share = np.nan
            if len(orc):
                o_sav = 1.0 - float(orc.loc[gi.index, "emissions_g"].sum()) / float(base.sum())
                share = sav / o_sav if o_sav else np.nan
            rows.append(
                {
                    "tag": tag, "scheduler": sched, "violation_rate": rate, "violation_lo": lo,
                    "violation_hi": hi, "saving_vs_s0": sav, "saving_lo": slo, "saving_hi": shi,
                    "share_of_oracle": share, "mean_wait_h": float(g["wait_h"].mean()),
                    "p95_wait_h": float(np.percentile(g["wait_h"], 95)),
                    "pool_gpus": float(g["pool_gpus"].mean()),
                    "mpc_fallbacks": int(g.groupby("seed")["fallbacks"].first().sum()),
                    "mpc_solves": int(g.groupby("seed")["solves"].first().sum()),
                    "unfinished": int((~g["finished"]).sum()),
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def choose_mpc_ci_mode(
    root: Path, *, smoke: bool, out: Path, load: tuple[str, float] = ("target_utilization", 0.3)
) -> str:
    """Pick the carbon risk level of the MPC on the calibration year (never on test data).

    Rule: highest carbon saving vs ASAP among modes whose violation rate is at most one
    percentage point above ASAP's; if none qualifies, the lowest violation rate.
    """
    modes = ("point", "cvar", "upper")
    weeks = 1 if smoke else 3
    cells = [
        Cell(
            f"calib_{m}",
            ("S0_asap", "S4_mpc"),
            alpha=0.05,
            mpc_ci_mode=m,
            n_weeks=weeks,
            overrides=(load,),
            period="calib",
        )
        for m in modes
    ]
    df = run_cells(
        cells, [100] if smoke else [100, 101], root, smoke=smoke, n_jobs=1 if smoke else 6
    )
    t = summarize_cells(df)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "mpc_ci_mode_calibration.csv", index=False)
    mpc = t[t["scheduler"] == "S4_mpc"].set_index("tag")
    s0_viol = float(t[t["scheduler"] == "S0_asap"]["violation_rate"].max())
    ok = mpc[mpc["violation_rate"] <= s0_viol + 0.01]
    pick = ok["saving_vs_s0"].idxmax() if len(ok) else mpc["violation_rate"].idxmin()
    return str(pick).removeprefix("calib_")
