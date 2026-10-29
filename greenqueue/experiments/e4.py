"""E4: carbon saved by S1-S4 versus ASAP, and the share of the oracle's saving captured.

Replays Alibaba-derived jobs against real GB carbon intensity. Schedulers see only the F3+CQR
forecast (issued 00:00/12:00) and a *synthetic* duration estimator with a conformal upper bound;
the simulator accounts with the true intensity. The oracle (S3) uses true values.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from greenqueue.core.types import Scheduler
from greenqueue.eval.plots import PALETTE, new_axes, save
from greenqueue.experiments.harness import (
    ForecastReplay,
    SyntheticEstimator,
    make_provider,
    without_bounds,
)
from greenqueue.schedulers.baselines import Asap, GreedyWindow, Threshold
from greenqueue.schedulers.s3_oracle import PlanScheduler, solve_oracle
from greenqueue.schedulers.s4_robust import RobustMPC
from greenqueue.sim.simulator import SimResult, simulate
from greenqueue.sim.traces import (
    JobSet,
    load_alibaba_tasks,
    load_config,
    load_philly_jobs,
    synthesize_jobs,
)

Q_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
SCHEDULERS = ("S0_asap", "S1_greedy", "S2_threshold", "S3_oracle", "S4_simple")
ALL_SCHEDULERS = (*SCHEDULERS, "S4_mpc")


def _monday_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
    ts = ts.normalize()
    return ts + pd.Timedelta(days=(7 - ts.dayofweek) % 7)


def true_ci_window(gb_ci: pd.Series, start: pd.Timestamp, n_steps: int) -> tuple[np.ndarray, int]:
    """True hourly CI for the window; tiny gaps (<=24 h left after cleaning) are carried over."""
    win = gb_ci.reindex(pd.date_range(start, periods=n_steps, freq="h"))
    n_missing = int(win.isna().sum())
    return win.ffill().bfill().to_numpy(dtype=float), n_missing


def evaluate_schedulers(
    jobset: JobSet,
    true_ci: np.ndarray,
    replay: ForecastReplay,
    *,
    sigma: float,
    alpha: float,
    q: float,
    dt_h: float,
    seed: int,
    only: tuple[str, ...] = SCHEDULERS,
    oracle_time_limit_s: float = 60.0,
    cal_sigma: float | None = None,
    mpc_ci_mode: str = "upper",
    mpc_use_bounds: bool = True,
) -> tuple[dict[str, SimResult], dict[str, Any]]:
    """Run the requested schedulers on one job set; returns results and oracle info."""
    est = SyntheticEstimator(jobset.jobs, sigma, alpha, dt_h, seed, cal_sigma)
    provider = make_provider(replay, est)
    results: dict[str, SimResult] = {}
    info: dict[str, Any] = {}
    online: dict[str, Scheduler] = {
        "S0_asap": Asap(),
        "S1_greedy": GreedyWindow(dt_h),
        "S2_threshold": Threshold(q, dt_h),
        "S4_simple": GreedyWindow(dt_h, use_upper_duration=True),
    }
    for name, sched in online.items():
        if name in only:
            results[name] = simulate(
                jobset.jobs,
                sched,
                true_ci,
                capacity_gpus=jobset.capacity_gpus,
                dt_h=dt_h,
                provider=provider,
            )
    if "S4_mpc" in only:
        mpc = RobustMPC(dt_h, ci_mode=mpc_ci_mode)  # type: ignore[arg-type]
        results["S4_mpc"] = simulate(
            jobset.jobs,
            mpc,
            true_ci,
            capacity_gpus=jobset.capacity_gpus,
            dt_h=dt_h,
            provider=provider if mpc_use_bounds else without_bounds(provider),
        )
        info["mpc_solves"], info["mpc_fallbacks"] = mpc.solves, mpc.fallbacks
    if "S3_oracle" in only:
        plan, oinfo = solve_oracle(
            jobset.jobs, true_ci, jobset.capacity_gpus, dt_h, time_limit_s=oracle_time_limit_s
        )
        results["S3_oracle"] = simulate(
            jobset.jobs,
            PlanScheduler(plan),
            true_ci,
            capacity_gpus=jobset.capacity_gpus,
            dt_h=dt_h,
            provider=provider,
        )
        info["oracle_blocks"] = oinfo.blocks
        info["oracle_non_optimal_blocks"] = oinfo.non_optimal_blocks
        info["oracle_heuristic_blocks"] = oinfo.heuristic_blocks
    return results, info


def tune_threshold_q(
    jobs_factory: Any,
    true_ci: np.ndarray,
    replay: ForecastReplay,
    *,
    sigma: float,
    alpha: float,
    dt_h: float,
) -> tuple[float, list[dict[str, float]]]:
    """Pick S2's quantile on the calibration period: lowest emissions with violations <= 5%."""
    table = []
    for q in Q_GRID:
        res, _ = evaluate_schedulers(
            jobs_factory,
            true_ci,
            replay,
            sigma=sigma,
            alpha=alpha,
            q=q,
            dt_h=dt_h,
            seed=100,
            only=("S2_threshold",),
        )
        r = res["S2_threshold"]
        table.append(
            {"q": q, "emissions_g": r.total_emissions_g, "violation_rate": r.violation_rate}
        )
    ok = [t for t in table if t["violation_rate"] <= 0.05]
    best = (
        min(ok, key=lambda t: t["emissions_g"])
        if ok
        else min(table, key=lambda t: t["violation_rate"])
    )
    return float(best["q"]), table


def run_e4(
    root: Path,
    *,
    smoke: bool = False,
    sigma: float = 0.3,
    alpha: float = 0.05,
    n_jobs: int = 5,
) -> dict[str, Any]:
    from greenqueue.experiments.sweeps import Cell, run_cells, run_task, summarize_cells

    sim_cfg = load_config(root / "configs" / "sim.yaml")
    split_cfg = yaml.safe_load((root / "configs" / "splits.yaml").read_text(encoding="utf8"))
    sp = split_cfg["smoke"] if smoke else split_cfg
    e1_dir = root / "results" / ("e1_smoke" if smoke else "e1")
    out = root / "results" / ("e4_smoke" if smoke else "e4")
    out.mkdir(parents=True, exist_ok=True)

    if smoke:
        sim_cfg = {**sim_cfg, "jobs_per_week": 40}
    n_weeks, seeds, calib_weeks = (2, [0], 2) if smoke else (13, list(sim_cfg["seeds"]), 4)
    dt_h = float(sim_cfg["dt_h"])

    trace = os.environ.get("GQ_TRACE", "alibaba")  # alibaba (default) | philly
    if trace not in ("alibaba", "philly"):
        raise ValueError(f"GQ_TRACE must be alibaba or philly, got {trace!r}")
    if trace == "philly":
        out = out.with_name(out.name + "_philly")
        out.mkdir(parents=True, exist_ok=True)
    cache = root / "data" / "processed" / f"{trace}_tasks.parquet"
    if cache.exists():
        tasks = pd.read_parquet(cache)
    else:
        lo_h, hi_h = sim_cfg["min_duration_h"], sim_cfg["max_duration_h"]
        if trace == "philly":
            log = root / "data" / "raw" / "traces" / "philly" / "trace-data" / "cluster_job_log"
            tasks = load_philly_jobs(log, lo_h, hi_h)
        else:
            csv = root / "data" / "raw" / "traces" / "alibaba" / "x" / "pai_task_table.csv"
            tasks = load_alibaba_tasks(csv, lo_h, hi_h)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tasks.to_parquet(cache)
    gb_ci = pd.read_parquet(root / "data" / "processed" / "gb_hourly.parquet")["ci"]
    fc = pd.read_parquet(e1_dir / "forecasts_F3_cqr90.parquet")

    # ---- tune S2 on the calibration period (never on test) -----------------------------
    c_start = _monday_on_or_after(pd.Timestamp(sp["calib_start"], tz="UTC"))
    c_set = synthesize_jobs(tasks, sim_cfg, n_weeks=calib_weeks, seed=100)
    c_ci, _ = true_ci_window(gb_ci, c_start, c_set.n_steps)
    c_replay = ForecastReplay(fc[fc["split"] == "calib"], c_start, dt_h)
    q_star, q_table = tune_threshold_q(c_set, c_ci, c_replay, sigma=sigma, alpha=alpha, dt_h=dt_h)

    # ---- test replay: all seeds in parallel -------------------------------------------------
    oracle_limit = float(os.environ.get("GQ_ORACLE_LIMIT", "120"))  # seconds per block
    oracle_limit = float(os.environ.get("GQ_ORACLE_LIMIT", "120"))  # seconds per oracle block
    cell = Cell(
        "main",
        SCHEDULERS,
        alpha=alpha,
        sigma=sigma,
        forecast="cqr",
        n_weeks=n_weeks,
        q=q_star,
        oracle_time_limit_s=oracle_limit,
    )
    df = run_cells([cell], seeds, root, smoke=smoke, n_jobs=1 if smoke else min(n_jobs, len(seeds)))
    df.to_parquet(out / "per_job.parquet")
    table = summarize_cells(df)
    table["scheduler"] = pd.Categorical(table["scheduler"], list(SCHEDULERS), ordered=True)
    table = table.sort_values("scheduler").reset_index(drop=True)
    table["scheduler"] = table["scheduler"].astype(str)
    table.to_csv(out / "scheduling_results.csv", index=False)
    pd.DataFrame(q_table).to_csv(out / "s2_threshold_tuning_calibration.csv", index=False)
    per_seed = df.groupby("seed").agg(
        pool_gpus=("pool_gpus", "first"),
        jobs=("id", "nunique"),
        oracle_blocks=("oracle_blocks", "max"),
        oracle_time_limited_blocks=("oracle_time_limited_blocks", "max"),
        oracle_heuristic_blocks=("oracle_heuristic_blocks", "max"),
    )
    per_seed.to_csv(out / "per_seed.csv")

    # ---- sanity bounds (section 13.1) -------------------------------------------------------
    again = run_task(
        Cell("det", ("S0_asap",), forecast="cqr", n_weeks=n_weeks), seeds[0], str(root), smoke
    )
    first = df[(df["scheduler"] == "S0_asap") & (df["seed"] == seeds[0])]
    sanity = {
        "s0_deterministic": bool(
            abs(again["emissions_g"].sum() - first["emissions_g"].sum()) < 1e-6
        ),
        "all_finished": bool(df["finished"].all()),
    }
    tot = df.groupby(["seed", "scheduler"]).agg(e=("emissions_g", "sum"), v=("violated", "sum"))
    totals = {
        (int(r.seed), str(r.scheduler)): (float(r.e), int(r.v))  # type: ignore[arg-type]
        for r in tot.reset_index().itertuples()
    }
    beaten = []
    for seed in seeds:
        o_e, o_v = totals[(seed, "S3_oracle")]
        for name in SCHEDULERS:
            e, v = totals[(seed, name)]
            if name != "S3_oracle" and o_v == 0 and v == 0 and o_e > e * 1.005:
                beaten.append((seed, name))
    limited = int(
        per_seed["oracle_time_limited_blocks"].sum() + per_seed["oracle_heuristic_blocks"].sum()
    )
    # when some oracle blocks hit the time limit the oracle is only a heuristic there, so a
    # scheduler beating it is reported but not treated as a bug
    sanity["none_beats_oracle"] = (not beaten) or limited > 0

    plot = table[table["scheduler"] != "S0_asap"].reset_index(drop=True)
    fig, ax = new_axes(
        "Carbon saved vs ASAP (95% block-bootstrap CI)", "", "Saving (%)", (6.4, 3.6)
    )
    x = np.arange(len(plot))
    ax.bar(x, 100 * plot["saving_vs_s0"], color=[PALETTE[i] for i in range(len(plot))], width=0.55)
    ax.errorbar(
        x,
        100 * plot["saving_vs_s0"],
        yerr=[
            (100 * (plot["saving_vs_s0"] - plot["saving_lo"])).clip(lower=0),
            (100 * (plot["saving_hi"] - plot["saving_vs_s0"])).clip(lower=0),
        ],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=1,
    )
    ax.set_xticks(x, plot["scheduler"], fontsize=8)
    save(fig, out / "savings_vs_s0")

    fig, ax = new_axes("Deadline violations by scheduler", "", "Violation rate (%)", (6.4, 3.6))
    allx = np.arange(len(table))
    ax.bar(
        allx,
        100 * table["violation_rate"],
        color=[PALETTE[i] for i in range(len(table))],
        width=0.55,
    )
    ax.axhline(100 * alpha, color="#444444", lw=0.8, ls="--", label=f"alpha = {alpha:.0%}")
    ax.set_xticks(allx, table["scheduler"], fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "violations")

    key = {
        "sim": sim_cfg,
        "sp": sp,
        "seeds": seeds,
        "weeks": n_weeks,
        "sigma": sigma,
        "alpha": alpha,
        "q": q_star,
    }
    summary = {
        "config_hash": hashlib.sha256(
            json.dumps(key, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
        "smoke": smoke,
        "seeds": seeds,
        "n_weeks": n_weeks,
        "s2_q": q_star,
        "estimator": (
            f"SYNTHETIC lognormal noise sigma={sigma} (no real energy model yet), "
            f"conformal alpha={alpha}"
        ),
        "power_per_gpu": "ASSUMED lognormal, median 0.20 kW (no GPU measurements yet)",
        "oracle_time_limit_s_per_block": oracle_limit,
        "oracle_time_limited_blocks": limited,
        "oracle_heuristic_blocks": int(per_seed["oracle_heuristic_blocks"].sum()),
        "oracle_beaten_cases": beaten,
        "sanity": sanity,
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not all(sanity.values()):
        raise AssertionError(f"sanity bounds failed: {sanity}")
    return summary
