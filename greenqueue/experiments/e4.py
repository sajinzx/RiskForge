"""E4: carbon saved by S1-S4 versus ASAP, and the share of the oracle's saving captured.

Replays Alibaba-derived jobs against real GB carbon intensity. Schedulers see only the F3+CQR
forecast (issued 00:00/12:00) and a *synthetic* duration estimator with a conformal upper bound;
the simulator accounts with the true intensity. The oracle (S3) uses true values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from greenqueue.core.types import Job, Scheduler
from greenqueue.eval.bootstrap import paired_saving_ci
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
from greenqueue.sim.traces import JobSet, load_alibaba_tasks, load_config, synthesize_jobs

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


def _per_job(res: SimResult, jobs: list[Job], seed: int, sched: str) -> pd.DataFrame:
    arrival = {j.id: j.arrival for j in jobs}
    return (
        pd.DataFrame(
            {
                "seed": seed,
                "scheduler": sched,
                "id": [r.id for r in res.jobs],
                "week": [arrival[r.id] // 168 for r in res.jobs],
                "emissions_g": [r.emissions_g for r in res.jobs],
                "energy_kwh": [r.energy_kwh for r in res.jobs],
                "violated": [r.violated for r in res.jobs],
                "wait_h": [r.wait_h for r in res.jobs],
            }
        )
        .sort_values(["seed", "id"])
        .reset_index(drop=True)
    )


def run_e4(
    root: Path,
    *,
    smoke: bool = False,
    tasks: pd.DataFrame | None = None,
    sigma: float = 0.3,
    alpha: float = 0.05,
) -> dict[str, Any]:
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

    if tasks is None:
        cache = root / "data" / "processed" / "alibaba_tasks.parquet"
        if cache.exists():
            tasks = pd.read_parquet(cache)
        else:
            csv = root / "data" / "raw" / "traces" / "alibaba" / "x" / "pai_task_table.csv"
            tasks = load_alibaba_tasks(csv, sim_cfg["min_duration_h"], sim_cfg["max_duration_h"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            tasks.to_parquet(cache)
    gb_ci = pd.read_parquet(root / "data" / "processed" / "gb_hourly.parquet")["ci"]
    fc = pd.read_parquet(e1_dir / "forecasts_F3_cqr90.parquet")

    # ---- tune S2 on the calibration period (never on test) -----------------------------
    c_start = _monday_on_or_after(pd.Timestamp(sp["calib_start"], tz="UTC"))
    c_set = synthesize_jobs(tasks, sim_cfg, n_weeks=calib_weeks, seed=100)
    c_ci, _ = true_ci_window(gb_ci, c_start, c_set.n_steps)
    c_fc = fc[(fc["split"] == "calib")]
    c_replay = ForecastReplay(c_fc, c_start, dt_h)
    q_star, q_table = tune_threshold_q(c_set, c_ci, c_replay, sigma=sigma, alpha=alpha, dt_h=dt_h)

    # ---- test replay ----------------------------------------------------------------------
    t_start = _monday_on_or_after(pd.Timestamp(sp["test_start"], tz="UTC"))
    probe = synthesize_jobs(tasks, sim_cfg, n_weeks=n_weeks, seed=seeds[0])
    t_fc = fc[
        (fc["split"] == "test")
        & (fc["target_time"] >= t_start - pd.Timedelta(days=1))
        & (fc["target_time"] <= t_start + pd.Timedelta(hours=probe.n_steps + 48))
    ]
    replay = ForecastReplay(t_fc, t_start, dt_h)

    frames, per_seed, oracle_info, gaps = [], [], [], 0
    sanity = {"s0_deterministic": True, "none_beats_oracle": True, "all_finished": True}
    for seed in seeds:
        js = synthesize_jobs(tasks, sim_cfg, n_weeks=n_weeks, seed=seed)
        ci, gaps = true_ci_window(gb_ci, t_start, js.n_steps)
        res, info = evaluate_schedulers(
            js, ci, replay, sigma=sigma, alpha=alpha, q=q_star, dt_h=dt_h, seed=seed
        )
        oracle_info.append(info)
        if seed == seeds[0]:
            again, _ = evaluate_schedulers(
                js,
                ci,
                replay,
                sigma=sigma,
                alpha=alpha,
                q=q_star,
                dt_h=dt_h,
                seed=seed,
                only=("S0_asap",),
            )
            sanity["s0_deterministic"] = (
                again["S0_asap"].total_emissions_g == res["S0_asap"].total_emissions_g
            )
        for name, r in res.items():
            if r.unfinished:
                sanity["all_finished"] = False
            frames.append(_per_job(r, js.jobs, seed, name))
        orc = res["S3_oracle"]
        for name, r in res.items():
            if (
                name != "S3_oracle"
                and r.violation_rate == 0
                and orc.violation_rate == 0
                and orc.total_emissions_g > r.total_emissions_g * 1.005
            ):
                sanity["none_beats_oracle"] = False
        per_seed.append(
            {
                "seed": seed,
                "pool_gpus": js.capacity_gpus,
                "jobs": len(js.jobs),
                "s0_busy_gpu_share": float(
                    res["S0_asap"].gpu_busy_steps[: n_weeks * 168].mean() / js.capacity_gpus
                ),
            }
        )

    jobs_df = pd.concat(frames)
    wide = jobs_df.pivot_table(
        index=["seed", "id", "week"], columns="scheduler", values="emissions_g"
    ).reset_index()
    blocks = (wide["seed"].astype(str) + "_" + wide["week"].astype(str)).to_numpy()
    rows = []
    for name in SCHEDULERS:
        sub = jobs_df[jobs_df["scheduler"] == name]
        row: dict[str, Any] = {
            "scheduler": name,
            "emissions_t": float(sub["emissions_g"].sum() / 1e6),
            "violation_rate": float(sub["violated"].mean()),
            "mean_wait_h": float(sub["wait_h"].mean()),
            "p95_wait_h": float(np.percentile(sub["wait_h"], 95)),
            "cvar90_job_g": float(
                np.sort(sub["emissions_g"].to_numpy())[-max(1, int(0.1 * len(sub))) :].mean()
            ),
        }
        if name != "S0_asap":
            s, lo, hi = paired_saving_ci(wide["S0_asap"].to_numpy(), wide[name].to_numpy(), blocks)
            row.update(saving_vs_s0=s, saving_lo=lo, saving_hi=hi)
            so, olo, ohi = paired_saving_ci(
                wide["S0_asap"].to_numpy(), wide["S3_oracle"].to_numpy(), blocks
            )
            row["share_of_oracle"] = s / so if so else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "scheduling_results.csv", index=False)
    pd.DataFrame(q_table).to_csv(out / "s2_threshold_tuning_calibration.csv", index=False)
    pd.DataFrame(per_seed).to_csv(out / "per_seed.csv", index=False)

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
            100 * (plot["saving_vs_s0"] - plot["saving_lo"]),
            100 * (plot["saving_hi"] - plot["saving_vs_s0"]),
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

    summary = {
        "config_hash": hashlib.sha256(
            json.dumps(
                {
                    "sim": sim_cfg,
                    "sp": sp,
                    "seeds": seeds,
                    "weeks": n_weeks,
                    "sigma": sigma,
                    "alpha": alpha,
                    "q": q_star,
                },
                sort_keys=True,
                default=str,
            ).encode()
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
        "true_ci_filled_hours": gaps,
        "oracle": oracle_info,
        "sanity": sanity,
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not all(sanity.values()):
        raise AssertionError(f"sanity bounds failed: {sanity}")
    return summary
