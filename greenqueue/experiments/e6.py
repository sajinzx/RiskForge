"""E6: ablations of the robust MPC scheduler (which component matters?).

All variants run at the E5 load (30%) with alpha = 0.05. Variants switch off one component of the
full system at a time: conformal bounds altogether, the carbon bound, the duration/energy
uncertainty, or replace the adaptive carbon intervals by static CQR; 'perfect carbon' gives the
value of a perfect forecast. A no-weather variant needs the v2 weather model, which is not built.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from greenqueue.eval.plots import PALETTE, new_axes, save
from greenqueue.experiments.sweeps import Cell, choose_mpc_ci_mode, run_cells, summarize_cells

LOAD = ("target_utilization", 0.3)
VARIANTS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_conformal_bounds": {"mpc_use_bounds": False},
    "carbon_point_only": {"mpc_ci_mode": "point"},
    "carbon_cvar": {"mpc_ci_mode": "cvar"},
    "perfect_energy_and_duration": {"sigma": 0.0},
    "static_cqr_carbon": {"forecast": "cqr"},
    "perfect_carbon_forecast": {"forecast": "perfect"},
}


def run_e6(root: Path, *, smoke: bool = False, n_jobs: int = 6) -> dict[str, Any]:
    out = root / "results" / ("e6_smoke" if smoke else "e6")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [0] if smoke else [0, 1, 2]
    n_weeks = 1 if smoke else 4
    variants = dict(list(VARIANTS.items())[:3]) if smoke else VARIANTS
    ci_mode = choose_mpc_ci_mode(root, smoke=smoke, out=out)
    cells = [Cell("ref", ("S0_asap", "S1_greedy", "S3_oracle"), n_weeks=n_weeks, overrides=(LOAD,))]
    for name, kw in variants.items():
        params = {"alpha": 0.05, "sigma": 0.3, "mpc_ci_mode": ci_mode, **kw}
        cells.append(
            Cell(name, ("S0_asap", "S4_mpc"), n_weeks=n_weeks, overrides=(LOAD,), **params)
        )
    df = run_cells(cells, seeds, root, smoke=smoke, n_jobs=1 if smoke else n_jobs)
    if df.groupby("seed")["pool_gpus"].nunique().max() != 1:
        raise AssertionError("cells of one experiment disagree on the GPU pool: load override lost")
    table = summarize_cells(df, oracle_tag="ref")
    table = table[(table["scheduler"] != "S0_asap") | (table["tag"] == "ref")]
    table.to_csv(out / "ablations.csv", index=False)

    mpc = table[table["scheduler"] == "S4_mpc"].reset_index(drop=True)
    fig, ax = new_axes("Ablations: carbon saved vs ASAP (95% CI)", "", "Saving (%)", (7.2, 3.8))
    x = list(range(len(mpc)))
    ax.bar(x, 100 * mpc["saving_vs_s0"], color=PALETTE[0], width=0.55)
    lower = (100 * (mpc["saving_vs_s0"] - mpc["saving_lo"])).clip(lower=0)
    upper = (100 * (mpc["saving_hi"] - mpc["saving_vs_s0"])).clip(lower=0)
    ax.errorbar(
        x,
        100 * mpc["saving_vs_s0"],
        yerr=[lower, upper],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=1,
    )
    ax.set_xticks(x, mpc["tag"], fontsize=7, rotation=25, ha="right")
    save(fig, out / "ablation_savings")

    key = {"seeds": seeds, "weeks": n_weeks, "variants": list(variants)}
    summary = {
        "config_hash": hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12],
        "smoke": smoke,
        "mpc_ci_mode_chosen_on_calibration": ci_mode,
        "not_run": ["no_weather (weather model F4 not built)"],
        "estimator": "SYNTHETIC lognormal duration/energy noise (sigma=0.3 unless stated)",
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
