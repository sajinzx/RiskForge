"""E5: does the realised deadline-violation rate track the target alpha?

Sweeps the conformal level alpha of the duration bound used by the robust schedulers (simple S4
and the full MPC S4) and measures the share of jobs that miss their deadline, in two scenarios:
``calibrated`` (the duration bound was calibrated on the same error distribution it is deployed on)
and ``shifted`` (deployment errors are 1.6x larger than the calibration errors). The duration
estimator is a synthetic lognormal-noise stand-in until real energy measurements exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenqueue.eval.bootstrap import paired_saving_ci, rate_ci
from greenqueue.eval.plots import PALETTE, new_axes, save
from greenqueue.experiments.sweeps import Cell, choose_mpc_ci_mode, run_cells

ALPHAS = (0.01, 0.05, 0.1, 0.2)
SIGMA = 0.3
SHIFT = 1.6
# At the guide's 60% load plain ASAP already misses 1-19% of deadlines from queueing alone, which
# swamps the effect of the duration bound; E5 therefore runs at 30% load (E7 sweeps load).
UTILISATION = ("target_utilization", 0.3)


def run_e5(
    root: Path, *, smoke: bool = False, n_jobs: int = 6, alphas: tuple[float, ...] | None = None
) -> dict[str, Any]:
    out = root / "results" / ("e5_smoke" if smoke else "e5")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [0] if smoke else [0, 1, 2]
    n_weeks = 1 if smoke else 4
    grid = alphas or ((0.1,) if smoke else ALPHAS)
    ci_mode = choose_mpc_ci_mode(root, smoke=smoke, out=out)
    scenarios = {"calibrated": None, "shifted": SIGMA / SHIFT}
    if smoke:
        scenarios = {"calibrated": None}

    cells = [
        Cell(
            "ref",
            ("S0_asap", "S1_greedy", "S3_oracle"),
            sigma=SIGMA,
            n_weeks=n_weeks,
            overrides=(UTILISATION,),
        )  # fmt: skip
    ]
    for scen, cal_sigma in scenarios.items():
        for a in grid:
            cells.append(
                Cell(
                    f"{scen}_a{a}",
                    ("S0_asap", "S4_simple", "S4_mpc"),
                    alpha=a,
                    sigma=SIGMA,
                    cal_sigma=cal_sigma,
                    mpc_ci_mode=ci_mode,
                    n_weeks=n_weeks,
                    overrides=(UTILISATION,),
                )  # fmt: skip
            )
    df = run_cells(cells, seeds, root, smoke=smoke, n_jobs=1 if smoke else n_jobs)
    if df.groupby("seed")["pool_gpus"].nunique().max() != 1:
        raise AssertionError("cells of one experiment disagree on the GPU pool: load override lost")
    df["block"] = df["seed"].astype(str) + "_" + df["week"].astype(str)
    df.to_parquet(out / "per_job.parquet")

    s0 = df[(df["tag"] == "ref") & (df["scheduler"] == "S0_asap")].set_index(["seed", "id"])
    orc = df[(df["tag"] == "ref") & (df["scheduler"] == "S3_oracle")].set_index(["seed", "id"])
    rows = []
    for (tag, sched), g in df.groupby(["tag", "scheduler"]):
        if sched == "S0_asap" and tag != "ref":
            continue
        rate, lo, hi = rate_ci(g["violated"].to_numpy(), g["block"].to_numpy())
        gi = g.set_index(["seed", "id"])
        base = s0.loc[gi.index, "emissions_g"].to_numpy()
        sav, slo, shi = paired_saving_ci(base, gi["emissions_g"].to_numpy(), gi["block"].to_numpy())
        o_sav, *_ = paired_saving_ci(
            base, orc.loc[gi.index, "emissions_g"].to_numpy(), gi["block"].to_numpy()
        )
        scen, _, a_str = str(tag).partition("_a")
        rows.append(
            {
                "tag": tag, "scenario": scen if a_str else "reference",
                "alpha": float(a_str) if a_str else np.nan,
                "scheduler": sched, "violation_rate": rate, "violation_lo": lo, "violation_hi": hi,
                "saving_vs_s0": sav, "saving_lo": slo, "saving_hi": shi,
                "share_of_oracle": sav / o_sav if o_sav else np.nan,
                "mpc_fallbacks": int(g.groupby("seed")["fallbacks"].first().sum()),
                "mpc_solves": int(g.groupby("seed")["solves"].first().sum()),
                "unfinished": int((~g["finished"]).sum()),
            }
        )  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(out / "alpha_tracking.csv", index=False)

    fig, ax = new_axes(
        "Realised deadline violations vs target alpha", "Target alpha", "Violation rate", (6.4, 4.0)
    )
    ax.plot([0, max(grid)], [0, max(grid)], color="#444444", lw=0.8, ls="--", label="y = alpha")
    style = {"calibrated": "-", "shifted": ":"}
    for i, sched in enumerate(("S4_simple", "S4_mpc")):
        for scen in scenarios:
            sub = table[(table["scheduler"] == sched) & (table["scenario"] == scen)].sort_values(
                "alpha"
            )
            ax.plot(
                sub["alpha"],
                sub["violation_rate"],
                style[scen],
                color=PALETTE[i],
                marker="o",
                ms=4,
                lw=1.5,
                label=f"{sched} ({scen})",
            )
            ax.fill_between(
                sub["alpha"],
                sub["violation_lo"],
                sub["violation_hi"],
                color=PALETTE[i],
                alpha=0.12,
                lw=0,
            )
    ref = table[(table["tag"] == "ref") & (table["scheduler"] == "S1_greedy")]
    if len(ref):
        ax.axhline(
            float(ref["violation_rate"].iloc[0]),
            color=PALETTE[3],
            lw=1.2,
            label="S1_greedy (no bound)",
        )
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, out / "alpha_vs_violations")

    summary = {
        "config_hash": hashlib.sha256(
            json.dumps(
                {"seeds": seeds, "weeks": n_weeks, "alphas": grid, "sigma": SIGMA, "shift": SHIFT}
            ).encode()
        ).hexdigest()[:12],
        "smoke": smoke,
        "mpc_ci_mode_chosen_on_calibration": ci_mode,
        "seeds": seeds,
        "n_weeks": n_weeks,
        "estimator": (
            f"SYNTHETIC lognormal noise sigma={SIGMA}; "
            f"shifted scenario calibrates on sigma={SIGMA / SHIFT:.3f}"
        ),
        "all_finished": bool((df["finished"]).all()),
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not summary["all_finished"]:
        raise AssertionError("some jobs did not finish inside the replay window")
    return summary
