"""E8: how much of the saving survives under marginal-emissions proxies?

Schedulers plan on the average-intensity forecasts exactly as in E4. Each finished schedule is then
re-scored with other accounting signals built from the GB generation mix (shares per fuel):

- ``average``: the grid average intensity used for planning;
- ``merit_gas``: proxy for the marginal unit by a merit-order rule. If gas supplies at least
  ``GAS_THRESHOLD`` of generation in the hour, the marginal unit is taken to be gas
  (``GAS_EF`` gCO2/kWh, IPCC lifecycle median in ``configs/emission_factors.yaml``), otherwise
  non-fossil (0);
- ``const_gas``: marginal always gas. Control: shifting cannot save carbon under it.

These are PROXIES, not measured marginal emissions. A regression of emissions on demand (the second
proxy named in the guide) is not possible because the data holds fuel shares, not demand in MW.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenqueue.eval.bootstrap import paired_saving_ci
from greenqueue.eval.plots import PALETTE, new_axes, save
from greenqueue.experiments.sweeps import Cell, run_cells

GAS_EF = 490.0
GAS_THRESHOLD = 0.10
SCHEDS = ("S0_asap", "S1_greedy", "S2_threshold", "S4_simple", "S3_oracle")


def accounting_signals(ci: pd.Series, generation: pd.DataFrame) -> pd.DataFrame:
    """Hourly UTC signals (gCO2/kWh) on the index of ``ci``."""
    shares = generation.resample("h").mean()
    gas = (shares["gas"] / shares.sum(axis=1)).reindex(ci.index)
    gas = gas.ffill(limit=24).bfill(limit=24)  # ~2% of hours are missing; bridge gaps up to a day
    sig = pd.DataFrame(index=ci.index)
    sig["average"] = ci
    sig["merit_gas"] = np.where(gas >= GAS_THRESHOLD, GAS_EF, 0.0)
    sig.loc[gas.isna(), "merit_gas"] = ci[gas.isna()]  # longer gaps: fall back to the average
    sig["const_gas"] = GAS_EF
    return sig


def rescore(df: pd.DataFrame, signals: pd.DataFrame, dt_h: float = 1.0) -> pd.DataFrame:
    out = df.drop(columns=["profile"]).copy()
    arr = {c: signals[c].to_numpy() for c in signals}
    idx = signals.index
    for col, vals in arr.items():
        em = np.zeros(len(df))
        for i, (prof, start) in enumerate(zip(df["profile"], df["start"], strict=True)):
            base = int(idx.searchsorted(start))
            em[i] = sum(kwh * vals[base + int(t * dt_h)] for t, kwh in prof)
        out[f"em_{col}"] = em
    return out


def run_e8(root: Path, *, smoke: bool = False, n_jobs: int = 6) -> dict[str, Any]:
    out = root / "results" / ("e8_smoke" if smoke else "e8")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [0] if smoke else [0, 1, 2]
    cell = Cell(
        "main", SCHEDS, n_weeks=1 if smoke else 4, oracle_time_limit_s=10.0 if smoke else 60.0,
        keep_profile=True,
    )  # fmt: skip
    df = run_cells([cell], seeds, root, smoke=smoke, n_jobs=1 if smoke else n_jobs)
    gb = pd.read_parquet(root / "data" / "processed" / "gb_hourly.parquet")["ci"]
    gen = pd.read_parquet(root / "data" / "raw" / "uk_ci" / "generation.parquet")
    signals = accounting_signals(gb, gen)
    scored = rescore(df, signals)
    scored["block"] = scored["seed"].astype(str) + "_" + scored["week"].astype(str)

    rows = []
    s0 = scored[scored["scheduler"] == "S0_asap"].set_index(["seed", "id"])
    for sched, g in scored.groupby("scheduler"):
        gi = g.set_index(["seed", "id"])
        for sig in signals.columns:
            sav, lo, hi = paired_saving_ci(
                s0.loc[gi.index, f"em_{sig}"].to_numpy(),
                gi[f"em_{sig}"].to_numpy(),
                gi["block"].to_numpy(),
            )
            rows.append(
                {"scheduler": sched, "accounting": sig, "saving_vs_s0": sav, "lo": lo, "hi": hi}
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "accounting_sensitivity.csv", index=False)

    fig, ax = new_axes("Saving vs ASAP by accounting", "Scheduler", "Saving (%)", (6.4, 3.8))
    scheds = [s for s in SCHEDS if s != "S0_asap"]
    w = 0.27
    for k, sig in enumerate(signals.columns):
        t = table[table["accounting"] == sig].set_index("scheduler").reindex(scheds)
        err = np.vstack([t["saving_vs_s0"] - t["lo"], t["hi"] - t["saving_vs_s0"]]).clip(min=0)
        ax.bar(np.arange(len(scheds)) + (k - 1) * w, 100 * t["saving_vs_s0"], w,
               yerr=100 * err, color=PALETTE[k], label=sig)  # fmt: skip
    ax.set_xticks(range(len(scheds)), scheds, fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "saving_by_accounting")

    key = {"seeds": seeds, "gas_ef": GAS_EF, "thr": GAS_THRESHOLD, "smoke": smoke}
    summary = {
        "config_hash": hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12],
        "smoke": smoke,
        "gas_share_ge_threshold": float((signals["merit_gas"] > 0).mean()),
        "proxies_are_not_measured_marginal_emissions": True,
        "not_run": ["regression of emissions on demand (no demand data)"],
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
