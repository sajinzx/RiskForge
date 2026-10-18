"""E7: sensitivity of savings to slack, load and interruptibility.

Runs the fast schedulers (greedy window, simple robust) and the oracle. The zone axis of the
guide is not covered because only Great Britain data is available (no ENTSO-E token).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from greenqueue.eval.plots import new_axes, save
from greenqueue.experiments.sweeps import Cell, run_cells, summarize_cells

SLACKS = {"6h": 6, "12h": 12, "24h": 24, "48h": 48}
LOADS = (0.4, 0.6, 0.8)
SCHEDS = ("S0_asap", "S1_greedy", "S4_simple", "S3_oracle")


def run_e7(root: Path, *, smoke: bool = False, n_jobs: int = 6) -> dict[str, Any]:
    out = root / "results" / ("e7_smoke" if smoke else "e7")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [0] if smoke else [0, 1, 2]
    n_weeks = 1 if smoke else 4
    slacks = {"12h": 12, "48h": 48} if smoke else SLACKS
    loads = (0.6,) if smoke else LOADS
    cells = []
    for sname, sh in slacks.items():
        for u in loads:
            over: tuple[tuple[str, Any], ...] = (
                ("slack_h", [sh]),
                ("slack_probs", [1.0]),
                ("target_utilization", u),
            )
            cells.append(Cell(f"slack{sname}_load{u}", SCHEDS, n_weeks=n_weeks, overrides=over))
    for share in (0.0, 1.0):
        over = (("interruptible_share", share), ("target_utilization", 0.6))
        cells.append(Cell(f"interruptible{share}", SCHEDS, n_weeks=n_weeks, overrides=over))
    df = run_cells(cells, seeds, root, smoke=smoke, n_jobs=1 if smoke else n_jobs)
    table = summarize_cells(df)
    table.to_csv(out / "sensitivity.csv", index=False)

    for sched in ("S1_greedy", "S4_simple", "S3_oracle"):
        sub = table[(table["scheduler"] == sched) & table["tag"].str.startswith("slack")].copy()
        sub["slack"] = sub["tag"].str.extract(r"slack(\w+)_")[0]
        sub["load"] = sub["tag"].str.extract(r"load([\d.]+)")[0].astype(float)
        grid = sub.pivot(index="slack", columns="load", values="saving_vs_s0").reindex(list(slacks))
        fig, ax = new_axes(
            f"{sched}: carbon saved vs ASAP (%)", "Cluster load", "Slack", (5.2, 3.6)
        )
        vals = 100 * grid.to_numpy(dtype=float)
        im = ax.imshow(vals, cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks(range(grid.shape[1]), [str(c) for c in grid.columns])
        ax.set_yticks(range(grid.shape[0]), list(grid.index))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="saving (%)")
        save(fig, out / f"heatmap_{sched}")

    key = {"seeds": seeds, "weeks": n_weeks, "slacks": list(slacks), "loads": list(loads)}
    summary = {
        "config_hash": hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12],
        "smoke": smoke,
        "not_run": ["zone axis (only GB available)"],
        "estimator": "SYNTHETIC lognormal duration noise (sigma=0.3), alpha=0.05",
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
