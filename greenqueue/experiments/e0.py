"""E0 (task T1.4): upper bound on temporal-shifting savings for single fixed-length jobs.

A job of ``d`` hours that could start now may instead start at any whole hour within the next
``w`` hours (it must finish before ``now + w + d``). With perfect knowledge of the carbon intensity
the saving is ``1 - min_start(mean CI over run) / mean CI of starting now``. Averaged over every
hour of the test period. This is the pipeline sanity check against the temporal-shifting
literature; compare the shape (more flexibility, more saving; short jobs gain most) rather than
exact numbers, which depend on the grid and period. Published numbers are NOT quoted here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from greenqueue.eval.bootstrap import block_bootstrap_ci
from greenqueue.eval.plots import PALETTE, new_axes, save

DURATIONS = (1, 4, 8, 24)
WINDOWS = (6, 12, 24, 48, 96, 168)


def shifting_saving(ci: np.ndarray, d: int, w: int) -> np.ndarray:
    """Per-hour saving of the best start within ``w`` hours versus starting now."""
    csum = np.concatenate([[0.0], np.cumsum(ci)])
    run_mean = (csum[d:] - csum[:-d]) / d  # mean CI of a run starting at each hour
    n = len(run_mean) - w
    now = run_mean[:n]
    best = np.min(np.lib.stride_tricks.sliding_window_view(run_mean, w + 1)[:n], axis=1)
    return np.asarray(1.0 - best / now)


def run_e0(root: Path, *, smoke: bool = False) -> dict[str, Any]:
    cfg = yaml.safe_load((root / "configs" / "splits.yaml").read_text(encoding="utf8"))
    sp = cfg["smoke"] if smoke else cfg
    ci = pd.read_parquet(root / "data" / "processed" / "gb_hourly.parquet")["ci"]
    ci = ci.loc[pd.Timestamp(sp["test_start"], tz="UTC") : pd.Timestamp(sp["test_end"], tz="UTC")]
    out = root / "results" / ("e0_smoke" if smoke else "e0")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DURATIONS:
        for w in WINDOWS:
            if len(ci) <= w + d + 24:
                continue
            s = shifting_saving(ci.to_numpy(), d, w)
            idx = pd.DatetimeIndex(ci.index[: len(s)])
            m, lo, hi = block_bootstrap_ci(s, idx)
            rows.append({"duration_h": d, "window_h": w, "saving": m, "lo": lo, "hi": hi})
    table = pd.DataFrame(rows)
    table.to_csv(out / "shifting_potential.csv", index=False)
    fig, ax = new_axes("Perfect-foresight shifting saving (GB)", "Window (h)", "Saving (%)")
    for i, d in enumerate(DURATIONS):
        t = table[table["duration_h"] == d]
        ax.plot(t["window_h"], 100 * t["saving"], marker="o", color=PALETTE[i], label=f"{d} h job")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "shifting_potential")
    key = json.dumps({"d": DURATIONS, "w": WINDOWS, "sp": str(sp)}, sort_keys=True)
    summary = {
        "config_hash": hashlib.sha256(key.encode()).hexdigest()[:12],
        "smoke": smoke,
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
