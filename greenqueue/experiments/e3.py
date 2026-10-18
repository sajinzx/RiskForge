"""E3: does the energy model generalise? Random vs leave-one-model-out vs leave-one-GPU-out.

Runs on the owner's measurements in ``data/raw/energy_own``. ``synthetic=True`` exercises the code
on generated data and writes to ``results/e3_SYNTHETIC``; those numbers are NOT evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenqueue.energy.models import GBM, GrayBox, LogLinear
from greenqueue.energy.preprocess import aggregate_configs, load_gpu_specs, load_runs
from greenqueue.energy.synthetic import make_synthetic
from greenqueue.energy.validation import Scheme, evaluate, summarize
from greenqueue.eval.plots import PALETTE, new_axes, save

MODELS = {"EN0_loglinear": LogLinear, "EN1_lightgbm": GBM, "EN2_graybox": GrayBox}
SCHEMES: tuple[Scheme, ...] = ("random", "lomo", "logo")


def run_e3(
    root: Path, *, synthetic: bool = False, alpha: float = 0.1, seed: int = 0
) -> dict[str, Any]:
    out = root / "results" / ("e3_SYNTHETIC" if synthetic else "e3")
    out.mkdir(parents=True, exist_ok=True)
    if synthetic:
        df = make_synthetic(seed)
    else:
        specs = load_gpu_specs(root / "configs" / "gpus.yaml")
        runs = load_runs(root / "data" / "raw" / "energy_own")
        df = aggregate_configs(runs, specs)
    n_gpus = df["gpu"].nunique()

    rows = []
    for name, factory in MODELS.items():
        for scheme in SCHEMES:
            if scheme == "logo" and n_gpus < 2:
                continue  # one GPU type: LOGO is undefined (reported as missing, not faked)
            rows.append(
                {
                    "model": name,
                    "scheme": scheme,
                    **summarize(evaluate(df, factory, scheme, alpha=alpha, seed=seed)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "generalisation.csv", index=False)

    fig, ax = new_axes(
        "Energy-model error by validation scheme" + (" (SYNTHETIC)" if synthetic else ""),
        "",
        "Median abs. % error (energy/step)",
    )
    width = 0.25
    for i, scheme in enumerate(SCHEMES):
        sub = table[table["scheme"] == scheme].set_index("model").reindex(list(MODELS))
        ax.bar(
            np.arange(len(MODELS)) + i * width,
            100 * sub["medape_energy"],
            width * 0.9,
            color=PALETTE[i],
            label=scheme,
        )
    ax.set_xticks(np.arange(len(MODELS)) + width, list(MODELS), fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "generalisation")

    summary = {
        "config_hash": hashlib.sha256(
            json.dumps(
                {"alpha": alpha, "seed": seed, "n": len(df), "gpus": sorted(df["gpu"].unique())}
            ).encode()
        ).hexdigest()[:12],
        "synthetic": synthetic,
        "n_configs": len(df),
        "gpus": sorted(df["gpu"].unique()),
        "results": table.round(4).to_dict("records"),
        "note": "SYNTHETIC DATA: pipeline check only, not evidence"
        if synthetic
        else "own measurements",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
