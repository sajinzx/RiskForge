"""SYNTHETIC measurement table for tests and pipeline checks only (never a result).

The generator is compute-bound physics with family-specific efficiency, a baseline power draw,
a precision-dependent power factor and 5% noise. It is close to the gray-box form, so accuracy on
it says nothing about real GPUs (in particular real LLM decoding is memory-bound, which this
ignores). It only exercises the code paths. Rows are tagged ``source="synthetic"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greenqueue.energy.harness.sweep import default_grid

# Spec-sheet-like values, for generating data only (see configs/gpus.yaml for the modelling copy).
SPECS = {
    "T4": (8.1, 65.0, 70.0, 0.25),
    "P100": (9.3, 18.7, 250.0, 0.12),
    "V100": (15.7, 125.0, 300.0, 0.12),
    "L4": (30.3, 121.0, 72.0, 0.2),
    "A100": (19.5, 312.0, 400.0, 0.1),
}
FAMILY_EFF = {"cnn": 0.35, "vit": 0.55, "bert": 0.5, "llm": 0.12}


def _flops(row: pd.Series) -> float:
    b, s = float(row["batch"]), float(row["seq"])
    fam = row["family"]
    if fam == "cnn":
        return 0.22e9 * b
    if fam == "vit":
        return 105e9 * b
    if fam == "bert":
        return 6 * 1.1e8 * b * s
    return 2 * 1.1e9 * b * s  # llm: prompt + generated tokens


def make_synthetic(seed: int = 0, noise: float = 0.05, repeats: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for gpu, (fp32, fp16, tdp, idle_frac) in SPECS.items():
        for cfg in default_grid():
            r = pd.Series({"batch": cfg.batch, "seq": cfg.seq, "family": cfg.family})
            flops = _flops(r)
            peak = (fp32 if cfg.precision == "fp32" else fp16) * 1e12
            work = cfg.batch * max(cfg.seq, 1)
            sat = work / (work + (30.0 if cfg.family == "llm" else 12.0))
            eff = FAMILY_EFF[cfg.family] * sat * (0.85 + 0.3 * rng.random() * 0.0)
            t = flops / (peak * eff)
            idle = idle_frac * tdp
            power = idle + (tdp - idle) * (0.25 + 0.75 * sat**1.2) * (
                0.9 if cfg.precision == "fp16" else 1.0
            )
            for rep in range(repeats):
                f = np.exp(rng.normal(0, noise))
                rows.append(
                    {
                        "gpu": gpu,
                        "config_key": cfg.key(),
                        "workload": cfg.workload,
                        "model": cfg.model,
                        "family": cfg.family,
                        "batch": cfg.batch,
                        "seq": cfg.seq,
                        "precision": cfg.precision,
                        "flops_per_step": flops,
                        "step_time_s": t * f,
                        "energy_per_step_kwh": (power - idle)
                        * t
                        * f
                        * np.exp(rng.normal(0, noise))
                        / 3.6e6,
                        "idle_w": idle,
                        "mean_util_pct": 100 * sat,
                        "tdp_w": tdp,
                        "peak_tflops": peak / 1e12,
                        "repeat": rep,
                        "source": "synthetic",
                    }
                )
    df = pd.DataFrame(rows)
    keys = ["gpu", "config_key"]
    agg = df.groupby(keys, as_index=False).agg(
        energy_per_step_kwh=("energy_per_step_kwh", "mean"),
        step_time_s=("step_time_s", "mean"),
        repeats=("repeat", "count"),
    )
    first = df.drop(columns=["energy_per_step_kwh", "step_time_s", "repeat"]).drop_duplicates(keys)
    return first.merge(agg, on=keys).reset_index(drop=True)
