"""Measured runs to a per-configuration modelling table (implementation.md §7.5).

All repeats are kept in the raw files; the modelling table has one row per configuration with the
mean and standard deviation of each target. Repeats whose energy is more than 3 MAD from the
configuration median are flagged (``outlier_repeats``), not deleted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TARGETS = ("energy_per_step_kwh", "step_time_s")


def load_gpu_specs(path: Path) -> dict[str, dict[str, object]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf8"))
    return {k: v for k, v in cfg["gpus"].items()}


def gpu_key(name: str, specs: dict[str, dict[str, object]]) -> str:
    """Map an NVML device name such as 'Tesla T4' to a key of ``gpus.yaml``."""
    for key, spec in specs.items():
        if str(spec["match"]) in name:
            return key
    raise KeyError(f"GPU {name!r} not in gpus.yaml; add its spec-sheet values first")


def load_runs(root: Path) -> pd.DataFrame:
    """Read every ``runs.jsonl`` below ``root`` (one folder per GPU and date)."""
    frames = []
    for path in sorted(root.rglob("runs.jsonl")):
        rows = [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise FileNotFoundError(f"no runs.jsonl under {root}")
    return pd.concat(frames, ignore_index=True)


def flag_outlier_repeats(df: pd.DataFrame, min_rel: float = 0.10) -> pd.Series:
    """True for repeats beyond 3 MAD and more than ``min_rel`` of their (GPU, config) median energy.

    Grouped per GPU when a ``gpu_name`` column exists: the same configuration on another GPU has a
    different energy, so pooling GPUs would flag whole GPUs instead of disturbed repeats.
    """
    keys = [df["gpu_name"], df["config_key"]] if "gpu_name" in df else [df["config_key"]]
    e = df["energy_per_step_kwh"]
    med = e.groupby(keys).transform("median")
    mad = (e - med).abs().groupby(keys).transform("median")
    beyond_mad = ((e - med).abs() > 3 * mad.replace(0, np.nan)).fillna(False)
    # Also require a material deviation: on a quiet GPU the repeats agree within ~1%, so 3 MAD alone
    # would flag harmless 1-2% wobble. Disturbances (other programs, throttling) are 25% or more.
    return beyond_mad & ((e - med).abs() > min_rel * med)


def aggregate_configs(runs: pd.DataFrame, specs: dict[str, dict[str, object]]) -> pd.DataFrame:
    """One row per (GPU, configuration) with mean and std of the targets."""
    runs = runs.copy()
    runs["gpu"] = runs["gpu_name"].map(lambda n: gpu_key(str(n), specs))
    runs["outlier_repeat"] = flag_outlier_repeats(runs)
    keys = ["gpu", "config_key"]
    first = runs.groupby(keys).first()
    n_out = runs.groupby(keys)["outlier_repeat"].sum()
    # Disturbed repeats (other programs on a laptop, throttling) are left out of the means.
    # A configuration keeps all its repeats if dropping would leave fewer than two.
    kept = runs[~runs["outlier_repeat"]]
    thin = kept.groupby(keys).size().reindex(n_out.index, fill_value=0) < 2
    runs = pd.concat([kept, runs[runs.set_index(keys).index.isin(thin[thin].index)]])
    runs = runs[~runs.index.duplicated()]
    agg = runs.groupby(keys).agg(
        energy_per_step_kwh=("energy_per_step_kwh", "mean"),
        energy_std=("energy_per_step_kwh", "std"),
        step_time_s=("step_time_s", "mean"),
        step_time_std=("step_time_s", "std"),
        idle_w=("idle_w", "mean"),
        mean_util_pct=("mean_util_pct", "mean"),
        repeats=("repeat", "count"),
    )
    agg["outlier_repeats"] = n_out
    cols = ["workload", "model", "family", "batch", "seq", "precision", "flops_per_step", "source"]
    out = first[cols].join(agg).reset_index()
    out["tdp_w"] = [float(specs[g]["tdp_w"]) for g in out["gpu"]]  # type: ignore[arg-type]
    out["peak_tflops"] = [
        float(specs[g]["peak_tflops"][p])  # type: ignore[index]
        for g, p in zip(out["gpu"], out["precision"], strict=True)
    ]
    return out
