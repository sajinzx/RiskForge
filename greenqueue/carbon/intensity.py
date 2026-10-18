"""Carbon intensity from a generation mix: CI = sum(G_f * EF_f) / sum(G_f) (§6.2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_emission_factors(path: Path) -> tuple[dict[str, float], set[str]]:
    """Return ``({production type: gCO2eq/kWh}, excluded types)`` from the YAML config."""
    cfg = yaml.safe_load(path.read_text(encoding="utf8"))
    factors = {k: float(v["ef"]) for k, v in cfg["factors"].items()}
    return factors, set(cfg.get("exclude_from_denominator", []))


def generation_columns(columns: pd.Index) -> dict[object, str]:
    """Map raw ENTSO-E columns to production-type names, dropping storage *consumption*.

    entsoe-py returns a (type, kind) MultiIndex with kind 'Actual Aggregated' or
    'Actual Consumption'. Consumption (pumping) is not generation, so it is excluded.
    """
    out: dict[object, str] = {}
    for col in columns:
        if isinstance(col, tuple):
            name, kind = col[0], str(col[1])
            if "consumption" in kind.lower():
                continue
            out[col] = str(name)
        else:
            out[col] = str(col)
    return out


def compute_intensity(
    generation: pd.DataFrame, factors: dict[str, float], excluded: set[str] | None = None
) -> pd.Series:
    """Weighted-average emission factor of the mix at each timestamp.

    Negative generation is clipped to 0. Unknown production types raise, because silently
    dropping a fuel would bias the intensity. Rows with zero total generation give NaN.
    """
    excluded = excluded or set()
    mapping = generation_columns(generation.columns)
    unknown = sorted({n for n in mapping.values() if n not in factors})
    if unknown:
        raise KeyError(f"no emission factor for production types: {unknown}")
    use = [c for c, n in mapping.items() if n not in excluded]
    g = generation[use].clip(lower=0.0)
    ef = pd.Series([factors[mapping[c]] for c in use], index=g.columns)
    total = g.sum(axis=1, min_count=1)
    weighted = (g * ef).sum(axis=1, min_count=1)
    ci = weighted / total.where(total > 0)
    ci.name = "ci"
    return ci
