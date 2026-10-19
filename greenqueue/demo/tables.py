"""Load the result tables the dashboard shows (no plotting, so testable without streamlit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = {
    "forecast": "e2/coverage_summary.csv",
    "scheduling": "e4/scheduling_results.csv",
    "alpha": "e5/alpha_tracking.csv",
    "accounting": "e8/accounting_sensitivity.csv",
    "energy": "e3/generalisation.csv",
}


def load_tables(results: Path) -> dict[str, pd.DataFrame | None]:
    """Each table, or None when that experiment has not been run."""
    out: dict[str, pd.DataFrame | None] = {}
    for key, rel in FILES.items():
        path = results / rel
        out[key] = pd.read_csv(path) if path.exists() else None
    return out


def alpha_vs_violation(alpha: pd.DataFrame) -> pd.DataFrame:
    """The robust MPC rows of the alpha-tracking table, by scenario."""
    sub = alpha[(alpha["scheduler"] == "S4_mpc") & alpha["alpha"].notna()]
    return sub[["scenario", "alpha", "violation_rate", "violation_lo", "violation_hi"]]
