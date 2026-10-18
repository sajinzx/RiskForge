"""End-to-end smoke test of E1 on a synthetic GB-like series (no network, no real data)."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]


def test_e1_runs_end_to_end_on_synthetic_data(smoke_root: tuple[Path, dict]) -> None:  # type: ignore[type-arg]
    tmp_path, summary = smoke_root

    out = tmp_path / "results" / "e1_smoke"
    for name in (
        "metrics.csv",
        "summary.json",
        "coverage_vs_horizon.png",
        "rolling_coverage.png",
        "calibration.png",
    ):
        assert (out / name).exists(), name
    assert all(summary["sanity"].values())
    m = pd.read_csv(out / "metrics.csv")
    f3 = m[(m["model"] == "F3") & (m["level"] == 0.9)].set_index("method")
    assert abs(f3.loc["U2_cqr", "coverage"] - 0.9) < 0.05  # conformal lands near target
    assert (
        f3.loc["U0_raw", "coverage"] < f3.loc["U2_cqr", "coverage"]
    )  # raw quantiles overconfident
    persistence = m[m["model"] == "F0_persistence"]["mae"].iloc[0]
    assert f3.loc["U2_cqr", "mae"] < persistence
    assert json.loads((out / "summary.json").read_text())["config_hash"] == summary["config_hash"]
