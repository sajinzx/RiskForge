"""End-to-end smoke test of E1 on a synthetic GB-like series (no network, no real data)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from greenqueue.experiments.e1 import run_e1

ROOT = Path(__file__).parents[2]


def test_e1_runs_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    idx = pd.date_range("2020-12-20", "2023-07-05", freq="30min", tz="UTC")
    t = np.arange(len(idx))
    rng = np.random.default_rng(0)
    ci = (
        170
        + 60 * np.sin(2 * np.pi * t / 48)
        + 25 * np.sin(2 * np.pi * t / (48 * 7))
        + rng.normal(0, 12, len(idx))
    )
    raw = pd.DataFrame(
        {"forecast": ci + rng.normal(0, 8, len(idx)), "actual": ci, "index": "low"}, index=idx
    )
    raw_dir = tmp_path / "data" / "raw" / "uk_ci"
    raw_dir.mkdir(parents=True)
    raw.to_parquet(raw_dir / "intensity.parquet")

    summary = run_e1(tmp_path, smoke=True, trials=1)

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
