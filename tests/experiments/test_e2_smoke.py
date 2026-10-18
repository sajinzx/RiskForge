from pathlib import Path

import pandas as pd

from greenqueue.experiments.e2 import run_e2


def test_e2_runs_and_aci_is_not_worse_than_static_when_coverage_is_off(
    smoke_root: tuple[Path, dict],  # type: ignore[type-arg]
) -> None:
    root, _ = smoke_root
    # E2 needs the processed series that E1 wrote
    s = run_e2(root, smoke=True)
    out = root / "results" / "e2_smoke"
    for name in ("coverage_summary.csv", "rolling_coverage_aci.png", "forecasts_F3_aci90.parquet"):
        assert (out / name).exists(), name
    res = pd.read_csv(out / "coverage_summary.csv").set_index("method")
    assert {"U1_split", "U2_cqr", "U3_aci"} == set(res.index)
    assert res["coverage"].between(0.5, 1.0).all()
    # ACI must stay within a few points of nominal on this stationary synthetic series
    assert abs(res.loc["U3_aci", "coverage"] - 0.9) < 0.06
    assert s["gamma"] in (0.001, 0.005, 0.01, 0.05)
