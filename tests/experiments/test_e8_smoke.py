from pathlib import Path

import numpy as np
import pandas as pd

from greenqueue.experiments.e8 import accounting_signals, run_e8


def test_accounting_signals_rule() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    ci = pd.Series([100.0, 200.0, 300.0, 400.0], index=idx)
    g_idx = pd.date_range("2024-01-01", periods=8, freq="30min", tz="UTC")
    gas = [0.0, 0.0, 50.0, 50.0, 5.0, 5.0, 50.0, 50.0]
    wind = [50.0, 50.0, 50.0, 50.0, 95.0, 95.0, 50.0, 50.0]
    gen = pd.DataFrame({"gas": gas, "wind": wind}, g_idx)
    s = accounting_signals(ci, gen)
    assert list(s["merit_gas"]) == [0.0, 490.0, 0.0, 490.0]
    assert (s["const_gas"] == 490.0).all() and (s["average"] == ci).all()


def test_e8_smoke_average_rescoring_matches_simulator(sched_root: Path) -> None:
    gb = pd.read_parquet(sched_root / "data" / "processed" / "gb_hourly.parquet")
    idx = pd.date_range(gb.index[0], gb.index[-1], freq="30min")
    rng = np.random.default_rng(0)
    gen = pd.DataFrame({"gas": rng.uniform(0, 40, len(idx)), "wind": 30.0}, index=idx)
    d = sched_root / "data" / "raw" / "uk_ci"
    d.mkdir(parents=True, exist_ok=True)
    gen.to_parquet(d / "generation.parquet")
    s = run_e8(sched_root, smoke=True)
    t = pd.read_csv(sched_root / "results" / "e8_smoke" / "accounting_sensitivity.csv")
    assert s["proxies_are_not_measured_marginal_emissions"]
    # constant marginal factor: shifting without pauses cannot change emissions
    c = t[(t["accounting"] == "const_gas") & (t["scheduler"] == "S1_greedy")]
    assert abs(float(c["saving_vs_s0"].iloc[0])) < 1e-9
