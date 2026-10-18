from pathlib import Path

import pandas as pd

from greenqueue.eval.bootstrap import rate_ci
from greenqueue.experiments.e5 import run_e5


def test_rate_ci_brackets_the_rate() -> None:
    ev = [True] * 5 + [False] * 95
    blocks = [i % 10 for i in range(100)]
    r, lo, hi = rate_ci(pd.Series(ev).to_numpy(), pd.Series(blocks).to_numpy(), n_resamples=300)
    assert r == 0.05 and lo <= r <= hi


def test_e5_smoke_runs_and_mpc_has_no_fallbacks(sched_root: Path) -> None:
    s = run_e5(sched_root, smoke=True)
    out = sched_root / "results" / "e5_smoke"
    assert (out / "alpha_tracking.csv").exists() and (out / "alpha_vs_violations.png").exists()
    assert s["all_finished"]
    rows = {(r["tag"], r["scheduler"]): r for r in s["results"]}
    mpc = rows[("calibrated_a0.1", "S4_mpc")]
    assert mpc["mpc_solves"] > 0 and mpc["mpc_fallbacks"] == 0
    assert 0.0 <= mpc["violation_rate"] <= 1.0
    # robust schedulers should still save carbon on a strongly cyclic grid
    assert rows[("calibrated_a0.1", "S4_simple")]["saving_vs_s0"] > 0
