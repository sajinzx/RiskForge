from pathlib import Path

import pandas as pd

from greenqueue.experiments.e6 import run_e6
from greenqueue.experiments.e7 import run_e7


def test_e6_smoke(sched_root: Path) -> None:
    s = run_e6(sched_root, smoke=True)
    out = sched_root / "results" / "e6_smoke"
    t = pd.read_csv(out / "ablations.csv")
    assert (out / "ablation_savings.png").exists()
    tags = set(t[t["scheduler"] == "S4_mpc"]["tag"])
    assert {"full", "no_conformal_bounds", "carbon_point_only"} <= tags
    assert (t["unfinished"] == 0).all() and s["not_run"]


def test_e7_smoke(sched_root: Path) -> None:
    s = run_e7(sched_root, smoke=True)
    out = sched_root / "results" / "e7_smoke"
    t = pd.read_csv(out / "sensitivity.csv")
    assert (out / "heatmap_S1_greedy.png").exists()
    # more slack gives the scheduler more room, so the oracle saves at least as much
    orc = t[t["scheduler"] == "S3_oracle"].set_index("tag")["saving_vs_s0"]
    assert orc["slack48h_load0.6"] >= orc["slack12h_load0.6"] - 1e-9
    assert s["smoke"]
