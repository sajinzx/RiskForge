"""E4 end to end on synthetic carbon, forecasts and trace tasks (no network, no real data)."""

from pathlib import Path

from greenqueue.experiments.e4 import run_e4


def test_e4_smoke_end_to_end(sched_root: Path) -> None:
    tmp_path = sched_root
    summary = run_e4(tmp_path, smoke=True)

    out = tmp_path / "results" / "e4_smoke"
    assert (out / "scheduling_results.csv").exists() and (out / "savings_vs_s0.png").exists()
    assert all(summary["sanity"].values())
    res = {r["scheduler"]: r for r in summary["results"]}
    # the oracle is the best possible, and any informed scheduler saves carbon on a cyclic grid
    assert (
        res["S3_oracle"]["saving_vs_s0"]
        >= max(res[s]["saving_vs_s0"] for s in ("S1_greedy", "S2_threshold", "S4_simple")) - 1e-6
    )
    assert res["S1_greedy"]["saving_vs_s0"] > 0
    assert res["S3_oracle"]["saving_vs_s0"] > 0
