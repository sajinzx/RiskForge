from pathlib import Path

import pandas as pd

from greenqueue.demo.tables import alpha_vs_violation, load_tables


def test_load_tables_marks_missing_experiments(tmp_path: Path) -> None:
    (tmp_path / "e5").mkdir()
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "e5" / "alpha_tracking.csv", index=False)
    t = load_tables(tmp_path)
    assert t["alpha"] is not None and t["scheduling"] is None


def test_alpha_vs_violation_keeps_mpc_rows_with_alpha() -> None:
    df = pd.DataFrame(
        {
            "scenario": ["calibrated", "calibrated", "reference"],
            "alpha": [0.1, 0.1, float("nan")],
            "scheduler": ["S4_mpc", "S4_simple", "S0_asap"],
            "violation_rate": [0.004, 0.008, 0.014],
            "violation_lo": [0.0, 0.0, 0.0],
            "violation_hi": [0.01, 0.01, 0.04],
        }
    )
    out = alpha_vs_violation(df)
    assert len(out) == 1 and out["violation_rate"].iloc[0] == 0.004
