import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from greenqueue.experiments.e9 import run_e9


def test_e9_scaling_removes_the_level_gap_between_zones(
    smoke_root: tuple[Path, dict],  # type: ignore[type-arg]
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    shutil.copytree(smoke_root[0], root)
    proc = root / "data" / "processed"
    gb = pd.read_parquet(proc / "gb_hourly.parquet")
    rng = np.random.default_rng(1)
    fr = pd.DataFrame(
        {"ci": 0.25 * gb["ci"] * (1 + rng.normal(0, 0.03, len(gb))), "imputed": 0}, index=gb.index
    )
    fr.to_parquet(proc / "FR_hourly.parquet")

    s = run_e9(root, smoke=True)
    t = pd.read_csv(root / "results" / "e9_smoke" / "transfer_matrix.csv")
    assert s["zones_used"] == ["FR", "GB"] and "DE_LU" in s["zones_skipped_no_data"]
    assert len(t) == 8  # 2 modes x 2 source zones x 2 target zones

    def skill(mode: str) -> float:
        row = t[(t["mode"] == mode) & (t["source"] == "GB") & (t["target"] == "FR")]
        return float(row["skill_vs_persistence"].iloc[0])

    assert skill("scaled") < skill("raw")  # a 4x lower level breaks the unscaled model
    own = t[(t["mode"] == "scaled") & (t["source"] == "FR") & (t["target"] == "FR")]
    assert (
        abs(float(own["coverage_as_is"].iloc[0]) - float(own["coverage_recalibrated"].iloc[0]))
        < 1e-9
    )
