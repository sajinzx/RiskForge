"""Shared fixture: a synthetic GB-like project root with E1 (smoke) already run once."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenqueue.experiments.e1 import run_e1

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:  # type: ignore[type-arg]
    root = tmp_path_factory.mktemp("smoke_root")
    shutil.copytree(ROOT / "configs", root / "configs")
    idx = pd.date_range("2020-12-20", "2023-07-05", freq="30min", tz="UTC")
    t = np.arange(len(idx))
    rng = np.random.default_rng(0)
    ci = 170 + 60 * np.sin(2 * np.pi * t / 48) + 25 * np.sin(2 * np.pi * t / (48 * 7))
    ci = ci + rng.normal(0, 12, len(idx))
    raw = pd.DataFrame(
        {"forecast": ci + rng.normal(0, 8, len(idx)), "actual": ci, "index": "low"}, index=idx
    )
    raw_dir = root / "data" / "raw" / "uk_ci"
    raw_dir.mkdir(parents=True)
    raw.to_parquet(raw_dir / "intensity.parquet")
    summary = run_e1(root, smoke=True, trials=1)
    return root, summary
