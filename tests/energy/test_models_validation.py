import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenqueue.energy.models import GBM, GrayBox, LogLinear
from greenqueue.energy.preprocess import (
    aggregate_configs,
    flag_outlier_repeats,
    gpu_key,
    load_gpu_specs,
    load_runs,
)
from greenqueue.energy.synthetic import make_synthetic
from greenqueue.energy.validation import GroupConformal, evaluate, make_folds, summarize
from greenqueue.experiments.e3 import run_e3

ROOT = Path(__file__).parents[2]
SPECS = load_gpu_specs(ROOT / "configs" / "gpus.yaml")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return make_synthetic(0)


def test_gpu_key_and_missing_gpu() -> None:
    assert gpu_key("Tesla T4", SPECS) == "T4" and gpu_key("NVIDIA A100-SXM4-40GB", SPECS) == "A100"
    with pytest.raises(KeyError):
        gpu_key("Mystery GPU", SPECS)


def test_aggregate_and_outlier_flag(tmp_path: Path) -> None:
    rows = []
    for rep, e in enumerate([1.0, 1.1, 0.9, 1.0, 9.0]):
        rows.append(
            {
                "config_key": "k",
                "repeat": rep,
                "gpu_name": "Tesla T4",
                "workload": "resnet18",
                "model": "resnet18",
                "family": "cnn",
                "batch": 32,
                "seq": 0,
                "precision": "fp16",
                "flops_per_step": 1e9,
                "source": "own",
                "energy_per_step_kwh": e * 1e-6,
                "step_time_s": 0.1,
                "idle_w": 10.0,
                "mean_util_pct": 50.0,
            }
        )
    d = tmp_path / "T4" / "2026-01-01"
    d.mkdir(parents=True)
    (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    runs = load_runs(tmp_path)
    assert flag_outlier_repeats(runs).tolist() == [False, False, False, False, True]
    agg = aggregate_configs(runs, SPECS)
    assert (
        len(agg) == 1 and agg["outlier_repeats"].iloc[0] == 1 and agg["peak_tflops"].iloc[0] == 65.0
    )


def test_folds_hold_out_whole_groups(df: pd.DataFrame) -> None:
    for scheme, col in (("lomo", "model"), ("logo", "gpu")):
        for tr, te in make_folds(df, scheme):  # type: ignore[arg-type]
            assert set(df.iloc[tr][col]).isdisjoint(set(df.iloc[te][col]))
    tr, te = make_folds(df, "random")[0]
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(df)


def test_models_fit_and_predict_positive_values(df: pd.DataFrame) -> None:
    for factory in (LogLinear, GBM, GrayBox):
        pred = factory().fit(df).predict(df)
        assert (pred["energy"] > 0).all() and (pred["time"] > 0).all()
        assert np.isfinite(pred.to_numpy()).all()


def test_graybox_params_respect_bounds_and_unseen_gpu_uses_pooled(df: pd.DataFrame) -> None:
    m = GrayBox().fit(df[df["gpu"] != "L4"])
    lo, hi = GrayBox.BOUNDS
    for p in m.params.values():
        assert np.all(p >= np.array(lo) - 1e-9) and np.all(p <= np.array(hi) + 1e-9)
    pred = m.predict(df[df["gpu"] == "L4"])  # never seen: must still work from spec-sheet values
    assert (pred["energy"] > 0).all()


def test_random_split_is_easier_than_leaving_a_model_out_for_blackbox(df: pd.DataFrame) -> None:
    rnd = summarize(evaluate(df, GBM, "random"))["medape_energy"]
    logo = summarize(evaluate(df, GBM, "logo"))["medape_energy"]
    assert logo > rnd  # the generalisation gap the guide asks us to report


def test_group_conformal_pooled_fallback_and_coverage() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    cal = pd.DataFrame({"family": rng.choice(["a", "b"], n), "gpu": rng.choice(["x", "y"], n)})
    y = np.exp(rng.normal(0, 0.1, n))
    gc = GroupConformal(0.1).fit(cal, y, np.ones(n))
    test = pd.DataFrame({"family": ["a"] * 2000, "gpu": ["x"] * 2000})
    yt = np.exp(rng.normal(0, 0.1, 2000))
    lo, hi = gc.interval(test, np.ones(2000))
    assert abs(((yt >= lo) & (yt <= hi)).mean() - 0.9) < 0.03
    small = GroupConformal(0.1, min_group=10_000).fit(cal, y, np.ones(n))  # all groups too small
    assert small.q["two_sided"]["family"] == {} and np.isfinite(small.pooled["two_sided"])


def test_e3_synthetic_run_is_labelled_and_complete(tmp_path: Path) -> None:
    s = run_e3(tmp_path, synthetic=True)
    assert s["synthetic"] and "SYNTHETIC" in s["note"]
    assert (tmp_path / "results" / "e3_SYNTHETIC" / "generalisation.png").exists()
    assert len(s["results"]) == 9
