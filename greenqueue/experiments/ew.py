"""EW: does weather help? F4 (lags + calendar + archived weather forecasts) versus F3.

Both models share one short split because archived forecasts only start in 2024-03, use the same
LightGBM parameters (tuned in E1 on 2018-2022) and the same CQR calibration, so the only
difference is the weather features. The E1 model trained on 2018-2022 is shown for context.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from greenqueue.carbon.conformal import CQR
from greenqueue.carbon.features import build_dataset, horizon_bucket
from greenqueue.carbon.forecasters import LGBQuantileForecaster
from greenqueue.carbon.weather import weather_feature_names
from greenqueue.eval.bootstrap import block_bootstrap_ci
from greenqueue.eval.metrics import pinball
from greenqueue.eval.plots import PALETTE, new_axes, save

PURGE = pd.Timedelta(days=2)


def run_ew(root: Path, *, smoke: bool = False, seed: int = 0) -> dict[str, Any]:
    cfg = yaml.safe_load((root / "configs" / "splits.yaml").read_text(encoding="utf8"))
    raw = cfg["weather_smoke" if smoke else "weather"]
    sp = {
        k: pd.Timestamp(raw[k], tz="UTC")
        for k in ("train_start", "calib_start", "test_start", "test_end")
    }
    out = root / "results" / ("ew_smoke" if smoke else "ew")
    out.mkdir(parents=True, exist_ok=True)
    e1_dir = root / "results" / ("e1_smoke" if smoke else "e1")
    params = json.loads((e1_dir / "best_params.json").read_text())["lightgbm"]
    zones = yaml.safe_load((root / "configs" / "zones.yaml").read_text(encoding="utf8"))
    wcols = weather_feature_names(zones["weather"]["variables"])

    gb = pd.read_parquet(root / "data" / "processed" / "gb_hourly.parquet")
    weather = pd.read_parquet(root / "data" / "raw" / "weather" / "GB_previous_day2.parquet")
    df = build_dataset(gb["ci"], weather=weather, country="GB")
    df = df[df["issue_time"] >= sp["train_start"]]
    it = df["issue_time"]
    train = df[it < sp["calib_start"] - PURGE]
    calib = df[(it >= sp["calib_start"]) & (it < sp["test_start"] - PURGE)].reset_index(drop=True)
    test = df[(it >= sp["test_start"]) & (it < sp["test_end"] + pd.Timedelta(days=1))].reset_index(
        drop=True
    )
    need = list(wcols)
    for name, part in (("train", train), ("calib", calib), ("test", test)):
        if part.empty or part[need].isna().mean().max() > 0.01:
            raise ValueError(f"{name} split is empty or lacks weather coverage")

    y = test["y"].to_numpy()
    h = test["h"].to_numpy()
    times = test["issue_time"]
    models = {
        "F3_no_weather": LGBQuantileForecaster(params, (0.05, 0.5, 0.95), seed=seed).fit(train),
        "F4_weather": LGBQuantileForecaster(
            params, (0.05, 0.5, 0.95), seed=seed, weather_cols=wcols
        ).fit(train),
    }
    rows, abs_err, hbucket = [], {}, horizon_bucket(h)
    for name, m in models.items():
        qc, qt = m.predict(calib), m.predict(test)
        cq = CQR(0.1).fit(calib["h"].to_numpy(), qc[0.05], qc[0.95], calib["y"].to_numpy())
        lo, hi = cq.predict(h, qt[0.05], qt[0.95])
        ae = np.abs(y - qt[0.5])
        abs_err[name] = ae
        mae, mlo, mhi = block_bootstrap_ci(ae, times)
        cov, clo, chi = block_bootstrap_ci(((y >= lo) & (y <= hi)).astype(float), times)
        rows.append(
            {
                "model": name, "mae": mae, "mae_lo": mlo, "mae_hi": mhi,
                "pinball_0.05": pinball(y, qt[0.05], 0.05),
                "pinball_0.95": pinball(y, qt[0.95], 0.95),
                "cqr_coverage": cov, "cqr_cov_lo": clo, "cqr_cov_hi": chi,
                "cqr_width": float((hi - lo).mean()),
                **{f"mae_bucket{b}": float(ae[hbucket == b].mean()) for b in range(4)},
            }
        )  # fmt: skip
    d = abs_err["F3_no_weather"] - abs_err["F4_weather"]  # positive = weather helps
    dm, dlo, dhi = block_bootstrap_ci(d, times)
    ref_path = e1_dir / "forecasts_F3_cqr90.parquet"
    if ref_path.exists():
        ref = pd.read_parquet(ref_path)
        ref = ref[
            (ref["split"] == "test")
            & (ref["issue_time"] >= sp["test_start"])
            & (ref["issue_time"] < sp["test_end"] + pd.Timedelta(days=1))
        ].dropna(subset=["y"])
        if len(ref):
            rows.append(
                {
                    "model": "F3_trained_2018_2022 (E1, context)",
                    "mae": float(np.abs(ref["y"] - ref["point"]).mean()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "weather_ablation.csv", index=False)

    fig, ax = new_axes("MAE by horizon: weather vs no weather", "Horizon (hours)", "MAE (gCO2/kWh)")
    for i, (name, ae) in enumerate(abs_err.items()):
        s = pd.Series(ae).groupby(h).mean()
        ax.plot(s.index, s.values, color=PALETTE[i], lw=1.6, label=name)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "mae_vs_horizon_weather")

    imp = pd.Series(
        models["F4_weather"].models[0.5].feature_importances_, index=models["F4_weather"].cols
    )
    imp.sort_values(ascending=False).head(15).to_csv(out / "f4_top_features.csv")

    key = {"sp": {k: str(v) for k, v in sp.items()}, "params": params, "seed": seed}
    summary = {
        "config_hash": hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12],
        "smoke": smoke,
        "rows": {"train": len(train), "calib": len(calib), "test": len(test)},
        "mae_improvement_from_weather": {"mean": dm, "lo": dlo, "hi": dhi},
        "weather_vintage": "previous_day2 (forecast issued 48 h before the target hour)",
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
