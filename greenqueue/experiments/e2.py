"""E2: coverage under regime shift, static CQR versus adaptive conformal inference (ACI).

Refits the tuned F3 (parameters from E1) and compares, on the test period, split conformal (U1),
static CQR (U2) and ACI with delayed feedback (U3). The ACI step size gamma is chosen on the
calibration year only: scores from its first half, ACI run over its second half.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from greenqueue.carbon.conformal import CQR, SplitConformal, aci_intervals
from greenqueue.carbon.features import build_dataset, horizon_bucket
from greenqueue.carbon.forecasters import LGBQuantileForecaster
from greenqueue.eval.bootstrap import block_bootstrap_ci
from greenqueue.eval.plots import PALETTE, new_axes, save

GAMMAS = (0.001, 0.005, 0.01, 0.05)
PURGE = pd.Timedelta(days=2)
ALPHA = 0.1


def _scores(d: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> dict[int, np.ndarray]:
    s = np.maximum(lo - d["y"].to_numpy(), d["y"].to_numpy() - hi)
    b = horizon_bucket(d["h"].to_numpy())
    return {int(k): s[b == k] for k in np.unique(b)}


def run_e2(root: Path, *, smoke: bool = False, seed: int = 0) -> dict[str, Any]:
    split_cfg = yaml.safe_load((root / "configs" / "splits.yaml").read_text(encoding="utf8"))
    raw = split_cfg["smoke"] if smoke else split_cfg
    sp = {
        k: pd.Timestamp(raw[k], tz="UTC")
        for k in ("train_start", "calib_start", "test_start", "test_end")
    }
    e1_dir = root / "results" / ("e1_smoke" if smoke else "e1")
    out = root / "results" / ("e2_smoke" if smoke else "e2")
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads((e1_dir / "best_params.json").read_text())["lightgbm"]

    gb = pd.read_parquet(root / "data" / "processed" / "gb_hourly.parquet")
    df = build_dataset(gb["ci"], country="GB")
    df = df[df["issue_time"] >= sp["train_start"]]
    it = df["issue_time"]
    train = df[it < sp["calib_start"] - PURGE]
    calib = df[(it >= sp["calib_start"]) & (it < sp["test_start"] - PURGE)].reset_index(drop=True)
    test = df[(it >= sp["test_start"]) & (it < sp["test_end"] + pd.Timedelta(days=1))].reset_index(
        drop=True
    )

    model = LGBQuantileForecaster(params, (0.05, 0.5, 0.95), seed=seed).fit(train)
    qc, qt = model.predict(calib), model.predict(test)

    # ---- choose gamma on the calibration year only ---------------------------------------
    mid = calib["issue_time"].min() + (calib["issue_time"].max() - calib["issue_time"].min()) / 2
    h1, h2 = calib["issue_time"] <= mid, calib["issue_time"] > mid + PURGE
    c1, c2 = calib[h1], calib[h2].reset_index(drop=True)
    lo1, hi1 = qc[0.05][h1.to_numpy()], qc[0.95][h1.to_numpy()]
    lo2, hi2 = qc[0.05][h2.to_numpy()], qc[0.95][h2.to_numpy()]
    cal_scores_h1 = _scores(c1, lo1, hi1)
    rows = []
    for g in GAMMAS:
        a_lo, a_hi, _ = aci_intervals(
            c2["issue_time"], c2["h"].to_numpy(), c2["target_time"], c2["y"].to_numpy(),
            lo2, hi2, cal_scores_h1, alpha=ALPHA, gamma=g,
        )  # fmt: skip
        inside = (c2["y"].to_numpy() >= a_lo) & (c2["y"].to_numpy() <= a_hi)
        rows.append(
            {"gamma": g, "coverage": float(inside.mean()), "width": float((a_hi - a_lo).mean())}
        )
    gtab = pd.DataFrame(rows)
    gtab["gap"] = (gtab["coverage"] - (1 - ALPHA)).abs()
    gamma = float(gtab.sort_values(["gap", "width"]).iloc[0]["gamma"])
    gtab.to_csv(out / "gamma_selection_calibration.csv", index=False)

    # ---- test period: three methods --------------------------------------------------------
    h_t, y = test["h"].to_numpy(), test["y"].to_numpy()
    cal_all = _scores(calib, qc[0.05], qc[0.95])
    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sc = SplitConformal(ALPHA).fit(calib["h"].to_numpy(), qc[0.5], calib["y"].to_numpy())
    methods["U1_split"] = sc.predict(h_t, qt[0.5])
    cq = CQR(ALPHA).fit(calib["h"].to_numpy(), qc[0.05], qc[0.95], calib["y"].to_numpy())
    methods["U2_cqr"] = cq.predict(h_t, qt[0.05], qt[0.95])
    a_lo, a_hi, a_used = aci_intervals(
        test["issue_time"], h_t, test["target_time"], y, qt[0.05], qt[0.95], cal_all,
        alpha=ALPHA, gamma=gamma,
    )  # fmt: skip
    methods["U3_aci"] = (a_lo, a_hi)

    tt = pd.DatetimeIndex(test["target_time"])
    table, rolling = [], {}
    colors = {"U1_split": PALETTE[2], "U2_cqr": PALETTE[0], "U3_aci": PALETTE[1]}
    for name, (lo, hi) in methods.items():
        inside = ((y >= lo) & (y <= hi)).astype(float)
        c, clo, chi = block_bootstrap_ci(inside, test["issue_time"])
        daily = pd.Series(inside, index=tt.normalize()).groupby(level=0).mean()
        roll = daily.rolling(30, min_periods=15).mean()
        rolling[name] = roll
        table.append(
            {
                "method": name,
                "coverage": c,
                "coverage_lo": clo,
                "coverage_hi": chi,
                "mean_width": float((hi - lo).mean()),
                "worst_30d_coverage": float(roll.min()),
                "share_30d_windows_below_0.85": float((roll < 0.85).mean()),
            }
        )
    res = pd.DataFrame(table)
    res.to_csv(out / "coverage_summary.csv", index=False)
    season = tt.month.map(
        lambda m: {
            12: "DJF",
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
        }.get(m, "SON")
    )
    seas = pd.DataFrame(
        {
            n: pd.Series(((y >= lo) & (y <= hi)).astype(float)).groupby(np.asarray(season)).mean()
            for n, (lo, hi) in methods.items()
        }
    )
    seas.to_csv(out / "coverage_by_season.csv")

    fig, ax = new_axes("Rolling 30-day coverage, nominal 90%", "Target date", "Coverage")
    for name, roll in rolling.items():
        ax.plot(roll.index, roll.values, color=colors[name], lw=1.5, label=name)
    ax.axhline(1 - ALPHA, color="#444444", lw=0.8, ls="--", label="target")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, out / "rolling_coverage_aci")

    # ---- export ACI forecasts for scheduling experiments -----------------------------------
    export = pd.DataFrame(
        {
            "split": "test",
            "issue_time": test["issue_time"].to_numpy(),
            "h": h_t,
            "target_time": test["target_time"].to_numpy(),
            "y": y,
            "point": qt[0.5],
            "lo": a_lo,
            "hi": a_hi,
        }
    )
    export.to_parquet(out / "forecasts_F3_aci90.parquet")

    summary = {
        "config_hash": hashlib.sha256(
            json.dumps(
                {"gamma": gamma, "params": params, "sp": {k: str(v) for k, v in sp.items()}},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12],
        "smoke": smoke,
        "gamma": gamma,
        "gamma_table": gtab.round(4).to_dict("records"),
        "results": res.round(4).to_dict("records"),
        "min_alpha_used": float(a_used.min()),
        "max_alpha_used": float(a_used.max()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
