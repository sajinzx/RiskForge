"""E9: cross-zone transfer of the carbon forecaster (train zone x test zone).

For every source zone a LightGBM quantile model (F3, parameters tuned on GB in E1) is trained on
2018-2022 data of that zone and applied to every target zone's test period. Two modes:

- ``raw``: features and target in gCO2/kWh. A model moved to a zone with a different level fails
  mostly because of scale.
- ``scaled``: each zone's series is divided by its own training-period mean, predictions are
  multiplied back. This removes the level and keeps the shape of the daily and weekly cycles.

For each pair the 90% interval is built by CQR calibrated either on the source zone's 2023 data
(``as_is``: what a transferred model would give with no target data) or on the target's 2023 data
(``recalibrated``). Skill is the MAE ratio to persistence on the target (below 1 is better).
Only zones whose hourly file exists are used; missing zones are listed in the summary.
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
from greenqueue.carbon.features import build_dataset
from greenqueue.carbon.forecasters import LGBQuantileForecaster
from greenqueue.carbon.zones import HOLIDAY_COUNTRY
from greenqueue.eval.bootstrap import block_bootstrap_ci
from greenqueue.eval.plots import new_axes, save

PURGE = pd.Timedelta(days=2)
ZONES = ("GB", "DE_LU", "FR", "PL", "ES")
TAUS = (0.05, 0.5, 0.95)


def _zone_ci(root: Path, zone: str) -> pd.Series | None:
    path = root / "data" / "processed" / f"{zone.lower() if zone == 'GB' else zone}_hourly.parquet"
    return pd.read_parquet(path)["ci"] if path.exists() else None


def split_dataset(
    ci: pd.Series, zone: str, sp: dict[str, pd.Timestamp], scale: float
) -> dict[str, pd.DataFrame]:
    df = build_dataset(ci / scale, country=HOLIDAY_COUNTRY[zone])
    df = df[df["issue_time"] >= sp["train_start"]]
    it = df["issue_time"]
    return {
        "train": df[it < sp["calib_start"] - PURGE],
        "calib": df[(it >= sp["calib_start"]) & (it < sp["test_start"] - PURGE)],
        "test": df[(it >= sp["test_start"]) & (it < sp["test_end"] + pd.Timedelta(days=1))],
    }


def run_e9(root: Path, *, smoke: bool = False, seed: int = 0) -> dict[str, Any]:
    cfg = yaml.safe_load((root / "configs" / "splits.yaml").read_text(encoding="utf8"))
    raw = cfg["smoke"] if smoke else cfg
    sp = {
        k: pd.Timestamp(raw[k], tz="UTC")
        for k in ("train_start", "calib_start", "test_start", "test_end")
    }
    out = root / "results" / ("e9_smoke" if smoke else "e9")
    out.mkdir(parents=True, exist_ok=True)
    e1 = root / "results" / ("e1_smoke" if smoke else "e1")
    params = json.loads((e1 / "best_params.json").read_text())["lightgbm"]

    series = {z: s for z in ZONES if (s := _zone_ci(root, z)) is not None}
    skipped = [z for z in ZONES if z not in series]
    if len(series) < 2:
        raise ValueError(f"E9 needs at least two zones with hourly data, found {sorted(series)}")
    train_mean = {
        z: float(s.loc[sp["train_start"] : sp["calib_start"] - PURGE].mean())
        for z, s in series.items()
    }

    rows = []
    for mode in ("raw", "scaled"):
        sets = {
            z: split_dataset(s, z, sp, train_mean[z] if mode == "scaled" else 1.0)
            for z, s in series.items()
        }
        for src in series:
            model = LGBQuantileForecaster(params, TAUS, seed=seed).fit(sets[src]["train"])
            q_src = model.predict(sets[src]["calib"])
            cq_src = CQR(0.1).fit(
                sets[src]["calib"]["h"].to_numpy(),
                q_src[0.05],
                q_src[0.95],
                sets[src]["calib"]["y"].to_numpy(),
            )
            for tgt in series:
                mult = train_mean[tgt] if mode == "scaled" else 1.0
                tcal, test = sets[tgt]["calib"], sets[tgt]["test"]
                q_cal, q_test = model.predict(tcal), model.predict(test)
                cq_tgt = CQR(0.1).fit(
                    tcal["h"].to_numpy(), q_cal[0.05], q_cal[0.95], tcal["y"].to_numpy()
                )
                y = test["y"].to_numpy() * mult
                h = test["h"].to_numpy()
                times = test["issue_time"]
                ae = np.abs(y - q_test[0.5] * mult)
                pers = np.abs(y - test["lag_1"].to_numpy() * mult)
                mae, lo, hi = block_bootstrap_ci(ae, times)
                rec: dict[str, Any] = {
                    "mode": mode,
                    "source": src,
                    "target": tgt,
                    "mae": mae,
                    "mae_lo": lo,
                    "mae_hi": hi,
                    # same hours for both: a gap in the target leaves lag_1 NaN for a few rows
                    "skill_vs_persistence": float(
                        ae[np.isfinite(pers)].mean() / pers[np.isfinite(pers)].mean()
                    ),
                }
                for name, cq in (("as_is", cq_src), ("recalibrated", cq_tgt)):
                    a, b = cq.predict(h, q_test[0.05], q_test[0.95])
                    a, b = a * mult, b * mult
                    rec[f"coverage_{name}"] = float(((y >= a) & (y <= b)).mean())
                    rec[f"width_{name}"] = float((b - a).mean())
                rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(out / "transfer_matrix.csv", index=False)

    for mode in ("raw", "scaled"):
        piv = table[table["mode"] == mode].pivot(
            index="source", columns="target", values="skill_vs_persistence"
        )
        fig, ax = new_axes(
            f"MAE / persistence MAE ({mode}); <1 beats persistence",
            "Test zone",
            "Training zone",
            (5.4, 4.2),
        )
        vals = piv.to_numpy(dtype=float)
        im = ax.imshow(vals, cmap="Blues_r", aspect="auto")
        ax.set_xticks(range(piv.shape[1]), list(piv.columns))
        ax.set_yticks(range(piv.shape[0]), list(piv.index))
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.figure.colorbar(im, ax=ax)
        save(ax.figure, out / f"transfer_skill_{mode}")

    key = {"sp": {k: str(v) for k, v in sp.items()}, "zones": sorted(series), "params": params}
    summary = {
        "config_hash": hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12],
        "smoke": smoke,
        "zones_used": sorted(series),
        "zones_skipped_no_data": skipped,
        "train_mean_ci": train_mean,
        "results": table.round(4).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
