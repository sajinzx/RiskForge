"""E1: forecast accuracy and calibration for Great Britain (implementation.md §12).

Fits F0-F3 + NESO baseline on the training period, calibrates conformal methods on the
calibration period, and reports test-period results with 7-day block-bootstrap CIs.

The NESO archived forecast has an unknown issue time (its error is flat from 12 to 48 h ahead, so
it is a short-lead forecast). Rows using it are labelled ``vintage_unknown`` and are not used by
any scheduler experiment. Main model is F3 (lags and calendar only).

Deviation from §7.4: models are fitted once on the training period (no monthly refit); metrics are
reported per month so regime changes stay visible. Recorded in QUESTIONS.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from greenqueue.carbon.conformal import CQR, SplitConformal
from greenqueue.carbon.features import build_dataset, horizon_bucket
from greenqueue.carbon.forecasters import (
    LGBQuantileForecaster,
    RidgeForecaster,
    neso_forecast,
    persistence,
    seasonal_naive,
)
from greenqueue.carbon.gb import build_gb_hourly
from greenqueue.carbon.tuning import assert_not_test, tune_lightgbm, tune_ridge
from greenqueue.eval.bootstrap import block_bootstrap_ci
from greenqueue.eval.metrics import interval_score_pinball, pinball
from greenqueue.eval.plots import PALETTE, new_axes, save

LEVELS = (0.8, 0.9, 0.95)
TAUS = (0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975)
PURGE = pd.Timedelta(days=2)
Q_FOR_LEVEL = {0.8: (0.1, 0.9), 0.9: (0.05, 0.95), 0.95: (0.025, 0.975)}


def _splits(cfg: dict[str, Any], smoke: bool) -> dict[str, pd.Timestamp]:
    c = cfg["smoke"] if smoke else cfg
    return {
        k: pd.Timestamp(c[k], tz="UTC")
        for k in ("train_start", "calib_start", "test_start", "test_end")
    }


def run_e1(
    root: Path, *, smoke: bool = False, trials: int | None = None, seed: int = 0
) -> dict[str, Any]:
    cfg = yaml.safe_load((root / "configs" / "splits.yaml").read_text(encoding="utf8"))
    sp = _splits(cfg, smoke)
    out = root / "results" / ("e1_smoke" if smoke else "e1")
    out.mkdir(parents=True, exist_ok=True)
    n_trials = trials if trials is not None else (3 if smoke else 50)

    gb_path = root / "data" / "processed" / "gb_hourly.parquet"
    if not gb_path.exists():
        build_gb_hourly(
            root / "data" / "raw" / "uk_ci",
            root / "data" / "processed",
            root / "configs" / "preprocess.yaml",
        )
    gb = pd.read_parquet(gb_path)
    df_all = build_dataset(gb["ci"], neso=gb["neso_fc"], country="GB", require_target=False)
    df_all = df_all[df_all["issue_time"] >= sp["train_start"]]
    df = df_all[df_all["y"].notna()]

    it = df["issue_time"]
    train = df[it < sp["calib_start"] - PURGE].reset_index(drop=True)
    calib = df[(it >= sp["calib_start"]) & (it < sp["test_start"] - PURGE)].reset_index(drop=True)
    test = df[(it >= sp["test_start"]) & (it < sp["test_end"] + pd.Timedelta(days=1))].reset_index(
        drop=True
    )
    assert_not_test(train, sp["test_start"])
    assert_not_test(calib, sp["test_start"])
    for part in (train, calib, test):
        if part.empty:
            raise ValueError("a split is empty; check configs/splits.yaml against the data")

    # ---- tuning on training data only -------------------------------------------------
    alpha_ridge = tune_ridge(train, sp["test_start"])
    lgb_params = tune_lightgbm(
        train, sp["test_start"], n_trials=n_trials, seed=seed, max_estimators=300 if smoke else 1000
    )
    (out / "best_params.json").write_text(
        json.dumps({"ridge_alpha": alpha_ridge, "lightgbm": lgb_params}, indent=2)
    )

    # ---- fit -----------------------------------------------------------------------------
    ridge = RidgeForecaster(alpha_ridge).fit(train)
    lgb_models = {
        "F3": LGBQuantileForecaster(lgb_params, TAUS, use_neso=False, seed=seed).fit(train),
        "F3_neso_vintage_unknown": LGBQuantileForecaster(
            lgb_params, TAUS, use_neso=True, seed=seed
        ).fit(train),
    }

    # ---- predictions: model name -> dict(point, intervals by (method, level)) -----------
    def point_models(d: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "F0_persistence": persistence(d),
            "F1_snaive24": seasonal_naive(d, 24),
            "F1_snaive168": seasonal_naive(d, 168),
            "FN_neso_vintage_unknown": neso_forecast(d),
            "F2_ridge": ridge.predict(d),
        }

    cal_pts, test_pts = point_models(calib), point_models(test)
    rows: list[dict[str, Any]] = []
    iv: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}

    y = test["y"].to_numpy()
    h = test["h"].to_numpy()
    times = test["issue_time"]

    def add_row(
        model: str,
        method: str,
        level: float | None,
        point: np.ndarray,
        lo: np.ndarray | None,
        hi: np.ndarray | None,
        q: dict[float, np.ndarray] | None = None,
    ) -> None:
        ok = np.isfinite(point) & np.isfinite(y)
        ae = np.abs(y - point)
        m, mlo, mhi = block_bootstrap_ci(ae[ok], times[ok])
        row: dict[str, Any] = {
            "model": model,
            "method": method,
            "level": level,
            "n": int(ok.sum()),
            "mae": m,
            "mae_lo": mlo,
            "mae_hi": mhi,
        }
        if lo is not None and hi is not None:
            inside = ((y >= lo) & (y <= hi)).astype(float)
            c, clo, chi = block_bootstrap_ci(inside, times)
            row.update(
                coverage=c,
                coverage_lo=clo,
                coverage_hi=chi,
                width=float(np.mean(hi - lo)),
                interval_pinball=interval_score_pinball(y, lo, hi, 1 - (level or 0.9)),
            )
            iv[(model, method, level or 0.9)] = (lo, hi)
        if q is not None:
            for tau in (0.05, 0.5, 0.95):
                row[f"pinball_{tau}"] = pinball(y, q[tau], tau)
        rows.append(row)

    for name, p in test_pts.items():
        add_row(name, "point", None, p, None, None)
    # U1 on the ridge point model
    for level in LEVELS:
        a = 1 - level
        sc = SplitConformal(a).fit(
            calib["h"].to_numpy(), cal_pts["F2_ridge"], calib["y"].to_numpy()
        )
        lo, hi = sc.predict(h, test_pts["F2_ridge"])
        add_row("F2_ridge", "U1_split", level, test_pts["F2_ridge"], lo, hi)

    for name, fm in lgb_models.items():
        qt, qc = fm.predict(test), fm.predict(calib)
        for level in LEVELS:
            a = 1 - level
            lq, uq = Q_FOR_LEVEL[level]
            lo0, hi0 = qt[lq], qt[uq]
            add_row(name, "U0_raw", level, qt[0.5], lo0, hi0, qt if level == 0.9 else None)
            sc = SplitConformal(a).fit(calib["h"].to_numpy(), qc[0.5], calib["y"].to_numpy())
            lo1, hi1 = sc.predict(h, qt[0.5])
            add_row(name, "U1_split", level, qt[0.5], lo1, hi1)
            cq = CQR(a).fit(calib["h"].to_numpy(), qc[lq], qc[uq], calib["y"].to_numpy())
            lo2, hi2 = cq.predict(h, qt[lq], qt[uq])
            add_row(name, "U2_cqr", level, qt[0.5], lo2, hi2)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)

    # ---- breakdowns ---------------------------------------------------------------------
    bucket = horizon_bucket(h)
    recs = []
    for (model, method, level), (lo, hi) in iv.items():
        if level != 0.9:
            continue
        inside = (y >= lo) & (y <= hi)
        for b in range(4):
            recs.append(
                {
                    "model": model,
                    "method": method,
                    "bucket": b,
                    "coverage": float(inside[bucket == b].mean()),
                    "width": float((hi - lo)[bucket == b].mean()),
                }
            )
    pd.DataFrame(recs).to_csv(out / "coverage_by_horizon_bucket.csv", index=False)

    tt = pd.DatetimeIndex(test["target_time"])
    month = tt.strftime("%Y-%m")
    mon = []
    for (model, method, level), (lo, hi) in iv.items():
        if level != 0.9:
            continue
        inside = (y >= lo) & (y <= hi)
        for m_ in sorted(set(month)):
            sel = month == m_
            mon.append(
                {
                    "model": model,
                    "method": method,
                    "month": m_,
                    "coverage": float(inside[sel].mean()),
                    "mae": float(np.abs(y - lgb_models[model].predict(test)[0.5])[sel].mean())
                    if model in lgb_models
                    else np.nan,
                }
            )
    pd.DataFrame(mon).to_csv(out / "monthly.csv", index=False)

    # ---- figures -------------------------------------------------------------------------
    main = "F3"
    colors = {"U0_raw": PALETTE[1], "U1_split": PALETTE[2], "U2_cqr": PALETTE[0]}
    fig, ax = new_axes(
        "Coverage by horizon (F3, nominal 90%)", "Horizon (hours)", "Empirical coverage"
    )
    for method in ("U0_raw", "U1_split", "U2_cqr"):
        lo, hi = iv[(main, method, 0.9)]
        inside = ((y >= lo) & (y <= hi)).astype(float)
        by_h = pd.Series(inside).groupby(h).mean()
        ax.plot(by_h.index, by_h.values, color=colors[method], lw=1.6, label=method)
    ax.axhline(0.9, color="#444444", lw=0.8, ls="--", label="target")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, out / "coverage_vs_horizon")

    fig, ax = new_axes("Rolling 30-day coverage (F3, nominal 90%)", "Target date", "Coverage")
    for method in ("U0_raw", "U1_split", "U2_cqr"):
        lo, hi = iv[(main, method, 0.9)]
        inside_s = pd.Series(((y >= lo) & (y <= hi)).astype(float), index=tt.normalize())
        daily = inside_s.groupby(level=0).mean()
        ax.plot(
            daily.index,
            daily.rolling(30, min_periods=15).mean(),
            color=colors[method],
            lw=1.6,
            label=method,
        )
    ax.axhline(0.9, color="#444444", lw=0.8, ls="--", label="target")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, out / "rolling_coverage")

    fig, ax = new_axes(
        "Calibration (F3): nominal vs empirical coverage",
        "Nominal coverage",
        "Empirical coverage",
        (4.6, 4.2),
    )
    ax.plot([0.7, 1], [0.7, 1], color="#444444", lw=0.8, ls="--", label="ideal")
    for method in ("U0_raw", "U1_split", "U2_cqr"):
        sub = metrics[(metrics["model"] == main) & (metrics["method"] == method)].sort_values(
            "level"
        )
        ax.plot(
            sub["level"],
            sub["coverage"],
            color=colors[method],
            lw=1.6,
            marker="o",
            ms=5,
            label=method,
        )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out / "calibration")

    fig, ax = new_axes("MAE by horizon (test)", "Horizon (hours)", "MAE (gCO2/kWh)")
    preds_for_plot = {
        **{k: v for k, v in test_pts.items()},
        "F3": lgb_models["F3"].predict(test)[0.5],
    }
    for i, (name, p) in enumerate(preds_for_plot.items()):
        if name == "F1_snaive168":
            continue
        s = pd.Series(np.abs(y - p)).groupby(h).mean()
        ax.plot(s.index, s.values, color=PALETTE[i % len(PALETTE)], lw=1.6, label=name)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "mae_vs_horizon")

    # ---- export F3 + CQR(90%) forecasts for scheduling experiments (calibration and test) ------
    f3 = lgb_models["F3"]
    cq_main = CQR(0.1).fit(
        calib["h"].to_numpy(),
        f3.predict(calib)[0.05],
        f3.predict(calib)[0.95],
        calib["y"].to_numpy(),
    )
    it_all = df_all["issue_time"]
    exp_parts = {
        "calib": df_all[(it_all >= sp["calib_start"]) & (it_all < sp["test_start"] - PURGE)],
        "test": df_all[
            (it_all >= sp["test_start"]) & (it_all < sp["test_end"] + pd.Timedelta(days=1))
        ],
    }
    frames = []
    for split_name, part in exp_parts.items():
        pr = f3.predict(part)
        lo_, hi_ = cq_main.predict(part["h"].to_numpy(), pr[0.05], pr[0.95])
        frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "issue_time": part["issue_time"].to_numpy(),
                    "h": part["h"].to_numpy(),
                    "target_time": part["target_time"].to_numpy(),
                    "y": part["y"].to_numpy(),
                    "point": pr[0.5],
                    "lo": lo_,
                    "hi": hi_,
                }
            )
        )
    pd.concat(frames).to_parquet(out / "forecasts_F3_cqr90.parquet")

    # ---- sanity bounds (§13.1) -------------------------------------------------------------
    sane = {
        "ci_in_bounds": bool(((test["y"] >= 0) & (test["y"] <= 1200)).all()),
        "no_degenerate_coverage": bool(metrics["coverage"].dropna().between(0.01, 0.999).all()),
        "preds_finite": bool(np.isfinite(lgb_models["F3"].predict(test)[0.5]).all()),
    }
    h_ = hashlib.sha256(
        json.dumps(
            {
                "splits": {k: str(v) for k, v in sp.items()},
                "trials": n_trials,
                "seed": seed,
                "params": lgb_params,
                "data_rows": len(df),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    head = metrics[(metrics["level"] == 0.9) | metrics["level"].isna()]
    summary = {
        "config_hash": h_,
        "smoke": smoke,
        "seed": seed,
        "rows": {"train": len(train), "calib": len(calib), "test": len(test)},
        "splits": {k: str(v) for k, v in sp.items()},
        "sanity": sane,
        "best_params": {"ridge_alpha": alpha_ridge, "lightgbm": lgb_params},
        "headline": head[
            [
                "model",
                "method",
                "mae",
                "mae_lo",
                "mae_hi",
                "coverage",
                "coverage_lo",
                "coverage_hi",
                "width",
            ]
        ]
        .round(4)
        .to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not all(sane.values()):
        raise AssertionError(f"sanity bounds failed: {sane}")
    return summary
