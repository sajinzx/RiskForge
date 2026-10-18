"""Random / leave-one-model-out / leave-one-GPU-out validation and group conformal (§9.3, §9.4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from greenqueue.carbon.conformal import conformal_quantile
from greenqueue.energy.models import StepModel

Scheme = Literal["random", "lomo", "logo"]


class GroupConformal:
    """Split conformal on log-residuals, calibrated per model family and per GPU type.

    The correction for a row is the larger of its family quantile and its GPU quantile (the
    conservative choice); a group with fewer than ``min_group`` calibration points falls back to the
    pooled quantile. ``two_sided`` uses ``|log y - log yhat|``; ``upper`` uses the signed residual
    and only bounds ``y`` from above (what protects a deadline).
    """

    def __init__(self, alpha: float, min_group: int = 20) -> None:
        self.alpha, self.min_group = alpha, min_group
        self.q: dict[str, dict[str, dict[str, float]]] = {}
        self.pooled = {"two_sided": float("inf"), "upper": float("inf")}

    def fit(self, cal: pd.DataFrame, y: np.ndarray, yhat: np.ndarray) -> GroupConformal:
        signed = np.log(y) - np.log(yhat)
        scores = {"two_sided": np.abs(signed), "upper": signed}
        for kind, s in scores.items():
            self.pooled[kind] = conformal_quantile(s, self.alpha)
            self.q[kind] = {}
            for col in ("family", "gpu"):
                self.q[kind][col] = {}
                for val, idx in cal.groupby(col).indices.items():
                    if len(idx) >= self.min_group:
                        self.q[kind][col][str(val)] = conformal_quantile(s[idx], self.alpha)
        return self

    def _corr(self, df: pd.DataFrame, kind: str) -> np.ndarray:
        out = np.empty(len(df))
        for i, (fam, gpu) in enumerate(zip(df["family"], df["gpu"], strict=True)):
            out[i] = max(
                self.q[kind]["family"].get(str(fam), self.pooled[kind]),
                self.q[kind]["gpu"].get(str(gpu), self.pooled[kind]),
            )
        return out

    def interval(self, df: pd.DataFrame, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self._corr(df, "two_sided")
        return yhat * np.exp(-c), yhat * np.exp(c)

    def upper(self, df: pd.DataFrame, yhat: np.ndarray) -> np.ndarray:
        return yhat * np.exp(self._corr(df, "upper"))


def make_folds(
    df: pd.DataFrame, scheme: Scheme, k: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(df))
    if scheme == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        parts = np.array_split(perm, k)
        return [(np.setdiff1d(idx, p), p) for p in parts]
    col = {"lomo": "model", "logo": "gpu"}[scheme]
    groups = df[col].to_numpy()
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError(f"{scheme} needs at least two distinct values of {col!r}")
    return [(idx[groups != g], idx[groups == g]) for g in uniq]


def evaluate(
    df: pd.DataFrame,
    factory: Callable[[], StepModel],
    scheme: Scheme,
    *,
    alpha: float = 0.1,
    cal_frac: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold predictions and intervals for every row of ``df``."""
    df = df.reset_index(drop=True)
    out = []
    for fold, (tr, te) in enumerate(make_folds(df, scheme, seed=seed)):
        rng = np.random.default_rng(seed + fold)
        perm = rng.permutation(tr)
        n_cal = max(2, int(len(tr) * cal_frac))
        cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
        model = factory().fit(df.iloc[fit_idx])
        cal, test = df.iloc[cal_idx], df.iloc[te]
        pc, pt = model.predict(cal), model.predict(test)
        res = pd.DataFrame(
            {
                "fold": fold,
                "scheme": scheme,
                "gpu": test["gpu"].to_numpy(),
                "model": test["model"].to_numpy(),
                "family": test["family"].to_numpy(),
                "e_true": test["energy_per_step_kwh"].to_numpy(),
                "e_pred": pt["energy"].to_numpy(),
                "t_true": test["step_time_s"].to_numpy(),
                "t_pred": pt["time"].to_numpy(),
            }
        )
        ce = GroupConformal(alpha).fit(
            cal, cal["energy_per_step_kwh"].to_numpy(), pc["energy"].to_numpy()
        )
        res["e_lo"], res["e_hi"] = ce.interval(test, pt["energy"].to_numpy())
        ct = GroupConformal(alpha).fit(cal, cal["step_time_s"].to_numpy(), pc["time"].to_numpy())
        res["t_hi"] = ct.upper(test, pt["time"].to_numpy())
        out.append(res)
    return pd.concat(out, ignore_index=True)


def summarize(res: pd.DataFrame) -> dict[str, float]:
    ape_e = (res["e_pred"] - res["e_true"]).abs() / res["e_true"]
    ape_t = (res["t_pred"] - res["t_true"]).abs() / res["t_true"]
    return {
        "n": float(len(res)),
        "mape_energy": float(ape_e.mean()),
        "medape_energy": float(ape_e.median()),
        "mape_time": float(ape_t.mean()),
        "coverage_energy": float(
            ((res["e_true"] >= res["e_lo"]) & (res["e_true"] <= res["e_hi"])).mean()
        ),
        "rel_width_energy": float(((res["e_hi"] - res["e_lo"]) / res["e_pred"]).mean()),
        "coverage_time_upper": float((res["t_true"] <= res["t_hi"]).mean()),
    }
