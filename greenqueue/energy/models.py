"""Energy and step-time models: EN0 log-linear, EN1 LightGBM, EN2 gray-box (§9.2).

Every model predicts, for a configuration table (see ``preprocess.aggregate_configs``), the energy
per step (kWh) and the time per step (s). Job energy is the per-step value times the step count,
so log-errors carry over unchanged.
"""

from __future__ import annotations

from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

PRECISIONS = {"fp32": 0, "fp16": 1, "bf16": 2}
FAMILIES = {"cnn": 0, "vit": 1, "bert": 2, "llm": 3}


class StepModel(Protocol):
    def fit(self, df: pd.DataFrame) -> StepModel: ...
    def predict(self, df: pd.DataFrame) -> pd.DataFrame: ...  # columns: energy, time


def _tokens(df: pd.DataFrame) -> np.ndarray:
    """Parallel work per step: batch for images, batch * sequence length for text."""
    batch = df["batch"].to_numpy(dtype=float)
    return np.asarray(batch * np.maximum(df["seq"].to_numpy(dtype=float), 1.0))


class LogLinear:
    """EN0: linear regression on log FLOPs, log batch and a GPU one-hot (as in the guide).

    An unseen GPU gets an all-zero one-hot, which is exactly the weakness LOGO should expose.
    """

    def __init__(self) -> None:
        self.gpus: list[str] = []
        self.m_e = LinearRegression()
        self.m_t = LinearRegression()

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        onehot = np.column_stack([(df["gpu"] == g).to_numpy(float) for g in self.gpus])
        return np.column_stack(
            [np.log(df["flops_per_step"]), np.log(df["batch"].astype(float)), onehot]
        )

    def fit(self, df: pd.DataFrame) -> LogLinear:
        self.gpus = sorted(df["gpu"].unique())
        x = self._x(df)
        self.m_e.fit(x, np.log(df["energy_per_step_kwh"]))
        self.m_t.fit(x, np.log(df["step_time_s"]))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._x(df)
        return pd.DataFrame(
            {"energy": np.exp(self.m_e.predict(x)), "time": np.exp(self.m_t.predict(x))},
            index=df.index,
        )


class GBM:
    """EN1: LightGBM on log targets with numeric hardware descriptors (no GPU id)."""

    def __init__(self, params: dict[str, float | int] | None = None, seed: int = 0) -> None:
        self.params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 5,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            **(params or {}),
        }
        self.seed = seed
        self.m_e: lgb.LGBMRegressor | None = None
        self.m_t: lgb.LGBMRegressor | None = None

    @staticmethod
    def _x(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "log_flops": np.log(df["flops_per_step"].to_numpy(float)),
                "log_batch": np.log(df["batch"].to_numpy(float)),
                "log_tokens": np.log(_tokens(df)),
                "precision": df["precision"].map(PRECISIONS).to_numpy(),
                "family": df["family"].map(FAMILIES).to_numpy(),
                "log_peak": np.log(df["peak_tflops"].to_numpy(float)),
                "log_tdp": np.log(df["tdp_w"].to_numpy(float)),
            },
            index=df.index,
        )

    def fit(self, df: pd.DataFrame) -> GBM:
        x = self._x(df)
        kw = dict(random_state=self.seed, verbose=-1, n_jobs=1, **self.params)
        self.m_e = lgb.LGBMRegressor(**kw).fit(x, np.log(df["energy_per_step_kwh"]))  # type: ignore[arg-type]
        self.m_t = lgb.LGBMRegressor(**kw).fit(x, np.log(df["step_time_s"]))  # type: ignore[arg-type]
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.m_e is not None and self.m_t is not None
        x = self._x(df)
        return pd.DataFrame(
            {"energy": np.exp(self.m_e.predict(x)), "time": np.exp(self.m_t.predict(x))},
            index=df.index,
        )


class GrayBox:
    """EN2: physics-shaped model with a few fitted numbers per GPU type plus family multipliers.

    ``u(b) = b / (b + b_half)`` is utilisation (saturating in parallel work ``b``);
    time/step  ``T = FLOPs / (peak * min(1, eta_max * m_family) * u)``;
    power above idle ``P = (TDP - idle) * (p0 + (1 - p0) * u**k)``;
    energy/step ``E = P * T / 3.6e6`` kWh.
    ``(eta_max, b_half, k)`` plus a baseline-draw fraction ``p0`` (a fourth number, an extension of
    the guide: a busy GPU never drops to idle power) are fitted per GPU on log-energy and log-time
    residuals. The family
    multipliers ``m_family`` (shared by all GPUs, ridge-shrunk towards 1) are an extension of the
    guide's three-number model: workload families use a GPU very differently (CNN vs transformer vs
    memory-bound decoding) and a single efficiency cannot cover that. An unseen GPU uses the
    geometric mean of the fitted GPU parameters with its spec-sheet peak and TDP; an unseen
    family uses m = 1.
    """

    BOUNDS = ([1e-3, 1e-3, 0.5, 0.0], [1.0, 1e9, 3.0, 0.8])
    SHRINK = 0.3

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.fam_mult: dict[str, float] = {}
        self.pooled = np.array([0.3, 64.0, 1.0, 0.2])
        self.idle_frac = 0.2

    @staticmethod
    def _predict_arrays(
        p: np.ndarray, df: pd.DataFrame, idle_w: np.ndarray, mult: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        eta_max, b_half, k, p0 = p
        b = _tokens(df)
        u = b / (b + b_half)
        peak = df["peak_tflops"].to_numpy(float) * 1e12
        eta = np.minimum(1.0, eta_max * mult)
        t = df["flops_per_step"].to_numpy(float) / (peak * eta * u)
        power = np.maximum(df["tdp_w"].to_numpy(float) - idle_w, 1.0) * (p0 + (1 - p0) * u**k)
        return power * t / 3.6e6, t

    def _mult(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.fam_mult.get(str(f), 1.0) for f in df["family"]])

    def fit(self, df: pd.DataFrame) -> GrayBox:
        self.idle_frac = float(np.mean((df["idle_w"] / df["tdp_w"]).to_numpy(float)))
        gpus = sorted(df["gpu"].unique())
        fams = sorted(df["family"].unique())
        gi = {g: i for i, g in enumerate(gpus)}
        fi = {f: i for i, f in enumerate(fams)}
        g_idx = df["gpu"].map(gi).to_numpy()
        f_idx = df["family"].map(fi).to_numpy()
        idle = df["idle_w"].to_numpy(float)
        ye = np.log(df["energy_per_step_kwh"].to_numpy(float))
        yt = np.log(df["step_time_s"].to_numpy(float))
        ng, nf = len(gpus), len(fams)

        def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return x[: 4 * ng].reshape(ng, 4), x[4 * ng :]

        def resid(x: np.ndarray) -> np.ndarray:
            gp, theta = unpack(x)
            e = np.empty(len(df))
            t = np.empty(len(df))
            for g in range(ng):
                sel = g_idx == g
                e[sel], t[sel] = self._predict_arrays(
                    gp[g], df[sel], idle[sel], np.exp(theta[f_idx[sel]])
                )
            return np.concatenate([np.log(e) - ye, np.log(t) - yt, self.SHRINK * theta])

        lo = np.concatenate([np.tile(self.BOUNDS[0], ng), np.full(nf, -4.0)])
        hi = np.concatenate([np.tile(self.BOUNDS[1], ng), np.full(nf, 4.0)])
        sol = None
        for start in (
            [0.3, 4.0, 1.0, 0.2],
            [0.3, 16.0, 1.5, 0.4],
            [0.3, 64.0, 1.0, 0.1],
            [0.1, 8.0, 2.0, 0.3],
        ):
            x0 = np.concatenate([np.tile(start, ng), np.zeros(nf)])
            cand = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")
            if sol is None or cand.cost < sol.cost:
                sol = cand
        assert sol is not None
        gp, theta = unpack(sol.x)
        self.params = {g: gp[i] for g, i in gi.items()}
        self.fam_mult = {f: float(np.exp(theta[i])) for f, i in fi.items()}
        self.pooled = np.exp(np.log(gp).mean(axis=0))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        e = np.zeros(len(df))
        t = np.zeros(len(df))
        gpus = df["gpu"].to_numpy()
        for gpu in np.unique(gpus):
            sel = gpus == gpu
            sub = df[sel]
            seen = str(gpu) in self.params
            p = self.params[str(gpu)] if seen else self.pooled
            idle = (
                sub["idle_w"].to_numpy(float)
                if "idle_w" in sub and seen
                else self.idle_frac * sub["tdp_w"].to_numpy(float)
            )
            e[sel], t[sel] = self._predict_arrays(p, sub, idle, self._mult(sub))
        return pd.DataFrame({"energy": e, "time": t}, index=df.index)
