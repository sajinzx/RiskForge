"""Forecast model ladder F0-F3 and the NESO baseline FN (implementation.md §8.2).

All models are direct multi-horizon: one row per (issue time, horizon), horizon ``h`` is a feature.
Each ``predict`` works on the dataframe produced by ``features.build_dataset``.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from greenqueue.carbon.features import feature_columns

QUANTILES = (0.05, 0.5, 0.95)
RIDGE_ALPHAS = tuple(np.logspace(-3, 3, 13))


def persistence(df: pd.DataFrame) -> np.ndarray:
    """F0: the last fully observed hour."""
    return df["lag_1"].to_numpy(dtype=float)


def seasonal_naive(df: pd.DataFrame, period_h: int = 24) -> np.ndarray:
    """F1: same hour yesterday (most recent known day) or one week ago."""
    col = {24: "snaive24", 168: "snaive168"}[period_h]
    return df[col].to_numpy(dtype=float)


def neso_forecast(df: pd.DataFrame) -> np.ndarray:
    """FN: NESO's own published forecast for the target hour (GB only)."""
    return df["neso_fc"].to_numpy(dtype=float)


class RidgeForecaster:
    """F2: standardised ridge on lags, rolling stats and calendar (dow one-hot)."""

    def __init__(self, alpha: float = 1.0, use_neso: bool = False) -> None:
        self.alpha = alpha
        self.cols = feature_columns(use_neso)
        num = [c for c in self.cols if c != "dow"]
        pre = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                    num,
                ),
                (
                    "dow",
                    OneHotEncoder(categories=[list(range(7))], handle_unknown="ignore"),
                    ["dow"],
                ),
            ]
        )
        self.model = Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])

    def fit(self, df: pd.DataFrame) -> RidgeForecaster:
        self.model.fit(df[self.cols], df["y"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(df[self.cols]))


DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
}


class LGBQuantileForecaster:
    """F3: one LightGBM model per quantile, same tuned structure for all quantiles."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        quantiles: tuple[float, ...] = QUANTILES,
        use_neso: bool = False,
        seed: int = 0,
    ) -> None:
        self.params = {**DEFAULT_LGB_PARAMS, **(params or {})}
        self.quantiles = quantiles
        self.cols = feature_columns(use_neso)
        self.seed = seed
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.cols].copy()
        x["dow"] = x["dow"].astype("category")
        x["dow"] = x["dow"].cat.set_categories(list(range(7)))
        return x

    def fit(self, df: pd.DataFrame) -> LGBQuantileForecaster:
        x = self._x(df)
        for tau in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=tau,
                random_state=self.seed,
                verbose=-1,
                n_jobs=-1,
                **self.params,
            )
            m.fit(x, df["y"])
            self.models[tau] = m
        return self

    def predict(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        x = self._x(df)
        raw = {tau: m.predict(x) for tau, m in self.models.items()}
        # quantile crossing is possible with independent models; sort per row
        taus = sorted(raw)
        stacked = np.sort(np.column_stack([raw[t] for t in taus]), axis=1)
        return {t: stacked[:, i] for i, t in enumerate(taus)}
