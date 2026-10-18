"""Rolling-origin folds inside the training period and Optuna tuning (§7.4, §8.5).

Test data is never touched here: ``assert_not_test`` guards every entry point.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from greenqueue.carbon.features import feature_columns
from greenqueue.carbon.forecasters import RIDGE_ALPHAS, RidgeForecaster
from greenqueue.eval.metrics import pinball

PURGE = pd.Timedelta(
    days=2
)  # a horizon-48 target of a train issue time must not overlap validation


def assert_not_test(df: pd.DataFrame, test_start: pd.Timestamp) -> None:
    """Raise if any issue time lies in the test period."""
    if len(df) and df["issue_time"].max() >= test_start:
        raise AssertionError("test-period rows reached tuning code")


def rolling_origin_folds(
    train: pd.DataFrame, n_folds: int = 4, val_months: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index pairs ``(fit, val)``; each val block follows its fit window (expanding window)."""
    end = train["issue_time"].max().normalize() + pd.Timedelta(days=1)
    folds = []
    for i in range(n_folds, 0, -1):
        v_end = end - pd.DateOffset(months=val_months * (i - 1))
        v_start = v_end - pd.DateOffset(months=val_months)
        fit = np.flatnonzero((train["issue_time"] < v_start - PURGE).to_numpy())
        val = np.flatnonzero(
            ((train["issue_time"] >= v_start) & (train["issue_time"] < v_end)).to_numpy()
        )
        if len(fit) and len(val):
            folds.append((fit, val))
    return folds


def tune_ridge(train: pd.DataFrame, test_start: pd.Timestamp, use_neso: bool = False) -> float:
    """Pick ridge alpha from the 13-value grid by mean fold MAE."""
    assert_not_test(train, test_start)
    folds = rolling_origin_folds(train)
    best, best_score = float(RIDGE_ALPHAS[0]), float("inf")
    for a in RIDGE_ALPHAS:
        scores = []
        for fit, val in folds:
            m = RidgeForecaster(a, use_neso).fit(train.iloc[fit])
            scores.append(
                np.mean(np.abs(m.predict(train.iloc[val]) - train["y"].iloc[val].to_numpy()))
            )
        if np.mean(scores) < best_score:
            best, best_score = float(a), float(np.mean(scores))
    return best


def tune_lightgbm(
    train: pd.DataFrame,
    test_start: pd.Timestamp,
    *,
    n_trials: int = 50,
    seed: int = 0,
    use_neso: bool = False,
    max_estimators: int = 1000,
) -> dict[str, Any]:
    """Optuna TPE search on median (tau=0.5) pinball loss; structure is reused for all taus."""
    assert_not_test(train, test_start)
    cols = feature_columns(use_neso)
    x = train[cols].copy()
    x["dow"] = pd.Categorical(x["dow"], categories=list(range(7)))
    y = train["y"].to_numpy()
    folds = rolling_origin_folds(train)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        scores = []
        for i, (fit, val) in enumerate(folds):
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=0.5,
                n_estimators=max_estimators,
                bagging_freq=1,
                max_bin=255,
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
                **params,
            )
            m.fit(
                x.iloc[fit],
                y[fit],
                eval_set=[(x.iloc[val], y[val])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            trial.set_user_attr(f"best_iter_{i}", int(m.best_iteration_ or max_estimators))
            scores.append(pinball(y[val], np.asarray(m.predict(x.iloc[val])), 0.5))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr(
            "n_estimators",
            int(np.mean([trial.user_attrs[f"best_iter_{i}"] for i in range(len(folds))])),
        )
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    best["n_estimators"] = int(study.best_trial.user_attrs["n_estimators"])
    best["bagging_freq"] = 1
    best["max_bin"] = 255
    return best
