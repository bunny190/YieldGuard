"""Yield / quality soft-sensor: predicts YIELD from current sensor readings.

The regressor never sees `is_anomaly` or `fault_label` — a real soft sensor
only has access to process measurements, not ground-truth fault labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from yieldguard.config import Config


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def build_regressor(cfg: Config):
    rcfg = cfg["model"]["regressor"]
    if rcfg["type"] == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=rcfg["n_estimators"],
            max_depth=rcfg["max_depth"],
            learning_rate=rcfg["learning_rate"],
            subsample=rcfg["subsample"],
            random_state=rcfg["random_state"],
        )
    elif rcfg["type"] == "random_forest":
        return RandomForestRegressor(
            n_estimators=rcfg["n_estimators"],
            max_depth=rcfg["max_depth"],
            random_state=rcfg["random_state"],
        )
    raise ValueError(f"Unknown regressor type: {rcfg['type']}")


def train_regressor(model, X: pd.DataFrame, y: pd.Series):
    model.fit(X, y)
    return model


def evaluate_regressor(model, X: pd.DataFrame, y: pd.Series) -> RegressionMetrics:
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = r2_score(y, preds)
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)


def feature_importances(model, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return imp_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
