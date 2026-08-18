"""Anomaly detection layer: fuses two independent signals.

1. Residual-based SPC (statistical process control):
   Track the EWMA of |actual_yield - predicted_yield| and flag samples that
   exceed a sigma-based control limit. This is the classic soft-sensor fault
   detection approach used on real plants.

2. Isolation Forest on the raw + engineered sensor features:
   Catches anomalies that are visible in the *sensor pattern* before they
   show up as a yield deviation — e.g. rising vibration ahead of a bearing
   failure, which is exactly the "predictive maintenance" half of the story.

The two signals are min-max normalized and blended into a single 0-100
"risk score" per timestamp, which is what an operator dashboard would show.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from yieldguard.config import Config


@dataclass
class AnomalyMetrics:
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return {
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def build_isolation_forest(cfg: Config) -> IsolationForest:
    icfg = cfg["model"]["anomaly"]["isolation_forest"]
    return IsolationForest(
        n_estimators=icfg["n_estimators"],
        contamination=icfg["contamination"],
        random_state=icfg["random_state"],
    )


def ewma_residual_score(residuals: pd.Series, alpha: float, sigma_limit: float) -> pd.DataFrame:
    """Return EWMA-smoothed residual, control limit, and boolean flag per row."""
    ewma = residuals.ewm(alpha=alpha, adjust=False).mean()
    rolling_std = residuals.rolling(50, min_periods=10).std().bfill().fillna(residuals.std())
    upper_limit = sigma_limit * rolling_std
    flag = (ewma.abs() > upper_limit).astype(int)
    return pd.DataFrame({"ewma_residual": ewma, "control_limit": upper_limit, "spc_flag": flag})


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def blended_risk_score(
    iso_scores: np.ndarray, ewma_residual: np.ndarray, cfg: Config
) -> np.ndarray:
    """Combine Isolation Forest anomaly score + residual magnitude into 0-100."""
    weights = cfg["model"]["risk_score"]
    iso_norm = _minmax(-iso_scores)  # IsolationForest: lower score = more anomalous
    resid_norm = _minmax(np.abs(ewma_residual))
    blended = (
        weights["isolation_weight"] * iso_norm + weights["residual_weight"] * resid_norm
    )
    return np.clip(blended * 100, 0, 100)


def evaluate_anomaly_detection(y_true: pd.Series, risk_score: np.ndarray, threshold: float = 50.0) -> AnomalyMetrics:
    y_pred = (risk_score >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, risk_score) if y_true.nunique() > 1 else float("nan")
    ap = average_precision_score(y_true, risk_score) if y_true.nunique() > 1 else float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return AnomalyMetrics(roc_auc=roc_auc, average_precision=ap, precision=precision, recall=recall, f1=f1)
