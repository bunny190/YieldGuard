"""Feature engineering for sensor time-series.

Produces:
  * lag features (value N samples ago)
  * rolling mean/std over several windows (short-term trend + volatility)
  * rate-of-change (first difference) for fast-moving tags
All engineered strictly causally (no centered windows, no leakage from
future samples) so this is safe to use in streaming/production inference.
"""
from __future__ import annotations

import pandas as pd

from yieldguard.config import Config

SENSOR_COLS = [
    "feed_flow",
    "feed_temp",
    "rxn_temp",
    "rxn_press",
    "cat_activity",
    "hx_dp",
    "vib_rms",
    "feed_comp_upset",
]


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for w in windows:
            df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}_std"] = df[col].rolling(w, min_periods=1).std().fillna(0.0)
    return df


def add_rate_of_change(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[f"{col}_roc"] = df[col].diff().fillna(0.0)
    return df


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the full engineered-feature pipeline described in the config."""
    feat_cfg = cfg["features"]
    df = add_lag_features(df, SENSOR_COLS, feat_cfg["lag_steps"])
    df = add_rolling_features(df, SENSOR_COLS, feat_cfg["rolling_windows"])
    df = add_rate_of_change(df, feat_cfg["rate_of_change_cols"])

    # Rows at the start won't have full lag history — drop them rather than
    # imputing, since imputed lags would be misleading during training.
    max_lag = max(feat_cfg["lag_steps"])
    df = df.iloc[max_lag:].reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column usable as a model input (excludes targets/metadata)."""
    exclude = {"timestamp", "yield_pct", "is_anomaly", "fault_label"}
    return [c for c in df.columns if c not in exclude]
