"""Time-aware data loading and splitting.

Cardinal rule for process/sensor time-series: never shuffle before splitting.
The test set must be strictly later in time than train/val, otherwise the
model gets to "see the future" through autocorrelated neighbors and reported
metrics will be optimistic and useless in production.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from yieldguard.config import Config, resolve_path


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw(cfg: Config) -> pd.DataFrame:
    path = resolve_path(cfg, "paths.raw_data")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, cfg: Config) -> Splits:
    n = len(df)
    train_frac = cfg["split"]["train_frac"]
    val_frac = cfg["split"]["val_frac"]

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return Splits(train=train, val=val, test=test)
