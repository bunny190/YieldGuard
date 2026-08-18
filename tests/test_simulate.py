import pandas as pd

from yieldguard.config import load_config
from yieldguard.data.simulate import simulate


def test_simulate_shape_and_columns():
    cfg = load_config("configs/default.yaml")
    df = simulate(cfg)

    expected_cols = {
        "timestamp",
        "feed_flow",
        "feed_temp",
        "rxn_temp",
        "rxn_press",
        "cat_activity",
        "hx_dp",
        "vib_rms",
        "feed_comp_upset",
        "yield_pct",
        "is_anomaly",
        "fault_label",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) > 1000
    assert df["timestamp"].is_monotonic_increasing


def test_simulate_has_anomalies_and_faults():
    cfg = load_config("configs/default.yaml")
    df = simulate(cfg)

    assert df["is_anomaly"].sum() > 0
    fault_kinds = set(df.loc[df["is_anomaly"] == 1, "fault_label"].unique())
    assert "none" not in fault_kinds
    assert len(fault_kinds) >= 3  # multiple distinct fault modes injected


def test_yield_within_valid_range():
    cfg = load_config("configs/default.yaml")
    df = simulate(cfg)
    assert (df["yield_pct"] >= 0).all()
    assert (df["yield_pct"] <= 100).all()
