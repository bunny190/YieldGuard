import pandas as pd

from yieldguard.config import load_config
from yieldguard.data.simulate import simulate
from yieldguard.features.engineering import build_features, feature_columns


def test_build_features_no_nans_and_drops_warmup_rows():
    cfg = load_config("configs/default.yaml")
    raw = simulate(cfg)
    feat_df = build_features(raw, cfg)

    max_lag = max(cfg["features"]["lag_steps"])
    assert len(feat_df) == len(raw) - max_lag
    assert not feat_df.isna().any().any(), "engineered features must not contain NaNs"


def test_feature_columns_excludes_targets_and_metadata():
    cfg = load_config("configs/default.yaml")
    raw = simulate(cfg)
    feat_df = build_features(raw, cfg)
    cols = feature_columns(feat_df)

    assert "yield_pct" not in cols
    assert "is_anomaly" not in cols
    assert "fault_label" not in cols
    assert "timestamp" not in cols
    assert len(cols) > 10  # lag + rolling + roc features should add plenty of columns


def test_features_are_causal_not_leaking_future():
    """Rolling/lag features at row i must only depend on rows <= i."""
    cfg = load_config("configs/default.yaml")
    raw = simulate(cfg)
    feat_df = build_features(raw, cfg)

    # perturb a single far-future value and confirm an earlier row's features are unchanged
    perturbed = raw.copy()
    perturbed.loc[perturbed.index[-1], "rxn_temp"] += 1000.0
    feat_perturbed = build_features(perturbed, cfg)

    early_row_original = feat_df.iloc[5]
    early_row_perturbed = feat_perturbed.iloc[5]
    pd.testing.assert_series_equal(early_row_original, early_row_perturbed, check_names=False)
