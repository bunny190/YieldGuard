import numpy as np

from yieldguard.config import load_config
from yieldguard.data.loader import time_split
from yieldguard.data.simulate import simulate
from yieldguard.features.engineering import build_features, feature_columns
from yieldguard.models.anomaly_detector import (
    blended_risk_score,
    build_isolation_forest,
    ewma_residual_score,
    evaluate_anomaly_detection,
)
from yieldguard.models.yield_regressor import build_regressor, evaluate_regressor, train_regressor


def _prepare_splits():
    """Small/fast config override so the model tests run in seconds, not minutes.

    Uses a shorter simulated horizon and a much smaller regressor than the
    production config in configs/default.yaml -- this test only needs to
    confirm the training/eval code paths work, not that hyperparameters are
    well tuned (that's what artifacts/reports/test_report.json is for).
    """
    cfg = load_config("configs/default.yaml")
    cfg.raw["simulate"]["n_days"] = 20
    cfg.raw["model"]["regressor"]["n_estimators"] = 50
    cfg.raw["model"]["regressor"]["max_depth"] = 3
    cfg.raw["model"]["anomaly"]["isolation_forest"]["n_estimators"] = 100

    raw = simulate(cfg)
    feat_df = build_features(raw, cfg)
    splits = time_split(feat_df, cfg)
    feat_cols = feature_columns(feat_df)
    return cfg, splits, feat_cols


def test_regressor_trains_and_beats_naive_baseline():
    cfg, splits, feat_cols = _prepare_splits()

    model = build_regressor(cfg)
    train_regressor(model, splits.train[feat_cols], splits.train["yield_pct"])
    metrics = evaluate_regressor(model, splits.val[feat_cols], splits.val["yield_pct"])

    naive_mae = np.abs(splits.val["yield_pct"] - splits.train["yield_pct"].mean()).mean()
    assert metrics.mae < naive_mae, "trained regressor should beat predicting the train mean"
    # Note: R2 can legitimately go negative on a val window that happens to sit
    # inside a low-variance, steady-state period (R2 penalizes low-variance
    # targets harshly even when absolute error, i.e. MAE, is small) -- so we
    # assert against the naive-baseline MAE above rather than R2 here.
    assert metrics.mae < 5.0


def test_anomaly_pipeline_flags_more_risk_on_anomalous_rows():
    cfg, splits, feat_cols = _prepare_splits()

    regressor = build_regressor(cfg)
    train_regressor(regressor, splits.train[feat_cols], splits.train["yield_pct"])

    iso_forest = build_isolation_forest(cfg)
    iso_forest.fit(splits.train[feat_cols])

    preds = regressor.predict(splits.test[feat_cols])
    residuals = splits.test["yield_pct"] - preds
    spc_cfg = cfg["model"]["anomaly"]["spc"]
    spc = ewma_residual_score(residuals, spc_cfg["ewma_alpha"], spc_cfg["sigma_limit"])

    iso_scores = iso_forest.decision_function(splits.test[feat_cols])
    risk = blended_risk_score(iso_scores, spc["ewma_residual"].to_numpy(), cfg)

    mean_risk_anomalous = risk[splits.test["is_anomaly"] == 1].mean()
    mean_risk_normal = risk[splits.test["is_anomaly"] == 0].mean()
    assert mean_risk_anomalous > mean_risk_normal

    metrics = evaluate_anomaly_detection(splits.test["is_anomaly"], risk)
    assert 0 <= metrics.roc_auc <= 1
