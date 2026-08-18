"""End-to-end training pipeline.

1. Load raw simulated (or real) sensor log.
2. Build causal engineered features.
3. Time-ordered train/val/test split.
4. Train yield regressor on train, compute residuals.
5. Fit EWMA/SPC control limits on train residuals.
6. Train Isolation Forest on train features.
7. Save all artifacts (model + scaler-free, since tree models don't need scaling).
"""
from __future__ import annotations

import argparse

import pandas as pd

from yieldguard.config import load_config, resolve_path
from yieldguard.data.loader import load_raw, time_split
from yieldguard.features.engineering import build_features, feature_columns
from yieldguard.models.anomaly_detector import build_isolation_forest, ewma_residual_score
from yieldguard.models.yield_regressor import build_regressor, evaluate_regressor, train_regressor
from yieldguard.utils.io import save_json, save_model
from yieldguard.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YieldGuard models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info("Loading raw data...")
    raw = load_raw(cfg)

    logger.info("Building engineered features...")
    feat_df = build_features(raw, cfg)
    feat_cols = feature_columns(feat_df)

    logger.info("Splitting train/val/test (time-ordered, no shuffling)...")
    splits = time_split(feat_df, cfg)
    logger.info(
        f"train={len(splits.train):,} val={len(splits.val):,} test={len(splits.test):,}"
    )

    # --- 1. Yield regressor ---
    logger.info("Training yield regressor...")
    regressor = build_regressor(cfg)
    train_regressor(regressor, splits.train[feat_cols], splits.train["yield_pct"])

    val_metrics = evaluate_regressor(regressor, splits.val[feat_cols], splits.val["yield_pct"])
    logger.info(f"Val regressor metrics: {val_metrics.as_dict()}")

    # --- 2. Residuals for SPC + anomaly fusion ---
    train_preds = regressor.predict(splits.train[feat_cols])
    train_residuals = splits.train["yield_pct"] - train_preds
    spc_cfg = cfg["model"]["anomaly"]["spc"]
    spc_train = ewma_residual_score(train_residuals, spc_cfg["ewma_alpha"], spc_cfg["sigma_limit"])
    logger.info(f"Train residual std: {train_residuals.std():.3f}")

    # --- 3. Isolation Forest on engineered sensor features ---
    logger.info("Training Isolation Forest anomaly detector...")
    iso_forest = build_isolation_forest(cfg)
    iso_forest.fit(splits.train[feat_cols])

    # --- Save artifacts ---
    model_dir = resolve_path(cfg, "paths.model_dir")
    save_model(regressor, model_dir / "yield_regressor.joblib")
    save_model(iso_forest, model_dir / "isolation_forest.joblib")
    save_json({"feature_columns": feat_cols}, model_dir / "feature_columns.json")
    save_json(
        {
            "train_residual_std": float(train_residuals.std()),
            "val_metrics": val_metrics.as_dict(),
        },
        model_dir / "train_summary.json",
    )

    # persist processed splits so evaluate.py doesn't need to redo feature engineering
    processed_dir = resolve_path(cfg, "paths.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_csv(processed_dir / "train.csv", index=False)
    splits.val.to_csv(processed_dir / "val.csv", index=False)
    splits.test.to_csv(processed_dir / "test.csv", index=False)

    logger.info(f"Artifacts saved to {model_dir}")
    logger.info("Training complete. Run `python -m yieldguard.pipelines.evaluate` next.")


if __name__ == "__main__":
    main()
