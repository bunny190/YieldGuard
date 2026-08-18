"""Batch inference: score a new sensor log CSV with the trained models.

Usage:
    python -m yieldguard.pipelines.infer --config configs/default.yaml \
        --input data/raw/new_log.csv --output artifacts/reports/scored.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from yieldguard.config import load_config, resolve_path
from yieldguard.features.engineering import build_features
from yieldguard.models.anomaly_detector import blended_risk_score, ewma_residual_score
from yieldguard.utils.io import load_json, load_model
from yieldguard.utils.logging import get_logger

logger = get_logger(__name__)


def score(df: pd.DataFrame, cfg, model_dir) -> pd.DataFrame:
    regressor = load_model(model_dir / "yield_regressor.joblib")
    iso_forest = load_model(model_dir / "isolation_forest.joblib")
    feat_cols = load_json(model_dir / "feature_columns.json")["feature_columns"]

    feat_df = build_features(df, cfg)
    preds = regressor.predict(feat_df[feat_cols])
    residuals = (
        feat_df["yield_pct"] - preds if "yield_pct" in feat_df.columns else pd.Series([0.0] * len(feat_df))
    )

    spc_cfg = cfg["model"]["anomaly"]["spc"]
    spc = ewma_residual_score(residuals, spc_cfg["ewma_alpha"], spc_cfg["sigma_limit"])
    iso_scores = iso_forest.decision_function(feat_df[feat_cols])
    risk = blended_risk_score(iso_scores, spc["ewma_residual"].to_numpy(), cfg)

    out = feat_df[["timestamp"]].copy()
    out["predicted_yield"] = preds
    out["risk_score"] = risk
    out["alert"] = out["risk_score"] >= 50.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a sensor log with trained YieldGuard models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = resolve_path(cfg, "paths.model_dir")

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    scored = score(df, cfg, model_dir)
    scored.to_csv(args.output, index=False)

    n_alerts = int(scored["alert"].sum())
    logger.info(f"Scored {len(scored):,} rows -> {args.output}")
    logger.info(f"Alerts raised: {n_alerts:,} ({n_alerts / len(scored):.1%})")


if __name__ == "__main__":
    main()
