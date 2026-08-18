"""Evaluate trained models on the held-out (strictly future-in-time) test split.

Produces:
  * regression metrics (MAE, RMSE, R2) on test yield predictions
  * anomaly detection metrics (ROC-AUC, PR-AUC, precision/recall/F1) against
    the ground-truth `is_anomaly` labels the simulator attached
  * a feature importance table
  * a time-series plot of actual vs predicted yield with flagged anomalies
All written to artifacts/reports/.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

from yieldguard.config import load_config, resolve_path
from yieldguard.models.anomaly_detector import (
    blended_risk_score,
    ewma_residual_score,
    evaluate_anomaly_detection,
)
from yieldguard.models.yield_regressor import evaluate_regressor, feature_importances
from yieldguard.utils.io import load_json, load_model, save_json
from yieldguard.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YieldGuard models on test set")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = resolve_path(cfg, "paths.model_dir")
    processed_dir = resolve_path(cfg, "paths.processed_dir")
    report_dir = resolve_path(cfg, "paths.report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading artifacts...")
    regressor = load_model(model_dir / "yield_regressor.joblib")
    iso_forest = load_model(model_dir / "isolation_forest.joblib")
    feat_cols = load_json(model_dir / "feature_columns.json")["feature_columns"]
    test = pd.read_csv(processed_dir / "test.csv", parse_dates=["timestamp"])

    # --- regression evaluation ---
    reg_metrics = evaluate_regressor(regressor, test[feat_cols], test["yield_pct"])
    logger.info(f"Test regression metrics: {reg_metrics.as_dict()}")

    preds = regressor.predict(test[feat_cols])
    residuals = test["yield_pct"] - preds
    spc_cfg = cfg["model"]["anomaly"]["spc"]
    spc = ewma_residual_score(residuals, spc_cfg["ewma_alpha"], spc_cfg["sigma_limit"])

    # --- anomaly evaluation ---
    iso_scores = iso_forest.decision_function(test[feat_cols])
    risk_score = blended_risk_score(iso_scores, spc["ewma_residual"].to_numpy(), cfg)
    anom_metrics = evaluate_anomaly_detection(test["is_anomaly"], risk_score)
    logger.info(f"Test anomaly detection metrics: {anom_metrics.as_dict()}")

    # --- feature importances ---
    imp_df = feature_importances(regressor, feat_cols)
    logger.info(f"Top features:\n{imp_df.head(10).to_string(index=False)}")

    # --- save numeric report ---
    save_json(
        {
            "regression": reg_metrics.as_dict(),
            "anomaly_detection": anom_metrics.as_dict(),
            "top_features": imp_df.to_dict(orient="records"),
        },
        report_dir / "test_report.json",
    )

    # --- plot: actual vs predicted yield with risk overlay ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(test["timestamp"], test["yield_pct"], label="Actual yield", linewidth=1)
    ax1.plot(test["timestamp"], preds, label="Predicted yield", linewidth=1, alpha=0.8)
    anom_mask = test["is_anomaly"] == 1
    ax1.scatter(
        test.loc[anom_mask, "timestamp"],
        test.loc[anom_mask, "yield_pct"],
        color="red",
        s=8,
        label="True anomaly window",
        zorder=5,
    )
    ax1.set_ylabel("Yield (%)")
    ax1.set_title("YieldGuard — Actual vs Predicted Yield (test set)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(test["timestamp"], risk_score, color="darkorange", linewidth=1, label="Risk score (0-100)")
    ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Alert threshold")
    ax2.set_ylabel("Risk score")
    ax2.set_xlabel("Timestamp")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig_path = report_dir / "test_yield_and_risk.png"
    fig.savefig(fig_path, dpi=150)
    logger.info(f"Saved plot to {fig_path}")
    logger.info(f"Saved report to {report_dir / 'test_report.json'}")


if __name__ == "__main__":
    main()
