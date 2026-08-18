"""FastAPI service exposing YieldGuard as a real-time scoring endpoint.

Run with:
    uvicorn yieldguard.pipelines.api:app --reload

POST /score expects a short recent history window (enough samples to cover
the largest lag/rolling window in the config) and returns the latest
predicted yield + risk score, so it mirrors how a real streaming client
(pulling the last N samples from a DCS historian) would call this service.
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from yieldguard.config import load_config, resolve_path
from yieldguard.features.engineering import build_features
from yieldguard.models.anomaly_detector import blended_risk_score, ewma_residual_score
from yieldguard.utils.io import load_json, load_model

app = FastAPI(title="YieldGuard API", version="0.1.0")

_cfg = load_config("configs/default.yaml")
_model_dir = resolve_path(_cfg, "paths.model_dir")

_regressor = None
_iso_forest = None
_feat_cols: list[str] | None = None


class SensorSample(BaseModel):
    timestamp: str
    feed_flow: float
    feed_temp: float
    rxn_temp: float
    rxn_press: float
    cat_activity: float
    hx_dp: float
    vib_rms: float
    feed_comp_upset: float
    yield_pct: float | None = None  # optional: only needed to compute residual-based SPC


class ScoreRequest(BaseModel):
    samples: list[SensorSample]  # ordered oldest -> newest


class ScoreResponse(BaseModel):
    timestamp: str
    predicted_yield: float
    risk_score: float
    alert: bool


def _lazy_load_models() -> None:
    global _regressor, _iso_forest, _feat_cols
    if _regressor is None:
        _regressor = load_model(_model_dir / "yield_regressor.joblib")
        _iso_forest = load_model(_model_dir / "isolation_forest.joblib")
        _feat_cols = load_json(_model_dir / "feature_columns.json")["feature_columns"]


@app.on_event("startup")
def _startup() -> None:
    try:
        _lazy_load_models()
    except FileNotFoundError:
        # Models not trained yet — endpoints will raise a clear 503 instead of crashing.
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": _regressor is not None}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if _regressor is None:
        try:
            _lazy_load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Run `python -m yieldguard.pipelines.train` first.",
            )

    df = pd.DataFrame([s.model_dump() for s in request.samples])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["yield_pct"].isna().any():
        df["yield_pct"] = df["yield_pct"].fillna(method="ffill").fillna(0.0)

    feat_df = build_features(df, _cfg)
    if feat_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Not enough samples to compute lag/rolling features. Send a longer history window.",
        )

    preds = _regressor.predict(feat_df[_feat_cols])
    residuals = feat_df["yield_pct"] - preds
    spc_cfg = _cfg["model"]["anomaly"]["spc"]
    spc = ewma_residual_score(residuals, spc_cfg["ewma_alpha"], spc_cfg["sigma_limit"])
    iso_scores = _iso_forest.decision_function(feat_df[_feat_cols])
    risk = blended_risk_score(iso_scores, spc["ewma_residual"].to_numpy(), _cfg)

    latest = feat_df.iloc[-1]
    latest_idx = feat_df.index[-1]
    return ScoreResponse(
        timestamp=str(latest["timestamp"]),
        predicted_yield=float(preds[-1]),
        risk_score=float(risk[-1]),
        alert=bool(risk[-1] >= 50.0),
    )
