# YieldGuard

**Predictive Maintenance & Yield Anomaly Detection for Process Plants**

YieldGuard is an end-to-end ML system that watches a process unit's sensor
time-series (temperatures, pressures, flow rates, catalyst/feed properties,
vibration, etc.) and does two things continuously:

1. **Yield regression** — predicts the expected product yield (or a proxy
   quality metric, e.g. conversion %, pour point depression efficiency) from
   current operating conditions, so operators know *what yield they should be
   getting right now*.
2. **Anomaly / drift classification** — flags when the unit is behaving
   abnormally (sensor drift, incipient fouling, catalyst deactivation, a
   failing pump/valve) *before* it shows up as a yield loss or a trip, using
   residuals from the regressor plus unsupervised anomaly detectors.

The framing is deliberately close to refinery/petrochemical unit operations
(distillation, catalytic reaction, blending) — the kind of process-engineering
context you'd find in a CPCL-style plant — rather than a generic Kaggle
tabular dataset. Sensor tags, units, and fault modes in the synthetic
generator are written to look like a real DCS historian export.

## Why this project is different from a generic Kaggle ML repo

- Uses **time-series-aware** splits (no shuffling across time — this is a
  cardinal sin in process data that most Kaggle solutions get wrong).
- Models **physical process behavior**: first-order lag dynamics, delayed
  response to feed changes, and multiple realistic fault modes (heat
  exchanger fouling, catalyst deactivation, sensor drift/stuck sensor,
  feed composition upset) instead of i.i.d. noise.
- Combines a **supervised regressor** (yield/quality prediction) with an
  **unsupervised anomaly detector** (Isolation Forest + residual-based
  control limits), which mirrors how predictive maintenance is actually done
  in industry — you rarely have labeled failure data, so you lean on
  soft-sensor residuals and statistical process control (SPC) style limits.
- Ships with a **Streamlit-style operator dashboard concept** (see
  `src/yieldguard/pipelines/dashboard.py`) and a **FastAPI inference service**
  so the project reads as deployable, not just a notebook.

## Repository layout

```
YieldGuard/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── Makefile
├── .gitignore
├── configs/
│   └── default.yaml            # all tunable parameters in one place
├── data/
│   ├── raw/                    # raw synthetic DCS-style export lands here
│   └── processed/              # feature-engineered train/val/test splits
├── src/yieldguard/
│   ├── __init__.py
│   ├── config.py                # config loading (dataclass + yaml)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── simulate.py          # physics-flavored synthetic sensor generator
│   │   └── loader.py            # time-aware train/val/test split, scalers
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py       # lag features, rolling stats, rate-of-change
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yield_regressor.py   # GradientBoosting/RandomForest yield model
│   │   └── anomaly_detector.py  # IsolationForest + residual SPC limits
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── train.py             # end-to-end training entrypoint
│   │   ├── evaluate.py          # metrics + plots
│   │   ├── infer.py             # batch/streaming inference
│   │   └── api.py               # FastAPI service wrapping both models
│   └── utils/
│       ├── __init__.py
│       ├── io.py                # save/load artifacts
│       └── logging.py           # consistent logging setup
├── tests/
│   ├── test_simulate.py
│   ├── test_features.py
│   └── test_models.py
├── notebooks/
│   └── 01_eda.md                # EDA outline (markdown, not committed ipynb)
└── .github/workflows/ci.yml     # lint + test on push
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic plant data (or drop a real historian CSV into data/raw/)
python -m yieldguard.data.simulate --config configs/default.yaml

# 2. Train both models (yield regressor + anomaly detector)
python -m yieldguard.pipelines.train --config configs/default.yaml

# 3. Evaluate on the held-out (future-in-time) test split
python -m yieldguard.pipelines.evaluate --config configs/default.yaml

# 4. Serve predictions
uvicorn yieldguard.pipelines.api:app --reload
```

## Data / sensor tags

The simulator produces a DCS-historian-like tag list:

| Tag         | Description                          | Units   |
|-------------|---------------------------------------|---------|
| `FEED_FLOW` | Feed flow rate                        | m3/h    |
| `FEED_TEMP` | Feed inlet temperature                | °C      |
| `RXN_TEMP`  | Reactor / process temperature         | °C      |
| `RXN_PRESS` | Reactor / process pressure            | bar     |
| `CAT_ACT`   | Catalyst activity index (latent)      | 0–1     |
| `HX_DP`     | Heat exchanger differential pressure  | kPa     |
| `VIB_RMS`   | Rotating equipment vibration (RMS)    | mm/s    |
| `FEED_COMP` | Feed composition upset indicator      | 0–1     |
| `YIELD`     | Target: product yield / quality index | %       |

Fault modes injected by the simulator: `fouling`, `catalyst_decay`,
`sensor_drift`, `feed_upset`, `stuck_sensor` — each with a ground-truth
`is_anomaly` label for evaluating the classifier, while the regressor never
sees this label (it only ever sees sensor readings, as a real soft sensor
would).

## Modeling approach

**Yield regressor** — Gradient Boosted Trees (with a Random Forest baseline)
trained on lagged + rolling-window features. Time-aware cross-validation
(`sklearn.model_selection.TimeSeriesSplit`) prevents leakage from the future.

**Anomaly detector** — two signals are fused:
1. *Residual-based SPC*: `|actual_yield − predicted_yield|` tracked with an
   EWMA and 3-sigma control limits — a classic soft-sensor fault detection
   approach.
2. *Isolation Forest* on the raw + engineered sensor features, to catch
   anomalies that don't necessarily show up as a yield deviation yet
   (leading indicators — e.g. rising vibration before a bearing fails).

A final "risk score" blends both signals so operators get a single 0–100
health index per timestamp, plus a plain-language explanation of which tags
are driving the flag.

## Status / extending this

This repo is a strong portfolio piece as-is (runs end to end on synthetic
data). To adapt it to real CPCL/plant data:
1. Point `data/raw/` at a historian export with the same tag structure (or
   edit `configs/default.yaml` → `data.tag_map` to match your tags).
2. Re-tune `configs/default.yaml` → `simulate.fault_params` is only used for
   synthetic data; skip it for real data.
3. Retrain — everything downstream (features, models, dashboard, API) is
   tag-name agnostic as long as the config is updated.

## License

MIT — see `LICENSE`.
