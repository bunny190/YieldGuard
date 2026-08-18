# 01 — Exploratory Data Analysis (outline)

This is written as markdown rather than a committed `.ipynb` so diffs stay
readable. Paste these cells into a notebook against
`data/raw/plant_sensor_log.csv` (generate it first with
`python -m yieldguard.data.simulate`).

## 1. Load & sanity check

```python
import pandas as pd
df = pd.read_csv("data/raw/plant_sensor_log.csv", parse_dates=["timestamp"])
df.describe()
df["fault_label"].value_counts()
```

## 2. Time-series overview

Plot each sensor tag over time, shading `is_anomaly == 1` regions, to build
intuition for what each fault mode looks like:
- `fouling` → `hx_dp` trending up, `rxn_temp` trending down
- `catalyst_decay` → `cat_activity` decaying, `yield_pct` drifting down slowly
- `sensor_drift` → `rxn_temp` reading high while yield stays normal (a red
  herring for the regressor — good discussion point on soft sensor limits)
- `feed_upset` → `feed_comp_upset` spiking, sharp `yield_pct` drop
- `stuck_sensor` → `vib_rms` flatlining (zero variance -> dead giveaway)

## 3. Correlation with yield

```python
df.corr(numeric_only=True)["yield_pct"].sort_values()
```

Expect `cat_activity` positively correlated, `feed_comp_upset` negatively
correlated, and a non-linear (inverted-U) relationship with `rxn_temp` /
`rxn_press` around the design point — worth a 2D contour or partial
dependence plot once the model is trained.

## 4. Stationarity / autocorrelation

Run an ADF test and plot the ACF/PACF for `rxn_temp` and `yield_pct` to
justify the lag windows chosen in `configs/default.yaml` (`features.lag_steps`).

## 5. Class balance for anomaly detection

```python
df["is_anomaly"].mean()
```

Confirms this is a highly imbalanced problem (~10-15% anomalous), which is
why ROC-AUC alone is insufficient — check `evaluate.py`'s use of
average precision (PR-AUC) alongside it.
