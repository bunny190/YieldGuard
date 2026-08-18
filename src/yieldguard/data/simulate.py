"""Synthetic DCS-historian-style sensor data generator for a process unit.

This is not i.i.d. noise dressed up as "sensor data" — it models:
  * first-order lag response of process variables to disturbances,
  * a yield response surface driven by temperature, pressure, catalyst
    activity, and feed composition (loosely modeled on a catalytic
    conversion / blending-type unit),
  * five injectable fault modes with realistic signatures:
      - fouling:          heat exchanger dP rises, RXN_TEMP drifts down
      - catalyst_decay:   CAT_ACT decays, yield drifts down slowly
      - sensor_drift:     one sensor (RXN_TEMP) drifts from true value
      - feed_upset:       FEED_COMP spikes, yield drops sharply then recovers
      - stuck_sensor:     VIB_RMS freezes at last value (classic stuck sensor)

Run as a script:
    python -m yieldguard.data.simulate --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from yieldguard.config import Config, load_config, resolve_path
from yieldguard.utils.logging import get_logger

logger = get_logger(__name__)


def _first_order_lag(target: np.ndarray, tau: float = 6.0) -> np.ndarray:
    """Smooth a target signal with a first-order lag (process inertia)."""
    out = np.empty_like(target, dtype=float)
    out[0] = target[0]
    alpha = 1.0 / tau
    for i in range(1, len(target)):
        out[i] = out[i - 1] + alpha * (target[i] - out[i - 1])
    return out


def _yield_response(rxn_temp, rxn_press, cat_activity, feed_comp_upset) -> np.ndarray:
    """Toy but plausible yield response surface.

    Peaks near the design operating point and degrades away from it, is
    boosted by higher catalyst activity, and is punished by feed upsets.
    Numbers are illustrative, not calibrated to a real unit.
    """
    temp_term = -0.008 * (rxn_temp - 182.0) ** 2
    press_term = -0.6 * (rxn_press - 8.6) ** 2
    cat_term = 18.0 * (cat_activity - 0.5)
    upset_penalty = -25.0 * feed_comp_upset
    base_yield = 78.0
    return base_yield + temp_term + press_term + cat_term + upset_penalty


def simulate(cfg: Config) -> pd.DataFrame:
    sim_cfg = cfg["simulate"]
    rng = np.random.default_rng(cfg.get("project", "seed", default=42))

    n_days = sim_cfg["n_days"]
    freq_min = sim_cfg["freq_minutes"]
    n_samples = int(n_days * 24 * 60 / freq_min)

    idx = pd.date_range(
        start=sim_cfg["start_date"], periods=n_samples, freq=f"{freq_min}min"
    )

    base = sim_cfg["base"]
    noise = sim_cfg["noise_std"]

    # --- baseline disturbance signals (slow random walk + diurnal wobble) ---
    t = np.arange(n_samples)
    diurnal = np.sin(2 * np.pi * t / (24 * 60 / freq_min)) * 0.5

    feed_flow_raw = base["feed_flow"] + diurnal * 3 + rng.normal(0, noise["feed_flow"], n_samples)
    feed_temp_raw = base["feed_temp"] + diurnal * 1 + rng.normal(0, noise["feed_temp"], n_samples)
    rxn_temp_raw = np.full(n_samples, base["rxn_temp"], dtype=float)
    rxn_press_raw = np.full(n_samples, base["rxn_press"], dtype=float)
    cat_activity = np.full(n_samples, base["cat_activity"], dtype=float)
    hx_dp_raw = np.full(n_samples, base["hx_dp"], dtype=float)
    vib_rms_raw = np.full(n_samples, base["vib_rms"], dtype=float)
    feed_comp_upset = np.full(n_samples, base["feed_comp_upset"], dtype=float)

    is_anomaly = np.zeros(n_samples, dtype=int)
    fault_label = np.array(["none"] * n_samples, dtype=object)

    # --- inject fault windows ---
    for fault in sim_cfg["fault_windows"]:
        kind = fault["kind"]
        start = int(fault["start_frac"] * n_samples)
        dur = int(fault["duration_frac"] * n_samples)
        end = min(start + dur, n_samples)
        sev = fault["severity"]
        window = slice(start, end)
        ramp = np.linspace(0, 1, end - start)

        if kind == "fouling":
            hx_dp_raw[window] += sev * 40 * ramp
            rxn_temp_raw[window] -= sev * 6 * ramp
        elif kind == "catalyst_decay":
            cat_activity[window] -= sev * 0.35 * ramp
        elif kind == "sensor_drift":
            rxn_temp_raw[window] += sev * 8 * ramp  # sensor reads high, process is fine
        elif kind == "feed_upset":
            feed_comp_upset[window] = sev * (0.6 + 0.4 * np.sin(np.linspace(0, 3, end - start)))
        elif kind == "stuck_sensor":
            frozen_value = vib_rms_raw[max(start - 1, 0)]
            vib_rms_raw[window] = frozen_value
        else:
            raise ValueError(f"Unknown fault kind: {kind}")

        is_anomaly[window] = 1
        fault_label[window] = kind

    # apply process inertia to key controlled variables
    rxn_temp = _first_order_lag(rxn_temp_raw, tau=8) + rng.normal(0, noise["rxn_temp"], n_samples)
    rxn_press = _first_order_lag(rxn_press_raw, tau=5) + rng.normal(0, noise["rxn_press"], n_samples)
    hx_dp = _first_order_lag(hx_dp_raw, tau=10) + rng.normal(0, noise["hx_dp"], n_samples)
    vib_rms = vib_rms_raw + rng.normal(0, noise["vib_rms"], n_samples)
    vib_rms = np.clip(vib_rms, 0, None)

    # A genuinely stuck sensor stops sampling altogether: it reports the exact
    # same digital value every scan, with zero noise, not "noisy around a
    # frozen setpoint". Re-apply the freeze *after* noise so the flatline is
    # real -- this is precisely what makes stuck sensors easy to catch via a
    # rolling-std-collapse feature.
    for fault in sim_cfg["fault_windows"]:
        if fault["kind"] != "stuck_sensor":
            continue
        start = int(fault["start_frac"] * n_samples)
        dur = int(fault["duration_frac"] * n_samples)
        end = min(start + dur, n_samples)
        frozen_value = vib_rms[max(start - 1, 0)]
        vib_rms[start:end] = frozen_value

    yield_pct = _yield_response(rxn_temp, rxn_press, cat_activity, feed_comp_upset)
    yield_pct += rng.normal(0, 0.4, n_samples)  # measurement noise on the lab/analyzer yield
    yield_pct = np.clip(yield_pct, 0, 100)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "feed_flow": feed_flow_raw,
            "feed_temp": feed_temp_raw,
            "rxn_temp": rxn_temp,
            "rxn_press": rxn_press,
            "cat_activity": cat_activity,
            "hx_dp": hx_dp,
            "vib_rms": vib_rms,
            "feed_comp_upset": feed_comp_upset,
            "yield_pct": yield_pct,
            "is_anomaly": is_anomaly,
            "fault_label": fault_label,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic YieldGuard sensor data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = simulate(cfg)

    out_path = resolve_path(cfg, "paths.raw_data")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_anom = int(df["is_anomaly"].sum())
    logger.info(f"Simulated {len(df):,} samples -> {out_path}")
    logger.info(f"Anomalous samples: {n_anom:,} ({n_anom / len(df):.1%})")
    logger.info(f"Fault breakdown:\n{df.loc[df.is_anomaly == 1, 'fault_label'].value_counts()}")


if __name__ == "__main__":
    main()
