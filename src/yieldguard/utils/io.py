"""Save/load helpers for models and reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_model(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: str | Path) -> Any:
    return joblib.load(Path(path))


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(Path(path), "r") as f:
        return json.load(f)
