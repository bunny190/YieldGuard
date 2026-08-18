"""Config loading utilities.

Everything downstream (simulator, features, models, pipelines) reads from a
single YAML file so there are no magic numbers scattered through the code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Thin wrapper around the parsed YAML dict.

    Kept as a dict-backed object (rather than nested dataclasses) on purpose:
    the config surface is large and still evolving, and `cfg["model"]["regressor"]`
    is more forgiving to extend than a rigid schema. Use `.get()` for optional keys.
    """

    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def project_root(self) -> Path:
        # configs/default.yaml -> repo root is two levels up
        return self.path.resolve().parent.parent


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=path)


def resolve_path(cfg: Config, key_path: str) -> Path:
    """Resolve a dotted config path (e.g. 'paths.raw_data') to an absolute Path."""
    node: Any = cfg.raw
    for k in key_path.split("."):
        node = node[k]
    return (cfg.project_root / node).resolve()
