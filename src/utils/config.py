from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload["_config_path"] = str(config_path)
    return payload


def project_root(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    root = Path(config.get("project_root", ".."))
    return (config_path.parent / root).resolve()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root(config) / candidate).resolve()


def require_full_training_enabled(config: dict[str, Any]) -> None:
    if not bool(config.get("allow_full_training", False)):
        raise RuntimeError(
            "Full training is disabled. Close the Phase A audit gates and explicitly set "
            "allow_full_training: true before starting Phases B-D."
        )

