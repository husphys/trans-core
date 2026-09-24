"""Configuration loading with project-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    return config


def load_pretraining_config(path: str | Path) -> dict[str, Any]:
    """Load a backbone identity plus the single shared pretraining configuration."""

    specific = load_config(path)
    common_value = specific.get("common_config")
    if common_value is None:
        return specific
    common_path = resolve_path(specific, common_value)
    with common_path.open("r", encoding="utf-8") as handle:
        common = yaml.safe_load(handle) or {}
    merged = deepcopy(common)
    merged.update(
        {
            key: value
            for key, value in specific.items()
            if key not in {"_config_path", "common_config"}
        }
    )
    merged["common_config"] = str(common_value)
    merged["_config_path"] = specific["_config_path"]
    merged["_common_config_path"] = str(common_path)
    return merged


def project_root(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    return (config_path.parent / config.get("project_root", "..")).resolve()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def require_training_enabled(config: dict[str, Any]) -> None:
    if not bool(config.get("allow_training", False)):
        raise RuntimeError(
            "Training is disabled. Set allow_training: true only after reviewing the audit, "
            "dataset fingerprint, split manifests, and compute environment."
        )
