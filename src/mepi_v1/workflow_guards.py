"""Machine-readable dependency and split-access guards for downstream notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STAGES = {
    "30_dataset_audit": "30_dataset_audit.json",
    "31_downstream_depth": "31_downstream_depth.json",
    "33_loss_weight": "33_loss_weight.json",
    "34_final_model": "34_final_model.json",
}

PREREQUISITES = {
    "31_downstream_depth": (("30_dataset_audit", "PASS"),),
    "33_loss_weight": (("31_downstream_depth", "COMPLETE"),),
    "34_final_model": (("33_loss_weight", "COMPLETE"),),
    "35_frequency_screening": (("34_final_model", "COMPLETE"),),
}


def status_path(project_root: str | Path, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unknown status-producing stage: {stage}")
    return Path(project_root).resolve() / "experiments/downstream_v1/status" / STAGES[stage]


def read_status(project_root: str | Path, stage: str) -> dict[str, Any]:
    path = status_path(project_root, stage)
    if not path.is_file():
        return {"stage": stage, "status": "MISSING", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage or not isinstance(payload.get("status"), str):
        raise ValueError(f"Malformed workflow status: {path}")
    return payload


def write_status(project_root: str | Path, stage: str, payload: dict[str, Any]) -> Path:
    path = status_path(project_root, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    document["stage"] = stage
    if not isinstance(document.get("status"), str):
        raise ValueError("Workflow status payload requires a string status")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def require_prerequisites(project_root: str | Path, stage: str) -> list[dict[str, Any]]:
    checks = []
    for dependency, expected in PREREQUISITES.get(stage, ()):
        actual = read_status(project_root, dependency)
        checks.append({"stage": dependency, "expected": expected, "actual": actual["status"]})
        if actual["status"] != expected:
            raise RuntimeError(
                f"{stage} is blocked: {dependency} must be {expected}, got {actual['status']}"
            )
    return checks


def require_test_access(*, stage: str, final_configuration_frozen: bool) -> None:
    if stage != "34_final_model" or not final_configuration_frozen:
        raise RuntimeError(
            "Test access is permitted only in notebook 34 after FINAL CONFIGURATION FROZEN: YES"
        )


def validate_downstream_resume_checkpoint(
    checkpoint: dict[str, Any], expected_compatibility: dict[str, Any]
) -> None:
    required = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "amp_scaler_state",
        "rng_state",
        "epoch",
        "compatibility",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Incomplete downstream resume checkpoint: {missing}")
    actual = checkpoint["compatibility"]
    mismatched = {
        key: (actual.get(key), value)
        for key, value in expected_compatibility.items()
        if actual.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Incompatible downstream resume checkpoint: {mismatched}")
