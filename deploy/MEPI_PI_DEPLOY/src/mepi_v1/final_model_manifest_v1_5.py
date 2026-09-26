"""Build and validate the MEPI v1.5 Notebook 34 -> 35 handoff.

This module reads only frozen provenance artifacts, the completed final-test
ledger, and checkpoint bytes for SHA-256 verification. It does not import,
instantiate, or read any scientific dataset and has no training entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

MANIFEST_RELATIVE_PATH = Path(
    "experiments/finetune_v1_5_xlstm_depth8/final_model_manifest.json"
)
FINAL_CONFIG_RELATIVE_PATH = Path("configs/final_model_v1_5.yaml")
FINAL_CONFIG_SHA256 = "4b18d275be0a56f09371f2fd91d3fe03bba72a7cee6ca732014fbbf043dcdf96"
PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.5"
PROTOCOL_SHA256 = "0f68d6ebd5d16177e5238471639f861dfa91522130ac6a4a3ac48634b9af1e39"
SELECTION_RULE_SHA256 = "8c4eda9035e7e5e4f75a4b0ccde643c8946fd3b990575ec0323977cb8497a5be"
SELECTED_CONFIGURATION_ID = "l3_1_l4_0p05"
SELECTED_CHECKPOINT_PATH = (
    "experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/"
    "l3_1_l4_0p05/best_checkpoint.pt"
)
SELECTED_CHECKPOINT_SHA256 = (
    "0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33"
)
SELECTED_WEIGHTS = {
    "lambda1": 1.0,
    "lambda2": 1.0,
    "lambda3": 1.0,
    "lambda4": 0.05,
}
SELECTION_METRIC = "VAL_TASK_SCORE"
SELECTION_SCORE = 0.2023820665829322


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(root: Path, relative: str | Path, label: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise AssertionError(f"{label} must be a project-relative path")
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} must contain a JSON object")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label} changed: expected {expected!r}, got {actual!r}")


def build_final_model_manifest(project_root: str | Path) -> dict[str, Any]:
    """Derive the final-model handoff only from frozen, existing evidence."""

    root = Path(project_root).resolve()
    final_config_path = _require_file(root, FINAL_CONFIG_RELATIVE_PATH, "final config")
    _require_equal(sha256_file(final_config_path), FINAL_CONFIG_SHA256, "final config SHA256")
    final_config = yaml.safe_load(final_config_path.read_text(encoding="utf-8"))
    if not isinstance(final_config, dict):
        raise AssertionError("Final config must contain a YAML mapping")

    _require_equal(final_config.get("protocol_version"), PROTOCOL_VERSION, "protocol version")
    _require_equal(final_config.get("protocol_sha256"), PROTOCOL_SHA256, "protocol SHA256")
    protocol_path_value = final_config.get("protocol_artifact")
    protocol_path = _require_file(root, protocol_path_value, "v1.5 protocol")
    _require_equal(sha256_file(protocol_path), PROTOCOL_SHA256, "v1.5 protocol SHA256")

    selection = final_config.get("selection")
    if not isinstance(selection, dict):
        raise AssertionError("Final config is missing the frozen selection mapping")
    _require_equal(selection.get("selected_by"), SELECTION_METRIC, "selection metric")
    _require_equal(
        selection.get("selected_configuration_id"),
        SELECTED_CONFIGURATION_ID,
        "selected configuration",
    )
    _require_equal(selection.get("VAL_TASK_SCORE"), SELECTION_SCORE, "selection score")
    selection_rule_path_value = selection.get("rule_artifact")
    selection_rule_path = _require_file(root, selection_rule_path_value, "selection rule")
    _require_equal(
        selection.get("rule_sha256"), SELECTION_RULE_SHA256, "recorded selection-rule SHA256"
    )
    _require_equal(
        sha256_file(selection_rule_path), SELECTION_RULE_SHA256, "selection-rule SHA256"
    )

    _require_equal(final_config.get("loss_weights"), SELECTED_WEIGHTS, "selected loss weights")
    status = final_config.get("status")
    if not isinstance(status, dict):
        raise AssertionError("Final config is missing the frozen status mapping")
    _require_equal(status.get("FINAL_CONFIG_FROZEN"), True, "final configuration freeze")

    completed_run = final_config.get("completed_run")
    if not isinstance(completed_run, dict):
        raise AssertionError("Final config is missing the completed-run mapping")
    checkpoint_path_value = completed_run.get("selected_checkpoint")
    _require_equal(checkpoint_path_value, SELECTED_CHECKPOINT_PATH, "selected checkpoint path")
    _require_equal(
        completed_run.get("selected_checkpoint_sha256"),
        SELECTED_CHECKPOINT_SHA256,
        "recorded selected-checkpoint SHA256",
    )
    checkpoint_path = _require_file(root, checkpoint_path_value, "selected checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require_equal(
        checkpoint_sha256, SELECTED_CHECKPOINT_SHA256, "selected-checkpoint SHA256"
    )

    final_evaluation = final_config.get("final_evaluation")
    if not isinstance(final_evaluation, dict):
        raise AssertionError("Final config is missing the final-evaluation mapping")
    for key in (
        "training_allowed",
        "retraining_allowed",
        "checkpoint_selection_allowed",
        "lambda_modification_allowed",
    ):
        _require_equal(final_evaluation.get(key), False, f"final-evaluation {key}")
    _require_equal(final_evaluation.get("exactly_once"), True, "exactly-once test policy")

    result_path_value = final_evaluation.get("results_json")
    result_path = _require_file(root, result_path_value, "final-test result ledger")
    result = _read_json_object(result_path, "final-test result ledger")
    _require_equal(result.get("EVALUATION_STATUS"), "FINAL_TEST_COMPLETE", "final-test status")
    _require_equal(result.get("FINAL_CONFIG_FROZEN"), True, "result-ledger freeze status")
    _require_equal(result.get("TEST_ACCESSED"), True, "result-ledger test access")
    _require_equal(result.get("TEST_EVALUATION_COUNT"), 1, "result-ledger evaluation count")
    _require_equal(result.get("SELECTED_CONFIGURATION_ID"), SELECTED_CONFIGURATION_ID, "result-ledger configuration")
    _require_equal(result.get("SELECTED_CHECKPOINT"), SELECTED_CHECKPOINT_PATH, "result-ledger checkpoint path")
    _require_equal(
        result.get("SELECTED_CHECKPOINT_SHA256"),
        SELECTED_CHECKPOINT_SHA256,
        "result-ledger checkpoint SHA256",
    )
    for key, expected in SELECTED_WEIGHTS.items():
        _require_equal(result.get(f"SELECTED_{key.upper()}"), expected, f"result-ledger {key}")

    source_model_config_path = final_config.get("source_model_config")
    source_model_config_sha256 = final_config.get("source_model_config_sha256")
    if not isinstance(source_model_config_path, str) or not isinstance(
        source_model_config_sha256, str
    ):
        raise AssertionError("Final config is missing source-model configuration provenance")

    return {
        "manifest_schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_path": str(protocol_path_value),
        "protocol_sha256": PROTOCOL_SHA256,
        "configuration_id": SELECTED_CONFIGURATION_ID,
        **SELECTED_WEIGHTS,
        "selection_metric": SELECTION_METRIC,
        "selection_score": SELECTION_SCORE,
        "checkpoint_path": SELECTED_CHECKPOINT_PATH,
        "checkpoint_sha256": checkpoint_sha256,
        "source_model_config_path": source_model_config_path,
        "source_model_config_sha256": source_model_config_sha256,
        "final_config_path": FINAL_CONFIG_RELATIVE_PATH.as_posix(),
        "final_config_sha256": FINAL_CONFIG_SHA256,
        "selection_rule_path": str(selection_rule_path_value),
        "selection_rule_sha256": SELECTION_RULE_SHA256,
        "final_test_result_path": str(result_path_value),
        "final_test_result_sha256": sha256_file(result_path),
        "final_config_frozen": True,
        "final_test_complete": True,
        "test_evaluation_count": 1,
        "frequency_screening_ready": True,
    }


def write_final_model_manifest(
    project_root: str | Path, manifest_path: str | Path | None = None
) -> Path:
    """Atomically write the evidence-derived manifest and return its path."""

    root = Path(project_root).resolve()
    output = root / MANIFEST_RELATIVE_PATH if manifest_path is None else Path(manifest_path)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_final_model_manifest(root)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, output)
    return output


def validate_final_model_manifest(
    project_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Fail closed unless the manifest exactly matches current frozen evidence."""

    root = Path(project_root).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Final-model manifest is missing: {path}")
    manifest = _read_json_object(path, "final-model manifest")
    expected = build_final_model_manifest(root)
    if manifest != expected:
        changed = sorted(
            key
            for key in set(manifest) | set(expected)
            if manifest.get(key) != expected.get(key)
        )
        raise AssertionError(f"Final-model manifest does not match frozen evidence: {changed}")
    return manifest
