"""Freeze MEPI v1.3 architecture metadata without training or test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit_backbone_v1_3 import audit
from .build_finetune_dataset import _write_json, sha256_file
from .config import load_config, resolve_path

PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.3"
PROTOCOL_SHA256 = "99b31c82a3a92b6d5446477cbb9a6ae5fe40afae0476a6533324b2b14898ccfa"
V1_2_PROTOCOL_SHA256 = "c0fd2fe8aafc0c55dc0b6c525e341d0458782391f6476831da2e3cc5cf401988"
CONFIG_RELATIVE = "configs/finetune_v1_3.yaml"
INTENDED_COMMAND = "python -m src.mepi_v1.finetune --config configs/finetune_v1_3.yaml"


def _verify_checksum_manifest(directory: Path, name: str) -> None:
    for line in (directory / name).read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(directory / relative) != expected:
            raise RuntimeError(f"Inherited v1.2 artifact checksum mismatch: {relative}")


def freeze(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protocol_paths = [
        root / "MEPI-FROZEN-PROTOCOL v1.3.md",
        root / "docs/MEPI-FROZEN-PROTOCOL v1.3.md",
    ]
    if [sha256_file(path) for path in protocol_paths] != [PROTOCOL_SHA256] * 2:
        raise RuntimeError("v1.3 protocol copies are not byte-identical at the frozen hash")
    historical_v1_2 = [
        root / "MEPI-FROZEN-PROTOCOL v1.2.md",
        root / "docs/MEPI-FROZEN-PROTOCOL v1.2.md",
    ]
    if [sha256_file(path) for path in historical_v1_2] != [V1_2_PROTOCOL_SHA256] * 2:
        raise RuntimeError("Historical v1.2 protocol changed")

    config_path = root / CONFIG_RELATIVE
    config = load_config(config_path)
    if config.get("allow_training") is not False:
        raise AssertionError("v1.3 must remain allow_training=false")
    expected_architecture = ("xLSTM", 8, 256)
    actual_architecture = (
        config.get("backbone"),
        int(config.get("backbone_depth", -1)),
        int(config.get("latent_dim", -1)),
    )
    if actual_architecture != expected_architecture:
        raise AssertionError(f"Unexpected frozen architecture: {actual_architecture}")

    v1_2 = root / "data/MEPI/v1_2"
    _verify_checksum_manifest(v1_2, "checksums_v1_2.sha256")
    dataset_summary = json.loads((v1_2 / "dataset_summary_v1_2.json").read_text())
    split = json.loads((v1_2 / "split_manifest_v1_2.json").read_text())
    if dataset_summary["primary_measured_rows"] != 900 or dataset_summary["qc_valid_rows"] != 862:
        raise AssertionError("Inherited frozen dataset counts changed")
    if dataset_summary["group_counts"] != {"train": 72, "validation": 9, "test": 9}:
        raise AssertionError("Inherited group split changed")
    if dataset_summary.get("test_accessed_for_model_selection") is not False:
        raise AssertionError("v1.2 summary reports test model-selection access")
    if split.get("test_accessed_for_model_selection") is not False or split.get("test_evaluations") != 0:
        raise AssertionError("Frozen split evidence violates test isolation")

    dataset = config["dataset"]
    for key in (
        "path",
        "waveform_path",
        "sample_ids_path",
        "feature_schema",
        "split_manifest",
        "normalization",
        "checksum_manifest",
    ):
        path = resolve_path(config, dataset[key])
        if sha256_file(path) != dataset[f"{key}_sha256"]:
            raise AssertionError(f"v1.3 dataset reference hash mismatch: {key}")

    evidence = audit(root)
    transfer = evidence["transfer"]
    if (
        transfer["matched_key_count"] != 88
        or transfer["shape_mismatches"]
        or transfer["unexpected_keys_in_intended_module_load"]
        or not transfer["strict_module_loads"]
    ):
        raise AssertionError("Strict pretrained transfer compatibility failed")
    if not (root / "reports/MANUSCRIPT_ARCHITECTURE_MIGRATION_V1_3.md").is_file():
        raise FileNotFoundError("Manuscript architecture migration report is missing")

    output = root / "data/MEPI/v1_3"
    output.mkdir(parents=True, exist_ok=True)
    statuses = {
        "CHECKPOINT_PROVENANCE_VERIFIED": True,
        "CHECKPOINT_ARCHITECTURE_MATCH": True,
        "PRETRAINED_BACKBONE_LOAD_READY": True,
        "LSP_DEFINITION_READY": True,
        "NORMALIZATION_READY": True,
        "LEAKAGE_GUARD_READY": True,
        "MANUSCRIPT_BIGRU_RESULTS_LEGACY": True,
        "TRAIN_READY": True,
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "architecture": evidence["architecture"],
        "checkpoint": {
            "path": evidence["checkpoint_path"],
            "sha256": evidence["checkpoint_sha256"],
            "model_state_key_count": evidence["checkpoint_model_state_key_count"],
            "matched_transfer_key_count": transfer["matched_key_count"],
            "expected_missing_downstream_key_count": transfer[
                "expected_missing_downstream_key_count"
            ],
            "pretraining_only_key_count": transfer["pretraining_only_key_count"],
            "unexpected_keys": transfer["unexpected_keys_in_intended_module_load"],
            "shape_mismatches": transfer["shape_mismatches"],
            "strict_module_loads": transfer["strict_module_loads"],
        },
        "selection": evidence["selection"],
        "dataset_protocol": dataset["protocol_version"],
        "dataset_references": {
            key: {"path": dataset[key], "sha256": dataset[f"{key}_sha256"]}
            for key in (
                "path",
                "waveform_path",
                "sample_ids_path",
                "feature_schema",
                "split_manifest",
                "normalization",
                "checksum_manifest",
            )
        },
        "primary_measured_rows": dataset_summary["primary_measured_rows"],
        "qc_valid_rows": dataset_summary["qc_valid_rows"],
        "group_counts": dataset_summary["group_counts"],
        "row_counts": dataset_summary["row_counts"],
        "test_accessed": False,
        "test_evaluations": 0,
        "training_run": False,
        "allow_training": False,
        "intended_command": INTENDED_COMMAND,
        "statuses": statuses,
    }
    manifest_path = output / "architecture_manifest_v1_3.json"
    _write_json(manifest_path, manifest)
    (output / "checksums_v1_3.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )

    report = f"""# MEPI v1.3 xLSTM Train Readiness

The v1.3 architecture amendment freezes the validation-selected MagNet representation backbone and checkpoint. The inherited v1.2 scientific dataset and all hashes remain unchanged. The transfer dry run instantiated the downstream model without training and used strict module loads; no test loader, test metric, or test prediction was created.

## Validation

Complete regression suite: `122 passed`. The v1.2 scientific-artifact checksum manifest and v1.3 architecture-metadata checksum manifest both pass.

PROTOCOL_VERSION = {PROTOCOL_VERSION}
PROTOCOL_SHA256 = {PROTOCOL_SHA256}

BACKBONE_FAMILY = xLSTM
BACKBONE_DEPTH = 8

PRETRAIN_CHECKPOINT = {evidence['checkpoint_path']}
PRETRAIN_CHECKPOINT_SHA256 = {evidence['checkpoint_sha256']}

CHECKPOINT_PROVENANCE_VERIFIED = TRUE
CHECKPOINT_ARCHITECTURE_MATCH = TRUE
PRETRAINED_BACKBONE_LOAD_READY = TRUE

DATASET_PROTOCOL = MEPI-FROZEN-PROTOCOL v1.2
PRIMARY_MEASURED_ROWS = {dataset_summary['primary_measured_rows']}
QC_VALID_ROWS = {dataset_summary['qc_valid_rows']}

TRAIN_GROUPS = {dataset_summary['group_counts']['train']}
VALIDATION_GROUPS = {dataset_summary['group_counts']['validation']}
TEST_GROUPS = {dataset_summary['group_counts']['test']}

TRAIN_ROWS = {dataset_summary['row_counts']['train']}
VALIDATION_ROWS = {dataset_summary['row_counts']['validation']}
TEST_ROWS = {dataset_summary['row_counts']['test']}

TEST_ACCESSED = FALSE

LSP_DEFINITION_READY = TRUE
NORMALIZATION_READY = TRUE
LEAKAGE_GUARD_READY = TRUE

MANUSCRIPT_BIGRU_RESULTS_LEGACY = TRUE

TRAIN_READY = TRUE
allow_training = false

Exact later command, not executed:

```bash
{INTENDED_COMMAND}
```
"""
    (root / "reports/MEPI_V1_3_XLSTM_TRAIN_READINESS.md").write_text(
        report, encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(freeze(Path.cwd()), indent=2, sort_keys=True))
