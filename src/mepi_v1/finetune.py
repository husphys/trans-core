"""Guarded downstream entry point for frozen MEPI protocol versions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import load_config, require_training_enabled, resolve_path
from .finetune_notebook import require_scientific_training_ready


def validate_finetune_readiness(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    blockers: list[str] = []
    dataset_value = config.get("dataset", {}).get("path")
    if not dataset_value:
        blockers.append("real fine-tuning dataset path is not configured")
    elif not resolve_path(config, dataset_value).exists():
        blockers.append("configured fine-tuning dataset does not exist")
    if config.get("protocol_version") == "MEPI-FROZEN-PROTOCOL v1.4":
        from .finetune_v1_4 import audit_frozen_evidence

        try:
            audit_frozen_evidence(resolve_path(config, "."), config_path)
        except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
            blockers.append(str(error))
        return {"status": "READY" if not blockers else "FAIL", "blockers": blockers}
    if config.get("protocol_version") == "MEPI-FROZEN-PROTOCOL v1.3":
        expected_hash = str(config.get("protocol_sha256", ""))
        for relative in (
            "MEPI-FROZEN-PROTOCOL v1.3.md",
            "docs/MEPI-FROZEN-PROTOCOL v1.3.md",
        ):
            path = resolve_path(config, relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                blockers.append(f"v1.3 protocol hash mismatch: {relative}")
        if (config.get("backbone"), config.get("backbone_depth"), config.get("latent_dim")) != (
            "xLSTM",
            8,
            256,
        ):
            blockers.append("v1.3 architecture must be xLSTM depth 8 width 256")
        checkpoint_value = config.get("pretrained_checkpoint")
        checkpoint = resolve_path(config, checkpoint_value) if checkpoint_value else None
        if checkpoint is None or not checkpoint.is_file():
            blockers.append("pretrained_checkpoint is missing")
        elif hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config.get(
            "pretrained_checkpoint_sha256"
        ):
            blockers.append("pretrained_checkpoint SHA-256 mismatch")
        dataset = config.get("dataset", {})
        artifact_keys = (
            "path",
            "waveform_path",
            "sample_ids_path",
            "feature_schema",
            "split_manifest",
            "normalization",
            "checksum_manifest",
        )
        for key in artifact_keys:
            value = dataset.get(key)
            path = resolve_path(config, value) if value else None
            expected = dataset.get(f"{key}_sha256")
            if path is None or not path.is_file():
                blockers.append(f"dataset.{key} is missing")
            elif not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                blockers.append(f"dataset.{key} SHA-256 mismatch")
        return {"status": "READY" if not blockers else "FAIL", "blockers": blockers}
    if config.get("protocol_version") == "MEPI-FROZEN-PROTOCOL v1.2":
        expected_hash = str(config.get("protocol_sha256", ""))
        for relative in (
            "MEPI-FROZEN-PROTOCOL v1.2.md",
            "docs/MEPI-FROZEN-PROTOCOL v1.2.md",
        ):
            path = resolve_path(config, relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                blockers.append(f"v1.2 protocol hash mismatch: {relative}")
        dataset = config.get("dataset", {})
        for key in (
            "waveform_path",
            "sample_ids_path",
            "feature_schema",
            "split_manifest",
            "normalization",
        ):
            value = dataset.get(key)
            if not value or not resolve_path(config, value).is_file():
                blockers.append(f"dataset.{key} is missing")
        lsp = config.get("lsp", {})
        expected_lsp = {
            "gas_constant_j_per_mol_k": 8.314462618,
            "effective_activation_energy_j_per_mol": 125000.0,
            "reference_temperature_k": 298.15,
        }
        for key, expected in expected_lsp.items():
            if lsp.get(key) != expected:
                blockers.append(f"lsp.{key} does not match frozen v1.2")
        if lsp.get("epsilon", "missing") is not None:
            blockers.append("v1.2 LSP must not use epsilon")
        checkpoint = config.get("model", {}).get("pretrained_checkpoint")
        if not checkpoint or not resolve_path(config, checkpoint).is_file():
            blockers.append("model.pretrained_checkpoint is missing")
        return {"status": "READY" if not blockers else "FAIL", "blockers": blockers}
    transformer = config.get("transformer", {})
    for key in (
        "primary_turns",
        "effective_area_m2",
        "primary_resistance_ohm",
        "secondary_resistance_ohm",
    ):
        if transformer.get(key) is None:
            blockers.append(f"transformer.{key} is not configured")
    arrhenius = config.get("arrhenius", {})
    for key in ("activation_energy_ev", "reference_temperature_k", "epsilon"):
        if arrhenius.get(key) is None:
            blockers.append(f"arrhenius.{key} is not configured")
    return {
        "status": "WAITING_FOR_REAL_FINETUNE_DATA" if blockers else "READY",
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_v1.yaml")
    parser.add_argument(
        "--run-directory",
        default="experiments/finetune_v1_4_xlstm_depth8/baseline_l3_0p3_l4_0p20",
    )
    parser.add_argument("--force-retrain", action="store_true")
    arguments = parser.parse_args()
    readiness = validate_finetune_readiness(arguments.config)
    print(json.dumps(readiness, indent=2))
    if readiness["status"] != "READY":
        return
    config = load_config(arguments.config)
    if not bool(config.get("allow_training", False)):
        print("TRAINING_NOT_AUTHORIZED: readiness passed; allow_training remains false")
        return
    require_training_enabled(config)
    if config.get("protocol_version") == "MEPI-FROZEN-PROTOCOL v1.4":
        from .finetune_v1_4 import run_v1_4_baseline

        result = run_v1_4_baseline(
            arguments.config,
            resolve_path(config, arguments.run_directory),
            force_retrain=arguments.force_retrain,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    require_scientific_training_ready(config)
    raise NotImplementedError("Shared MEPI v1.3 fine-tuning execution is not implemented")


if __name__ == "__main__":
    main()
