"""MEPI v1.5 exact-state continuation with no test-dataset constructor.

The v1.4 scientific engine is reused unchanged. v1.5 changes only the maximum
training budget from 50 to 100 epochs and accepts the audited v1.4 rolling
checkpoint exactly once as the parent continuation state.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpointing import load_transfer_weights, save_checkpoint
from .config import load_config, resolve_path
from .finetune_notebook import set_reproducibility_seed
from .finetune_v1_4 import (
    CHECKPOINT_SHA256, STEINMETZ_SHA256, StrictEarlyStopping,
    _atomic_json, build_development_datasets, build_epoch_train_loader,
    build_validation_loader, capture_rng_state, load_steinmetz_prior,
    make_amp_scaler, restore_rng_state, sha256_file, train_one_epoch,
    validate_one_epoch, validate_v1_4_config,
)
from .models import DownstreamMEPIV14

PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.5"
PROTOCOL_SHA256 = "0f68d6ebd5d16177e5238471639f861dfa91522130ac6a4a3ac48634b9af1e39"
PARENT_PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.4"
PARENT_PROTOCOL_SHA256 = "685f79b6b700ae131ff2a19af524417ead46ee9c073f1c2ddf9d96191f59b487"
PARENT_STATE_SHA256 = "22aca18c37b4526c46e3044eeeda4e36a6a0364891e767cace755961eef44138"
RESUME_NEXT_EPOCH = 50
MAX_EPOCHS = 100


def validate_v1_5_config(config: dict[str, Any]) -> None:
    """Prove v1.4 equivalence after normalizing the sole scientific change."""
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise AssertionError("Configuration is not MEPI-FROZEN-PROTOCOL v1.5")
    if config.get("protocol_sha256") != PROTOCOL_SHA256:
        raise AssertionError("v1.5 protocol hash mismatch")
    if config.get("training", {}).get("max_epochs") != MAX_EPOCHS:
        raise AssertionError("v1.5 max_epochs must be 100")
    inherited = copy.deepcopy(config)
    inherited["protocol_version"] = PARENT_PROTOCOL_VERSION
    inherited["protocol_sha256"] = PARENT_PROTOCOL_SHA256
    inherited["training"]["max_epochs"] = 50
    validate_v1_4_config(inherited)
    resume = config.get("resume", {})
    expected = {
        "parent_protocol": PARENT_PROTOCOL_VERSION,
        "parent_state_sha256": PARENT_STATE_SHA256,
        "resume_next_epoch": RESUME_NEXT_EPOCH,
        "exact_state_required": True,
        "weights_only_resume_forbidden": True,
    }
    mismatches = {key: (resume.get(key), value) for key, value in expected.items() if resume.get(key) != value}
    if mismatches:
        raise AssertionError(f"v1.5 continuation metadata mismatch: {mismatches}")


def audit_v1_5_frozen_evidence(project_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(config_path)
    validate_v1_5_config(config)
    protocol_paths = [root / "MEPI-FROZEN-PROTOCOL v1.5.md", root / "docs/MEPI_FROZEN_PROTOCOL_v1.5.md"]
    if [sha256_file(path) for path in protocol_paths] != [PROTOCOL_SHA256, PROTOCOL_SHA256]:
        raise AssertionError("v1.5 protocol copies are not frozen and byte-identical")
    if sha256_file(root / "MEPI-FROZEN-PROTOCOL v1.4.md") != PARENT_PROTOCOL_SHA256:
        raise AssertionError("parent v1.4 protocol changed")
    if sha256_file(resolve_path(config, config["pretrained_checkpoint"])) != CHECKPOINT_SHA256:
        raise AssertionError("selected representation checkpoint changed")
    if sha256_file(resolve_path(config, config["physics"]["steinmetz_prior"])) != STEINMETZ_SHA256:
        raise AssertionError("Steinmetz prior changed")
    for key in ("path", "waveform_path", "sample_ids_path", "feature_schema", "split_manifest", "normalization", "checksum_manifest"):
        if sha256_file(resolve_path(config, config["dataset"][key])) != config["dataset"][f"{key}_sha256"]:
            raise AssertionError(f"frozen dataset artifact changed: {key}")
    return {"status": "PASS", "protocol_sha256": PROTOCOL_SHA256, "max_epochs": 100, "patience": 10, "scheduler": None, "test_dataset_created": False, "test_loader_created": False}


def construct_v1_5_model(config_path: str | Path) -> tuple[DownstreamMEPIV14, dict[str, Any]]:
    config = load_config(config_path)
    validate_v1_5_config(config)
    model = DownstreamMEPIV14(config["backbone"], backbone_layers=int(config["backbone_depth"]), latent_dim=int(config["latent_dim"]))
    transfer = load_transfer_weights(model, resolve_path(config, config["pretrained_checkpoint"]))
    if "p_loss_aux_z" not in transfer["newly_initialized"]:
        transfer["newly_initialized"].insert(3, "p_loss_aux_z")
    return model, {"status": "PASS", "transfer": transfer, "parameter_count": sum(p.numel() for p in model.parameters())}


def make_optimizer_v1_5(model: nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    validate_v1_5_config(config)
    training = config["training"]
    return torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))


def v1_5_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": PROTOCOL_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256, "steinmetz_sha256": STEINMETZ_SHA256, "optimizer": "AdamW", "learning_rate": 5e-6, "weight_decay": 1e-4, "scheduler": None, "lambda3": 0.3, "lambda4": 0.2}


def parent_compatibility() -> dict[str, Any]:
    return {"protocol_version": PARENT_PROTOCOL_VERSION, "protocol_sha256": PARENT_PROTOCOL_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256, "steinmetz_sha256": STEINMETZ_SHA256, "optimizer": "AdamW", "learning_rate": 5e-6, "weight_decay": 1e-4, "scheduler": None, "lambda3": 0.3, "lambda4": 0.2}


def audit_exact_parent_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if sha256_file(path) != PARENT_STATE_SHA256:
        raise AssertionError("parent rolling-state SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"epoch", "model_state", "optimizer_state", "best_validation_loss", "best_epoch", "patience_counter", "amp_scaler_state", "rng_state", "compatibility", "history"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"incomplete parent resume state: {missing}")
    if checkpoint["compatibility"] != parent_compatibility():
        raise ValueError("parent resume compatibility mismatch")
    if checkpoint["epoch"] != RESUME_NEXT_EPOCH or len(checkpoint["history"]) != RESUME_NEXT_EPOCH:
        raise ValueError("parent state is not the exact epoch-49/next-epoch-50 state")
    if checkpoint["best_epoch"] != 49 or checkpoint["best_validation_loss"] != 0.4636393350713393:
        raise ValueError("parent best-state metadata mismatch")
    if set(checkpoint["rng_state"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("parent RNG state is incomplete")
    return {"status": "PASS", "parent_state_sha256": PARENT_STATE_SHA256, "resume_next_epoch": 50, "history_rows": 50, "optimizer_state": True, "amp_scaler_state": True, "rng_state": True}


def prepare_v1_5_continuation(config_path: str | Path, run_directory: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_v1_5_config(config)
    parent = resolve_path(config, config["resume"]["parent_run"])
    parent_state = parent / "last_checkpoint.pt"
    audit = audit_exact_parent_state(parent_state)
    target = Path(run_directory)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("last_checkpoint.pt", "best_checkpoint.pt", "history.json"):
        source, destination = parent / name, target / name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"refusing to overwrite incompatible imported state: {destination}")
        shutil.copy2(source, destination)
    provenance = {**audit, "status": "EXACT_STATE_IMPORTED_NOT_EXECUTED", "parent_protocol": PARENT_PROTOCOL_VERSION, "parent_run": str(parent), "parent_checkpoint": str(parent_state), "parent_checkpoint_sha256": PARENT_STATE_SHA256, "scientific_training_run": False, "test_accessed": False}
    _atomic_json(target / "continuation_provenance.json", provenance)
    return provenance


def save_v1_5_resume(path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, epoch: int, early_stopping: StrictEarlyStopping, compatibility: dict[str, Any], history: list[dict[str, Any]]) -> None:
    save_checkpoint({"format_version": "mepi-v1.5-finetune-resume-v1", "epoch": int(epoch), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_validation_loss": early_stopping.best_validation_total_loss, "best_epoch": early_stopping.best_epoch, "patience_counter": early_stopping.patience_counter, "amp_scaler_state": scaler.state_dict(), "rng_state": capture_rng_state(), "compatibility": compatibility, "history": history}, path)


def load_v1_5_resume(path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, config: dict[str, Any]) -> tuple[int, StrictEarlyStopping, list[dict[str, Any]], str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"epoch", "model_state", "optimizer_state", "best_validation_loss", "best_epoch", "patience_counter", "amp_scaler_state", "rng_state", "compatibility", "history"}
    missing = sorted(required - checkpoint.keys())
    if missing or "scheduler_state" in checkpoint:
        raise ValueError(f"invalid v1.5 resume checkpoint; missing={missing}")
    compatibility = checkpoint["compatibility"]
    if compatibility == parent_compatibility():
        source = "v1.4_exact_parent_state"
        if sha256_file(path) != PARENT_STATE_SHA256 or checkpoint["epoch"] != 50:
            raise ValueError("weights-only or altered parent continuation is forbidden")
    elif compatibility == v1_5_compatibility(config):
        source = "v1.5_rolling_state"
    else:
        raise ValueError("resume compatibility mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if any(group["lr"] != 5e-6 or group["weight_decay"] != 1e-4 for group in optimizer.param_groups):
        raise ValueError("restored AdamW state violates fixed optimizer contract")
    if scaler.is_enabled():
        scaler.load_state_dict(checkpoint["amp_scaler_state"])
    restore_rng_state(checkpoint["rng_state"])
    stopping = StrictEarlyStopping(patience=10, min_delta=0.0, best_validation_total_loss=float(checkpoint["best_validation_loss"]), best_epoch=int(checkpoint["best_epoch"]), patience_counter=int(checkpoint["patience_counter"]))
    return int(checkpoint["epoch"]), stopping, list(checkpoint["history"]), source


def run_v1_5_baseline(config_path: str | Path, run_directory: str | Path, *, force_retrain: bool = False) -> dict[str, Any]:
    """Continue the official baseline only after explicit caller authorization."""
    if force_retrain:
        raise RuntimeError("v1.5 forbids discarding the exact v1.4 continuation state")
    config = load_config(config_path)
    validate_v1_5_config(config)
    if config.get("allow_training") is not True:
        raise RuntimeError("v1.5 configuration does not authorize training")
    audit_v1_5_frozen_evidence(Path(config_path).resolve().parents[1], config_path)
    run_path = Path(run_directory)
    completed_path = run_path / "completed.json"
    if completed_path.exists():
        return json.loads(completed_path.read_text(encoding="utf-8"))
    last_path, best_path = run_path / "last_checkpoint.pt", run_path / "best_checkpoint.pt"
    if not last_path.exists() or not (run_path / "continuation_provenance.json").exists():
        raise RuntimeError("exact v1.4 continuation state was not prepared")

    set_reproducibility_seed(int(config["seed"]))
    train_dataset, validation_dataset = build_development_datasets(config_path)
    validation_loader = build_validation_loader(validation_dataset, batch_size=int(config["training"]["batch_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = construct_v1_5_model(config_path)
    model.to(device)
    optimizer = make_optimizer_v1_5(model, config)
    scaler = make_amp_scaler(device)
    prior = load_steinmetz_prior(config)
    start_epoch, stopping, history, resume_source = load_v1_5_resume(last_path, model=model, optimizer=optimizer, scaler=scaler, config=config)
    stopped_early = False
    compatibility = v1_5_compatibility(config)
    for epoch in range(start_epoch, MAX_EPOCHS):
        train_loader = build_epoch_train_loader(train_dataset, batch_size=int(config["training"]["batch_size"]), seed=int(config["seed"]), epoch=epoch)
        train_losses = train_one_epoch(model, train_loader, optimizer=optimizer, scaler=scaler, prior=prior, device=device, gradient_clip_norm=float(config["training"]["gradient_clip_norm"]))
        validation_losses = validate_one_epoch(model, validation_loader, prior=prior, device=device)
        improved, should_stop = stopping.update(validation_losses["total"], epoch)
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_losses, "validation": validation_losses, "strict_improvement": improved, "patience_counter": stopping.patience_counter})
        if improved:
            save_v1_5_resume(best_path, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch + 1, early_stopping=stopping, compatibility=compatibility, history=history)
        save_v1_5_resume(last_path, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch + 1, early_stopping=stopping, compatibility=compatibility, history=history)
        _atomic_json(run_path / "history.json", {"epochs": history})
        if should_stop:
            stopped_early = True
            break
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    result = {"status": "TRAINING_COMPLETE_VALIDATION_ONLY", "parent_protocol": PARENT_PROTOCOL_VERSION, "parent_state_sha256": PARENT_STATE_SHA256, "resume_source": resume_source, "resume_next_epoch": start_epoch, "best_epoch": stopping.best_epoch, "best_validation_total_loss": stopping.best_validation_total_loss, "epochs_completed": len(history), "stopped_early": stopped_early, "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256_file(best_path), "scheduler": None, "test_accessed": False, "test_dataset_created": False, "test_loader_created": False}
    _atomic_json(completed_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory", required=True)
    args = parser.parse_args()
    print(json.dumps(run_v1_5_baseline(args.config, args.run_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
