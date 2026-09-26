"""Executable, test-isolated MEPI v1.4 downstream fine-tuning contract.

This module contains no test-dataset constructor.  It supports only the frozen
train and validation subsets and is inert unless its training entry point is
called explicitly by the v1.4 notebook with ``RUN_TRAINING=True``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .checkpointing import load_transfer_weights, save_checkpoint
from .config import load_config, resolve_path
from .constants import FINETUNE_TABULAR_FEATURES
from .finetune_notebook import set_reproducibility_seed
from .models import DownstreamMEPIV14

PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.4"
CHECKPOINT_SHA256 = "fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c"
STEINMETZ_SHA256 = "a5e110924093455cf6e0912ed4fcf82c046fec82326f0480d56358532662d5f8"
WAVEFORM_MEAN = -3.1640557780983195e-13
WAVEFORM_SCALE = 0.06941968146373301
VARIANCE_FLOOR = 1e-6
ALLOWED_SPLITS = frozenset({"train", "validation"})
EXPECTED_ROWS = {"train": 687, "validation": 85}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_development_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise RuntimeError(
            f"MEPI v1.4 permits only train/validation model development, got {split!r}"
        )


def _read_allowed_rows(path: Path, split: str) -> list[dict[str, str]]:
    assert_development_split(split)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if len(rows) != EXPECTED_ROWS[split]:
        raise AssertionError(f"Unexpected {split} row count: {len(rows)}")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v1_4_config(config: dict[str, Any]) -> None:
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 5e-6,
        "weight_decay": 1e-4,
        "scheduler": None,
        "batch_size": 64,
        "max_epochs": 50,
        "early_stopping_patience": 10,
        "min_delta": 0.0,
        "gradient_clip_norm": 1.0,
    }
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise AssertionError("Configuration is not MEPI-FROZEN-PROTOCOL v1.4")
    if (config.get("backbone"), config.get("backbone_depth"), config.get("latent_dim")) != (
        "xLSTM",
        8,
        256,
    ):
        raise AssertionError("v1.4 architecture must be xLSTM depth 8 width 256")
    training = config.get("training", {})
    mismatches = {
        key: (training.get(key), expected)
        for key, expected in expected_training.items()
        if training.get(key, "MISSING") != expected
    }
    if mismatches:
        raise AssertionError(f"v1.4 training contract mismatch: {mismatches}")
    if training.get("checkpoint_monitor") != "validation_total_loss":
        raise AssertionError("Checkpoint monitor must be validation total loss")
    if training.get("strict_improvement") is not True:
        raise AssertionError("v1.4 requires strict validation-loss improvement")
    if training.get("amp") != "cuda_conditional":
        raise AssertionError("v1.4 AMP behavior must be CUDA-conditional")
    if training.get("loss_weights") != {
        "electrical": 1.0,
        "LSP": 1.0,
        "physics": 0.3,
        "uncertainty_nll": 0.2,
    }:
        raise AssertionError("v1.4 baseline loss weights changed")


def audit_frozen_evidence(project_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(config_path)
    validate_v1_4_config(config)
    expected_protocol_hash = str(config["protocol_sha256"])
    protocol_paths = [
        root / "MEPI-FROZEN-PROTOCOL v1.4.md",
        root / "docs/MEPI_FROZEN_PROTOCOL_v1.4.md",
    ]
    protocol_hashes = [sha256_file(path) for path in protocol_paths]
    if protocol_hashes != [expected_protocol_hash, expected_protocol_hash]:
        raise AssertionError(f"v1.4 protocol hash mismatch: {protocol_hashes}")
    checkpoint = resolve_path(config, config["pretrained_checkpoint"])
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise AssertionError("Selected xLSTM checkpoint hash mismatch")
    prior = resolve_path(config, config["physics"]["steinmetz_prior"])
    if sha256_file(prior) != STEINMETZ_SHA256:
        raise AssertionError("Frozen Steinmetz prior hash mismatch")
    for key in (
        "path",
        "waveform_path",
        "sample_ids_path",
        "feature_schema",
        "split_manifest",
        "normalization",
        "checksum_manifest",
    ):
        artifact = resolve_path(config, config["dataset"][key])
        if sha256_file(artifact) != config["dataset"][f"{key}_sha256"]:
            raise AssertionError(f"Frozen dataset artifact hash mismatch: {key}")
    return {
        "status": "PASS",
        "protocol_sha256": expected_protocol_hash,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "steinmetz_prior_sha256": STEINMETZ_SHA256,
        "scheduler": None,
        "test_dataset_created": False,
        "test_loader_created": False,
    }


class FrozenV14Dataset(Dataset[dict[str, torch.Tensor]]):
    """A train-or-validation-only view of the frozen v1.2 scientific arrays."""

    def __init__(self, config_path: str | Path, split: str) -> None:
        assert_development_split(split)
        self.config = load_config(config_path)
        self.split = split
        dataset = self.config["dataset"]
        self.rows = _read_allowed_rows(resolve_path(self.config, dataset["path"]), split)
        all_ids = np.load(
            resolve_path(self.config, dataset["sample_ids_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        id_to_index = {str(sample_id): index for index, sample_id in enumerate(all_ids)}
        self.indices = [id_to_index[row["sample_id"]] for row in self.rows]
        self.waveforms = np.load(
            resolve_path(self.config, dataset["waveform_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.normalization = _load_json(resolve_path(self.config, dataset["normalization"]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        feature_values = []
        for name in FINETUNE_TABULAR_FEATURES:
            scaler = self.normalization["feature_scalers"][name]
            feature_values.append(
                (float(row[name]) - float(scaler["mean"][0]))
                / float(scaler["scale"][0])
            )
        target_z = []
        for name, column in (
            ("efficiency_percent", "efficiency_percent"),
            ("P_loss", "P_loss"),
        ):
            scaler = self.normalization["target_scalers"][name]
            target_z.append(
                (float(row[column]) - float(scaler["mean"][0]))
                / float(scaler["scale"][0])
            )
        waveform = np.array(self.waveforms[self.indices[index]], dtype=np.float64, copy=True)
        waveform = ((waveform - WAVEFORM_MEAN) / WAVEFORM_SCALE).astype(np.float32)
        return {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "tabular": torch.tensor(feature_values, dtype=torch.float32),
            "frequency_hz": torch.tensor(float(row["frequency_hz"]), dtype=torch.float32),
            "B_peak_t": torch.tensor(float(row["B_peak_t"]), dtype=torch.float32),
            "electrical_z": torch.tensor(target_z, dtype=torch.float32),
            "LSP_z": torch.tensor(float(row["LSP_z"]), dtype=torch.float32),
        }


def build_development_datasets(
    config_path: str | Path,
) -> tuple[FrozenV14Dataset, FrozenV14Dataset]:
    return FrozenV14Dataset(config_path, "train"), FrozenV14Dataset(
        config_path, "validation"
    )


def build_validation_loader(
    dataset: FrozenV14Dataset, *, batch_size: int
) -> DataLoader[dict[str, torch.Tensor]]:
    if dataset.split != "validation":
        raise ValueError("Validation loader requires the validation dataset")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def build_epoch_train_loader(
    dataset: FrozenV14Dataset, *, batch_size: int, seed: int, epoch: int
) -> DataLoader[dict[str, torch.Tensor]]:
    if dataset.split != "train":
        raise ValueError("Training loader requires the training dataset")
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def load_steinmetz_prior(config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(config, config["physics"]["steinmetz_prior"])
    if sha256_file(path) != STEINMETZ_SHA256:
        raise AssertionError("Frozen Steinmetz prior hash mismatch")
    prior = _load_json(path)
    expected = {
        "k": 1.638586240032685e-05,
        "alpha": 1.4329982981833669,
        "beta": 1.8648536845101025,
    }
    if any(prior["coefficients"][key] != value for key, value in expected.items()):
        raise AssertionError("Frozen Steinmetz coefficients changed")
    if prior["core_id_used"] is not False or prior["fit_row_count"] != 687:
        raise AssertionError("Steinmetz prior is not the audited train-only common fit")
    return prior


def steinmetz_reference_z(
    frequency_hz: torch.Tensor,
    b_peak_t: torch.Tensor,
    prior: dict[str, Any],
) -> torch.Tensor:
    if torch.any(frequency_hz <= 0) or torch.any(b_peak_t <= 0):
        raise ValueError("Steinmetz inputs must be positive")
    coefficients = prior["coefficients"]
    scaler = prior["target_scaler"]
    p_st_raw = (
        float(coefficients["k"])
        * frequency_hz.pow(float(coefficients["alpha"]))
        * b_peak_t.pow(float(coefficients["beta"]))
    )
    return ((p_st_raw - float(scaler["mean_W"])) / float(scaler["scale_W"])).unsqueeze(-1)


def construct_v1_4_model(config_path: str | Path) -> tuple[DownstreamMEPIV14, dict[str, Any]]:
    config = load_config(config_path)
    validate_v1_4_config(config)
    model = DownstreamMEPIV14(
        config["backbone"],
        backbone_layers=int(config["backbone_depth"]),
        latent_dim=int(config["latent_dim"]),
    )
    transfer = load_transfer_weights(model, resolve_path(config, config["pretrained_checkpoint"]))
    if "p_loss_aux_z" not in transfer["newly_initialized"]:
        transfer["newly_initialized"].insert(3, "p_loss_aux_z")
    return model, {
        "status": "PASS",
        "transfer": transfer,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def electrical_loss(prediction_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    return 0.7 * F.mse_loss(prediction_z, target_z) + 0.3 * F.l1_loss(
        prediction_z, target_z
    )


def lsp_loss(mu_lsp_z: torch.Tensor, lsp_z: torch.Tensor) -> torch.Tensor:
    return 0.7 * F.mse_loss(mu_lsp_z, lsp_z) + 0.3 * F.l1_loss(mu_lsp_z, lsp_z)


def gaussian_nll_full_mean(
    mu_lsp_z: torch.Tensor, var_lsp_z: torch.Tensor, lsp_z: torch.Tensor
) -> torch.Tensor:
    if torch.any(var_lsp_z < VARIANCE_FLOOR):
        raise ValueError("LSP variance is below the frozen floor")
    return torch.mean(
        0.5
        * (
            torch.log(2.0 * math.pi * var_lsp_z)
            + (lsp_z - mu_lsp_z).square() / var_lsp_z
        )
    )


def inverse_lsp_distribution(
    mu_lsp_z: torch.Tensor,
    var_lsp_z: torch.Tensor,
    *,
    mean: float,
    scale: float,
) -> dict[str, torch.Tensor]:
    if scale <= 0:
        raise ValueError("LSP scale must be positive")
    raw_mean = mu_lsp_z * scale + mean
    raw_variance = var_lsp_z * (scale**2)
    return {
        "LSP_raw_pred": raw_mean,
        "var_LSP_raw": raw_variance,
        "std_LSP_raw": torch.sqrt(raw_variance),
    }


def compute_v1_4_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda3: float = 0.3,
    lambda4: float = 0.2,
) -> dict[str, torch.Tensor]:
    prediction_electrical = torch.stack(
        (outputs["efficiency_z"], outputs["P_loss_z"]), dim=-1
    )
    loss_electrical = electrical_loss(prediction_electrical, batch["electrical_z"])
    loss_lsp = lsp_loss(outputs["mu_LSP_z"], batch["LSP_z"])
    loss_physics = torch.mean(outputs["r_St"].square())
    loss_uq = gaussian_nll_full_mean(
        outputs["mu_LSP_z"], outputs["var_LSP_z"], batch["LSP_z"]
    )
    total = loss_electrical + loss_lsp + lambda3 * loss_physics + lambda4 * loss_uq
    return {
        "electrical": loss_electrical,
        "LSP": loss_lsp,
        "physics": loss_physics,
        "uncertainty_nll": loss_uq,
        "total": total,
    }


@dataclass
class StrictEarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    best_validation_total_loss: float = math.inf
    best_epoch: int = -1
    patience_counter: int = 0

    def update(self, validation_total_loss: float, epoch: int) -> tuple[bool, bool]:
        threshold = self.best_validation_total_loss - self.min_delta
        improved = validation_total_loss < threshold
        if improved:
            self.best_validation_total_loss = float(validation_total_loss)
            self.best_epoch = int(epoch)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return improved, self.patience_counter >= self.patience


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    validate_v1_4_config(config)
    training = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def make_amp_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def resume_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["protocol_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "steinmetz_sha256": STEINMETZ_SHA256,
        "optimizer": "AdamW",
        "learning_rate": 5e-6,
        "weight_decay": 1e-4,
        "scheduler": None,
        "lambda3": 0.3,
        "lambda4": 0.2,
    }


def save_resume_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    early_stopping: StrictEarlyStopping,
    compatibility: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    checkpoint = {
        "format_version": "mepi-v1.4-finetune-resume-v1",
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_validation_loss": early_stopping.best_validation_total_loss,
        "best_epoch": early_stopping.best_epoch,
        "patience_counter": early_stopping.patience_counter,
        "amp_scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "compatibility": compatibility,
        "history": history,
    }
    save_checkpoint(checkpoint, path)


def load_resume_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    expected_compatibility: dict[str, Any],
) -> tuple[int, StrictEarlyStopping, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "epoch",
        "model_state",
        "optimizer_state",
        "best_validation_loss",
        "best_epoch",
        "patience_counter",
        "amp_scaler_state",
        "rng_state",
        "compatibility",
        "history",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Incomplete v1.4 resume checkpoint: {missing}")
    if "scheduler_state" in checkpoint:
        raise ValueError("v1.4 resume checkpoint must not contain scheduler state")
    if checkpoint["compatibility"] != expected_compatibility:
        raise ValueError("Incompatible v1.4 resume checkpoint")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        if group["lr"] != 5e-6 or group["weight_decay"] != 1e-4:
            raise ValueError("Resumed AdamW state violates the frozen fixed-LR contract")
    if scaler.is_enabled():
        scaler.load_state_dict(checkpoint["amp_scaler_state"])
    restore_rng_state(checkpoint["rng_state"])
    early_stopping = StrictEarlyStopping(
        patience=10,
        min_delta=0.0,
        best_validation_total_loss=float(checkpoint["best_validation_loss"]),
        best_epoch=int(checkpoint["best_epoch"]),
        patience_counter=int(checkpoint["patience_counter"]),
    )
    return int(checkpoint["epoch"]), early_stopping, list(checkpoint["history"])


def smoke_test_resume_roundtrip() -> dict[str, Any]:
    """Exercise an actual save/reload roundtrip without scientific training."""

    set_reproducibility_seed(42)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    scaler = make_amp_scaler(torch.device("cpu"))
    early_stopping = StrictEarlyStopping(
        best_validation_total_loss=1.25, best_epoch=2, patience_counter=3
    )
    compatibility = {"smoke": "v1.4", "scheduler": None}
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with tempfile.TemporaryDirectory(prefix="mepi-v1-4-resume-") as directory:
        path = Path(directory) / "last_checkpoint.pt"
        save_resume_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=4,
            early_stopping=early_stopping,
            compatibility=compatibility,
            history=[{"epoch": 3, "validation_total_loss": 1.25}],
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        epoch, restored_stop, history = load_resume_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_compatibility=compatibility,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
        raise AssertionError("Model state changed across resume roundtrip")
    if optimizer.param_groups[0]["lr"] != 5e-6:
        raise AssertionError("Optimizer learning rate changed across resume roundtrip")
    return {
        "status": "PASS",
        "epoch": epoch,
        "best_epoch": restored_stop.best_epoch,
        "patience_counter": restored_stop.patience_counter,
        "history_rows": len(history),
        "scheduler_state_present": "scheduler_state" in payload,
    }


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _accumulate(
    totals: dict[str, float], losses: dict[str, torch.Tensor], batch_size: int
) -> None:
    for key, value in losses.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size


def _finish_totals(totals: dict[str, float], sample_count: int) -> dict[str, float]:
    return {key: value / sample_count for key, value in totals.items()}


def _forward_losses(
    model: DownstreamMEPIV14,
    batch: dict[str, torch.Tensor],
    prior: dict[str, Any],
) -> dict[str, torch.Tensor]:
    p_st_z = steinmetz_reference_z(batch["frequency_hz"], batch["B_peak_t"], prior)
    outputs = model(batch["waveform"], batch["tabular"], p_st_z)
    return compute_v1_4_losses(outputs, batch)


def train_one_epoch(
    model: DownstreamMEPIV14,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    prior: dict[str, Any],
    device: torch.device,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            losses = _forward_losses(model, batch, prior)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        size = int(batch["LSP_z"].shape[0])
        _accumulate(totals, losses, size)
        sample_count += size
    return _finish_totals(totals, sample_count)


@torch.no_grad()
def validate_one_epoch(
    model: DownstreamMEPIV14,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    prior: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            losses = _forward_losses(model, batch, prior)
        size = int(batch["LSP_z"].shape[0])
        _accumulate(totals, losses, size)
        sample_count += size
    return _finish_totals(totals, sample_count)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def run_v1_4_baseline(
    config_path: str | Path,
    run_directory: str | Path,
    *,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Run the frozen baseline on train/validation only when explicitly called."""

    config = load_config(config_path)
    validate_v1_4_config(config)
    if config.get("allow_training") is not True:
        raise RuntimeError("v1.4 configuration does not authorize training")
    set_reproducibility_seed(int(config["seed"]))
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    completed_path = run_path / "completed.json"
    if completed_path.exists() and not force_retrain:
        return _load_json(completed_path)
    last_path = run_path / "last_checkpoint.pt"
    best_path = run_path / "best_checkpoint.pt"
    if force_retrain:
        for path in (last_path, best_path, completed_path, run_path / "history.json"):
            path.unlink(missing_ok=True)

    train_dataset, validation_dataset = build_development_datasets(config_path)
    validation_loader = build_validation_loader(
        validation_dataset, batch_size=int(config["training"]["batch_size"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = construct_v1_4_model(config_path)
    model.to(device)
    optimizer = make_optimizer(model, config)
    scaler = make_amp_scaler(device)
    prior = load_steinmetz_prior(config)
    compatibility = resume_compatibility(config)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    early_stopping = StrictEarlyStopping(
        patience=int(config["training"]["early_stopping_patience"]),
        min_delta=float(config["training"]["min_delta"]),
    )
    if last_path.exists() and not force_retrain:
        start_epoch, early_stopping, history = load_resume_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_compatibility=compatibility,
        )

    stopped_early = False
    for epoch in range(start_epoch, int(config["training"]["max_epochs"])):
        train_loader = build_epoch_train_loader(
            train_dataset,
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["seed"]),
            epoch=epoch,
        )
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            prior=prior,
            device=device,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        validation_losses = validate_one_epoch(
            model, validation_loader, prior=prior, device=device
        )
        improved, should_stop = early_stopping.update(validation_losses["total"], epoch)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_losses,
            "validation": validation_losses,
            "strict_improvement": improved,
            "patience_counter": early_stopping.patience_counter,
        }
        history.append(row)
        if improved:
            save_resume_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                early_stopping=early_stopping,
                compatibility=compatibility,
                history=history,
            )
        save_resume_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            early_stopping=early_stopping,
            compatibility=compatibility,
            history=history,
        )
        _atomic_json(run_path / "history.json", {"epochs": history})
        if should_stop:
            stopped_early = True
            break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    result = {
        "status": "TRAINING_COMPLETE_VALIDATION_ONLY",
        "best_epoch": early_stopping.best_epoch,
        "best_validation_total_loss": early_stopping.best_validation_total_loss,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "scheduler": None,
        "test_accessed": False,
        "test_dataset_created": False,
        "test_loader_created": False,
    }
    _atomic_json(completed_path, result)
    return result
