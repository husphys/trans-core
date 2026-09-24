"""Read-only v1.3 fine-tuning notebook support.

This module deliberately implements the audited preparation path only.  The
repository does not yet contain a complete authoritative downstream optimizer
and loss contract, so :func:`require_scientific_training_ready` fails closed
instead of inventing one.  No function here constructs a test dataset/loader
or indexes test waveforms and targets for model use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .checkpointing import load_transfer_weights
from .config import load_config, resolve_path
from .constants import (
    FINETUNE_TABULAR_FEATURES,
    FORBIDDEN_FINETUNE_INPUTS,
    WAVEFORM_LENGTH,
    validate_finetune_feature_names,
)
from .models import DownstreamMEPI

PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.3"
PROTOCOL_SHA256 = "99b31c82a3a92b6d5446477cbb9a6ae5fe40afae0476a6533324b2b14898ccfa"
CHECKPOINT_SHA256 = "fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c"
EXPECTED_ROWS = {"train": 687, "validation": 85, "test": 90}
EXPECTED_GROUPS = {"train": 72, "validation": 9, "test": 9}
ALLOWED_MODEL_DEVELOPMENT_SPLITS = frozenset({"train", "validation"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def assert_model_development_split(split: str) -> None:
    if split == "test":
        raise RuntimeError("TEST ACCESS DENIED: v1.3 model development is train/validation only")
    if split not in ALLOWED_MODEL_DEVELOPMENT_SPLITS:
        raise ValueError(f"split must be train or validation, got {split!r}")


def scientific_training_blockers(config: dict[str, Any]) -> list[str]:
    """Return unresolved scientific contracts without supplying defaults."""

    training = config.get("training", {})
    required = (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
    )
    blockers = [f"training.{key} is not authoritatively configured" for key in required if training.get(key) is None]
    weights = training.get("loss_weights", {})
    for key in ("efficiency", "P_loss", "LSP", "physics", "uncertainty_nll"):
        if weights.get(key) is None:
            blockers.append(f"training.loss_weights.{key} is not authoritatively configured")
    blockers.extend(
        [
            "no authoritative downstream waveform preprocessing is configured",
            "no v1.3 implementation constructs the two PIRL residual inputs without target leakage",
            "no shared downstream fine-tuning loss/evaluation loop is implemented",
        ]
    )
    return blockers


def require_scientific_training_ready(config: dict[str, Any]) -> None:
    blockers = scientific_training_blockers(config)
    if blockers:
        formatted = "\n- ".join(blockers)
        raise RuntimeError(
            "SCIENTIFIC_TRAINING_BLOCKED: MEPI v1.3 is not execution-ready. "
            "Do not infer settings from legacy BiGRU code.\n- " + formatted
        )


def audit_frozen_evidence(project_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(config_path)
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise AssertionError("Configuration is not MEPI-FROZEN-PROTOCOL v1.3")
    protocol_paths = [root / "MEPI-FROZEN-PROTOCOL v1.3.md", root / "docs/MEPI-FROZEN-PROTOCOL v1.3.md"]
    protocol_hashes = [sha256_file(path) for path in protocol_paths]
    if protocol_hashes != [PROTOCOL_SHA256, PROTOCOL_SHA256]:
        raise AssertionError(f"v1.3 protocol hash mismatch: {protocol_hashes}")
    checkpoint = resolve_path(config, config["pretrained_checkpoint"])
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise AssertionError(f"pretrained checkpoint hash mismatch: {checkpoint_hash}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_copies": [str(path) for path in protocol_paths],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "status": "PASS",
    }


def audit_dataset_metadata(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = config["dataset"]
    rows = _read_csv(resolve_path(config, dataset["path"]))
    waveforms = np.load(resolve_path(config, dataset["waveform_path"]), mmap_mode="r", allow_pickle=False)
    identifiers = np.load(resolve_path(config, dataset["sample_ids_path"]), mmap_mode="r", allow_pickle=False)
    split_rows = Counter(row["split"] for row in rows)
    split_groups = {name: {row["condition_group_id"] for row in rows if row["split"] == name} for name in EXPECTED_ROWS}
    split_manifest = json.loads(resolve_path(config, "data/MEPI/v1_2/split_manifest_v1_2.json").read_text())
    master = _read_csv(resolve_path(config, "data/MEPI/v1_2/samples_master_v1_2.csv"))
    primary_rows = sum(row["condition_role"] == "primary_paired" for row in master)
    if primary_rows != 900 or len(rows) != 862:
        raise AssertionError(f"unexpected row counts: primary={primary_rows}, qc_valid={len(rows)}")
    if tuple(waveforms.shape) != (862, WAVEFORM_LENGTH) or len(identifiers) != 862:
        raise AssertionError("frozen waveform/sample-ID arrays are not aligned at (862, 1024)")
    if dict(split_rows) != EXPECTED_ROWS:
        raise AssertionError(f"unexpected split row counts: {dict(split_rows)}")
    if split_manifest["group_counts"] != EXPECTED_GROUPS:
        raise AssertionError(f"unexpected split group counts: {split_manifest['group_counts']}")
    # One fully QC-invalid measured group is retained in the JSON manifest but has no candidate rows.
    if {name: len(groups) for name, groups in split_groups.items()} != {
        "train": 71,
        "validation": 9,
        "test": 9,
    }:
        raise AssertionError("candidate-row group assignments differ from frozen v1.2 evidence")
    return {
        "primary_measured_rows": primary_rows,
        "qc_valid_rows": len(rows),
        "B_shape": list(waveforms.shape),
        "group_counts": EXPECTED_GROUPS,
        "row_counts": EXPECTED_ROWS,
        "test_dataloader_created": False,
        "status": "PASS",
    }


def audit_leakage(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    schema = json.loads(resolve_path(config, config["dataset"]["feature_schema"]).read_text())
    features = tuple(schema["tabular_features"])
    validate_finetune_feature_names(features)
    required_denied = {
        "core_id",
        "temperature_core_c",
        "T_core_K",
        "LSP_raw",
        "efficiency_percent",
        "P_loss",
    }
    denied = set(schema["forbidden_predictive_inputs"]) | set(FORBIDDEN_FINETUNE_INPUTS)
    if not required_denied.issubset(denied):
        raise AssertionError("frozen leakage denylist is incomplete")
    return {"features": list(features), "forbidden": sorted(denied), "status": "PASS"}


def audit_normalization(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    metadata = json.loads(resolve_path(config, config["dataset"]["normalization"]).read_text())
    expected = (0.6030550333971313, 0.08375144701048931, 0)
    actual = (metadata["mu_LSP_train"], metadata["sigma_LSP_train"], metadata["numpy_ddof"])
    if actual != expected or metadata["fitted_split"] != "train" or metadata["validation_refit"] is not False:
        raise AssertionError(f"train-only normalization mismatch: {actual}")
    return {
        "LSP_mean": actual[0],
        "LSP_std": actual[1],
        "ddof": actual[2],
        "feature_scalers": metadata["feature_scalers"],
        "validation_uses_train_statistics": True,
        "status": "PASS",
    }


class FrozenV13Dataset(Dataset[dict[str, torch.Tensor]]):
    """Train/validation-only view over frozen v1.2 arrays."""

    def __init__(self, config_path: str | Path, split: str) -> None:
        assert_model_development_split(split)
        self.config = load_config(config_path)
        dataset = self.config["dataset"]
        all_rows = _read_csv(resolve_path(self.config, dataset["path"]))
        all_ids = np.load(resolve_path(self.config, dataset["sample_ids_path"]), mmap_mode="r", allow_pickle=False)
        id_to_index = {str(sample_id): index for index, sample_id in enumerate(all_ids)}
        self.rows = [row for row in all_rows if row["split"] == split]
        self.indices = [id_to_index[row["sample_id"]] for row in self.rows]
        self.waveforms = np.load(resolve_path(self.config, dataset["waveform_path"]), mmap_mode="r", allow_pickle=False)
        self.normalization = json.loads(resolve_path(self.config, dataset["normalization"]).read_text())
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        feature_values = []
        for name in FINETUNE_TABULAR_FEATURES:
            scaler = self.normalization["feature_scalers"][name]
            feature_values.append((float(row[name]) - float(scaler["mean"][0])) / float(scaler["scale"][0]))
        return {
            "waveform_raw": torch.from_numpy(np.array(self.waveforms[self.indices[index]], dtype=np.float32, copy=True)).unsqueeze(0),
            "tabular": torch.tensor(feature_values, dtype=torch.float32),
            "efficiency_percent": torch.tensor(float(row["efficiency_percent"]), dtype=torch.float32),
            "P_loss": torch.tensor(float(row["P_loss"]), dtype=torch.float32),
            "LSP_raw": torch.tensor(float(row["LSP_raw"]), dtype=torch.float32),
            "LSP_z": torch.tensor(float(row["LSP_z"]), dtype=torch.float32),
        }


def build_model_development_loaders(
    config_path: str | Path, *, audit_batch_size: int = 8
) -> tuple[FrozenV13Dataset, FrozenV13Dataset, DataLoader, DataLoader]:
    """Construct only train/validation loaders for a non-training software audit."""

    train = FrozenV13Dataset(config_path, "train")
    validation = FrozenV13Dataset(config_path, "validation")
    train_loader = DataLoader(train, batch_size=audit_batch_size, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=audit_batch_size, shuffle=False, num_workers=0)
    return train, validation, train_loader, validation_loader


def construct_v1_3_model(config_path: str | Path) -> tuple[DownstreamMEPI, dict[str, Any]]:
    config = load_config(config_path)
    identity = (config.get("backbone"), config.get("backbone_depth"), config.get("latent_dim"))
    if identity != ("xLSTM", 8, 256):
        raise AssertionError(f"unexpected v1.3 architecture: {identity}")
    model = DownstreamMEPI(identity[0], backbone_layers=identity[1], latent_dim=identity[2])
    transfer = load_transfer_weights(model, resolve_path(config, config["pretrained_checkpoint"]))
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "waveform_encoder": sum(parameter.numel() for parameter in model.waveform_encoder.parameters()),
        "backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "downstream_new": sum(
            parameter.numel()
            for module in (model.tabular_encoder_finetune, model.fusion, model.pirl, model.mtph)
            for parameter in module.parameters()
        ),
    }
    return model, {"transfer": transfer, "parameter_counts": counts, "status": "PASS"}


def set_reproducibility_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass
    return {
        "python": seed,
        "numpy": seed,
        "torch_cpu": seed,
        "torch_cuda": seed if torch.cuda.is_available() else "CUDA_UNAVAILABLE",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
