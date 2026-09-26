"""Named checkpoint format and restricted transfer utility."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def make_pretraining_checkpoint(
    model: nn.Module,
    *,
    preprocessing_metadata: dict[str, Any],
    config: dict[str, Any],
    git_commit: str | None,
    random_seed: int,
    material_names: list[str],
) -> dict[str, Any]:
    return {
        "format_version": "mepi-pretraining-v1",
        "waveform_encoder": model.waveform_encoder.state_dict(),
        "operating_encoder_pretrain": model.operating_encoder_pretrain.state_dict(),
        "fusion": model.fusion.state_dict(),
        "backbone": model.backbone.state_dict(),
        "temporary_heads": model.temporary_heads.state_dict(),
        "preprocessing_metadata": preprocessing_metadata,
        "config": config,
        "git_commit": git_commit,
        "random_seed": int(random_seed),
        "material_names": list(material_names),
    }


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Atomically replace a checkpoint after a complete write and fsync.

    The temporary file deliberately does not end in ``.pt`` so an interrupted
    write can never be mistaken for a usable checkpoint.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_transfer_weights(
    downstream_model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Load exactly the waveform encoder and backbone, reporting all new layers."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != "mepi-pretraining-v1":
        raise ValueError("Checkpoint is not a named MEPI v1 pretraining checkpoint")
    required = {"waveform_encoder", "backbone"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint lacks transferable components: {missing}")
    downstream_model.waveform_encoder.load_state_dict(checkpoint["waveform_encoder"], strict=True)
    downstream_model.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    newly_initialized = [
        name
        for name in (
            "tabular_encoder_finetune",
            "fusion",
            "pirl",
            "p_loss_aux_z",
            "mtph",
        )
        if hasattr(downstream_model, name)
    ]
    return {
        "transferred": ["waveform_encoder", "backbone"],
        "newly_initialized": newly_initialized,
        "not_transferred": [
            "operating_encoder_pretrain",
            "temporary_heads",
            "pretraining_fusion",
        ],
        "checkpoint": str(Path(checkpoint_path)),
    }


def _state_shapes(state: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in sorted(state.items())}


def audit_xlstm_transfer_checkpoint(
    project_root: str | Path, depth: int
) -> dict[str, Any]:
    """Audit one completed depth checkpoint without fine-tuning any model."""

    from .models import DownstreamMEPI

    allowed_depths = (2, 4, 6, 8, 10)
    if depth not in allowed_depths:
        raise ValueError(f"depth must be one of {allowed_depths}")
    root = Path(project_root).resolve()
    checkpoint_path = root / f"experiments/xlstm_depth_v1/depth_{depth}/best_checkpoint.pt"
    summary_path = checkpoint_path.with_name("training_summary.json")
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Completed depth-{depth} evidence is missing")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or summary.get("depth") != depth:
        raise ValueError(f"Depth-{depth} summary is not COMPLETE and compatible")
    if digest != summary.get("best_checkpoint_sha256"):
        raise ValueError(f"Depth-{depth} checkpoint SHA256 does not match its summary")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != "mepi-pretraining-v1":
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("compatibility", {}).get("backbone") != "xLSTM":
        raise ValueError("Checkpoint is not an xLSTM checkpoint")

    model = DownstreamMEPI("xLSTM", latent_dim=256, backbone_layers=depth)
    protected_before = {
        name: {key: value.detach().clone() for key, value in module.state_dict().items()}
        for name, module in {
            "tabular_encoder_finetune": model.tabular_encoder_finetune,
            "fusion": model.fusion,
            "pirl": model.pirl,
            "mtph": model.mtph,
        }.items()
    }
    transfer = load_transfer_weights(model, checkpoint_path)
    for name, before in protected_before.items():
        after = getattr(model, name).state_dict()
        if before.keys() != after.keys() or any(
            not torch.equal(before[key], after[key]) for key in before
        ):
            raise AssertionError(f"New downstream module changed during transfer: {name}")
    if model.tabular_encoder_finetune.input_dim != 9:
        raise AssertionError("Downstream tabular encoder is not the new 9-feature encoder")
    if len(model.backbone.blocks) != depth:
        raise AssertionError("Downstream xLSTM depth does not match its checkpoint")

    return {
        "status": "PASS",
        "depth": depth,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "waveform_encoder_keys": sorted(checkpoint["waveform_encoder"]),
        "backbone_keys": sorted(checkpoint["backbone"]),
        "temporary_material_head_keys": sorted(checkpoint.get("temporary_heads", {})),
        "magnet_operating_encoder_keys": sorted(checkpoint.get("operating_encoder_pretrain", {})),
        "transferred_shapes": {
            "waveform_encoder": _state_shapes(checkpoint["waveform_encoder"]),
            "backbone": _state_shapes(checkpoint["backbone"]),
        },
        "discarded_shapes": {
            "operating_encoder_pretrain": _state_shapes(
                checkpoint.get("operating_encoder_pretrain", {})
            ),
            "temporary_heads": _state_shapes(checkpoint.get("temporary_heads", {})),
            "pretraining_fusion": _state_shapes(checkpoint.get("fusion", {})),
        },
        "transfer": transfer,
        "downstream_tabular_encoder": "NEWLY_INITIALIZED_9_FEATURE",
    }
