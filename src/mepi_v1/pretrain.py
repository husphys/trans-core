"""Shared candidate-backbone MagNet training runner."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpointing import make_pretraining_checkpoint, save_checkpoint
from .config import load_pretraining_config, project_root, require_training_enabled, resolve_path
from .constants import CANDIDATE_BACKBONES
from .data import MagNetDataset
from .metrics import regression_metrics
from .models import PretrainingModel
from .reproducibility import runtime_metadata, sha256_file
from .scaling import CoreLossTransform


def _seed_everything(seed: int, runtime: dict[str, Any] | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    settings = runtime or {}
    torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(settings.get("cudnn_deterministic", True))
    torch.use_deterministic_algorithms(
        bool(settings.get("deterministic_algorithms", True)), warn_only=True
    )
    allow_tf32 = bool(settings.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision(str(settings.get("float32_matmul_precision", "highest")))


def _loader_kwargs(training: dict[str, Any]) -> dict[str, Any]:
    workers = int(training["num_workers"])
    kwargs: dict[str, Any] = {
        "batch_size": int(training["batch_size"]),
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    return kwargs


def _read_indices(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return np.asarray([int(row["sample_index"]) for row in csv.DictReader(handle)], dtype=np.int64)


def _fit_preprocessing(h5_path: Path, train_indices: np.ndarray) -> tuple[dict[str, Any], CoreLossTransform]:
    import h5py

    with h5py.File(h5_path, "r") as handle:
        frequency = np.asarray(handle["f"][train_indices], dtype=np.float64).reshape(-1)
        temperature = np.asarray(handle["T"][train_indices], dtype=np.float64).reshape(-1)
        loss = np.asarray(handle["P"][train_indices], dtype=np.float64).reshape(-1)
        waveform_sum = waveform_squared_sum = 0.0
        waveform_count = 0
        for start in range(0, len(train_indices), 2048):
            block = np.asarray(handle["B"][train_indices[start : start + 2048]], dtype=np.float64)
            waveform_sum += float(block.sum())
            waveform_squared_sum += float(np.square(block).sum())
            waveform_count += block.size
    waveform_mean = waveform_sum / waveform_count
    waveform_variance = max(waveform_squared_sum / waveform_count - waveform_mean**2, 0.0)
    waveform_scale = max(waveform_variance**0.5, 1e-12)
    operating = np.column_stack((frequency, temperature))
    operating_mean = operating.mean(axis=0)
    operating_scale = operating.std(axis=0)
    operating_scale[operating_scale == 0.0] = 1.0
    target = CoreLossTransform().fit(loss, split="train")
    metadata = {
        "fitted_split": "train",
        "train_sample_count": int(len(train_indices)),
        "waveform": {"mean": waveform_mean, "scale": waveform_scale},
        "operating": {"mean": operating_mean.tolist(), "scale": operating_scale.tolist()},
        "target": target.metadata(),
        "training_loss": "MSE + 0.3 * MAE in normalized log1p core-loss space",
        "evaluation": ["normalized_mae", "rmse", "r2"],
    }
    return metadata, target


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["waveform"].to(device, non_blocking=non_blocking),
                batch["tabular"].to(device, non_blocking=non_blocking),
                batch["material_index"].to(device, non_blocking=non_blocking),
            )
            target = batch["target"].to(device, non_blocking=non_blocking)
            loss = F.mse_loss(output, target) + 0.3 * F.l1_loss(output, target)
            losses.append(float(loss))
            truth.append(target.cpu().numpy())
            prediction.append(output.cpu().numpy())
    metrics = regression_metrics(np.concatenate(truth), np.concatenate(prediction))
    return float(np.mean(losses)), metrics


def _update_comparison(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for backbone in CANDIDATE_BACKBONES:
        summary_path = root / "experiments" / "pretrain" / backbone / "training_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "COMPLETED_MEASURED_RERUN":
                rows.append(
                    {
                        "backbone": backbone,
                        "validation_loss": summary["best_validation_loss"],
                        "normalized_mae": summary["best_validation_metrics"]["normalized_mae"],
                        "rmse": summary["best_validation_metrics"]["rmse"],
                        "r2": summary["best_validation_metrics"]["r2"],
                        "parameter_count": summary["parameter_count"],
                        "dataset_sha256": summary["dataset_sha256"],
                    }
                )
    destination = root / "reports" / "pretraining_backbone_comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backbone",
        "validation_loss",
        "normalized_mae",
        "rmse",
        "r2",
        "parameter_count",
        "dataset_sha256",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manuscript = root / "artifacts" / "manuscript" / "pretraining_backbone_comparison.csv"
    manuscript.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination, manuscript)


def run_pretraining(config_path: str | Path) -> dict[str, Any]:
    config = load_pretraining_config(config_path)
    require_training_enabled(config)
    root = project_root(config)
    seed = int(config["seed"])
    _seed_everything(seed, config.get("runtime"))
    backbone_name = str(config["backbone"])
    if backbone_name not in CANDIDATE_BACKBONES:
        raise ValueError(f"backbone must be one of {CANDIDATE_BACKBONES}")
    h5_path = resolve_path(config, config["dataset"]["path"])
    split_dir = resolve_path(config, config["dataset"]["split_dir"])
    train_manifest = split_dir / "magnet_train.csv"
    validation_manifest = split_dir / "magnet_validation.csv"
    if not train_manifest.exists() or not validation_manifest.exists():
        raise FileNotFoundError("Run the MagNet audit before pretraining")
    # The test manifest is intentionally not read or instantiated in this runner.
    train_indices = _read_indices(train_manifest)
    validation_indices = _read_indices(validation_manifest)
    preprocessing, target_transform = _fit_preprocessing(h5_path, train_indices)

    import h5py

    with h5py.File(h5_path, "r") as handle:
        material_names = sorted(
            {
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["material"][train_indices]
            }
        )
    material_to_index = {name: index for index, name in enumerate(material_names)}
    dataset_arguments = {
        "material_to_index": material_to_index,
        "waveform_mean": preprocessing["waveform"]["mean"],
        "waveform_scale": preprocessing["waveform"]["scale"],
        "operating_mean": preprocessing["operating"]["mean"],
        "operating_scale": preprocessing["operating"]["scale"],
        "target_transform": target_transform,
    }
    cache_mode = str(config["training"].get("cache_mode", "hdf5"))
    train_dataset = MagNetDataset(h5_path, train_indices, cache_mode=cache_mode, **dataset_arguments)
    validation_dataset = MagNetDataset(h5_path, validation_indices, cache_mode=cache_mode, **dataset_arguments)
    generator = torch.Generator().manual_seed(seed)
    loader_arguments = _loader_kwargs(config["training"])
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_arguments)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_arguments)
    device_name = config["training"].get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    )
    model = PretrainingModel(
        backbone_name,
        material_count=len(material_names),
        latent_dim=int(config["model"]["latent_dim"]),
        backbone_layers=int(config["model"]["backbone_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    experiment = root / "experiments" / "pretrain" / backbone_name
    experiment.mkdir(parents=True, exist_ok=True)
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    (experiment / "config.yaml").write_text(
        yaml.safe_dump(clean_config, sort_keys=False), encoding="utf-8"
    )
    metrics_path = experiment / "metrics.csv"
    fields = ["epoch", "training_loss", "validation_loss", "normalized_mae", "rmse", "r2"]
    best_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            training_losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    non_blocking = bool(loader_arguments["pin_memory"])
                    output = model(
                        batch["waveform"].to(device, non_blocking=non_blocking),
                        batch["tabular"].to(device, non_blocking=non_blocking),
                        batch["material_index"].to(device, non_blocking=non_blocking),
                    )
                    target = batch["target"].to(device, non_blocking=non_blocking)
                    loss = F.mse_loss(output, target) + 0.3 * F.l1_loss(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                training_losses.append(float(loss.detach()))
            validation_loss, validation_metrics = _evaluate(
                model,
                validation_loader,
                device,
                non_blocking=bool(loader_arguments["pin_memory"] and device.type == "cuda"),
            )
            row = {
                "epoch": epoch,
                "training_loss": float(np.mean(training_losses)),
                "validation_loss": validation_loss,
                **validation_metrics,
            }
            writer.writerow(row)
            metrics_file.flush()
            if validation_metrics["normalized_mae"] < best_mae:
                best_mae = validation_metrics["normalized_mae"]
                best_payload = row
                checkpoint = make_pretraining_checkpoint(
                    model,
                    preprocessing_metadata=preprocessing,
                    config=clean_config,
                    git_commit=runtime_metadata(root)["git_commit"],
                    random_seed=seed,
                    material_names=material_names,
                )
                save_checkpoint(checkpoint, experiment / "best_checkpoint.pt")
    assert best_payload is not None
    summary = {
        "status": "COMPLETED_MEASURED_RERUN",
        "backbone": backbone_name,
        "best_epoch": best_payload["epoch"],
        "best_validation_loss": best_payload["validation_loss"],
        "best_validation_metrics": {
            name: best_payload[name] for name in ("normalized_mae", "rmse", "r2")
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dataset_sha256": sha256_file(h5_path),
        "config_file": str(experiment / "config.yaml"),
        "dependency_lockfile": {
            "path": str(root / "requirements-lock.txt"),
            "sha256": sha256_file(root / "requirements-lock.txt"),
        },
        "sample_counts": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": "NOT_ACCESSED",
        },
        "test_split_accessed": False,
        "runtime": runtime_metadata(root),
        "preprocessing": preprocessing,
    }
    (experiment / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _update_comparison(root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_pretraining(arguments.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
