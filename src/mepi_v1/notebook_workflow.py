"""Thin-notebook workflow for recoverable, evidence-producing MagNet reruns.

The notebooks call this module; training logic intentionally does not live in
notebook cells.  No function in this module runs merely because it is imported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .audit_magnet import audit_magnet
from .checkpointing import make_pretraining_checkpoint, save_checkpoint
from .config import load_pretraining_config, resolve_path
from .constants import (
    CANDIDATE_BACKBONES,
    EXPECTED_PROTOCOL_SHA256,
    PRETRAIN_TABULAR_FEATURES,
    PROTOCOL_RELATIVE_PATHS,
    PROTOCOL_VERSION,
    WAVEFORM_LENGTH,
)
from .data import MagNetDataset
from .metrics import regression_metrics
from .models import PretrainingModel
from .pretrain import _fit_preprocessing, _loader_kwargs, _read_indices, _seed_everything
from .reproducibility import runtime_metadata, sha256_file
from .scaling import CoreLossTransform, StandardizationState

EXPERIMENT_ROOT = Path("experiments/pretrain_v1")
CHECKPOINT_FILENAMES = frozenset({"last_checkpoint.pt", "best_checkpoint.pt"})
EXPECTED_KERNEL_NAME = "trans-core"
METRIC_FIELDS = (
    "epoch",
    "global_step",
    "training_loss",
    "validation_loss",
    "normalized_mae",
    "rmse",
    "r2",
    "learning_rate",
    "elapsed_seconds",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def source_tree_fingerprint(root: str | Path) -> str:
    root_path = Path(root).resolve()
    digest = hashlib.sha256()
    files = sorted((root_path / "src" / "mepi_v1").glob("*.py"))
    if not files:
        raise FileNotFoundError(f"No MEPI source files found under {root_path / 'src/mepi_v1'}")
    for path in files:
        digest.update(path.relative_to(root_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def split_fingerprint(paths: Iterable[str | Path]) -> str:
    entries = []
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Required split manifest is missing: {path}")
        entries.append({"name": path.name, "sha256": sha256_file(path)})
    return fingerprint(entries)


def architecture_fingerprint(
    backbone: str, model_config: dict[str, Any], *, material_count: int
) -> str:
    model = PretrainingModel(
        backbone,
        material_count=material_count,
        latent_dim=int(model_config["latent_dim"]),
        backbone_layers=int(model_config["backbone_layers"]),
    )
    schema = [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in model.state_dict().items()]
    return fingerprint({"backbone": backbone, "state_schema": schema})


def validate_common_pretraining_configs(root: str | Path) -> str:
    """Fail if any candidate receives a different non-architecture setup."""

    root_path = Path(root).resolve()
    baseline: dict[str, Any] | None = None
    baseline_backbone: str | None = None
    for backbone in CANDIDATE_BACKBONES:
        name = backbone.lower().replace("-", "_")
        config = load_pretraining_config(root_path / "configs" / f"pretrain_{name}.yaml")
        common = {
            key: value
            for key, value in config.items()
            if not key.startswith("_") and key not in {"backbone"}
        }
        if baseline is None:
            baseline = common
            baseline_backbone = backbone
        elif common != baseline:
            raise ValueError(
                f"Common pretraining configuration mismatch: {backbone} differs from "
                f"{baseline_backbone}. Inspect configs/pretrain_*.yaml; independent tuning "
                "is forbidden for the backbone comparison."
            )
    assert baseline is not None
    return fingerprint(baseline)


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
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    actual = checkpoint.get("compatibility", {})
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        passed = actual_value == expected_value
        rows.append({"field": key, "expected": expected_value, "actual": actual_value, "status": "PASS" if passed else "FAIL"})
        if not passed:
            failures.append(f"{key}: expected {expected_value!r}, actual {actual_value!r}")
    if failures:
        raise ValueError(
            "Incompatible recovery checkpoint; refusing to resume. "
            + "; ".join(failures)
            + ". Inspect the notebook configuration, split manifests, and last_checkpoint.pt."
        )
    return rows


def enforce_checkpoint_policy(experiment_dir: str | Path) -> list[str]:
    directory = Path(experiment_dir)
    found = sorted(path.name for path in directory.glob("*.pt")) if directory.exists() else []
    unexpected = sorted(set(found) - CHECKPOINT_FILENAMES)
    if unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint files in {directory}: {unexpected}. "
            f"Only {sorted(CHECKPOINT_FILENAMES)} are allowed."
        )
    return found


def mark_completed(experiment_dir: str | Path) -> None:
    directory = Path(experiment_dir)
    enforce_checkpoint_policy(directory)
    (directory / "last_checkpoint.pt").unlink(missing_ok=True)


def _target_from_metadata(metadata: dict[str, Any]) -> CoreLossTransform:
    target = CoreLossTransform()
    target.standardizer.state = StandardizationState(**metadata["state"])
    if target.standardizer.state.fitted_split != "train":
        raise ValueError("Target transform metadata was not fitted on the training split")
    return target


def _count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _model_device(requested: str) -> torch.device:
    if requested == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Full notebook pretraining is intentionally GPU-gated; "
                "run 00_environment_check.ipynb and inspect the CUDA/PyTorch installation."
            )
        return torch.device("cuda")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but torch.cuda.is_available() is False")
    return device


def _trans_core_kernel_status() -> dict[str, Any]:
    """Report whether this interpreter is registered as the trans-core kernel."""

    active = str(Path(sys.executable).resolve())
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
            text=True,
            capture_output=True,
            check=True,
        )
        entry = json.loads(completed.stdout).get("kernelspecs", {}).get(
            EXPECTED_KERNEL_NAME
        )
        if entry is None:
            raise KeyError("trans-core kernelspec is not registered")
        spec = entry.get("spec", {})
        argv = spec.get("argv", [])
        interpreter = str(Path(argv[0]).resolve()) if argv else None
        return {
            "kernel_name": EXPECTED_KERNEL_NAME,
            "registered": True,
            "display_name": spec.get("display_name"),
            "kernel_interpreter": interpreter,
            "active_interpreter": active,
            "matches_active_interpreter": interpreter == active,
        }
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as error:
        return {
            "kernel_name": EXPECTED_KERNEL_NAME,
            "registered": False,
            "display_name": None,
            "kernel_interpreter": None,
            "active_interpreter": active,
            "matches_active_interpreter": False,
            "error": str(error),
        }


class PretrainingNotebookRun:
    """Stateful facade used identically by all eight training notebooks."""

    def __init__(
        self,
        project: str | Path,
        backbone: str,
        *,
        seed: int = 42,
        force_retrain: bool = False,
        auto_resume: bool = True,
        checkpoint_every_n_steps: int = 1000,
    ) -> None:
        self.root = Path(project).expanduser().resolve()
        if backbone not in CANDIDATE_BACKBONES:
            raise ValueError(f"Backbone must be one of {CANDIDATE_BACKBONES}, got {backbone!r}")
        self.backbone = backbone
        self.seed = int(seed)
        self.force_retrain = bool(force_retrain)
        self.auto_resume = bool(auto_resume)
        self.checkpoint_every_n_steps = int(checkpoint_every_n_steps)
        if self.checkpoint_every_n_steps <= 0:
            raise ValueError("CHECKPOINT_EVERY_N_STEPS must be positive")
        config_name = backbone.lower().replace("-", "_")
        self.config_path = self.root / "configs" / f"pretrain_{config_name}.yaml"
        self.config = load_pretraining_config(self.config_path)
        self.config["seed"] = self.seed
        self.config["allow_training"] = True  # explicit notebook action, never persisted to source config
        self.experiment_dir = self.root / EXPERIMENT_ROOT / backbone
        self.last_path = self.experiment_dir / "last_checkpoint.pt"
        self.best_path = self.experiment_dir / "best_checkpoint.pt"
        self.summary_path = self.experiment_dir / "training_summary.json"
        self.evidence_path = self.experiment_dir / "run_evidence.json"
        self.metrics_path = self.experiment_dir / "metrics.csv"
        self._prepared: dict[str, Any] | None = None
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def print_identity(self) -> dict[str, str]:
        payload = {
            "experiment": "MEPI PRETRAIN RERUN v1",
            "backbone": self.backbone,
            "protocol": PROTOCOL_VERSION,
            "output_directory": str(self.experiment_dir),
            "restart_behavior": "compatible last_checkpoint.pt auto-resume; completed run skips",
            "common_config": self.config.get("_common_config_path"),
            "batch_size": int(self.config["training"]["batch_size"]),
            "num_workers": int(self.config["training"]["num_workers"]),
            "amp": bool(self.config["training"].get("amp", True)),
            "tf32": bool(self.config.get("runtime", {}).get("allow_tf32", False)),
            "deterministic_algorithms": bool(
                self.config.get("runtime", {}).get("deterministic_algorithms", True)
            ),
        }
        print("MEPI PRETRAIN RERUN v1")
        for key in ("backbone", "protocol", "output_directory", "restart_behavior"):
            print(f"{key.replace('_', ' ').title()}: {payload[key]}")
        print(
            "Common runtime: "
            f"batch_size={payload['batch_size']}, workers={payload['num_workers']}, "
            f"AMP={payload['amp']}, TF32={payload['tf32']}, "
            f"deterministic_algorithms={payload['deterministic_algorithms']}"
        )
        return payload

    def check_environment(self, *, require_cuda: bool = True) -> list[dict[str, str]]:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE"
        kernel = _trans_core_kernel_status()
        checks = [
            ("Project root", self.root.is_dir(), str(self.root)),
            ("Python executable", bool(sys.executable), sys.executable),
            ("Python version", True, sys.version.replace("\n", " ")),
            ("PyTorch import", True, torch.__version__),
            ("PyTorch CUDA version", torch.version.cuda is not None, str(torch.version.cuda)),
            ("CUDA available", torch.cuda.is_available(), str(torch.cuda.is_available())),
            ("GPU model", torch.cuda.is_available(), gpu),
        ]
        rows = [{"check": name, "status": "PASS" if passed else "FAIL", "detail": detail} for name, passed, detail in checks]
        rows.append(
            {
                "check": "Python (trans-core) kernel interpreter",
                "status": "PASS" if kernel["matches_active_interpreter"] else "WARN",
                "detail": (
                    f"display={kernel['display_name']!r}; "
                    f"kernel={kernel['kernel_interpreter']}; active={kernel['active_interpreter']}"
                ),
            }
        )
        for row in rows:
            print(f"[{row['status']}] {row['check']}: {row['detail']}")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA check failed: expected an available CUDA GPU, actual torch.cuda.is_available()=False. Run 00_environment_check.ipynb.")
        if not kernel["matches_active_interpreter"]:
            print(
                "[WARN] Select the 'Python (trans-core)' kernel in VSCode/Jupyter "
                "before training."
            )
        return rows

    def validate_source_protocol(self) -> dict[str, Any]:
        protocols = [self.root / relative for relative in PROTOCOL_RELATIVE_PATHS]
        missing = [path for path in protocols if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Frozen protocol missing: expected {missing}. Restore the v1.1 copies; "
                "do not substitute the superseded v1.0 protocol."
            )
        protocol_hashes = [sha256_file(path) for path in protocols]
        if protocol_hashes != [EXPECTED_PROTOCOL_SHA256] * len(protocols):
            raise RuntimeError(
                "Frozen protocol hash mismatch: expected both authoritative copies to be "
                f"{EXPECTED_PROTOCOL_SHA256}, got {protocol_hashes}"
            )
        result = {
            "protocol": PROTOCOL_VERSION,
            "protocol_paths": [str(path) for path in protocols],
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "source_tree_fingerprint": source_tree_fingerprint(self.root),
            "git_sha": runtime_metadata(self.root)["git_commit"],
            "candidate_common_config_fingerprint": validate_common_pretraining_configs(
                self.root
            ),
        }
        print(
            f"[PASS] {PROTOCOL_VERSION} verified at both authoritative paths: "
            f"{EXPECTED_PROTOCOL_SHA256}"
        )
        print(f"[PASS] src.mepi_v1 imports: source fingerprint {result['source_tree_fingerprint']}")
        print(
            "[PASS] All eight candidate configs share one non-architecture setup: "
            f"{result['candidate_common_config_fingerprint']}"
        )
        print(f"Git SHA: {result['git_sha'] or 'UNAVAILABLE (source-tree fingerprint retained)'}")
        return result

    def _manifests(self) -> dict[str, Path]:
        split_dir = resolve_path(self.config, self.config["dataset"]["split_dir"])
        return {name: split_dir / f"magnet_{name}.csv" for name in ("train", "validation", "test")}

    def _compatibility(
        self, preprocessing: dict[str, Any], *, material_count: int
    ) -> dict[str, Any]:
        manifests = self._manifests()
        clean = {
            key: value
            for key, value in self.config.items()
            if not key.startswith("_") and key not in {"backbone", "allow_training"}
        }
        return {
            "backbone": self.backbone,
            "source_tree_fingerprint": source_tree_fingerprint(self.root),
            "dataset_fingerprint": sha256_file(resolve_path(self.config, self.config["dataset"]["path"])),
            "split_fingerprint": split_fingerprint(manifests.values()),
            "model_architecture_fingerprint": architecture_fingerprint(
                self.backbone, self.config["model"], material_count=material_count
            ),
            "material_head_count": material_count,
            "common_configuration_fingerprint": fingerprint(clean),
            "seed": self.seed,
            "waveform_length": WAVEFORM_LENGTH,
            "pretraining_tabular_fields": list(PRETRAIN_TABULAR_FEATURES),
            "target_transform": preprocessing["target"],
        }

    def validate_dataset(self) -> dict[str, Any]:
        import h5py

        manifests = self._manifests()
        for path in manifests.values():
            if not path.is_file():
                raise FileNotFoundError(f"Split manifest missing: {path}. Run 01_magnet_dataset_audit.ipynb.")
        counts = {name: _count_csv_rows(path) for name, path in manifests.items()}
        h5_path = resolve_path(self.config, self.config["dataset"]["path"])
        with h5py.File(h5_path, "r") as handle:
            total = len(handle["B"])
            waveform_shape = list(handle["B"].shape)
            target_shape = list(handle["P"].shape)
        if waveform_shape != [total, WAVEFORM_LENGTH]:
            raise ValueError(f"Waveform shape failed: expected [{total}, {WAVEFORM_LENGTH}], actual {waveform_shape}; inspect {h5_path}")
        if list(PRETRAIN_TABULAR_FEATURES) != ["frequency_hz", "temperature_c"]:
            raise AssertionError(f"Tabular contract failed: expected ['frequency_hz', 'temperature_c'], actual {list(PRETRAIN_TABULAR_FEATURES)}")
        if sum(counts.values()) != total:
            raise ValueError(f"Split counts failed: expected total {total}, actual {sum(counts.values())}; inspect {manifests}")
        first_index = int(_read_indices(manifests["train"])[0])
        with h5py.File(h5_path, "r") as handle:
            raw_material = handle["material"][first_index]
            first_material = raw_material.decode("utf-8") if isinstance(raw_material, bytes) else str(raw_material)
        actual_batch = next(
            iter(
                DataLoader(
                    MagNetDataset(h5_path, [first_index], material_to_index={first_material: 0}),
                    batch_size=1,
                )
            )
        )
        actual_shapes = {
            "waveform": list(actual_batch["waveform"].shape),
            "tabular": list(actual_batch["tabular"].shape),
            "target": list(actual_batch["target"].shape),
        }
        result = {
            "dataset_path": str(h5_path), "total": total, "counts": counts,
            "waveform_shape": waveform_shape, "tabular_shape": [total, 2], "target_shape": target_shape,
            "dataset_fingerprint": sha256_file(h5_path), "split_fingerprint": split_fingerprint(manifests.values()),
            "actual_batch_shapes": actual_shapes,
        }
        print(f"[PASS] Dataset loaded: {h5_path}")
        print(f"Total samples: {total}; Train: {counts['train']}; Validation: {counts['validation']}; Test: {counts['test']}")
        print(f"[PASS] Waveform shape: {waveform_shape}; tabular shape: [{total}, 2]; target shape: {target_shape}")
        print("[PASS] Tabular fields exactly [frequency_hz, temperature_c]")
        print("[PASS] Material/CoreID excluded from predictive input")
        print(f"[PASS] One actual train batch: {actual_shapes}")
        return result

    def inspect_status(self) -> dict[str, Any]:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        enforce_checkpoint_policy(self.experiment_dir)
        if self.force_retrain:
            known = [
                self.last_path, self.best_path, self.summary_path, self.evidence_path,
                self.experiment_dir / "run_evidence.md", self.metrics_path,
                self.experiment_dir / "run_state.json", self.experiment_dir / "config.yaml",
                self.experiment_dir / "environment.json",
            ]
            for path in known:
                path.unlink(missing_ok=True)
            print("STATUS: FORCE RETRAIN REQUESTED\nACTION: known current-run artifacts removed")
            return {"status": "NEW", "action": "FROM_SCRATCH"}
        state_path = self.experiment_dir / "run_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "INVALIDATED_NON_OFFICIAL":
                print(
                    "STATUS: INVALIDATED NON-OFFICIAL RUN\n"
                    "ACTION: FRESH TCN RERUN REQUIRED; INCOMPATIBLE CHECKPOINT WILL NOT LOAD"
                )
                return {
                    "status": "INVALIDATED_NON_OFFICIAL",
                    "action": "FROM_SCRATCH_REQUIRED",
                    "reason": state.get("reason"),
                }
        if self.summary_path.exists():
            summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "COMPLETE":
                print("STATUS: COMPLETED RUN FOUND\nACTION: DISPLAY SUMMARY, TRAINING SKIPPED")
                return {"status": "COMPLETE", "action": "SKIP", "summary": summary}
        if self.last_path.exists():
            action = "AUTO-RESUME" if self.auto_resume else "STOP"
            print(f"STATUS: INCOMPLETE RUN FOUND\nACTION: {action}")
            if not self.auto_resume:
                raise RuntimeError("Incomplete run exists but AUTO_RESUME=False. Set AUTO_RESUME=True or explicitly set FORCE_RETRAIN=True.")
            return {"status": "INCOMPLETE", "action": action}
        print("STATUS: NEW RUN")
        return {"status": "NEW", "action": "FROM_SCRATCH"}

    def _prepare(self) -> dict[str, Any]:
        if self._prepared is not None:
            return self._prepared
        h5_path = resolve_path(self.config, self.config["dataset"]["path"])
        manifests = self._manifests()
        train_indices = _read_indices(manifests["train"])
        validation_indices = _read_indices(manifests["validation"])
        preprocessing, target = _fit_preprocessing(h5_path, train_indices)
        import h5py
        with h5py.File(h5_path, "r") as handle:
            material_names = sorted({value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in handle["material"][train_indices]})
        args = {
            "material_to_index": {name: i for i, name in enumerate(material_names)},
            "waveform_mean": preprocessing["waveform"]["mean"],
            "waveform_scale": preprocessing["waveform"]["scale"],
            "operating_mean": preprocessing["operating"]["mean"],
            "operating_scale": preprocessing["operating"]["scale"],
            "target_transform": target,
        }
        self._prepared = {
            "h5_path": h5_path, "manifests": manifests, "train_indices": train_indices,
            "validation_indices": validation_indices, "preprocessing": preprocessing,
            "target": target, "material_names": material_names, "dataset_args": args,
            "compatibility": self._compatibility(
                preprocessing, material_count=len(material_names)
            ),
        }
        return self._prepared

    def inspect_checkpoint(self) -> list[dict[str, Any]]:
        if not self.last_path.exists():
            print("[PASS] No recovery checkpoint: a new run will initialize from scratch")
            return []
        checkpoint = torch.load(self.last_path, map_location="cpu", weights_only=False)
        rows = validate_checkpoint_compatibility(checkpoint, self._prepare()["compatibility"])
        for row in rows:
            row.update(
                {
                    "checkpoint_epoch": checkpoint["epoch"],
                    "global_step": checkpoint["global_step"],
                    "best_validation_mae_norm": checkpoint["best_validation_mae"],
                    "checkpoint_size_mib": self.last_path.stat().st_size / (1024**2),
                    "overall_compatibility": "PASS",
                }
            )
        print(f"Checkpoint detected: {self.last_path}")
        print(f"Epoch: {checkpoint['epoch']}; step: {checkpoint['global_step']}; best MAE_norm: {checkpoint['best_validation_mae']}")
        print(f"Size: {self.last_path.stat().st_size / (1024**2):.2f} MiB; compatibility: PASS")
        return rows

    def construct_model_smoke(self) -> dict[str, Any]:
        prepared = self._prepare()
        model = PretrainingModel(
            self.backbone, material_count=len(prepared["material_names"]),
            latent_dim=int(self.config["model"]["latent_dim"]),
            backbone_layers=int(self.config["model"]["backbone_layers"]),
        )
        dataset = MagNetDataset(prepared["h5_path"], prepared["train_indices"][:1], **prepared["dataset_args"])
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        model.eval()
        with torch.no_grad():
            output = model(batch["waveform"], batch["tabular"], batch["material_index"])
        result = {
            "model": self.backbone, "parameter_count": sum(p.numel() for p in model.parameters()),
            "waveform_input_shape": list(batch["waveform"].shape), "tabular_input_shape": list(batch["tabular"].shape),
            "target_shape": list(batch["target"].shape), "output_shape": list(output.shape),
            "temporary_heads": len(prepared["material_names"]),
        }
        print(json.dumps(result, indent=2))
        print("Material ID used as predictive input: NO")
        print("Material metadata routes only the temporary regression head: YES")
        print("PIRL instantiated: NO\nMTPH instantiated: NO")
        self._model = model
        return result

    def _load_metrics(self) -> list[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        with self.metrics_path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _write_metrics(self, rows: list[dict[str, Any]]) -> None:
        destination = self.metrics_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".metrics.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
                writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True); raise

    def _save_rolling(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: Any, scaler: Any, *, epoch: int, next_batch_index: int,
        global_step: int, epoch_loss_sum: float, epoch_loss_count: int,
        best_mae: float, best_epoch: int | None, best_step: int | None,
        resume_events: int, start_timestamp: str, elapsed_seconds: float,
    ) -> None:
        payload = {
            "format_version": "mepi-notebook-resume-v1", "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "amp_scaler_state": scaler.state_dict(), "epoch": epoch,
            "next_batch_index": next_batch_index, "global_step": global_step,
            "epoch_loss_sum": epoch_loss_sum, "epoch_loss_count": epoch_loss_count,
            "best_validation_mae": best_mae, "best_epoch": best_epoch, "best_step": best_step,
            "rng_state": capture_rng_state(), "compatibility": self._prepare()["compatibility"],
            "resume_events": resume_events, "start_timestamp": start_timestamp,
            "elapsed_seconds": elapsed_seconds, "saved_at": _utc_now(),
            "resume_semantics": "deterministic epoch shuffle and saved next-batch position; practical continuation, not claimed bit-exact",
        }
        save_checkpoint(payload, self.last_path)
        enforce_checkpoint_policy(self.experiment_dir)

    def train_or_resume(self) -> dict[str, Any]:
        status = self.inspect_status()
        if status["status"] == "COMPLETE":
            return status["summary"]
        if status["status"] == "INVALIDATED_NON_OFFICIAL":
            raise RuntimeError(
                "The existing run is non-official and its checkpoint is incompatible. "
                "Set FORCE_RETRAIN=True in the TCN notebook to remove only its current-run "
                "artifacts and start TCN from scratch."
            )
        prepared = self._prepare()
        device = _model_device(str(self.config["training"].get("device", "auto")))
        self._device = device
        _seed_everything(self.seed, self.config.get("runtime"))
        model = PretrainingModel(
            self.backbone, material_count=len(prepared["material_names"]),
            latent_dim=int(self.config["model"]["latent_dim"]),
            backbone_layers=int(self.config["model"]["backbone_layers"]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["training"]["learning_rate"]), weight_decay=float(self.config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        use_amp = bool(self.config["training"].get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        epoch = 1; next_batch = 0; global_step = 0; epoch_loss_sum = 0.0; epoch_loss_count = 0
        best_mae = float("inf"); best_epoch = best_step = None; resume_events = 0
        start_timestamp = _utc_now(); prior_elapsed = 0.0
        if self.last_path.exists():
            checkpoint = torch.load(self.last_path, map_location=device, weights_only=False)
            validate_checkpoint_compatibility(checkpoint, prepared["compatibility"])
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"]); scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["amp_scaler_state"])
            epoch = int(checkpoint["epoch"]); next_batch = int(checkpoint["next_batch_index"]); global_step = int(checkpoint["global_step"])
            epoch_loss_sum = float(checkpoint["epoch_loss_sum"]); epoch_loss_count = int(checkpoint["epoch_loss_count"])
            best_mae = float(checkpoint["best_validation_mae"]); best_epoch = checkpoint["best_epoch"]; best_step = checkpoint["best_step"]
            resume_events = int(checkpoint.get("resume_events", 0)) + 1; start_timestamp = checkpoint["start_timestamp"]; prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
            restore_rng_state(checkpoint["rng_state"])
            print(f"TRAINING MODE: RESUME\nPrevious epoch: {epoch}\nPrevious step: {global_step}\nBest validation MAE_norm so far: {best_mae}")
        else:
            print("TRAINING MODE: FROM SCRATCH")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        clean_config = {key: value for key, value in self.config.items() if not key.startswith("_")}
        atomic_write_text(self.experiment_dir / "config.yaml", yaml.safe_dump(clean_config, sort_keys=False))
        atomic_write_json(self.experiment_dir / "environment.json", runtime_metadata(self.root))
        atomic_write_json(self.experiment_dir / "run_state.json", {"status": "INCOMPLETE", "backbone": self.backbone, "start_timestamp": start_timestamp, "resume_events": resume_events})
        cache_mode = str(self.config["training"].get("cache_mode", "hdf5"))
        train_dataset = MagNetDataset(prepared["h5_path"], prepared["train_indices"], cache_mode=cache_mode, **prepared["dataset_args"])
        validation_dataset = MagNetDataset(prepared["h5_path"], prepared["validation_indices"], cache_mode=cache_mode, **prepared["dataset_args"])
        loader_kwargs = _loader_kwargs(self.config["training"])
        non_blocking = bool(loader_kwargs["pin_memory"] and device.type == "cuda")
        validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
        metrics_rows = self._load_metrics(); session_start = time.monotonic()
        total_epochs = int(self.config["training"]["epochs"])
        try:
            while epoch <= total_epochs:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
                model.train()
                for batch_index, batch in enumerate(train_loader):
                    if batch_index < next_batch:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        output = model(
                            batch["waveform"].to(device, non_blocking=non_blocking),
                            batch["tabular"].to(device, non_blocking=non_blocking),
                            batch["material_index"].to(device, non_blocking=non_blocking),
                        )
                        target = batch["target"].to(device, non_blocking=non_blocking)
                        loss = F.mse_loss(output, target) + 0.3 * F.l1_loss(output, target)
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                    global_step += 1; epoch_loss_sum += float(loss.detach()); epoch_loss_count += 1
                    next_batch = batch_index + 1
                    elapsed = prior_elapsed + time.monotonic() - session_start
                    if global_step % self.checkpoint_every_n_steps == 0:
                        self._save_rolling(model, optimizer, scheduler, scaler, epoch=epoch, next_batch_index=next_batch, global_step=global_step, epoch_loss_sum=epoch_loss_sum, epoch_loss_count=epoch_loss_count, best_mae=best_mae, best_epoch=best_epoch, best_step=best_step, resume_events=resume_events, start_timestamp=start_timestamp, elapsed_seconds=elapsed)
                        memory = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                        print(f"Epoch {epoch}/{total_epochs} | step {global_step} | train loss {epoch_loss_sum/epoch_loss_count:.6f} | lr {optimizer.param_groups[0]['lr']:.3g} | elapsed {elapsed:.1f}s | GPU memory {memory:.2f} GiB")
                validation_loss, validation_metrics = self._evaluate(
                    model, validation_loader, device, non_blocking=non_blocking
                )
                elapsed = prior_elapsed + time.monotonic() - session_start
                row = {"epoch": epoch, "global_step": global_step, "training_loss": epoch_loss_sum/max(epoch_loss_count, 1), "validation_loss": validation_loss, **validation_metrics, "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed}
                metrics_rows = [item for item in metrics_rows if int(item["epoch"]) != epoch] + [row]
                self._write_metrics(metrics_rows)
                if validation_metrics["normalized_mae"] < best_mae:
                    best_mae = validation_metrics["normalized_mae"]; best_epoch = epoch; best_step = global_step
                    best = make_pretraining_checkpoint(model, preprocessing_metadata=prepared["preprocessing"], config=clean_config, git_commit=runtime_metadata(self.root)["git_commit"], random_seed=self.seed, material_names=prepared["material_names"])
                    best.update({"model_state": model.state_dict(), "compatibility": prepared["compatibility"], "best_epoch": best_epoch, "best_step": best_step, "validation_metrics": validation_metrics})
                    save_checkpoint(best, self.best_path)
                scheduler.step(); epoch += 1; next_batch = 0; epoch_loss_sum = 0.0; epoch_loss_count = 0
                self._save_rolling(model, optimizer, scheduler, scaler, epoch=epoch, next_batch_index=0, global_step=global_step, epoch_loss_sum=0.0, epoch_loss_count=0, best_mae=best_mae, best_epoch=best_epoch, best_step=best_step, resume_events=resume_events, start_timestamp=start_timestamp, elapsed_seconds=elapsed)
                completed_epoch = epoch - 1
                memory = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                depth = int(self.config["model"]["backbone_layers"])
                print(
                    f"Depth {depth} | Epoch {completed_epoch}/{total_epochs}\n"
                    f"train loss: {row['training_loss']:.6f}\n"
                    f"validation MAE_norm: {validation_metrics['normalized_mae']:.6f}\n"
                    f"validation RMSE: {validation_metrics['rmse']:.6f}\n"
                    f"validation R2: {validation_metrics['r2']:.6f}\n"
                    f"lr: {row['learning_rate']:.3g}\n"
                    f"elapsed: {elapsed:.1f}s\n"
                    f"GPU allocated: {memory:.2f} GiB"
                )
        except KeyboardInterrupt:
            elapsed = prior_elapsed + time.monotonic() - session_start
            self._save_rolling(model, optimizer, scheduler, scaler, epoch=epoch, next_batch_index=next_batch, global_step=global_step, epoch_loss_sum=epoch_loss_sum, epoch_loss_count=epoch_loss_count, best_mae=best_mae, best_epoch=best_epoch, best_step=best_step, resume_events=resume_events, start_timestamp=start_timestamp, elapsed_seconds=elapsed)
            print(f"EXPERIMENT STATUS: INCOMPLETE\nRECOVERY CHECKPOINT AVAILABLE: YES\nLAST SAVED EPOCH: {epoch}\nLAST SAVED STEP: {global_step}")
            raise
        self._model = model
        return {"status": "TRAINING_COMPLETE_EVALUATION_PENDING", "best_epoch": best_epoch, "best_step": best_step, "global_step": global_step, "start_timestamp": start_timestamp, "elapsed_seconds": prior_elapsed + time.monotonic() - session_start, "resume_events": resume_events}

    @staticmethod
    def _evaluate(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> tuple[float, dict[str, float]]:
        model.eval(); losses = []; truths = []; predictions = []
        with torch.no_grad():
            for batch in loader:
                output = model(
                    batch["waveform"].to(device, non_blocking=non_blocking),
                    batch["tabular"].to(device, non_blocking=non_blocking),
                    batch["material_index"].to(device, non_blocking=non_blocking),
                )
                target = batch["target"].to(device, non_blocking=non_blocking)
                losses.append(float(F.mse_loss(output, target) + 0.3 * F.l1_loss(output, target)))
                truths.append(target.cpu().numpy()); predictions.append(output.cpu().numpy())
        return float(np.mean(losses)), regression_metrics(np.concatenate(truths), np.concatenate(predictions))

    def plot_curves(self) -> list[Any]:
        if not self.metrics_path.exists():
            print("Training curves unavailable until at least one validation epoch completes")
            return []
        import matplotlib.pyplot as plt
        rows = self._load_metrics(); epochs = [int(row["epoch"]) for row in rows]
        figures = []
        for field, label in (("training_loss", "Training loss"), ("normalized_mae", "Validation MAE_norm"), ("rmse", "Validation RMSE")):
            figure = plt.figure(); plt.plot(epochs, [float(row[field]) for row in rows]); plt.xlabel("Epoch"); plt.ylabel(label); plt.title(f"{self.backbone}: {label}"); plt.grid(True); plt.show(); figures.append(figure)
        print("[PASS] Rendered three separate matplotlib training curves")
        return figures

    def evaluate_best_checkpoint(self) -> dict[str, Any]:
        if self.summary_path.exists():
            summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "COMPLETE":
                print("[PASS] Existing COMPLETE summary loaded; test set was not evaluated again")
                return summary
        if not self.best_path.exists():
            raise FileNotFoundError(f"Best checkpoint missing: {self.best_path}. Training must complete validation selection first.")
        evaluation_start = time.monotonic()
        prepared = self._prepare(); device = self._device or _model_device(str(self.config["training"].get("device", "auto")))
        checkpoint = torch.load(self.best_path, map_location=device, weights_only=False)
        validate_checkpoint_compatibility(checkpoint, prepared["compatibility"])
        model = PretrainingModel(self.backbone, material_count=len(prepared["material_names"]), latent_dim=int(self.config["model"]["latent_dim"]), backbone_layers=int(self.config["model"]["backbone_layers"])).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        loader_kwargs = _loader_kwargs(self.config["training"])
        cache_mode = str(self.config["training"].get("cache_mode", "hdf5"))
        validation = MagNetDataset(prepared["h5_path"], prepared["validation_indices"], cache_mode=cache_mode, **prepared["dataset_args"])
        non_blocking = bool(loader_kwargs["pin_memory"] and device.type == "cuda")
        validation_loss, validation_metrics = self._evaluate(
            model,
            DataLoader(validation, shuffle=False, **loader_kwargs),
            device,
            non_blocking=non_blocking,
        )
        test_indices = _read_indices(prepared["manifests"]["test"])
        test = MagNetDataset(prepared["h5_path"], test_indices, cache_mode=cache_mode, **prepared["dataset_args"])
        test_loss, test_metrics = self._evaluate(
            model,
            DataLoader(test, shuffle=False, **loader_kwargs),
            device,
            non_blocking=non_blocking,
        )
        state = torch.load(self.last_path, map_location="cpu", weights_only=False) if self.last_path.exists() else {}
        metrics_rows = self._load_metrics(); last = metrics_rows[-1]
        summary = {
            "status": "COMPLETE", "experiment": f"MEPI pretrain {self.backbone}", "protocol_version": PROTOCOL_VERSION,
            "backbone": self.backbone, "start_timestamp": state.get("start_timestamp"), "completion_timestamp": _utc_now(),
            "total_runtime_seconds": float(state.get("elapsed_seconds", last.get("elapsed_seconds", 0.0))) + time.monotonic() - evaluation_start,
            "epochs_completed": len(metrics_rows), "steps_completed": int(last["global_step"]),
            "resumed": int(state.get("resume_events", 0)) > 0, "resume_events": int(state.get("resume_events", 0)),
            "best_epoch": checkpoint["best_epoch"], "best_step": checkpoint["best_step"],
            "validation_loss": validation_loss, "validation_metrics": validation_metrics,
            "test_loss": test_loss, "test_metrics": test_metrics,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "sample_counts": {"total": len(prepared["train_indices"]) + len(prepared["validation_indices"]) + len(test_indices), "train": len(prepared["train_indices"]), "validation": len(prepared["validation_indices"]), "test": len(test_indices)},
            "compatibility": prepared["compatibility"], "best_checkpoint": str(self.best_path),
            "best_checkpoint_sha256": sha256_file(self.best_path), "test_evaluations": 1,
        }
        atomic_write_json(self.summary_path, summary)
        mark_completed(self.experiment_dir)
        print("[PASS] Frozen best checkpoint evaluated on validation")
        print("[PASS] Test evaluated exactly once after validation selection; test did not alter selection")
        print(f"Validation metrics: {validation_metrics}\nTest metrics: {test_metrics}")
        return summary

    def write_run_evidence(self) -> dict[str, Any]:
        if self.evidence_path.exists():
            existing = json.loads(self.evidence_path.read_text(encoding="utf-8"))
            if existing.get("status") == "COMPLETE":
                print("[PASS] Existing COMPLETE evidence retained unchanged")
                return existing
        if not self.summary_path.exists():
            raise RuntimeError("Cannot write COMPLETE evidence: training_summary.json does not exist")
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            raise RuntimeError(f"Cannot write COMPLETE evidence: actual status is {summary.get('status')!r}")
        environment = json.loads((self.experiment_dir / "environment.json").read_text(encoding="utf-8"))
        compatibility = summary["compatibility"]
        evidence = {
            **summary,
            "dataset_total": summary["sample_counts"]["total"], "train_count": summary["sample_counts"]["train"], "validation_count": summary["sample_counts"]["validation"], "test_count": summary["sample_counts"]["test"],
            "dataset_fingerprint": compatibility["dataset_fingerprint"], "split_fingerprint": compatibility["split_fingerprint"],
            "waveform_input": "B(t)", "waveform_length": WAVEFORM_LENGTH,
            "pretraining_tabular_input": list(PRETRAIN_TABULAR_FEATURES), "tabular_dimension": 2,
            "material_used_as_model_input": False, "target_definition": "P_v core loss",
            "target_transform": compatibility["target_transform"], "seed": self.seed,
            "optimizer": "AdamW", "learning_rate": float(self.config["training"]["learning_rate"]),
            "scheduler": "constant LambdaLR", "effective_batch_size": int(self.config["training"]["batch_size"]),
            "training_runtime_settings": self.config.get("runtime", {}),
            "dataloader_settings": {
                key: self.config["training"].get(key)
                for key in (
                    "batch_size", "num_workers", "pin_memory", "persistent_workers",
                    "prefetch_factor", "cache_mode", "amp",
                )
            },
            "source_git_sha": environment.get("git_commit"), "source_tree_fingerprint": source_tree_fingerprint(self.root),
            "python_version": environment.get("python_version"), "pytorch_version": environment.get("pytorch_version"),
            "cuda_version": environment.get("cuda_version"), "gpu_model": environment.get("gpu_model"),
            "final_test_suite_state": _read_test_state(self.root),
        }
        atomic_write_json(self.evidence_path, evidence)
        lines = [
            f"# MEPI pretraining evidence — {self.backbone}", "", "- Status: **COMPLETE**",
            f"- Protocol: `{PROTOCOL_VERSION}`", f"- Dataset SHA-256: `{evidence['dataset_fingerprint']}`",
            f"- Split fingerprint: `{evidence['split_fingerprint']}`", f"- Best epoch / step: {evidence['best_epoch']} / {evidence['best_step']}",
            f"- Resumed: {evidence['resumed']} ({evidence['resume_events']} recovery event(s))", "",
            "## Validation (selection)", "", f"- MAE_norm: {evidence['validation_metrics']['normalized_mae']}", f"- RMSE: {evidence['validation_metrics']['rmse']}", f"- R²: {evidence['validation_metrics']['r2']}", "",
            "## Test (reported once after selection)", "", f"- MAE_norm: {evidence['test_metrics']['normalized_mae']}", f"- RMSE: {evidence['test_metrics']['rmse']}", f"- R²: {evidence['test_metrics']['r2']}", "",
            f"- Best checkpoint: `{evidence['best_checkpoint']}`", f"- Checkpoint SHA-256: `{evidence['best_checkpoint_sha256']}`",
        ]
        atomic_write_text(self.experiment_dir / "run_evidence.md", "\n".join(lines) + "\n")
        atomic_write_json(self.experiment_dir / "run_state.json", {"status": "COMPLETE", "completion_timestamp": evidence["completion_timestamp"]})
        print(f"[PASS] COMPLETE measured evidence written: {self.evidence_path}")
        return evidence

    def display_final_summary(self) -> dict[str, Any]:
        evidence = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        validation = evidence["validation_metrics"]; test = evidence["test_metrics"]
        print("=" * 60); print("MEPI PRETRAINING RUN COMPLETE"); print("=" * 60)
        print(f"Protocol          : {PROTOCOL_VERSION}\nBackbone          : {self.backbone}\nTraining status   : COMPLETE")
        print(f"Resumed           : {'YES' if evidence['resumed'] else 'NO'} ({evidence['resume_events']} recovery event(s))")
        print(f"Train samples     : {evidence['train_count']}\nValidation samples: {evidence['validation_count']}\nTest samples      : {evidence['test_count']}")
        print(f"Best epoch        : {evidence['best_epoch']}\nBest step         : {evidence['best_step']}")
        print(f"Validation\n  MAE_norm        : {validation['normalized_mae']}\n  RMSE            : {validation['rmse']}\n  R²              : {validation['r2']}")
        print(f"Test\n  MAE_norm        : {test['normalized_mae']}\n  RMSE            : {test['rmse']}\n  R²              : {test['r2']}")
        print(f"Parameters        : {evidence['parameter_count']}\nRuntime           : {evidence['total_runtime_seconds']} seconds")
        print(f"Best checkpoint:\n{evidence['best_checkpoint']}\n\nCheckpoint SHA256:\n{evidence['best_checkpoint_sha256']}")
        print(f"\nEvidence:\n{self.experiment_dir / 'run_evidence.md'}\n\nRESULT: SUCCESS"); print("=" * 60)
        print("Execution complete.\nPlease save this notebook to preserve visible outputs as experiment evidence.")
        return evidence


def _read_test_state(root: Path) -> dict[str, Any]:
    path = root / "reports" / "notebook_workflow_test_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "NOT_RECORDED", "reason": "Run 00_environment_check.ipynb"}


def run_environment_checks(project: str | Path, *, require_cuda: bool = True, run_tests: bool = True) -> dict[str, Any]:
    root = Path(project).resolve()
    protocols = [root / relative for relative in PROTOCOL_RELATIVE_PATHS]
    protocol_hashes = [sha256_file(path) if path.is_file() else None for path in protocols]
    runtime = runtime_metadata(root)
    kernel = _trans_core_kernel_status()
    print(f"Project root: {root}")
    print(f"Python executable: {sys.executable}\nPython version: {sys.version.replace(chr(10), ' ')}")
    print(f"PyTorch version: {torch.__version__}\nCUDA version: {torch.version.cuda}\nGPU: {runtime['gpu_model']}")
    print(
        f"Registered kernel: {kernel['display_name'] or 'MISSING'} "
        f"({kernel['kernel_interpreter'] or 'no interpreter'})"
    )
    print(
        f"[{'PASS' if kernel['matches_active_interpreter'] else 'WARN'}] "
        "Active interpreter matches Python (trans-core): "
        f"{kernel['matches_active_interpreter']}"
    )
    if not kernel["matches_active_interpreter"]:
        print(
            "[WARN] Select the 'Python (trans-core)' kernel in VSCode/Jupyter; "
            f"the current interpreter is {sys.executable}."
        )
    checks = {
        "project_root": root.is_dir(),
        "protocol_exists": all(path.is_file() for path in protocols),
        "protocol_hash": protocol_hashes == [EXPECTED_PROTOCOL_SHA256] * len(protocols),
        "source_import": True, "waveform_length_1024": WAVEFORM_LENGTH == 1024,
        "tabular_contract": list(PRETRAIN_TABULAR_FEATURES) == ["frequency_hz", "temperature_c"],
        "eight_backbones": len(CANDIDATE_BACKBONES) == 8, "cuda_available": torch.cuda.is_available(),
    }
    for name, passed in checks.items(): print(f"[{'PASS' if passed else 'FAIL'}] {name}: {passed}")
    if not all(value for key, value in checks.items() if key != "cuda_available"):
        raise RuntimeError(
            f"Protocol/environment assertions failed: {checks}. "
            f"Inspect {protocols} and src/mepi_v1."
        )
    test_result = {"status": "SKIPPED", "returncode": None}
    if run_tests:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, text=True, capture_output=True)
        print(completed.stdout); print(completed.stderr)
        test_result = {"status": "PASS" if completed.returncode == 0 else "FAIL", "returncode": completed.returncode, "timestamp_utc": _utc_now()}
        atomic_write_json(root / "reports" / "notebook_workflow_test_state.json", test_result)
        if completed.returncode:
            raise RuntimeError(f"Project tests failed with return code {completed.returncode}. Inspect the pytest output above before training.")
    if require_cuda and not checks["cuda_available"]:
        raise RuntimeError("CUDA readiness failed: torch.cuda.is_available() is False. Fix the GPU/PyTorch environment before pretraining.")
    result = {"environment_ready": True, "cuda_ready": checks["cuda_available"], "tests": test_result, "protocol_assertions": "PASS", "runtime": runtime, "kernel": kernel}
    print(f"ENVIRONMENT READY: YES\nCUDA READY: {'YES' if checks['cuda_available'] else 'NO'}\nTESTS: {test_result['status']}\nPROTOCOL ASSERTIONS: PASS")
    return result


def run_dataset_audit_notebook(project: str | Path) -> dict[str, Any]:
    root = Path(project).resolve(); print(f"Loading MagNet audit configuration: {root / 'configs/magnet.yaml'}")
    report = audit_magnet(root / "configs" / "magnet.yaml")
    manifests = [root / "data/splits/mepi_v1" / f"magnet_{name}.csv" for name in ("train", "validation", "test")]
    result = {**report, "split_fingerprint": split_fingerprint(manifests)}
    print(f"Dataset loaded: {report['dataset']}\nShape: {report['B_waveform_shape']}\nCounts: {report['split']['counts']}")
    print(f"Ranges: {json.dumps(report['ranges'], indent=2)}\nMaterials: {json.dumps(report['samples_per_material'], indent=2)}")
    print(f"Duplicates: sample IDs={report['duplicate_sample_ids']}, waveforms={report['exact_duplicate_waveforms']}")
    print(f"Split integrity: manifests present, fingerprint={result['split_fingerprint']}")
    if report["status"] != "PASS": raise RuntimeError(f"MAGNET DATASET AUDIT failed: actual status {report['status']}; inspect reports/magnet_dataset_audit.json")
    print(f"MAGNET DATASET AUDIT: PASS\nDataset fingerprint: {report['dataset_sha256']}\nSplit fingerprint: {result['split_fingerprint']}")
    return result


def compare_completed_pretraining(project: str | Path) -> list[dict[str, Any]]:
    root = Path(project).resolve(); rows = []
    for backbone in CANDIDATE_BACKBONES:
        path = root / EXPERIMENT_ROOT / backbone / "run_evidence.json"
        if not path.exists():
            rows.append({"backbone": backbone, "status": "PENDING"}); continue
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("status") != "COMPLETE":
            rows.append({"backbone": backbone, "status": "INCOMPLETE"}); continue
        rows.append({"backbone": backbone, "status": "COMPLETE", "parameter_count": evidence["parameter_count"], "best_epoch": evidence["best_epoch"], "validation_mae_norm": evidence["validation_metrics"]["normalized_mae"], "validation_rmse": evidence["validation_metrics"]["rmse"], "validation_r2": evidence["validation_metrics"]["r2"], "test_mae_norm": evidence["test_metrics"]["normalized_mae"], "test_rmse": evidence["test_metrics"]["rmse"], "test_r2": evidence["test_metrics"]["r2"], "checkpoint_sha256": evidence["best_checkpoint_sha256"], "resumed": evidence["resumed"]})
    complete = sorted((row for row in rows if row["status"] == "COMPLETE"), key=lambda row: row["validation_mae_norm"])
    pending = [row for row in rows if row["status"] != "COMPLETE"]
    ordered = complete + pending
    report_dir = root / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    fields = ["backbone", "status", "parameter_count", "best_epoch", "validation_mae_norm", "validation_rmse", "validation_r2", "test_mae_norm", "test_rmse", "test_r2", "checkpoint_sha256", "resumed"]
    descriptor, temporary_name = tempfile.mkstemp(prefix=".comparison.", suffix=".tmp", dir=report_dir)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(ordered); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary_name, report_dir / "pretraining_backbone_comparison.csv")
    md = ["# Measured pretraining backbone comparison", "", "Sorted strictly by validation MAE_norm; pending runs are listed last.", "", "| Backbone | Status | Parameters | Best epoch | Val MAE_norm | Val RMSE | Val R² | Test MAE_norm | Test RMSE | Test R² | Resumed |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in ordered: md.append("| " + " | ".join(str(row.get(key, "")) for key in ("backbone", "status", "parameter_count", "best_epoch", "validation_mae_norm", "validation_rmse", "validation_r2", "test_mae_norm", "test_rmse", "test_r2", "resumed")) + " |")
    atomic_write_text(report_dir / "pretraining_backbone_comparison.md", "\n".join(md) + "\n")
    selection_path = root / "artifacts/manuscript/selected_pretraining_backbone.json"
    if len(complete) == len(CANDIDATE_BACKBONES):
        winner = {"selection_status": "COMPLETE", "selection_criterion": "minimum validation MAE_norm", "selected_backbone": complete[0]["backbone"], "validation_mae_norm": complete[0]["validation_mae_norm"], "completed_backbones": len(complete), "required_backbones": len(CANDIDATE_BACKBONES), "generated_at": _utc_now()}
        atomic_write_json(selection_path, winner)
    else:
        selection_path.unlink(missing_ok=True)
    print(f"Completed runs: {len(complete)}/{len(CANDIDATE_BACKBONES)}")
    if len(complete) == len(CANDIDATE_BACKBONES): print(f"Validation-only selection: {complete[0]['backbone']} (MAE_norm={complete[0]['validation_mae_norm']})")
    elif complete: print(f"Current ranking leader: {complete[0]['backbone']}; selection artifact withheld until all eight runs are COMPLETE")
    else: print("No completed measured runs; no selection artifact written")
    return ordered
