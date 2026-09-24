"""Validation-only xLSTM depth experiment on the audited MagNet splits.

This module deliberately does not expose a test-evaluation path.  It reuses the
current reproducible pretraining workflow while changing only the number of
stacked xLSTM blocks and the manuscript-defined ten-epoch depth-study horizon.
"""

from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data import MagNetDataset
from .models import PretrainingModel
from .notebook_workflow import (
    PROTOCOL_VERSION,
    PretrainingNotebookRun,
    _count_csv_rows,
    _model_device,
    _read_test_state,
    _target_from_metadata,
    atomic_write_json,
    atomic_write_text,
    mark_completed,
    source_tree_fingerprint,
    validate_checkpoint_compatibility,
)
from .pretrain import _loader_kwargs, _read_indices
from .reproducibility import runtime_metadata, sha256_file


DEPTHS = (2, 4, 6, 8, 10)
DEPTH_EXPERIMENT_ROOT = Path("experiments/xlstm_depth_v1")
OFFICIAL_PRETRAINING_ROOT = Path("experiments/pretrain_v1")


def _tree_sha256(root: Path) -> str:
    """Hash relative paths and bytes for an immutable evidence tree."""

    import hashlib

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def official_pretraining_snapshot(project: str | Path) -> dict[str, Any]:
    root = Path(project).resolve() / OFFICIAL_PRETRAINING_ROOT
    files = sorted(item for item in root.rglob("*") if item.is_file())
    return {
        "root": str(root),
        "file_count": len(files),
        "tree_sha256": _tree_sha256(root),
    }


class XLSTMDepthRun(PretrainingNotebookRun):
    """One crash-safe xLSTM depth candidate with no test-set access."""

    def __init__(
        self,
        project: str | Path,
        depth: int,
        *,
        seed: int = 42,
        force_retrain: bool = False,
        auto_resume: bool = True,
        checkpoint_every_n_steps: int = 1000,
    ) -> None:
        if int(depth) not in DEPTHS:
            raise ValueError(f"Depth must be one of {DEPTHS}, received {depth!r}")
        super().__init__(
            project,
            "xLSTM",
            seed=seed,
            force_retrain=force_retrain,
            auto_resume=auto_resume,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
        )
        self.depth = int(depth)
        depth_config = load_config(self.root / "configs/xlstm_depth_v1.yaml")
        if tuple(int(value) for value in depth_config["depth_experiment"]["depths"]) != DEPTHS:
            raise ValueError("The configured depth set differs from the audited manuscript design")
        self.config = deepcopy(self.config)
        self.config["model"] = deepcopy(self.config["model"])
        self.config["training"] = deepcopy(self.config["training"])
        self.config["model"]["backbone_layers"] = self.depth
        self.config["training"]["epochs"] = int(depth_config["depth_experiment"]["epochs"])
        self.config["experiment"] = {
            "kind": "xlstm_depth_v1",
            "depth": self.depth,
            "selection_metric": "validation_normalized_mae",
            "test_access": "FORBIDDEN_DURING_SELECTION",
            "preprocessing_reuse": "official xLSTM train-fitted metadata, fingerprint verified",
        }
        self.experiment_dir = self.root / DEPTH_EXPERIMENT_ROOT / f"depth_{self.depth}"
        official = (self.root / OFFICIAL_PRETRAINING_ROOT).resolve()
        if self.experiment_dir.resolve() == official or official in self.experiment_dir.resolve().parents:
            raise RuntimeError("Depth output must not be inside official pretraining evidence")
        self.last_path = self.experiment_dir / "last_checkpoint.pt"
        self.best_path = self.experiment_dir / "best_checkpoint.pt"
        self.summary_path = self.experiment_dir / "training_summary.json"
        self.evidence_path = self.experiment_dir / "run_evidence.json"
        self.metrics_path = self.experiment_dir / "metrics.csv"

    def _manifests(self) -> dict[str, Path]:
        """Return only the two splits that depth selection is authorized to use."""

        split_dir = resolve_path(self.config, self.config["dataset"]["split_dir"])
        return {
            name: split_dir / f"magnet_{name}.csv"
            for name in ("train", "validation")
        }

    def validate_dataset(self) -> dict[str, Any]:
        """Validate train/validation inputs without opening the test manifest."""

        import h5py

        manifests = self._manifests()
        for path in manifests.values():
            if not path.is_file():
                raise FileNotFoundError(f"Required split manifest is missing: {path}")
        counts = {name: _count_csv_rows(path) for name, path in manifests.items()}
        h5_path = resolve_path(self.config, self.config["dataset"]["path"])
        with h5py.File(h5_path, "r") as handle:
            waveform_shape = list(handle["B"].shape)
        result = {
            "dataset_path": str(h5_path),
            "waveform_shape": waveform_shape,
            "counts": counts | {"test": "NOT_ACCESSED"},
            "test_split_accessed": False,
        }
        print(f"[PASS] Depth-study dataset: {h5_path}")
        print(f"[PASS] Train/validation counts: {counts}")
        print("[PASS] Test manifest and test dataset not opened")
        return result

    def validate_completed_compatibility(self) -> dict[str, Any]:
        """Refuse to skip a completed candidate unless all frozen identities match."""

        if not self.summary_path.is_file() or not self.best_path.is_file():
            raise FileNotFoundError("A COMPLETE candidate is missing its summary or best checkpoint")
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        expected = self._prepare()["compatibility"]
        validate_checkpoint_compatibility(
            {"compatibility": summary.get("compatibility", {})}, expected
        )
        if summary.get("best_checkpoint_sha256") != sha256_file(self.best_path):
            raise ValueError("Completed candidate checkpoint SHA-256 does not match its summary")
        if summary.get("test_split_accessed") is not False or summary.get("test_evaluations") != 0:
            raise ValueError("Completed depth candidate violates test-isolation requirements")
        print(f"[PASS] Depth {self.depth} completed evidence is configuration-compatible")
        return summary

    def _prepare(self) -> dict[str, Any]:
        """Reuse the official train-fitted transform after strict identity checks.

        The five candidates share the exact dataset and train manifest used by the
        completed xLSTM screening run. Re-reading its immutable preprocessing
        metadata avoids five redundant full-HDF5 statistics scans.
        """

        if self._prepared is not None:
            return self._prepared
        h5_path = resolve_path(self.config, self.config["dataset"]["path"])
        manifests = self._manifests()
        train_indices = _read_indices(manifests["train"])
        validation_indices = _read_indices(manifests["validation"])
        official_path = self.root / OFFICIAL_PRETRAINING_ROOT / "xLSTM/best_checkpoint.pt"
        official = torch.load(official_path, map_location="cpu", weights_only=False)
        if official.get("config", {}).get("backbone") != "xLSTM":
            raise ValueError("Official preprocessing source is not an xLSTM checkpoint")
        official_compatibility = official.get("compatibility", {})
        dataset_fingerprint = sha256_file(h5_path)
        if official_compatibility.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Dataset fingerprint differs from the official xLSTM screening run")
        preprocessing = official["preprocessing_metadata"]
        if preprocessing.get("fitted_split") != "train":
            raise ValueError("Official preprocessing was not fitted on the training split")
        if int(preprocessing.get("train_sample_count", -1)) != len(train_indices):
            raise ValueError("Training-manifest count differs from official preprocessing metadata")
        material_names = list(official["material_names"])
        target = _target_from_metadata(preprocessing["target"])
        dataset_args = {
            "material_to_index": {name: index for index, name in enumerate(material_names)},
            "waveform_mean": preprocessing["waveform"]["mean"],
            "waveform_scale": preprocessing["waveform"]["scale"],
            "operating_mean": preprocessing["operating"]["mean"],
            "operating_scale": preprocessing["operating"]["scale"],
            "target_transform": target,
        }
        self._prepared = {
            "h5_path": h5_path,
            "manifests": manifests,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "preprocessing": preprocessing,
            "target": target,
            "material_names": material_names,
            "dataset_args": dataset_args,
            "preprocessing_source": {
                "checkpoint": str(official_path),
                "checkpoint_sha256": sha256_file(official_path),
                "dataset_fingerprint": dataset_fingerprint,
                "fitted_split": "train",
                "train_sample_count": len(train_indices),
            },
        }
        self._prepared["compatibility"] = self._compatibility(
            preprocessing, material_count=len(material_names)
        )
        return self._prepared

    def finalize_validation_only(self, *, peak_gpu_memory_bytes: int) -> dict[str, Any]:
        """Freeze the validation-selected checkpoint without constructing a test dataset."""

        if self.summary_path.exists():
            summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "COMPLETE":
                return summary
        if not self.best_path.exists() or not self.last_path.exists():
            raise FileNotFoundError("Training must finish before validation-only finalization")
        evaluation_start = time.monotonic()
        prepared = self._prepare()
        device = self._device or _model_device(str(self.config["training"].get("device", "auto")))
        checkpoint = torch.load(self.best_path, map_location=device, weights_only=False)
        validate_checkpoint_compatibility(checkpoint, prepared["compatibility"])
        model = PretrainingModel(
            "xLSTM",
            material_count=len(prepared["material_names"]),
            latent_dim=int(self.config["model"]["latent_dim"]),
            backbone_layers=self.depth,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        loader_kwargs = _loader_kwargs(self.config["training"])
        validation = MagNetDataset(
            prepared["h5_path"],
            prepared["validation_indices"],
            cache_mode=str(self.config["training"].get("cache_mode", "hdf5")),
            **prepared["dataset_args"],
        )
        validation_loss, validation_metrics = self._evaluate(
            model,
            DataLoader(validation, shuffle=False, **loader_kwargs),
            device,
            non_blocking=bool(loader_kwargs["pin_memory"] and device.type == "cuda"),
        )
        state = torch.load(self.last_path, map_location="cpu", weights_only=False)
        metrics_rows = self._load_metrics()
        final_epoch = int(self.config["training"]["epochs"])
        best_epoch = int(checkpoint["best_epoch"])
        summary = {
            "status": "COMPLETE",
            "experiment": "MEPI xLSTM depth sensitivity on MagNet",
            "protocol_version": PROTOCOL_VERSION,
            "backbone": "xLSTM",
            "depth": self.depth,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "best_step": int(checkpoint["best_step"]),
            "epochs_completed": len(metrics_rows),
            "final_epoch": final_epoch,
            "training_horizon_boundary_reached": best_epoch == final_epoch,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "selection_metric": "validation_normalized_mae",
            "selected_with_validation_only": True,
            "test_split_accessed": False,
            "test_evaluations": 0,
            "sample_counts": {
                "train": len(prepared["train_indices"]),
                "validation": len(prepared["validation_indices"]),
                "test": "NOT_ACCESSED",
            },
            "start_timestamp": state.get("start_timestamp"),
            "completion_timestamp": runtime_metadata(self.root)["timestamp_utc"],
            "training_runtime_seconds": float(state.get("elapsed_seconds", 0.0)),
            "finalization_runtime_seconds": time.monotonic() - evaluation_start,
            "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
            "peak_gpu_memory_gib": float(peak_gpu_memory_bytes / (1024**3)),
            "resumed": int(state.get("resume_events", 0)) > 0,
            "resume_events": int(state.get("resume_events", 0)),
            "compatibility": prepared["compatibility"],
            "preprocessing_source": prepared["preprocessing_source"],
            "best_checkpoint": str(self.best_path),
            "best_checkpoint_sha256": sha256_file(self.best_path),
        }
        atomic_write_json(self.summary_path, summary)
        mark_completed(self.experiment_dir)
        return summary

    def write_depth_evidence(
        self, summary: dict[str, Any], official_snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        evidence = {
            **summary,
            "official_pretraining_evidence": official_snapshot,
            "source_tree_fingerprint": source_tree_fingerprint(self.root),
            "environment": runtime_metadata(self.root),
            "final_test_suite_state": _read_test_state(self.root),
        }
        atomic_write_json(self.evidence_path, evidence)
        metric = evidence["validation_metrics"]
        lines = [
            f"# xLSTM depth {self.depth} evidence",
            "",
            "- Status: **COMPLETE**",
            "- Selection scope: validation only",
            "- Test split accessed: **NO**",
            f"- Parameters: {evidence['parameter_count']}",
            f"- Best epoch: {evidence['best_epoch']} / {evidence['final_epoch']}",
            f"- TRAINING_HORIZON_BOUNDARY_REACHED: `{str(evidence['training_horizon_boundary_reached']).lower()}`",
            f"- Validation MAE_norm: {metric['normalized_mae']}",
            f"- Validation RMSE: {metric['rmse']}",
            f"- Validation R2: {metric['r2']}",
            f"- Training runtime: {evidence['training_runtime_seconds']} seconds",
            f"- Peak GPU memory: {evidence['peak_gpu_memory_gib']} GiB",
            f"- Best checkpoint SHA-256: `{evidence['best_checkpoint_sha256']}`",
            f"- Official pretraining tree SHA-256: `{official_snapshot['tree_sha256']}`",
        ]
        atomic_write_text(self.experiment_dir / "run_evidence.md", "\n".join(lines) + "\n")
        atomic_write_json(
            self.experiment_dir / "run_state.json",
            {"status": "COMPLETE", "completion_timestamp": evidence["completion_timestamp"]},
        )
        return evidence


def summarize_depth_experiment(project: str | Path) -> list[dict[str, Any]]:
    root = Path(project).resolve()
    rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        path = root / DEPTH_EXPERIMENT_ROOT / f"depth_{depth}" / "training_summary.json"
        if not path.exists():
            rows.append({"depth": depth, "status": "PENDING"})
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary["validation_metrics"]
        rows.append(
            {
                "depth": depth,
                "status": summary["status"],
                "parameter_count": summary["parameter_count"],
                "best_epoch": summary["best_epoch"],
                "validation_mae_norm": metrics["normalized_mae"],
                "validation_rmse": metrics["rmse"],
                "validation_r2": metrics["r2"],
                "training_runtime_seconds": summary["training_runtime_seconds"],
                "peak_gpu_memory_gib": summary["peak_gpu_memory_gib"],
                "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
                "training_horizon_boundary_reached": summary[
                    "training_horizon_boundary_reached"
                ],
            }
        )
    complete = sorted(
        (row for row in rows if row["status"] == "COMPLETE"),
        key=lambda row: row["validation_mae_norm"],
    )
    pending = [row for row in rows if row["status"] != "COMPLETE"]
    ordered = complete + pending
    output_root = root / DEPTH_EXPERIMENT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    fields = (
        "depth",
        "status",
        "parameter_count",
        "best_epoch",
        "validation_mae_norm",
        "validation_rmse",
        "validation_r2",
        "training_runtime_seconds",
        "peak_gpu_memory_gib",
        "best_checkpoint_sha256",
        "training_horizon_boundary_reached",
    )
    with (output_root / "depth_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    if len(complete) == len(DEPTHS):
        winner = complete[0]
        atomic_write_json(
            output_root / "selected_depth.json",
            {
                "status": "COMPLETE",
                "backbone": "xLSTM",
                "selected_depth": winner["depth"],
                "parameter_count": winner["parameter_count"],
                "selection_criterion": "minimum validation MAE_norm",
                "validation_mae_norm": winner["validation_mae_norm"],
                "test_split_accessed": False,
                "scope": "selected under the MagNet depth-study protocol only",
            },
        )
    return ordered
