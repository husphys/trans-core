"""Preparation utilities for the proposed MEPI v1.4 training contract.

These utilities perform no neural-network training and never construct a test
dataset or loader.  They only fit the explicitly authorized common Steinmetz
prior from frozen training rows and recover representation-compatible waveform
preprocessing from the selected MagNet checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config, resolve_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train":
                rows.append(row)
    return rows


def recover_waveform_preprocessing(config_path: str | Path) -> dict[str, Any]:
    """Recover the exact global train-statistic transform stored in the checkpoint."""

    config = load_config(config_path)
    checkpoint_path = resolve_path(config, config["pretrained_checkpoint"])
    expected_hash = str(config["pretrained_checkpoint_sha256"])
    actual_hash = _sha256(checkpoint_path)
    if actual_hash != expected_hash:
        raise AssertionError("Selected xLSTM checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("preprocessing_metadata", {})
    waveform = metadata.get("waveform", {})
    mean = float(waveform["mean"])
    scale = float(waveform["scale"])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Checkpoint waveform preprocessing is incomplete")
    return {
        "status": "RECOVERED_FROM_EXECUTABLE_EVIDENCE",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "transform_order": [
            "validate B(t) shape equals 1024",
            "cast source values to float64 while fitting global statistics",
            "subtract MagNet training-sample global element mean",
            "divide by MagNet training-sample global element population standard deviation",
            "cast transformed waveform to float32",
            "add singleton channel dimension",
        ],
        "formula": "B_model = (B_raw - mean) / scale",
        "mean": mean,
        "scale": scale,
        "statistics_scope": "all waveform elements from the MagNet training split",
        "per_waveform": False,
        "clipping": None,
        "epsilon_or_floor": 1e-12,
        "fit_sample_count": int(metadata["train_sample_count"]),
        "source": "checkpoint.preprocessing_metadata plus src/mepi_v1/pretrain.py::_fit_preprocessing",
    }


def fit_common_steinmetz_prior(config_path: str | Path) -> dict[str, Any]:
    """Fit one leakage-safe common prior using only QC-valid frozen train rows."""

    config = load_config(config_path)
    dataset = config["dataset"]
    candidate_path = resolve_path(config, dataset["path"])
    rows = _train_rows(candidate_path)
    accepted: list[dict[str, str]] = []
    for row in rows:
        frequency = float(row["frequency_hz"])
        b_peak = float(row["B_peak_t"])
        p_loss = float(row["P_loss"])
        if all(math.isfinite(value) and value > 0.0 for value in (frequency, b_peak, p_loss)):
            accepted.append(row)
    if len(accepted) != 687:
        raise AssertionError(f"Expected 687 positive finite training rows, received {len(accepted)}")

    frequency = np.asarray([float(row["frequency_hz"]) for row in accepted], dtype=np.float64)
    b_peak = np.asarray([float(row["B_peak_t"]) for row in accepted], dtype=np.float64)
    p_loss = np.asarray([float(row["P_loss"]) for row in accepted], dtype=np.float64)
    design = np.column_stack((np.ones(len(accepted), dtype=np.float64), np.log(frequency), np.log(b_peak)))
    response = np.log(p_loss)
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(design, response, rcond=None)
    intercept, alpha, beta = (float(value) for value in coefficients)
    k = float(np.exp(intercept))

    split_json = resolve_path(config, "data/MEPI/v1_2/split_manifest_v1_2.json")
    split = json.loads(split_json.read_text(encoding="utf-8"))
    split_csv = resolve_path(config, dataset["split_manifest"])
    normalization_path = resolve_path(config, dataset["normalization"])
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    train_group_ids = list(split["groups"]["train"])
    observed_group_ids = sorted({row["condition_group_id"] for row in accepted})
    if len(train_group_ids) != 72 or len(observed_group_ids) != 71:
        raise AssertionError("Frozen train-group evidence changed")

    return {
        "artifact": "steinmetz_prior_train_v1_4.json",
        "status": "DRAFT_TRAIN_ONLY_PRIOR_READY_PROTOCOL_NOT_FROZEN",
        "core_id_used": False,
        "fit_method": {
            "library": "NumPy",
            "function": "numpy.linalg.lstsq",
            "numpy_dtype": "float64",
            "rcond": None,
            "model": "log(P_loss_W) = intercept + alpha*log(frequency_Hz) + beta*log(B_peak_T)",
            "intercept": True,
            "filter": "split == train and finite positive frequency_hz, B_peak_t, P_loss",
            "numerical_epsilon": None,
            "logarithm": "natural log",
        },
        "coefficients": {"k": k, "alpha": alpha, "beta": beta, "log_k": intercept},
        "fit_row_count": len(accepted),
        "fit_observed_group_count": len(observed_group_ids),
        "frozen_train_group_count": len(train_group_ids),
        "train_group_ids": train_group_ids,
        "observed_train_group_ids": observed_group_ids,
        "split_manifest_csv": str(split_csv),
        "split_manifest_sha256": _sha256(split_csv),
        "candidate_dataset": str(candidate_path),
        "candidate_dataset_sha256": _sha256(candidate_path),
        "target_scaler": {
            "name": "P_loss",
            "mean_W": float(normalization["target_scalers"]["P_loss"]["mean"][0]),
            "scale_W": float(normalization["target_scalers"]["P_loss"]["scale"][0]),
            "fitted_split": "train",
            "ddof": 0,
            "formula": "P_St_z = (P_St_raw_W - mean_W) / scale_W",
        },
        "application": {
            "raw_formula": "P_St_raw_W = k * frequency_Hz**alpha * B_peak_T**beta",
            "residual": "r_St = P_loss_aux_z - P_St_z",
            "validation_refit": False,
            "test_refit": False,
        },
        "units": {
            "frequency_hz": "Hz",
            "B_peak_t": "T",
            "P_loss": "W",
            "k": "W / (Hz**alpha * T**beta)",
        },
        "least_squares_diagnostics": {
            "rank": int(rank),
            "singular_values": [float(value) for value in singular_values],
            "sum_squared_log_residuals": float(residuals[0]) if residuals.size else 0.0,
        },
        "test_samples_accessed": False,
    }


def write_steinmetz_prior(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    payload = fit_common_steinmetz_prior(config_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
