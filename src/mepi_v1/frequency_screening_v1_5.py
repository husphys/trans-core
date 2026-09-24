"""Inference-only MEPI v1.5 manuscript frequency-screening runtime.

The screening source is the dedicated demo manifest and its per-condition
processed B1024 files. Model-development splits and final-test data are never
opened here. Evidence is written only after an explicit ``authorize=True``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch

from .config import load_config, resolve_path
from .constants import FINETUNE_TABULAR_FEATURES
from .final_model_manifest_v1_5 import (
    MANIFEST_RELATIVE_PATH,
    SELECTED_CHECKPOINT_SHA256,
    sha256_file,
    validate_final_model_manifest,
)
from .finetune_v1_4 import (
    WAVEFORM_MEAN,
    WAVEFORM_SCALE,
    load_steinmetz_prior,
    steinmetz_reference_z,
)
from .finetune_v1_5 import construct_v1_5_model

SCREENING_INPUT_RELATIVE_PATH = Path("data/MEPI/demo_manifest_v2.csv")
SCREENING_INPUT_SHA256 = "812ea7e87ad69c02702a411d0f1fe6ef241b9753586a50f1d38522d25d7f9556"
REPORT_JSON_RELATIVE_PATH = Path("reports/MEPI_V1_5_FREQUENCY_SCREENING.json")
REPORT_MARKDOWN_RELATIVE_PATH = Path("reports/MEPI_V1_5_FREQUENCY_SCREENING.md")
PREDICTIONS_CSV_RELATIVE_PATH = Path(
    "reports/MEPI_V1_5_FREQUENCY_SCREENING_PREDICTIONS.csv"
)
EXPECTED_CANDIDATE_FREQUENCIES_HZ = tuple(range(1000, 4501, 250))
PREDICTION_COLUMNS = (
    "sample_id",
    "condition_group_id",
    "core_id",
    "repeat_id",
    "candidate_frequency_hz",
    "frequency_hz",
    "vin_rms_v",
    "predicted_efficiency_percent",
    "predicted_P_loss",
    "predicted_LSP_raw",
    "predicted_LSP_std_raw",
    "qc_status",
    "qc_warning_reasons",
)


@dataclass(frozen=True)
class ScreeningInputBundle:
    rows: tuple[dict[str, Any], ...]
    waveforms: np.ndarray
    input_path: Path
    input_sha256: str
    waveform_bundle_sha256: str
    excluded_hard_fail_count: int
    warning_reason_counts: dict[str, int]


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _resolve_demo_waveform(row: dict[str, str], project_root: Path) -> Path:
    source_samples = Path(row["source_samples_csv"])
    if not source_samples.is_absolute() or "demo" not in source_samples.parts:
        raise AssertionError("Demo source_samples_csv must be an absolute demo path")
    relative = PureWindowsPath(row["processed_B1024_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise AssertionError("processed_B1024_file must be session-relative")
    path = source_samples.parent.joinpath(*relative.parts)
    if path.is_file():
        return path
    portable_path = (
        project_root
        / "data/MEPI/demo_waveforms"
        / row["source_session_id"]
    ).joinpath(*relative.parts)
    if portable_path.is_file():
        return portable_path
    raise FileNotFoundError(
        f"Missing dedicated demo B1024 file: {path}; portable copy: {portable_path}"
    )


def _load_b1024(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["index", "time_rel_s", "phase_fraction", "B_t"]:
            raise AssertionError(f"Unexpected B1024 schema: {path}")
        rows = list(reader)
    if len(rows) != 1024:
        raise AssertionError(f"B1024 must contain exactly 1024 rows: {path}")
    if [int(row["index"]) for row in rows] != list(range(1024)):
        raise AssertionError(f"B1024 index is not contiguous: {path}")
    waveform = np.asarray([float(row["B_t"]) for row in rows], dtype=np.float64)
    if waveform.shape != (1024,) or not np.isfinite(waveform).all():
        raise ValueError(f"B1024 contains invalid values: {path}")
    return waveform


def _load_screening_inputs(project_root: Path) -> ScreeningInputBundle:
    input_path = project_root / SCREENING_INPUT_RELATIVE_PATH
    if not input_path.is_file():
        raise FileNotFoundError(f"Dedicated screening manifest is missing: {input_path}")
    input_sha256 = sha256_file(input_path)
    if input_sha256 != SCREENING_INPUT_SHA256:
        raise AssertionError("Dedicated screening manifest SHA256 mismatch")
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 150:
        raise AssertionError("Dedicated screening manifest must contain 150 demo rows")
    if {row["dataset_mode"] for row in source_rows} != {"demo"}:
        raise AssertionError("Screening input must contain demo rows only")
    if "split" in (source_rows[0].keys() if source_rows else ()):
        raise AssertionError("Screening input must not contain model-development splits")

    selected = [
        row
        for row in source_rows
        if row["qc_status"] != "HARD_FAIL" and row["qc_pass"] == "1"
    ]
    excluded_hard_fail_count = sum(
        row["qc_status"] == "HARD_FAIL" for row in source_rows
    )
    if len(selected) != 146 or excluded_hard_fail_count != 4:
        raise AssertionError("Expected 146 usable demo rows and four HARD_FAIL exclusions")
    if {int(float(row["frequency_set_hz"])) for row in selected} != set(
        EXPECTED_CANDIDATE_FREQUENCIES_HZ
    ):
        raise AssertionError("Predefined demo candidate-frequency list changed")
    if {float(row["vin_set_group_v"]) for row in selected} != {3.9}:
        raise AssertionError("Dedicated demo is no longer the measured 3.9-V screening sweep")

    projected_rows: list[dict[str, Any]] = []
    waveforms: list[np.ndarray] = []
    waveform_digest = hashlib.sha256()
    waveform_digest.update(f"{input_sha256}\n".encode())
    warning_counts: Counter[str] = Counter()
    for row in sorted(
        selected,
        key=lambda value: (
            float(value["frequency_set_hz"]),
            value["core_id"],
            int(value["repeat_id"]),
        ),
    ):
        if row["B1024_source"] != "raw Scope #2 CH1 primary Vin":
            raise AssertionError("Demo B1024 provenance changed")
        feature_values = {
            name: float(row[name]) for name in FINETUNE_TABULAR_FEATURES
        }
        if not all(math.isfinite(value) for value in feature_values.values()):
            raise ValueError(f"Non-finite screening feature: {row['sample_id']}")
        if (
            feature_values["frequency_hz"] <= 0
            or feature_values["vin_rms_v"] <= 0
            or feature_values["B_peak_t"] <= 0
        ):
            raise ValueError(f"Non-positive screening input: {row['sample_id']}")
        waveform_path = _resolve_demo_waveform(row, project_root)
        waveform = _load_b1024(waveform_path)
        waveform_sha256 = sha256_file(waveform_path)
        waveform_digest.update(f"{row['sample_id']}\0{waveform_sha256}\n".encode())
        waveforms.append(waveform)
        warning_reasons = row["qc_warning_reasons"]
        for reason in filter(None, warning_reasons.split(";")):
            warning_counts[reason] += 1
        projected_rows.append(
            {
                "sample_id": row["sample_id"],
                "condition_group_id": row["condition_group_id"],
                "core_id": row["core_id"],
                "repeat_id": int(row["repeat_id"]),
                "candidate_frequency_hz": float(row["frequency_set_hz"]),
                **feature_values,
                "qc_status": row["qc_status"],
                "qc_warning_reasons": warning_reasons,
                "source_session_id": row["source_session_id"],
                "waveform_sha256": waveform_sha256,
            }
        )
    waveform_array = np.stack(waveforms)
    if waveform_array.shape != (146, 1024):
        raise AssertionError(f"Unexpected screening waveform shape: {waveform_array.shape}")
    return ScreeningInputBundle(
        rows=tuple(projected_rows),
        waveforms=waveform_array,
        input_path=input_path,
        input_sha256=input_sha256,
        waveform_bundle_sha256=waveform_digest.hexdigest(),
        excluded_hard_fail_count=excluded_hard_fail_count,
        warning_reason_counts=dict(sorted(warning_counts.items())),
    )


def _load_frozen_runtime_evidence(
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    manifest_path = project_root / MANIFEST_RELATIVE_PATH
    manifest = validate_final_model_manifest(project_root, manifest_path)
    config_path = project_root / manifest["source_model_config_path"]
    if sha256_file(config_path) != manifest["source_model_config_sha256"]:
        raise AssertionError("Frozen source-model configuration SHA256 mismatch")
    config = load_config(config_path)
    normalization_path = resolve_path(config, config["dataset"]["normalization"])
    if sha256_file(normalization_path) != config["dataset"]["normalization_sha256"]:
        raise AssertionError("Frozen train-only normalization SHA256 mismatch")
    normalization = _read_json_object(normalization_path, "train-only normalization")
    if (
        normalization.get("status") != "FROZEN_TRAIN_ONLY"
        or normalization.get("fitted_split") != "train"
        or normalization.get("validation_refit") is not False
        or normalization.get("test_refit") is not False
    ):
        raise AssertionError("Normalization is not the frozen train-only artifact")
    if set(normalization.get("feature_scalers", {})) != set(FINETUNE_TABULAR_FEATURES):
        raise AssertionError("Frozen nine-feature scaler schema changed")
    checkpoint_path = project_root / manifest["checkpoint_path"]
    if sha256_file(checkpoint_path) != SELECTED_CHECKPOINT_SHA256:
        raise AssertionError("Frozen selected-checkpoint SHA256 mismatch")
    return manifest, config, normalization, config_path, checkpoint_path


def _load_frozen_model(
    config_path: Path, checkpoint_path: Path
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    model, model_audit = construct_v1_5_model(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        raise AssertionError("Frozen selected checkpoint lacks model_state")
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError("Frozen selected checkpoint did not load strictly")
    model.eval()
    if model.training:
        raise AssertionError("Frequency-screening model must remain in eval mode")
    return model, model_audit, _state_sha256(model)


def audit_frequency_screening_inputs(project_root: str | Path) -> dict[str, Any]:
    """Run all model/input provenance checks without running inference."""

    root = Path(project_root).resolve()
    manifest, config, normalization, config_path, checkpoint_path = (
        _load_frozen_runtime_evidence(root)
    )
    bundle = _load_screening_inputs(root)
    model, model_audit, model_state_sha256 = _load_frozen_model(
        config_path, checkpoint_path
    )
    del model
    frequencies = [float(row["frequency_hz"]) for row in bundle.rows]
    return {
        "status": "PASS",
        "FREQUENCY_SCREENING_RUNTIME_READY": True,
        "FINAL_MODEL_READY": True,
        "SELECTED_CHECKPOINT_SHA256_MATCH": True,
        "FREQUENCY_SCREENING_INPUT_READY": True,
        "screening_input_artifact": SCREENING_INPUT_RELATIVE_PATH.as_posix(),
        "screening_input_sha256": bundle.input_sha256,
        "screening_waveform_bundle_sha256": bundle.waveform_bundle_sha256,
        "screening_condition_count": len(bundle.rows),
        "excluded_hard_fail_count": bundle.excluded_hard_fail_count,
        "candidate_frequency_count": len(EXPECTED_CANDIDATE_FREQUENCIES_HZ),
        "candidate_frequency_range_hz": [
            min(EXPECTED_CANDIDATE_FREQUENCIES_HZ),
            max(EXPECTED_CANDIDATE_FREQUENCIES_HZ),
        ],
        "measured_frequency_range_hz": [min(frequencies), max(frequencies)],
        "warning_reason_counts": bundle.warning_reason_counts,
        "feature_order": list(FINETUNE_TABULAR_FEATURES),
        "waveform_shape": [len(bundle.rows), 1024],
        "normalization_sha256": config["dataset"]["normalization_sha256"],
        "normalization_fitted_split": normalization["fitted_split"],
        "final_model_manifest_sha256": sha256_file(root / MANIFEST_RELATIVE_PATH),
        "protocol_sha256": manifest["protocol_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "model_parameter_count": model_audit["parameter_count"],
        "model_state_sha256": model_state_sha256,
        "TRAINING_PERFORMED": False,
        "TEST_DATA_USED_FOR_SCREENING": False,
        "TEST_TUNING": False,
        "FINAL_TEST_RERUN": False,
        "TEST_EVALUATION_COUNT": manifest["test_evaluation_count"],
    }


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _inverse_normalize(
    value: torch.Tensor, normalization: dict[str, Any], target: str
) -> torch.Tensor:
    scaler = normalization["target_scalers"][target]
    return value * float(scaler["scale"][0]) + float(scaler["mean"][0])


def _frequency_summary(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[float(row["candidate_frequency_hz"])].append(row)
    summary = []
    for frequency_hz in sorted(grouped):
        members = grouped[frequency_hz]
        summary.append(
            {
                "frequency_hz": frequency_hz,
                "n_samples": len(members),
                "mean_predicted_efficiency": statistics.fmean(
                    row["predicted_efficiency_percent"] for row in members
                ),
                "mean_predicted_P_loss": statistics.fmean(
                    row["predicted_P_loss"] for row in members
                ),
                "mean_predicted_LSP": statistics.fmean(
                    row["predicted_LSP_raw"] for row in members
                ),
                "mean_predicted_LSP_std": statistics.fmean(
                    row["predicted_LSP_std_raw"] for row in members
                ),
            }
        )
    return summary


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MEPI v1.5 frequency-screening demonstration",
        "",
        "Status: **INFERENCE_COMPLETE**",
        "",
        "This is an offline proof-of-concept screening demonstration within the measured demo domain. It is not an automated hardware sweep, an independently validated operating-frequency recommendation, or a new optimization study. No single optimal frequency is declared.",
        "",
        "## Provenance and isolation",
        "",
        "```text",
        f"PROTOCOL_SHA256 = {payload['protocol_sha256']}",
        f"FINAL_MODEL_MANIFEST_SHA256 = {payload['final_model_manifest_sha256']}",
        f"SELECTED_CHECKPOINT_SHA256 = {payload['selected_checkpoint_sha256']}",
        f"SCREENING_INPUT_ARTIFACT = {payload['screening_input_artifact']}",
        f"SCREENING_INPUT_SHA256 = {payload['screening_input_sha256']}",
        f"SCREENING_WAVEFORM_BUNDLE_SHA256 = {payload['screening_waveform_bundle_sha256']}",
        f"SCREENING_CONDITION_COUNT = {payload['screening_condition_count']}",
        f"TRAINING_PERFORMED = {str(payload['TRAINING_PERFORMED']).upper()}",
        f"TEST_DATA_USED_FOR_SCREENING = {str(payload['TEST_DATA_USED_FOR_SCREENING']).upper()}",
        f"TEST_TUNING = {str(payload['TEST_TUNING']).upper()}",
        f"FINAL_TEST_RERUN = {str(payload['FINAL_TEST_RERUN']).upper()}",
        f"TEST_EVALUATION_COUNT = {payload['TEST_EVALUATION_COUNT']}",
        f"ANY_NAN = {str(payload['ANY_NAN']).upper()}",
        f"ANY_INF = {str(payload['ANY_INF']).upper()}",
        "```",
        "",
        "The four source rows with existing `VOUT_THD` hard failures were excluded. Remaining source QC warnings are retained in the per-condition evidence and do not constitute target labels.",
        "",
        "## Descriptive frequency-level prediction summary",
        "",
        "| Frequency (Hz) | n | Mean predicted efficiency (%) | Mean predicted P_loss | Mean predicted LSP | Mean predicted LSP std |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["frequency_summary"]:
        lines.append(
            f"| {row['frequency_hz']:.0f} | {row['n_samples']} | "
            f"{row['mean_predicted_efficiency']:.8g} | "
            f"{row['mean_predicted_P_loss']:.8g} | "
            f"{row['mean_predicted_LSP']:.8g} | "
            f"{row['mean_predicted_LSP_std']:.8g} |"
        )
    lines.extend(
        [
            "",
            "The table is descriptive only. It exposes the predicted tradeoff across the predefined measured frequencies without adding a weighted score, tuning a threshold, or declaring an optimum.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_reports(project_root: Path, payload: dict[str, Any]) -> None:
    predictions_path = project_root / PREDICTIONS_CSV_RELATIVE_PATH
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=predictions_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        writer.writerows(payload["predictions"])
        temporary = Path(handle.name)
    os.replace(temporary, predictions_path)
    _atomic_text(
        project_root / REPORT_JSON_RELATIVE_PATH,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(
        project_root / REPORT_MARKDOWN_RELATIVE_PATH,
        _render_markdown(payload),
    )


def run_frequency_screening(
    project_root: str | Path, *, authorize: bool, batch_size: int = 64
) -> dict[str, Any]:
    """Run the authorized inference-only demonstration and write evidence."""

    if authorize is not True:
        raise RuntimeError("Frequency screening remains locked unless explicitly authorized")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(project_root).resolve()
    manifest, config, normalization, config_path, checkpoint_path = (
        _load_frozen_runtime_evidence(root)
    )
    bundle = _load_screening_inputs(root)
    final_result_path = root / manifest["final_test_result_path"]
    final_result_sha256_before = sha256_file(final_result_path)

    model, model_audit, state_sha256_before = _load_frozen_model(
        config_path, checkpoint_path
    )

    feature_matrix = np.asarray(
        [
            [
                (
                    float(row[name])
                    - float(normalization["feature_scalers"][name]["mean"][0])
                )
                / float(normalization["feature_scalers"][name]["scale"][0])
                for name in FINETUNE_TABULAR_FEATURES
            ]
            for row in bundle.rows
        ],
        dtype=np.float32,
    )
    waveforms = (
        (bundle.waveforms - WAVEFORM_MEAN) / WAVEFORM_SCALE
    ).astype(np.float32)[:, None, :]
    if feature_matrix.shape != (146, 9) or waveforms.shape != (146, 1, 1024):
        raise AssertionError("Frozen screening tensor shapes changed")
    if not np.isfinite(feature_matrix).all() or not np.isfinite(waveforms).all():
        raise ValueError("Screening tensors contain NaN or Inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prior = load_steinmetz_prior(config)
    output_batches: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(bundle.rows), batch_size):
            stop = min(start + batch_size, len(bundle.rows))
            waveform = torch.from_numpy(waveforms[start:stop]).to(device)
            tabular = torch.from_numpy(feature_matrix[start:stop]).to(device)
            frequency_hz = torch.tensor(
                [row["frequency_hz"] for row in bundle.rows[start:stop]],
                dtype=torch.float32,
                device=device,
            )
            b_peak_t = torch.tensor(
                [row["B_peak_t"] for row in bundle.rows[start:stop]],
                dtype=torch.float32,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                p_st_z = steinmetz_reference_z(frequency_hz, b_peak_t, prior)
                outputs = model(waveform, tabular, p_st_z)
            for key in ("efficiency_z", "P_loss_z", "mu_LSP_z", "var_LSP_z"):
                output_batches[key].append(outputs[key].float().cpu())

    outputs = {key: torch.cat(values) for key, values in output_batches.items()}
    predicted_efficiency = _inverse_normalize(
        outputs["efficiency_z"], normalization, "efficiency_percent"
    )
    predicted_p_loss = _inverse_normalize(outputs["P_loss_z"], normalization, "P_loss")
    predicted_lsp = _inverse_normalize(outputs["mu_LSP_z"], normalization, "LSP_raw")
    lsp_scale = float(normalization["target_scalers"]["LSP_raw"]["scale"][0])
    predicted_lsp_std = torch.sqrt(outputs["var_LSP_z"].double()) * lsp_scale
    numerical = torch.cat(
        [
            predicted_efficiency.double(),
            predicted_p_loss.double(),
            predicted_lsp.double(),
            predicted_lsp_std.double(),
        ]
    )
    any_nan = bool(torch.isnan(numerical).any())
    any_inf = bool(torch.isinf(numerical).any())
    if any_nan or any_inf:
        raise ValueError("Frequency-screening prediction contains NaN or Inf")

    predictions = []
    for index, row in enumerate(bundle.rows):
        predictions.append(
            {
                key: value
                for key, value in {
                    **{name: row[name] for name in PREDICTION_COLUMNS[:7]},
                    "predicted_efficiency_percent": float(predicted_efficiency[index]),
                    "predicted_P_loss": float(predicted_p_loss[index]),
                    "predicted_LSP_raw": float(predicted_lsp[index]),
                    "predicted_LSP_std_raw": float(predicted_lsp_std[index]),
                    "qc_status": row["qc_status"],
                    "qc_warning_reasons": row["qc_warning_reasons"],
                }.items()
                if key in PREDICTION_COLUMNS
            }
        )
    state_sha256_after = _state_sha256(model)
    if state_sha256_after != state_sha256_before:
        raise AssertionError("Frozen model state changed during screening")

    payload = {
        "status": "INFERENCE_COMPLETE",
        "protocol_version": manifest["protocol_version"],
        "protocol_sha256": manifest["protocol_sha256"],
        "final_model_manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "final_model_manifest_sha256": sha256_file(root / MANIFEST_RELATIVE_PATH),
        "selected_configuration": manifest["configuration_id"],
        "selected_checkpoint": manifest["checkpoint_path"],
        "selected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "selected_checkpoint_sha256_match": True,
        "model_parameter_count": model_audit["parameter_count"],
        "model_state_sha256_before": state_sha256_before,
        "model_state_sha256_after": state_sha256_after,
        "screening_input_artifact": SCREENING_INPUT_RELATIVE_PATH.as_posix(),
        "screening_input_sha256": bundle.input_sha256,
        "screening_waveform_bundle_sha256": bundle.waveform_bundle_sha256,
        "screening_condition_count": len(bundle.rows),
        "excluded_hard_fail_count": bundle.excluded_hard_fail_count,
        "frequency_range_hz": [
            min(row["frequency_hz"] for row in bundle.rows),
            max(row["frequency_hz"] for row in bundle.rows),
        ],
        "candidate_frequency_range_hz": [
            min(EXPECTED_CANDIDATE_FREQUENCIES_HZ),
            max(EXPECTED_CANDIDATE_FREQUENCIES_HZ),
        ],
        "prediction_columns": list(PREDICTION_COLUMNS),
        "warning_reason_counts": bundle.warning_reason_counts,
        "device": str(device),
        "ANY_NAN": any_nan,
        "ANY_INF": any_inf,
        "TRAINING_PERFORMED": False,
        "SCIENTIFIC_TRAINING_RUN": False,
        "TEST_DATA_USED_FOR_SCREENING": False,
        "TEST_TUNING": False,
        "FINAL_TEST_RERUN": False,
        "TEST_EVALUATION_COUNT": manifest["test_evaluation_count"],
        "claim_boundary": "OFFLINE_PROOF_OF_CONCEPT_NO_OPTIMUM_DECLARED",
        "predictions": predictions,
        "frequency_summary": _frequency_summary(predictions),
    }
    _write_reports(root, payload)
    if sha256_file(final_result_path) != final_result_sha256_before:
        raise AssertionError("Final-test result ledger changed during screening")
    return payload
