"""Fail-closed final evaluation for the frozen MEPI v1.5 selected model.

The preparation audit never constructs a test dataset or test loader. The test
path is available only through :func:`run_final_test`, which records the single
authorized access before reading any test row. This module contains no training
or checkpoint-selection entry point.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .config import load_config, resolve_path
from .constants import FINETUNE_TABULAR_FEATURES
from .final_model_manifest_v1_5 import write_final_model_manifest
from .finetune_v1_4 import (
    VARIANCE_FLOOR,
    WAVEFORM_MEAN,
    WAVEFORM_SCALE,
    compute_v1_4_losses,
    load_steinmetz_prior,
    sha256_file,
    steinmetz_reference_z,
)
from .finetune_v1_5 import construct_v1_5_model

FINAL_CONFIG_SHA256 = "4b18d275be0a56f09371f2fd91d3fe03bba72a7cee6ca732014fbbf043dcdf96"
PROTOCOL_SHA256 = "0f68d6ebd5d16177e5238471639f861dfa91522130ac6a4a3ac48634b9af1e39"
SELECTION_RULE_SHA256 = "8c4eda9035e7e5e4f75a4b0ccde643c8946fd3b990575ec0323977cb8497a5be"
GRID_RESULT_SHA256 = "908c872450e0811ec2e9cc8ca2e193925f9c302100e413e51d66f80d3419245f"
SELECTED_CHECKPOINT_SHA256 = "0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33"
SELECTED_CONFIGURATION_ID = "l3_1_l4_0p05"
SELECTED_WEIGHTS = {"lambda1": 1.0, "lambda2": 1.0, "lambda3": 1.0, "lambda4": 0.05}
EXPECTED_TEST_ROWS = 90
EXPECTED_TEST_GROUPS = 9


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _project_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, config[key])


def load_final_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if sha256_file(path) != FINAL_CONFIG_SHA256:
        raise AssertionError("Frozen final-model configuration SHA256 mismatch")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    if config.get("protocol_version") != "MEPI-FROZEN-PROTOCOL v1.5":
        raise AssertionError("Final configuration protocol changed")
    if config.get("protocol_sha256") != PROTOCOL_SHA256:
        raise AssertionError("Final configuration protocol SHA256 changed")
    selection = config.get("selection", {})
    if (
        selection.get("selected_by") != "VAL_TASK_SCORE"
        or selection.get("formula") != "validation_L_electrical + validation_L_LSP"
        or selection.get("direction") != "minimize"
        or selection.get("weighted_validation_total_loss_used_for_cross_configuration_selection") is not False
        or selection.get("selected_configuration_id") != SELECTED_CONFIGURATION_ID
    ):
        raise AssertionError("Frozen selection rule or selected configuration changed")
    if config.get("loss_weights") != SELECTED_WEIGHTS:
        raise AssertionError("Frozen selected lambda values changed")
    completed = config.get("completed_run", {})
    if (
        completed.get("best_epoch") != 91
        or completed.get("epoch_indexing") != "zero_based"
        or completed.get("completed_epochs") != 100
        or completed.get("max_epochs") != 100
        or completed.get("patience") != 10
        or completed.get("seed") != 42
    ):
        raise AssertionError("Frozen completed-run metadata changed")
    evaluation = config.get("final_evaluation", {})
    forbidden = (
        evaluation.get("training_allowed"),
        evaluation.get("retraining_allowed"),
        evaluation.get("checkpoint_selection_allowed"),
        evaluation.get("lambda_modification_allowed"),
    )
    if forbidden != (False, False, False, False):
        raise AssertionError("Final evaluation must prohibit training, reselection, and lambda changes")
    if (
        evaluation.get("run_final_test_default") is not False
        or evaluation.get("exactly_once") is not True
        or evaluation.get("expected_test_rows") != EXPECTED_TEST_ROWS
        or evaluation.get("expected_test_groups") != EXPECTED_TEST_GROUPS
    ):
        raise AssertionError("Final-test gate or expected frozen test size changed")
    status = config.get("status", {})
    if status != {
        "HYPERPARAMETER_SELECTION_COMPLETE": True,
        "FINAL_CONFIG_FROZEN": True,
        "TEST_ACCESSED": False,
        "TEST_EVALUATION_COUNT": 0,
    }:
        raise AssertionError("Frozen pre-test status changed")
    return config


def _validate_grid_selection(config: dict[str, Any]) -> dict[str, Any]:
    selection = config["selection"]
    rule_path = resolve_path(config, selection["rule_artifact"])
    grid_path = resolve_path(config, selection["complete_grid_result_artifact"])
    if sha256_file(rule_path) != SELECTION_RULE_SHA256:
        raise AssertionError("Selection-rule SHA256 mismatch")
    if sha256_file(grid_path) != GRID_RESULT_SHA256:
        raise AssertionError("Complete grid-result SHA256 mismatch")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    rows = grid.get("comparison_table", [])
    if len(rows) != 12 or grid.get("configuration_count") != 12:
        raise AssertionError("Completed grid must contain exactly 12 configurations")
    if any(not row.get("eligible_for_selection") or row.get("numerical_failure") for row in rows):
        raise AssertionError("Every completed grid configuration must be numerically eligible")
    for row in rows:
        recomputed = float(row["validation_L_electrical"]) + float(row["validation_L_LSP"])
        if not math.isclose(recomputed, float(row["VAL_TASK_SCORE"]), rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"VAL_TASK_SCORE mismatch for {row['configuration_id']}")
    eligible = sorted(rows, key=lambda row: float(row["VAL_TASK_SCORE"]))
    selected = eligible[0]
    expected_selection = {
        "winner_declared": True,
        "selected_configuration_id": SELECTED_CONFIGURATION_ID,
        "tied_configuration_ids": [SELECTED_CONFIGURATION_ID],
        "minimum_VAL_TASK_SCORE": 0.2023820665829322,
        "reason": "UNIQUE_MINIMUM_VAL_TASK_SCORE",
    }
    if grid.get("selection") != expected_selection:
        raise AssertionError("Completed grid does not record the unique frozen minimum")
    if grid.get("weighted_validation_total_loss_used_for_cross_configuration_ranking") is not False:
        raise AssertionError("Weighted validation total loss was improperly used for grid ranking")
    expected_selected = {
        **SELECTED_WEIGHTS,
        "configuration_id": SELECTED_CONFIGURATION_ID,
        "best_epoch": 91,
        "epochs_completed": 100,
        "validation_L_electrical": 0.0504307309494299,
        "validation_L_LSP": 0.1519513356335023,
        "VAL_TASK_SCORE": 0.2023820665829322,
    }
    for key, expected in expected_selected.items():
        if selected.get(key) != expected:
            raise AssertionError(f"Selected grid field changed: {key}")
    if selected.get("test_accessed") is not False:
        raise AssertionError("Test was accessed during hyperparameter selection")
    return selected


def _audit_and_load(
    project_root: str | Path, final_config_path: str | Path
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    root = Path(project_root).resolve()
    config = load_final_config(final_config_path)
    if root != resolve_path(config, "."):
        raise AssertionError("Final config project_root does not match the active project")
    hash_checks = {
        "FINAL_CONFIG_SHA256": sha256_file(Path(final_config_path).resolve()) == FINAL_CONFIG_SHA256,
        "PROTOCOL_SHA256": sha256_file(_project_path(config, "protocol_artifact")) == PROTOCOL_SHA256,
        "PROTOCOL_DOC_COPY_SHA256": sha256_file(root / "docs/MEPI_FROZEN_PROTOCOL_v1.5.md") == PROTOCOL_SHA256,
        "SOURCE_MODEL_CONFIG_SHA256": sha256_file(_project_path(config, "source_model_config")) == config["source_model_config_sha256"],
        "SENSITIVITY_CONFIG_SHA256": sha256_file(_project_path(config, "sensitivity_config")) == config["sensitivity_config_sha256"],
        "SELECTION_RULE_SHA256": sha256_file(resolve_path(config, config["selection"]["rule_artifact"])) == SELECTION_RULE_SHA256,
        "SELECTION_RULE_DOC_COPY_SHA256": sha256_file(root / "docs/MEPI_V1_5_LOSS_WEIGHT_SELECTION_RULE.md") == SELECTION_RULE_SHA256,
        "GRID_RESULT_SHA256": sha256_file(resolve_path(config, config["selection"]["complete_grid_result_artifact"])) == GRID_RESULT_SHA256,
        "SELECTED_CHECKPOINT_SHA256": sha256_file(resolve_path(config, config["completed_run"]["selected_checkpoint"])) == SELECTED_CHECKPOINT_SHA256,
    }
    if not all(hash_checks.values()):
        raise AssertionError(f"Frozen hash check failed: {hash_checks}")
    selected = _validate_grid_selection(config)
    checkpoint_path = resolve_path(config, config["completed_run"]["selected_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state", "optimizer_state", "amp_scaler_state", "rng_state", "compatibility",
        "epoch", "best_epoch", "best_validation_loss", "patience_counter", "history",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise AssertionError(f"Selected checkpoint is incomplete: {missing}")
    expected_compatibility = {
        "experiment": "MEPI_V1_5_LOSS_WEIGHT_SENSITIVITY",
        "protocol_version": "MEPI-FROZEN-PROTOCOL v1.5",
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": "fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c",
        "optimizer": "AdamW",
        "learning_rate": 5e-6,
        "weight_decay": 1e-4,
        "scheduler": None,
        **SELECTED_WEIGHTS,
        "seed": 42,
        "fresh_representation_initialization": True,
    }
    if checkpoint.get("compatibility") != expected_compatibility:
        raise AssertionError("Selected checkpoint compatibility metadata changed")
    if checkpoint.get("best_epoch") != 91 or checkpoint.get("epoch") != 92:
        raise AssertionError("Selected best-checkpoint epoch metadata changed")
    source_config_path = _project_path(config, "source_model_config")
    model, model_audit = construct_v1_5_model(source_config_path)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError("Selected checkpoint did not load strictly")
    loaded = model.state_dict()
    if loaded.keys() != checkpoint["model_state"].keys() or any(
        not torch.equal(loaded[key].cpu(), checkpoint["model_state"][key].cpu()) for key in loaded
    ):
        raise AssertionError("Loaded model state is not exactly equal to the selected checkpoint")
    model.eval()
    results_path = resolve_path(config, config["final_evaluation"]["results_json"])
    evidence = json.loads(results_path.read_text(encoding="utf-8"))
    if evidence.get("TEST_ACCESSED") is not False or evidence.get("TEST_EVALUATION_COUNT") != 0:
        raise RuntimeError("Exactly-once test ledger is not in the unopened state")
    audit = {
        "status": "PASS",
        "hash_checks": hash_checks,
        "selected_configuration_id": selected["configuration_id"],
        "selected_lambda3": selected["lambda3"],
        "selected_lambda4": selected["lambda4"],
        "checkpoint_strict_load": True,
        "model_state_exact_match": True,
        "model_parameter_count": model_audit["parameter_count"],
        "training_performed": False,
        "test_dataset_created": False,
        "test_loader_created": False,
        "test_accessed": False,
        "test_evaluation_count": 0,
    }
    return audit, model, config


def audit_final_preparation(project_root: str | Path, final_config_path: str | Path) -> dict[str, Any]:
    """Verify the frozen final model without reading or instantiating test data."""
    audit, _model, _config = _audit_and_load(project_root, final_config_path)
    return audit


class FrozenV15FinalTestDataset(Dataset[dict[str, torch.Tensor]]):
    """The frozen test-only dataset; construct only after the access ledger is set."""

    def __init__(self, source_config_path: str | Path) -> None:
        self.config = load_config(source_config_path)
        dataset = self.config["dataset"]
        data_path = resolve_path(self.config, dataset["path"])
        with data_path.open(newline="", encoding="utf-8-sig") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
        if len(self.rows) != EXPECTED_TEST_ROWS:
            raise AssertionError(f"Expected {EXPECTED_TEST_ROWS} frozen test rows, got {len(self.rows)}")
        self.group_count = len({row["condition_group_id"] for row in self.rows})
        if self.group_count != EXPECTED_TEST_GROUPS:
            raise AssertionError(f"Expected {EXPECTED_TEST_GROUPS} frozen test groups, got {self.group_count}")
        all_ids = np.load(resolve_path(self.config, dataset["sample_ids_path"]), mmap_mode="r", allow_pickle=False)
        id_to_index = {str(sample_id): index for index, sample_id in enumerate(all_ids)}
        self.indices = [id_to_index[row["sample_id"]] for row in self.rows]
        self.waveforms = np.load(resolve_path(self.config, dataset["waveform_path"]), mmap_mode="r", allow_pickle=False)
        self.normalization = json.loads(resolve_path(self.config, dataset["normalization"]).read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        feature_values = []
        for name in FINETUNE_TABULAR_FEATURES:
            scaler = self.normalization["feature_scalers"][name]
            feature_values.append((float(row[name]) - float(scaler["mean"][0])) / float(scaler["scale"][0]))
        electrical_z = []
        for name in ("efficiency_percent", "P_loss"):
            scaler = self.normalization["target_scalers"][name]
            electrical_z.append((float(row[name]) - float(scaler["mean"][0])) / float(scaler["scale"][0]))
        waveform = np.array(self.waveforms[self.indices[index]], dtype=np.float64, copy=True)
        waveform = ((waveform - WAVEFORM_MEAN) / WAVEFORM_SCALE).astype(np.float32)
        return {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "tabular": torch.tensor(feature_values, dtype=torch.float32),
            "frequency_hz": torch.tensor(float(row["frequency_hz"]), dtype=torch.float32),
            "B_peak_t": torch.tensor(float(row["B_peak_t"]), dtype=torch.float32),
            "electrical_z": torch.tensor(electrical_z, dtype=torch.float32),
            "LSP_z": torch.tensor(float(row["LSP_z"]), dtype=torch.float32),
        }


def _regression_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target, prediction = target.double(), prediction.double()
    error = prediction - target
    denominator = (target - target.mean()).square().sum()
    if denominator == 0:
        raise ValueError("R2 is undefined for a constant test target")
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "r2": float(1.0 - error.square().sum() / denominator),
    }


def _evaluate_test(
    model: torch.nn.Module, dataset: FrozenV15FinalTestDataset, source_config_path: Path
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    prior = load_steinmetz_prior(load_config(source_config_path))
    predicted = {key: [] for key in ("efficiency_z", "P_loss_z", "mu_LSP_z", "var_LSP_z")}
    observed = {key: [] for key in ("efficiency_z", "P_loss_z", "LSP_z")}
    losses: dict[str, float] = {}
    count = 0
    any_nan = False
    any_inf = False
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p_st_z = steinmetz_reference_z(batch["frequency_hz"], batch["B_peak_t"], prior)
                outputs = model(batch["waveform"], batch["tabular"], p_st_z)
                batch_losses = compute_v1_4_losses(outputs, batch, lambda3=1.0, lambda4=0.05)
            tensors = [*batch.values(), *outputs.values(), *batch_losses.values()]
            any_nan |= any(bool(torch.isnan(value).any()) for value in tensors)
            any_inf |= any(bool(torch.isinf(value).any()) for value in tensors)
            size = len(batch["LSP_z"])
            count += size
            for key, value in batch_losses.items():
                losses[key] = losses.get(key, 0.0) + float(value.detach().cpu()) * size
            for key in predicted:
                predicted[key].append(outputs[key].float().cpu())
            observed["efficiency_z"].append(batch["electrical_z"][:, 0].float().cpu())
            observed["P_loss_z"].append(batch["electrical_z"][:, 1].float().cpu())
            observed["LSP_z"].append(batch["LSP_z"].float().cpu())
    if count != EXPECTED_TEST_ROWS:
        raise AssertionError("Final test loader did not evaluate exactly 90 rows")
    predicted = {key: torch.cat(value) for key, value in predicted.items()}
    observed = {key: torch.cat(value) for key, value in observed.items()}
    losses = {key: value / count for key, value in losses.items()}
    scalers = dataset.normalization["target_scalers"]

    def inverse(value: torch.Tensor, name: str) -> torch.Tensor:
        return value * float(scalers[name]["scale"][0]) + float(scalers[name]["mean"][0])

    efficiency_true = inverse(observed["efficiency_z"], "efficiency_percent")
    efficiency_pred = inverse(predicted["efficiency_z"], "efficiency_percent")
    p_loss_true = inverse(observed["P_loss_z"], "P_loss")
    p_loss_pred = inverse(predicted["P_loss_z"], "P_loss")
    lsp_true = inverse(observed["LSP_z"], "LSP_raw")
    lsp_pred = inverse(predicted["mu_LSP_z"], "LSP_raw")
    zero_target_count = int((lsp_true == 0).sum())
    if zero_target_count:
        raise ValueError("LSP MAPE is undefined because the frozen test set contains a zero target")
    efficiency = _regression_metrics(efficiency_true, efficiency_pred)
    p_loss = _regression_metrics(p_loss_true, p_loss_pred)
    lsp = _regression_metrics(lsp_true, lsp_pred)
    lsp["mape_percent"] = float(((lsp_pred - lsp_true).abs() / lsp_true.abs()).mean() * 100.0)
    var_z = predicted["var_LSP_z"].double()
    lsp_scale = float(scalers["LSP_raw"]["scale"][0])
    std_raw = torch.sqrt(var_z * (lsp_scale**2))
    numerical_values = [*efficiency.values(), *p_loss.values(), *lsp.values(), *losses.values(), *var_z.tolist(), *std_raw.tolist()]
    numerical_failure = any_nan or any_inf or not all(math.isfinite(float(value)) for value in numerical_values)
    return {
        "device": str(device),
        "TEST_ROWS": count,
        "TEST_GROUPS": dataset.group_count,
        "TEST_EFFICIENCY_MAE": efficiency["mae"],
        "TEST_EFFICIENCY_RMSE": efficiency["rmse"],
        "TEST_EFFICIENCY_R2": efficiency["r2"],
        "TEST_PLOSS_MAE": p_loss["mae"],
        "TEST_PLOSS_RMSE": p_loss["rmse"],
        "TEST_PLOSS_R2": p_loss["r2"],
        "TEST_LSP_MAE": lsp["mae"],
        "TEST_LSP_RMSE": lsp["rmse"],
        "TEST_LSP_R2": lsp["r2"],
        "TEST_LSP_MAPE_PERCENT": lsp["mape_percent"],
        "TEST_LSP_GAUSSIAN_NLL_NORMALIZED": losses["uncertainty_nll"],
        "TEST_MIN_PREDICTED_VAR_LSP_Z": float(var_z.min()),
        "TEST_MEDIAN_PREDICTED_VAR_LSP_Z": float(var_z.median()),
        "TEST_MAX_PREDICTED_VAR_LSP_Z": float(var_z.max()),
        "TEST_VARIANCE_FLOOR": VARIANCE_FLOOR,
        "TEST_VARIANCE_FLOOR_HIT": bool((var_z <= VARIANCE_FLOOR).any()),
        "TEST_MIN_PREDICTED_STD_LSP_RAW": float(std_raw.min()),
        "TEST_MEDIAN_PREDICTED_STD_LSP_RAW": float(std_raw.median()),
        "TEST_MAX_PREDICTED_STD_LSP_RAW": float(std_raw.max()),
        "NUMERICAL_FAILURE": numerical_failure,
        "ANY_NAN": any_nan,
        "ANY_INF": any_inf,
    }


def _report_markdown(payload: dict[str, Any]) -> str:
    def display(key: str) -> str:
        value = payload.get(key)
        return "MISSING" if value is None else str(value).upper() if isinstance(value, bool) else str(value)

    required = [
        "FINAL_CONFIG_FROZEN", "SELECTED_LAMBDA1", "SELECTED_LAMBDA2", "SELECTED_LAMBDA3", "SELECTED_LAMBDA4",
        "SELECTED_CHECKPOINT", "SELECTED_CHECKPOINT_SHA256", "TEST_ROWS", "TEST_GROUPS",
        "TEST_EFFICIENCY_MAE", "TEST_EFFICIENCY_RMSE", "TEST_EFFICIENCY_R2",
        "TEST_PLOSS_MAE", "TEST_PLOSS_RMSE", "TEST_PLOSS_R2",
        "TEST_LSP_MAE", "TEST_LSP_RMSE", "TEST_LSP_R2", "TEST_LSP_MAPE_PERCENT",
        "NUMERICAL_FAILURE", "TEST_ACCESSED", "TEST_EVALUATION_COUNT",
    ]
    lines = [
        "# MEPI v1.5 final test results",
        "",
        f"Status: **{payload['EVALUATION_STATUS']}**",
        "",
        "This file is the exactly-once final-test evidence ledger. `MISSING` means the frozen test has not been accessed; it is not an estimated result.",
        "",
        "```text",
    ]
    lines.extend(f"{key} = {display(key)}" for key in required)
    lines.extend(["```", "", "## Frozen uncertainty diagnostics", "", "```text"])
    uncertainty = [
        "TEST_LSP_GAUSSIAN_NLL_NORMALIZED", "TEST_MIN_PREDICTED_VAR_LSP_Z",
        "TEST_MEDIAN_PREDICTED_VAR_LSP_Z", "TEST_MAX_PREDICTED_VAR_LSP_Z",
        "TEST_VARIANCE_FLOOR", "TEST_VARIANCE_FLOOR_HIT",
        "TEST_MIN_PREDICTED_STD_LSP_RAW", "TEST_MEDIAN_PREDICTED_STD_LSP_RAW",
        "TEST_MAX_PREDICTED_STD_LSP_RAW", "ANY_NAN", "ANY_INF",
    ]
    lines.extend(f"{key} = {display(key)}" for key in uncertainty)
    lines.extend([
        "```", "", "No uncertainty calibration or post-test tuning is performed. The normalized-space full Gaussian NLL and the frozen model's predicted variance/raw-space standard deviation summaries are reported directly.", "",
    ])
    return "\n".join(lines)


def _write_report_pair(config: dict[str, Any], payload: dict[str, Any]) -> None:
    json_path = resolve_path(config, config["final_evaluation"]["results_json"])
    markdown_path = resolve_path(config, config["final_evaluation"]["results_markdown"])
    _atomic_json(json_path, payload)
    _atomic_text(markdown_path, _report_markdown(payload))


def run_final_test(
    project_root: str | Path, final_config_path: str | Path, *, authorize: bool
) -> dict[str, Any]:
    """Perform the single final test after an explicit in-notebook authorization."""
    if authorize is not True:
        raise RuntimeError("Final test remains locked unless RUN_FINAL_TEST is exactly True")
    audit, model, config = _audit_and_load(project_root, final_config_path)
    results_path = resolve_path(config, config["final_evaluation"]["results_json"])
    ledger = json.loads(results_path.read_text(encoding="utf-8"))
    if ledger.get("TEST_ACCESSED") is not False or ledger.get("TEST_EVALUATION_COUNT") != 0:
        raise RuntimeError("Exactly-once final test is already consumed or in progress")
    ledger.update({
        "EVALUATION_STATUS": "TEST_ACCESS_IN_PROGRESS",
        "RUN_FINAL_TEST": True,
        "TEST_ACCESSED": True,
        "TEST_EVALUATION_COUNT": 1,
    })
    _write_report_pair(config, ledger)
    try:
        source_config_path = _project_path(config, "source_model_config")
        dataset = FrozenV15FinalTestDataset(source_config_path)
        metrics = _evaluate_test(model, dataset, source_config_path)
        result = {
            **ledger,
            **metrics,
            "EVALUATION_STATUS": "FINAL_TEST_COMPLETE",
            "PRE_TEST_AUDIT": audit,
            "TEST_ACCESSED": True,
            "TEST_EVALUATION_COUNT": 1,
        }
        _write_report_pair(config, result)
    except Exception as error:
        ledger.update({
            "EVALUATION_STATUS": "FAILED_AFTER_TEST_ACCESS",
            "FAILURE_TYPE": type(error).__name__,
            "FAILURE_MESSAGE": str(error),
            "NUMERICAL_FAILURE": None,
        })
        _write_report_pair(config, ledger)
        raise
    # Derive the handoff only after the exactly-once result ledger has been
    # durably completed. A manifest failure must never rewrite that scientific
    # result as a failed test evaluation.
    write_final_model_manifest(project_root)
    return result

