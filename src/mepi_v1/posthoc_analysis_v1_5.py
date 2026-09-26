"""Post-hoc reporting for the already-frozen MEPI v1.5 final test.

There is deliberately no training, optimizer, selection, or checkpoint-writing
entry point here. The exactly-once final-test ledger is read and hash-checked,
never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
from src.evaluation.metrics import regression_metrics
from src.mepi_v1.config import resolve_path
from src.mepi_v1.constants import FINETUNE_TABULAR_FEATURES
from src.mepi_v1.final_model_manifest_v1_5 import MANIFEST_RELATIVE_PATH, sha256_file
from src.mepi_v1.finetune_v1_4 import WAVEFORM_MEAN, WAVEFORM_SCALE, steinmetz_reference_z

ERR95_EPSILON = 1e-12
ERR95_SOURCE = "src/evaluation/metrics.py"


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = regression_metrics(target, prediction, near_zero=False)
    return {
        "mae": result["mae"],
        "rmse": result["rmse"],
        "r2": result["r2"],
        "err95_percent": result["err95"],
    }


def _predict_batch(
    engine: FrozenMEPIEngine,
    waveforms: np.ndarray,
    feature_rows: list[dict[str, float]],
) -> dict[str, np.ndarray]:
    tabular = torch.cat([engine._feature_tensor(row) for row in feature_rows], dim=0)
    waveform_z = ((waveforms.astype(np.float64) - WAVEFORM_MEAN) / WAVEFORM_SCALE).astype(np.float32)
    waveform_tensor = torch.from_numpy(waveform_z).unsqueeze(1).to(engine.device)
    frequency = torch.tensor([row["frequency_hz"] for row in feature_rows], dtype=torch.float32, device=engine.device)
    b_peak = torch.tensor([row["B_peak_t"] for row in feature_rows], dtype=torch.float32, device=engine.device)
    collected: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("efficiency_z", "P_loss_z", "mu_LSP_z", "var_LSP_z")
    }
    with torch.inference_mode():
        for start in range(0, len(feature_rows), 64):
            stop = min(start + 64, len(feature_rows))
            with torch.autocast(
                device_type=engine.device.type,
                dtype=torch.float16,
                enabled=engine.device.type == "cuda",
            ):
                batch = engine.model(
                    waveform_tensor[start:stop],
                    tabular[start:stop],
                    steinmetz_reference_z(frequency[start:stop], b_peak[start:stop], engine.prior),
                )
            for key in collected:
                collected[key].append(batch[key].float().cpu())
    outputs = {key: torch.cat(value) for key, value in collected.items()}
    scalers = engine.normalization["target_scalers"]

    def inverse(name: str, key: str) -> np.ndarray:
        value = outputs[key].double().numpy()
        return value * float(scalers[name]["scale"][0]) + float(scalers[name]["mean"][0])

    lsp_scale = float(scalers["LSP_raw"]["scale"][0])
    return {
        "efficiency": inverse("efficiency_percent", "efficiency_z"),
        "core_loss": inverse("P_loss", "P_loss_z"),
        "lsp": inverse("LSP_raw", "mu_LSP_z"),
        "lsp_sigma": torch.sqrt(outputs["var_LSP_z"].double()).numpy() * lsp_scale,
    }


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    panels = (
        ("efficiency_measured_percent", "efficiency_predicted_percent", "Efficiency (%)", "(a) Efficiency"),
        ("core_loss_measured_w", "core_loss_predicted_w", "Core loss (W)", "(b) Core loss"),
        ("lsp_measured", "lsp_predicted", "LSP (dimensionless)", "(c) LSP"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    for axis, (truth_key, pred_key, label, title) in zip(axes, panels):
        truth = np.asarray([row[truth_key] for row in rows])
        pred = np.asarray([row[pred_key] for row in rows])
        lower, upper = min(truth.min(), pred.min()), max(truth.max(), pred.max())
        padding = max((upper - lower) * 0.05, np.finfo(float).eps)
        axis.scatter(truth, pred, s=20, alpha=0.72, edgecolors="none")
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], "k--", lw=1)
        axis.set(xlabel=f"Measured {label}", ylabel=f"Predicted {label}", title=title)
        axis.grid(alpha=0.25)
    base = directory / "MEPI_final_test_measured_vs_predicted"
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    figure.savefig(png, dpi=300)
    figure.savefig(pdf)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    for axis, (truth_key, pred_key, label, title) in zip(axes, panels):
        truth = np.asarray([row[truth_key] for row in rows])
        residual = np.asarray([row[pred_key] for row in rows]) - truth
        axis.scatter(truth, residual, s=20, alpha=0.72, edgecolors="none")
        axis.axhline(0.0, color="black", linestyle="--", lw=1)
        axis.set(xlabel=f"Measured {label}", ylabel="Residual (prediction - measured)", title=title)
        axis.grid(alpha=0.25)
    residual = directory / "MEPI_final_test_residual_diagnostics.png"
    figure.savefig(residual, dpi=300)
    plt.close(figure)
    return [png, pdf, residual]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MEPI v1.5 post-hoc final-test metrics", "", "Status: **POST-HOC REPORTING ONLY**", "",
        "No training, model selection, hyperparameter change, checkpoint change, or uncertainty calibration was performed.",
        "", "## Guard state", "", "```text",
    ]
    for key in ("POSTHOC_ANALYSIS_ONLY", "TRAINING_PERFORMED", "MODEL_SELECTION_PERFORMED", "HYPERPARAMETER_CHANGED", "CHECKPOINT_CHANGED", "FINAL_TEST_LEDGER_MODIFIED"):
        lines.append(f"{key} = {str(payload[key]).upper()}")
    lines += ["```", "", "## Metrics", "", "| Target | MAE | RMSE | R2 | MAPE (%) | Err95 (%) |", "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("efficiency", "Efficiency (percentage points)"), ("core_loss", "Core loss (W)"), ("lsp", "LSP")):
        metric = payload["metrics"][key]
        mape = metric.get("mape_percent")
        mape_text = "—" if mape is None else f"{mape:.12g}"
        lines.append(f"| {label} | {metric['mae']:.12g} | {metric['rmse']:.12g} | {metric['r2']:.12g} | {mape_text} | {metric['err95_percent']:.12g} |")
    lines += [
        "", f"Err95 uses epsilon `{payload['err95']['epsilon']}` from `{payload['err95']['source']}`.",
        "", "Predicted LSP sigma is an uncalibrated model output; it is not a confidence interval or calibrated uncertainty.",
        "", "LSP is a dimensionless Arrhenius-informed relative thermal-stress proxy, not lifetime, RUL, time-to-failure, or measured degradation.",
    ]
    return "\n".join(lines) + "\n"


def run_posthoc_analysis(project_root: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    root = Path(project_root).resolve()
    ledger_path = root / "reports/MEPI_V1_5_FINAL_TEST_RESULTS.json"
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    if ledger.get("EVALUATION_STATUS") != "FINAL_TEST_COMPLETE" or ledger.get("TEST_EVALUATION_COUNT") != 1:
        raise RuntimeError("Completed exactly-once final-test evidence is required")
    engine = FrozenMEPIEngine(root, device=device)
    dataset_path = resolve_path(engine.config, engine.config["dataset"]["path"])
    with dataset_path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if len(source_rows) != 90 or len({row["condition_group_id"] for row in source_rows}) != 9:
        raise AssertionError("Exact frozen test manifest changed")
    sample_ids = np.load(resolve_path(engine.config, engine.config["dataset"]["sample_ids_path"]), allow_pickle=False)
    index = {str(value): position for position, value in enumerate(sample_ids)}
    all_waveforms = np.load(resolve_path(engine.config, engine.config["dataset"]["waveform_path"]), mmap_mode="r", allow_pickle=False)
    waveforms = np.stack([np.asarray(all_waveforms[index[row["sample_id"]]]) for row in source_rows])
    feature_rows = [{name: float(row[name]) for name in FINETUNE_TABULAR_FEATURES} for row in source_rows]
    predicted = _predict_batch(engine, waveforms, feature_rows)

    output_rows: list[dict[str, Any]] = []
    for position, source in enumerate(source_rows):
        record: dict[str, Any] = {
            "sample_id": source["sample_id"], "split": source["split"],
            "condition_group_id": source["condition_group_id"], "session_id": source["session_id"],
            "core_id": source["core_id"], "repeat_id": int(source["repeat_id"]),
            "frequency_hz": float(source["frequency_hz"]), "vin_rms_v": float(source["vin_rms_v"]),
        }
        targets = {
            "efficiency": (float(source["efficiency_percent"]), float(predicted["efficiency"][position]), "percent"),
            "core_loss": (float(source["P_loss"]), float(predicted["core_loss"][position]), "w"),
            "lsp": (float(source["LSP_raw"]), float(predicted["lsp"][position]), ""),
        }
        for name, (truth, estimate, unit) in targets.items():
            ending = f"_{unit}" if unit else ""
            record[f"{name}_measured{ending}"] = truth
            record[f"{name}_predicted{ending}"] = estimate
            record[f"{name}_residual{ending}"] = estimate - truth
            record[f"{name}_absolute_error{ending}"] = abs(estimate - truth)
            record[f"{name}_relative_error_percent"] = 100.0 * abs(estimate - truth) / (abs(truth) + ERR95_EPSILON)
        record["lsp_predicted_sigma"] = float(predicted["lsp_sigma"][position])
        output_rows.append(record)

    arrays = {
        "efficiency": (np.asarray([r["efficiency_measured_percent"] for r in output_rows]), np.asarray([r["efficiency_predicted_percent"] for r in output_rows])),
        "core_loss": (np.asarray([r["core_loss_measured_w"] for r in output_rows]), np.asarray([r["core_loss_predicted_w"] for r in output_rows])),
        "lsp": (np.asarray([r["lsp_measured"] for r in output_rows]), np.asarray([r["lsp_predicted"] for r in output_rows])),
    }
    metrics = {name: _metrics(*values) for name, values in arrays.items()}
    metrics["lsp"]["mape_percent"] = float(np.mean(np.abs(arrays["lsp"][1] - arrays["lsp"][0]) / (np.abs(arrays["lsp"][0]) + ERR95_EPSILON)) * 100.0)
    sigma = predicted["lsp_sigma"]
    metrics["lsp"]["predicted_sigma"] = {
        "minimum": float(sigma.min()),
        "median": float(np.sort(sigma)[(len(sigma) - 1) // 2]),
        "mean": float(sigma.mean()), "maximum": float(sigma.max()),
        "calibrated_uncertainty": False,
    }
    frozen_keys = {
        "efficiency": ("TEST_EFFICIENCY_MAE", "TEST_EFFICIENCY_RMSE", "TEST_EFFICIENCY_R2"),
        "core_loss": ("TEST_PLOSS_MAE", "TEST_PLOSS_RMSE", "TEST_PLOSS_R2"),
        "lsp": ("TEST_LSP_MAE", "TEST_LSP_RMSE", "TEST_LSP_R2"),
    }
    agreement = {}
    for name, keys in frozen_keys.items():
        differences = [abs(metrics[name][metric] - float(ledger[key])) for metric, key in zip(("mae", "rmse", "r2"), keys)]
        agreement[name] = {"absolute_differences": differences, "within_1e-5": all(value <= 1e-5 for value in differences)}
    if not all(value["within_1e-5"] for value in agreement.values()):
        raise AssertionError("Post-hoc metrics do not reproduce the frozen final-test report")

    prediction_path = root / "reports/MEPI_V1_5_POSTHOC_TEST_PREDICTIONS.csv"
    _write_predictions(prediction_path, output_rows)
    figures = _write_figures(output_rows, root / "reports/figures")
    if ledger_path.read_bytes() != ledger_bytes:
        raise AssertionError("Exactly-once final-test ledger was modified")
    payload: dict[str, Any] = {
        "status": "POSTHOC_REPORTING_COMPLETE",
        "POSTHOC_ANALYSIS_ONLY": True, "TRAINING_PERFORMED": False,
        "MODEL_SELECTION_PERFORMED": False, "HYPERPARAMETER_CHANGED": False,
        "CHECKPOINT_CHANGED": False, "FINAL_TEST_LEDGER_MODIFIED": False,
        "test_rows": 90, "test_groups": 9, "checkpoint_sha256": engine.checkpoint_sha256,
        "final_model_manifest": MANIFEST_RELATIVE_PATH.as_posix(),
        "final_model_manifest_sha256": sha256_file(root / MANIFEST_RELATIVE_PATH),
        "test_dataset_path": str(dataset_path.relative_to(root)),
        "test_dataset_sha256": sha256_file(dataset_path),
        "err95": {"formula": "100 * Q_0.95(abs(y-y_hat)/(abs(y)+epsilon))", "epsilon": ERR95_EPSILON, "source": ERR95_SOURCE},
        "metrics": metrics, "agreement_with_frozen_final_report": agreement,
        "predictions_csv": str(prediction_path.relative_to(root)),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "figures": [str(path.relative_to(root)) for path in figures],
    }
    json_path = root / "reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json"
    md_path = root / "reports/MEPI_V1_5_POSTHOC_TEST_METRICS.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return payload
