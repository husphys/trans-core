from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_posthoc_reports_are_complete_and_scientifically_guarded() -> None:
    payload = json.loads((ROOT / "reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json").read_text())
    assert payload["status"] == "POSTHOC_REPORTING_COMPLETE"
    assert payload["POSTHOC_ANALYSIS_ONLY"] is True
    assert payload["TRAINING_PERFORMED"] is False
    assert payload["MODEL_SELECTION_PERFORMED"] is False
    assert payload["HYPERPARAMETER_CHANGED"] is False
    assert payload["CHECKPOINT_CHANGED"] is False
    assert payload["FINAL_TEST_LEDGER_MODIFIED"] is False
    assert payload["test_rows"] == 90 and payload["test_groups"] == 9
    assert payload["err95"]["epsilon"] == 1e-12
    assert payload["err95"]["source"] == "src/evaluation/metrics.py"
    assert all(value["within_1e-5"] for value in payload["agreement_with_frozen_final_report"].values())
    assert payload["metrics"]["lsp"]["predicted_sigma"]["calibrated_uncertainty"] is False


def test_sample_predictions_are_real_aligned_test_rows() -> None:
    with (ROOT / "reports/MEPI_V1_5_POSTHOC_TEST_PREDICTIONS.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 90
    assert {row["split"] for row in rows} == {"test"}
    assert len({row["condition_group_id"] for row in rows}) == 9
    for field in (
        "sample_id", "session_id", "core_id", "frequency_hz", "vin_rms_v",
        "efficiency_measured_percent", "efficiency_predicted_percent", "efficiency_residual_percent",
        "core_loss_measured_w", "core_loss_predicted_w", "core_loss_residual_w",
        "lsp_measured", "lsp_predicted", "lsp_residual", "lsp_predicted_sigma",
    ):
        assert field in rows[0]


def test_frozen_final_ledger_and_checkpoint_hashes_remain_unchanged() -> None:
    assert hashlib.sha256((ROOT / "reports/MEPI_V1_5_FINAL_TEST_RESULTS.json").read_bytes()).hexdigest() == "aa6178404deea9d48fdee436ba9ef6d033a569df76f84f9ad9e33b8e2ec7c1df"
    assert hashlib.sha256((ROOT / "experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt").read_bytes()).hexdigest() == "0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33"


def test_posthoc_notebook_executed_to_completion() -> None:
    notebook = json.loads((ROOT / "notebooks/37_mepi_v1_5_posthoc_test_analysis.ipynb").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code and all(cell["execution_count"] is not None for cell in code)
    output = "\n".join(
        "".join(item.get("text", []))
        for cell in code
        for item in cell.get("outputs", [])
        if item.get("output_type") == "stream"
    )
    assert "POSTHOC_REPORTING_COMPLETE = TRUE" in output
    assert "TRAINING_PERFORMED = FALSE" in output
