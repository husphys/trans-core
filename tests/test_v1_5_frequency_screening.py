from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.mepi_v1.frequency_screening_v1_5 import (
    EXPECTED_CANDIDATE_FREQUENCIES_HZ,
    REPORT_JSON_RELATIVE_PATH,
    REPORT_MARKDOWN_RELATIVE_PATH,
    SCREENING_INPUT_RELATIVE_PATH,
    SCREENING_INPUT_SHA256,
    audit_frequency_screening_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/mepi_v1/frequency_screening_v1_5.py"
NOTEBOOK = ROOT / "notebooks/35_xlstm_v1_5_frequency_screening.ipynb"
FINAL_LEDGER = ROOT / "reports/MEPI_V1_5_FINAL_TEST_RESULTS.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_dedicated_demo_input_preflight_is_ready_and_fail_closed() -> None:
    assert _sha256(ROOT / SCREENING_INPUT_RELATIVE_PATH) == SCREENING_INPUT_SHA256
    audit = audit_frequency_screening_inputs(ROOT)
    assert audit["status"] == "PASS"
    assert audit["FREQUENCY_SCREENING_INPUT_READY"] is True
    assert audit["FREQUENCY_SCREENING_RUNTIME_READY"] is True
    assert audit["screening_condition_count"] == 146
    assert audit["excluded_hard_fail_count"] == 4
    assert audit["candidate_frequency_count"] == 15
    assert EXPECTED_CANDIDATE_FREQUENCIES_HZ == tuple(range(1000, 4501, 250))
    assert audit["waveform_shape"] == [146, 1024]
    assert audit["normalization_fitted_split"] == "train"
    assert audit["TEST_EVALUATION_COUNT"] == 1


def test_input_preflight_does_not_write_reports_or_final_ledger() -> None:
    watched = [
        FINAL_LEDGER,
        ROOT / REPORT_JSON_RELATIVE_PATH,
        ROOT / REPORT_MARKDOWN_RELATIVE_PATH,
    ]
    before = {
        path: _sha256(path) if path.is_file() else None
        for path in watched
    }
    audit_frequency_screening_inputs(ROOT)
    after = {
        path: _sha256(path) if path.is_file() else None
        for path in watched
    }
    assert after == before


def test_runtime_is_inference_only_and_test_isolated() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "validate_final_model_manifest",
        "construct_v1_5_model",
        'load_state_dict(checkpoint["model_state"], strict=True)',
        "model.eval()",
        "with torch.no_grad():",
        "FINETUNE_TABULAR_FEATURES",
        "WAVEFORM_MEAN",
        "WAVEFORM_SCALE",
        "TRAINING_PERFORMED",
        "TEST_DATA_USED_FOR_SCREENING",
        "FINAL_TEST_RERUN",
    ):
        assert required in source
    for forbidden in (
        "FrozenV15FinalTestDataset",
        "final_evaluation_v1_5",
        "run_final_test",
        "DataLoader",
        "samples_qc_valid_v1_2.csv",
        "split_manifest_v1_2",
        "make_optimizer",
        ".backward(",
        ".train(",
        "minmax_score",
        "screening_score",
    ):
        assert forbidden not in source


def test_runtime_declares_required_evidence_outputs_and_no_optimum() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert REPORT_JSON_RELATIVE_PATH.as_posix() in source
    assert REPORT_MARKDOWN_RELATIVE_PATH.as_posix() in source
    for column in (
        "predicted_efficiency_percent",
        "predicted_P_loss",
        "predicted_LSP_raw",
        "predicted_LSP_std_raw",
        "mean_predicted_efficiency",
        "mean_predicted_P_loss",
        "mean_predicted_LSP",
        "mean_predicted_LSP_std",
    ):
        assert column in source
    assert "NO_OPTIMUM_DECLARED" in source


def test_notebook_35_defaults_to_dry_run_but_has_authorized_execution_path() -> None:
    source = _notebook_source()
    assert "RUN_FREQUENCY_SCREENING = False" in source
    assert "audit_frequency_screening_inputs(PROJECT_ROOT)" in source
    assert "if RUN_FREQUENCY_SCREENING:" in source
    assert "run_frequency_screening(PROJECT_ROOT, authorize=True)" in source
    assert "Scientific frequency screening is outside this consistency-only dry run" not in source
    for required in (
        'print("FINAL_MODEL_READY =", str(FINAL_MODEL_READY).upper())',
        'print("FREQUENCY_SCREENING_INPUT_READY =", str(FREQUENCY_SCREENING_INPUT_READY).upper())',
        'print("FREQUENCY_SCREENING_RUNTIME_READY =", str(FREQUENCY_SCREENING_RUNTIME_READY).upper())',
        "FREQUENCY_SCREENING_RUN =",
        "TRAINING_PERFORMED = FALSE",
        "TEST_DATA_USED_FOR_SCREENING = FALSE",
        "FINAL_TEST_RERUN = FALSE",
        "TEST_EVALUATION_COUNT =",
    ):
        assert required in source
