from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.mepi_v1.final_model_manifest_v1_5 import (
    MANIFEST_RELATIVE_PATH,
    SELECTED_CHECKPOINT_PATH,
    SELECTED_CHECKPOINT_SHA256,
    build_final_model_manifest,
    sha256_file,
    validate_final_model_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / MANIFEST_RELATIVE_PATH
NOTEBOOK = ROOT / "notebooks/35_xlstm_v1_5_frequency_screening.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_completed_final_test_generates_manifest_without_rerun() -> None:
    ledger = json.loads(
        (ROOT / "reports/MEPI_V1_5_FINAL_TEST_RESULTS.json").read_text(encoding="utf-8")
    )
    assert ledger["EVALUATION_STATUS"] == "FINAL_TEST_COMPLETE"
    assert ledger["TEST_ACCESSED"] is True
    assert ledger["TEST_EVALUATION_COUNT"] == 1

    source = (ROOT / "src/mepi_v1/final_model_manifest_v1_5.py").read_text(
        encoding="utf-8"
    )
    assert "run_final_test" not in source
    assert "FrozenV15FinalTestDataset" not in source
    assert "DataLoader" not in source
    assert build_final_model_manifest(ROOT)["final_test_complete"] is True


def test_manifest_is_exactly_derived_from_existing_evidence() -> None:
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert recorded == build_final_model_manifest(ROOT)
    assert validate_final_model_manifest(ROOT, MANIFEST) == recorded
    assert recorded["configuration_id"] == "l3_1_l4_0p05"
    assert recorded["selection_metric"] == "VAL_TASK_SCORE"
    assert recorded["selection_score"] == 0.2023820665829322
    assert recorded["final_config_frozen"] is True
    assert recorded["test_evaluation_count"] == 1
    assert recorded["frequency_screening_ready"] is True


def test_selected_checkpoint_exists_and_sha256_is_verified() -> None:
    checkpoint = ROOT / SELECTED_CHECKPOINT_PATH
    assert checkpoint.is_file()
    assert sha256_file(checkpoint) == SELECTED_CHECKPOINT_SHA256
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == SELECTED_CHECKPOINT_SHA256


def test_manifest_validation_fails_closed_when_missing_or_corrupt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_final_model_manifest(ROOT, tmp_path / "missing.json")

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        validate_final_model_manifest(ROOT, corrupt)

    wrong_hash = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wrong_hash["checkpoint_sha256"] = "0" * 64
    corrupt.write_text(json.dumps(wrong_hash), encoding="utf-8")
    with pytest.raises(AssertionError, match="checkpoint_sha256"):
        validate_final_model_manifest(ROOT, corrupt)


def test_notebook_35_preflight_is_fail_closed_inference_only() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = _notebook_source()
    for required in (
        "FINAL_MODEL_MANIFEST.is_file()",
        'manifest["final_config_frozen"] is not True',
        'manifest["final_test_complete"] is not True',
        'manifest["test_evaluation_count"] != 1',
        'manifest["configuration_id"] != "l3_1_l4_0p05"',
        'manifest["lambda3"] != 1.0',
        'manifest["lambda4"] != 0.05',
        "checkpoint_path.is_file()",
        'sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]',
        'print("FINAL_MODEL_READY =", str(FINAL_MODEL_READY).upper())',
        'print("FREQUENCY_SCREENING_INPUT_READY =", str(FREQUENCY_SCREENING_INPUT_READY).upper())',
        "FREQUENCY_SCREENING_RUN =",
        "TEST_TUNING = FALSE",
    ):
        assert required in source

    for forbidden in (
        "FrozenV15FinalTestDataset",
        "run_final_test",
        "DataLoader",
        "np.load",
        "optimizer",
        ".backward(",
        ".train(",
        "RUN_TRAINING",
    ):
        assert forbidden not in source


def test_future_final_evaluation_writes_handoff_after_completed_ledger() -> None:
    source = (ROOT / "src/mepi_v1/final_evaluation_v1_5.py").read_text(encoding="utf-8")
    assert "write_final_model_manifest(project_root)" in source
    assert source.index("write_final_model_manifest(project_root)") > source.index(
        '"EVALUATION_STATUS": "FINAL_TEST_COMPLETE"'
    )
