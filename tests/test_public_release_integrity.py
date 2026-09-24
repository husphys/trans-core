from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_public_hashes() -> None:
    expected = {
        "experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt": "0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33",
        "experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt": "fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c",
        "experiments/finetune_v1_5_xlstm_depth8/final_model_manifest.json": "607bad9d532777f11a6659459cfa59e6a78e054bbd6ed95abe0e4f3dfffe67c1",
        "data/MEPI/demo_manifest_v2.csv": "812ea7e87ad69c02702a411d0f1fe6ef241b9753586a50f1d38522d25d7f9556",
        "MEPI-FROZEN-PROTOCOL v1.5.md": "0f68d6ebd5d16177e5238471639f861dfa91522130ac6a4a3ac48634b9af1e39",
        "configs/final_model_v1_5.yaml": "4b18d275be0a56f09371f2fd91d3fe03bba72a7cee6ca732014fbbf043dcdf96",
        "reports/MEPI_V1_5_FINAL_TEST_RESULTS.json": "aa6178404deea9d48fdee436ba9ef6d033a569df76f84f9ad9e33b8e2ec7c1df",
        "reports/MEPI_V1_5_FREQUENCY_SCREENING.json": "5b287be4ad4d457898afcaed5deda3ebddef5022c9fa74964c35f7aa0dce6f5c",
    }
    assert {relative: sha256(ROOT / relative) for relative in expected} == expected


def test_checksum_defined_data_bundles() -> None:
    for directory, manifest_name in (
        (ROOT / "data/MEPI/v1_1", "checksums_v4.sha256"),
        (ROOT / "data/MEPI/v1_2", "checksums_v1_2.sha256"),
        (ROOT / "data/MEPI/v1_3", "checksums_v1_3.sha256"),
    ):
        for line in (directory / manifest_name).read_text(encoding="utf-8").splitlines():
            expected, name = line.split(maxsplit=1)
            assert sha256(directory / name.strip()) == expected


def test_public_source_and_notebooks_parse_without_execution() -> None:
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in ROOT.rglob("*.json"):
        if ".git" not in path.parts:
            json.loads(path.read_text(encoding="utf-8"))
    for path in [*ROOT.rglob("*.yaml"), *ROOT.rglob("*.yml")]:
        if ".git" not in path.parts:
            yaml.safe_load(path.read_text(encoding="utf-8"))
    for path in ROOT.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert not [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "/home/" not in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:cell{index}")


def test_demo_bundle_and_final_metrics_are_recorded_not_recomputed() -> None:
    with (ROOT / "data/MEPI/demo_manifest_v2.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 150
    assert len(list((ROOT / "data/MEPI/demo_waveforms").rglob("*_B1024.csv"))) == 150

    report = json.loads((ROOT / "reports/MEPI_V1_5_FINAL_TEST_RESULTS.json").read_text())
    assert report["TEST_EVALUATION_COUNT"] == 1
    expected_metrics = {
        "TEST_EFFICIENCY_MAE": 1.1456656562,
        "TEST_EFFICIENCY_RMSE": 1.4833748448,
        "TEST_EFFICIENCY_R2": 0.9817636937,
        "TEST_PLOSS_MAE": 0.0136925422,
        "TEST_PLOSS_RMSE": 0.0164594169,
        "TEST_PLOSS_R2": 0.9706166407,
        "TEST_LSP_MAE": 0.0279506859,
        "TEST_LSP_RMSE": 0.0363657696,
        "TEST_LSP_R2": 0.9162509237,
        "TEST_LSP_MAPE_PERCENT": 4.7880477905,
    }
    assert {key: round(report[key], 10) for key in expected_metrics} == expected_metrics
