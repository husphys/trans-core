"""Build the frozen MEPI v1.2 LSP, split, and normalization artifacts.

This is a preparation-only builder. It reads the immutable v4 evidence, creates
new v1.2 outputs, and never launches fine-tuning or model evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .build_finetune_dataset import _write_csv, _write_json, sha256_file
from .checkpointing import audit_xlstm_transfer_checkpoint
from .constants import FINETUNE_TABULAR_FEATURES, WAVEFORM_LENGTH, validate_finetune_feature_names
from .scaling import TrainOnlyStandardizer
from .splitting import assert_disjoint_splits, deterministic_exact_group_split
from .targets import (
    GAS_CONSTANT_J_PER_MOL_K,
    LSP_V1_2_EFFECTIVE_ACTIVATION_ENERGY_J_PER_MOL,
    LSP_V1_2_REFERENCE_TEMPERATURE_K,
    arrhenius_log_lsp_v1_2,
    arrhenius_lsp_v1_2,
    normalize_lsp_v1_2,
)

PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.2"
PROTOCOL_SHA256 = "c0fd2fe8aafc0c55dc0b6c525e341d0458782391f6476831da2e3cc5cf401988"
V1_1_PROTOCOL_SHA256 = "a515052cd2f8cf2970b731cfb48dee055c316a5d4acc0f0ca90313436c344c0f"
PROCESSING_VERSION = "mepi-v1.2-lsp-split-normalization-v1"
LSP_DEFINITION_VERSION = "v1.2"
TEMPERATURE_MAPPING = "SWAPPED_CORE_ROOM_CHANNELS_PHYSICALLY_VERIFIED"
PRIMARY_ROLE = "primary_paired"
SUPPLEMENTARY_ROLE = "supplementary_excluded_from_primary"
OLD_HIGH_SESSION = "session_20260802_075027"
SPLIT_SEED = 42
INTENDED_COMMAND = "python -m src.mepi_v1.finetune --config configs/finetune_v1_2.yaml"
CHECKPOINT_RELATIVE = "experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _verify_checksums(directory: Path, manifest_name: str) -> None:
    for line in (directory / manifest_name).read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(directory / name) != expected:
            raise RuntimeError(f"Checksum mismatch for historical artifact: {name}")


def _fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    return fields


def _scaler_metadata(values: np.ndarray) -> tuple[TrainOnlyStandardizer, dict[str, Any]]:
    scaler = TrainOnlyStandardizer().fit(np.asarray(values, dtype=np.float64), split="train")
    if scaler.state is None:  # pragma: no cover - guarded by fit
        raise AssertionError("Train scaler did not produce state")
    return scaler, {
        **scaler.state.to_dict(),
        "implementation": "src.mepi_v1.scaling.TrainOnlyStandardizer",
        "standard_deviation": "population",
        "numpy_ddof": 0,
    }


def _copy_and_validate_rows(v4_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in v4_rows:
        row: dict[str, Any] = dict(source)
        if source["temperature_mapping_correction"] != TEMPERATURE_MAPPING:
            raise AssertionError("Temperature row is not marked as exactly-once corrected")
        legacy_ambient = float(source["temperature_ambient_raw_legacy_c"])
        legacy_core = float(source["temperature_core_raw_legacy_c"])
        if not (
            float(source["temperature_ambient_c"]) == legacy_core
            and float(source["temperature_core_c"]) == legacy_ambient
        ):
            raise AssertionError("Temperature mapping differs from the verified one-swap definition")
        row.update(
            {
                "source_protocol_version": source["protocol_version"],
                "protocol_version": PROTOCOL_VERSION,
                "protocol_sha256": PROTOCOL_SHA256,
                "processing_version": PROCESSING_VERSION,
                "condition_role": (
                    SUPPLEMENTARY_ROLE
                    if source["source_session_id"] == OLD_HIGH_SESSION
                    else source["condition_role"]
                ),
                "T_core_K": float(source["temperature_core_c"]) + 273.15,
                "log_LSP_raw": "",
                "LSP_raw": "",
                "LSP": "",
                "LSP_z": "",
                "LSP_Arr_raw": "",
                "LSP_Arr_z": "",
                "lsp_definition_version": "",
                "lsp_target_eligible": 0,
                "split": "",
            }
        )
        result.append(row)
    return result


def _write_split_manifest(
    output: Path, candidates: list[dict[str, Any]], group_assignments: dict[str, str]
) -> tuple[list[dict[str, Any]], str]:
    rows = [
        {
            "sample_id": row["sample_id"],
            "condition_group_id": row["condition_group_id"],
            "canonical_vin_set_group_v": row["canonical_vin_set_group_v"],
            "frequency_set_hz": row["frequency_set_hz"],
            "core_id": row["core_id"],
            "repeat_id": row["repeat_id"],
            "split": row["split"],
        }
        for row in candidates
    ]
    fields = list(rows[0])
    path = output / "split_manifest_v1_2.csv"
    _write_csv(path, rows, fields)
    split_hash = sha256_file(path)
    group_lists = {
        name: sorted(group for group, split in group_assignments.items() if split == name)
        for name in ("train", "validation", "test")
    }
    candidate_groups = {str(row["condition_group_id"]) for row in candidates}
    row_counts = Counter(row["split"] for row in candidates)
    _write_json(
        output / "split_manifest_v1_2.json",
        {
            "status": "FROZEN",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "seed": SPLIT_SEED,
            "method": "seeded permutation of sorted operating-condition group identities",
            "group_key": ["canonical_vin_set_group_v", "frequency_set_hz"],
            "groups": group_lists,
            "group_counts": {name: len(values) for name, values in group_lists.items()},
            "groups_with_zero_qc_valid_rows": sorted(set(group_assignments) - candidate_groups),
            "row_counts": {name: row_counts[name] for name in group_lists},
            "split_manifest_csv_sha256": split_hash,
            "test_accessed_for_model_selection": False,
            "test_evaluations": 0,
        },
    )
    return rows, split_hash


def _write_checksums(output: Path, names: list[str]) -> None:
    (output / "checksums_v1_2.sha256").write_text(
        "".join(f"{sha256_file(output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def build(project_root: Path) -> dict[str, Any]:
    v1_1 = project_root / "data/MEPI/v1_1"
    output = project_root / "data/MEPI/v1_2"
    reports = project_root / "reports"
    output.mkdir(parents=True, exist_ok=True)

    protocol_paths = [
        project_root / "MEPI-FROZEN-PROTOCOL v1.2.md",
        project_root / "docs/MEPI-FROZEN-PROTOCOL v1.2.md",
    ]
    if [sha256_file(path) for path in protocol_paths] != [PROTOCOL_SHA256, PROTOCOL_SHA256]:
        raise RuntimeError("Frozen v1.2 protocol copies are not byte-identical at the expected hash")
    historical_protocols = [
        project_root / "MEPI-FROZEN-PROTOCOL v1.1.md",
        project_root / "docs/MEPI-FROZEN-PROTOCOL v1.1.md",
    ]
    if [sha256_file(path) for path in historical_protocols] != [V1_1_PROTOCOL_SHA256] * 2:
        raise RuntimeError("Historical v1.1 protocol evidence changed")
    _verify_checksums(v1_1, "checksums_v4.sha256")
    validate_finetune_feature_names(FINETUNE_TABULAR_FEATURES)

    source_master = _read_csv(v1_1 / "samples_master_reprocessed_v4.csv")
    source_primary = _read_csv(v1_1 / "samples_primary_paired_v4.csv")
    source_candidate = _read_csv(v1_1 / "samples_candidate_finetune_v4.csv")
    if len(source_master) != 975 or len(source_primary) != 900:
        raise AssertionError("Expected 975 v4 master rows and 900 primary measured rows")
    if len({row["sample_id"] for row in source_master}) != len(source_master):
        raise AssertionError("Duplicate sample IDs in v4 master")
    if any(row["dataset_mode"] != "finetune" for row in source_master):
        raise AssertionError("Demo row entered the finetune master")

    master = _copy_and_validate_rows(source_master)
    primary = [row for row in master if row["condition_role"] == PRIMARY_ROLE]
    supplementary = [row for row in master if row["condition_role"] == SUPPLEMENTARY_ROLE]
    candidates = [row for row in primary if int(row["qc_pass"]) == 1]
    if len(primary) != 900 or len(supplementary) != 75:
        raise AssertionError("v1.2 primary/supplementary counts must be 900/75")
    if any(row["source_session_id"] == OLD_HIGH_SESSION for row in primary + candidates):
        raise AssertionError("Old Commercial 6.x session entered active data")
    if [row["sample_id"] for row in candidates] != [row["sample_id"] for row in source_candidate]:
        raise AssertionError("Candidate membership/order differs from audited v4")
    if any(row["B_source"] != "scope2_ch1_primary_vin" for row in primary):
        raise AssertionError("An active primary row does not use Scope #2 CH1 primary Vin for B")

    core_c = np.asarray([float(row["temperature_core_c"]) for row in candidates], dtype=np.float64)
    log_lsp = arrhenius_log_lsp_v1_2(core_c)
    lsp_raw = arrhenius_lsp_v1_2(core_c)
    for row, log_value, raw_value in zip(candidates, log_lsp, lsp_raw, strict=True):
        row.update(
            {
                "log_LSP_raw": float(log_value),
                "LSP_raw": float(raw_value),
                "LSP": float(raw_value),
                "LSP_Arr_raw": float(raw_value),
                "lsp_definition_version": LSP_DEFINITION_VERSION,
                "lsp_target_eligible": 1,
            }
        )

    primary_group_ids = [str(row["condition_group_id"]) for row in primary]
    primary_splits = deterministic_exact_group_split(primary_group_ids, seed=SPLIT_SEED)
    group_assignments: dict[str, str] = {}
    for group, split in zip(primary_group_ids, primary_splits, strict=True):
        previous = group_assignments.setdefault(group, str(split))
        if previous != str(split):
            raise AssertionError("Operating-condition group crossed subsets")
    group_ids = [str(row["condition_group_id"]) for row in candidates]
    splits = np.asarray([group_assignments[group] for group in group_ids], dtype=str)
    sample_ids = [str(row["sample_id"]) for row in candidates]
    waveform_hashes = [
        f"{row['source_scope1_sha256']}:{row['source_scope2_sha256']}" for row in candidates
    ]
    assert_disjoint_splits(sample_ids, group_ids, splits, waveform_hashes)
    for row, split in zip(candidates, splits, strict=True):
        row["split"] = str(split)
        if group_assignments[str(row["condition_group_id"])] != str(split):
            raise AssertionError("Operating-condition group crossed subsets")
    group_counts = Counter(group_assignments.values())
    if group_counts != {"train": 72, "validation": 9, "test": 9}:
        raise AssertionError(f"Exact 72/9/9 group split not achieved: {dict(group_counts)}")
    zero_valid_groups = sorted(set(primary_group_ids) - set(group_ids))

    _, split_hash = _write_split_manifest(output, candidates, group_assignments)
    train = [row for row in candidates if row["split"] == "train"]
    train_group_ids = sorted({str(row["condition_group_id"]) for row in train})
    train_lsp = np.asarray([float(row["LSP_raw"]) for row in train], dtype=np.float64)
    lsp_scaler, lsp_state = _scaler_metadata(train_lsp)
    if lsp_scaler.state is None:  # pragma: no cover
        raise AssertionError("Missing LSP scaler state")
    mu_lsp = float(lsp_scaler.state.mean[0])
    sigma_lsp = float(lsp_scaler.state.scale[0])
    for row in candidates:
        normalized = float(normalize_lsp_v1_2(float(row["LSP_raw"]), mean=mu_lsp, scale=sigma_lsp))
        row["LSP_z"] = normalized
        row["LSP_Arr_z"] = normalized

    feature_matrix = np.asarray(
        [[float(row[name]) for name in FINETUNE_TABULAR_FEATURES] for row in candidates],
        dtype=np.float64,
    )
    required_targets = np.asarray(
        [[float(row[name]) for name in ("efficiency_percent", "P_loss", "LSP_raw")] for row in candidates],
        dtype=np.float64,
    )
    if not np.isfinite(feature_matrix).all() or not np.isfinite(required_targets).all():
        raise AssertionError("Candidate model inputs or required targets contain NaN/Inf")
    if np.any(lsp_raw <= 0.0) or not np.isfinite(lsp_raw).all():
        raise AssertionError("LSP is not finite and positive")
    if not np.isclose(float(arrhenius_lsp_v1_2(25.0)), 1.0, rtol=0.0, atol=1e-14):
        raise AssertionError("LSP(T_ref) != 1")
    monotonic = arrhenius_lsp_v1_2(np.asarray([20.0, 25.0, 30.0], dtype=np.float64))
    if not np.all(np.diff(monotonic) < 0.0):
        raise AssertionError("LSP is not strictly decreasing with core temperature")

    feature_scalers: dict[str, Any] = {}
    train_feature_matrix = np.asarray(
        [[float(row[name]) for name in FINETUNE_TABULAR_FEATURES] for row in train],
        dtype=np.float64,
    )
    for index, name in enumerate(FINETUNE_TABULAR_FEATURES):
        _, feature_scalers[name] = _scaler_metadata(train_feature_matrix[:, index])
    target_scalers: dict[str, Any] = {}
    for name in ("efficiency_percent", "P_loss"):
        _, target_scalers[name] = _scaler_metadata(
            np.asarray([float(row[name]) for row in train], dtype=np.float64)
        )
    target_scalers["LSP_raw"] = lsp_state
    normalization = {
        "status": "FROZEN_TRAIN_ONLY",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "processing_version": PROCESSING_VERSION,
        "fitted_split": "train",
        "fit_order": "after frozen group split",
        "implementation": "src.mepi_v1.scaling.TrainOnlyStandardizer",
        "standard_deviation": "population standard deviation",
        "numpy_ddof": 0,
        "split_hash": split_hash,
        "training_group_ids": train_group_ids,
        "training_row_count": len(train),
        "mu_LSP_train": mu_lsp,
        "sigma_LSP_train": sigma_lsp,
        "feature_scalers": feature_scalers,
        "target_scalers": target_scalers,
        "arrhenius_reference_scaler": "IDENTICAL_TO_target_scalers.LSP_raw",
        "separate_arrhenius_residual_scaler": False,
        "validation_refit": False,
        "test_refit": False,
    }
    _write_json(output / "train_normalization_v1_2.json", normalization)

    primary_b = np.load(v1_1 / "B1024_primary_paired_v4.npy", allow_pickle=False)
    candidate_b = np.load(v1_1 / "B1024_candidate_finetune_v4.npy", allow_pickle=False)
    candidate_ids = np.load(v1_1 / "sample_ids_candidate_finetune_v4.npy", allow_pickle=False)
    if primary_b.shape != (900, WAVEFORM_LENGTH):
        raise AssertionError("Primary B array is not 900x1024")
    if candidate_b.shape != (len(candidates), WAVEFORM_LENGTH):
        raise AssertionError("Candidate B array width/count mismatch")
    if candidate_ids.tolist() != sample_ids:
        raise AssertionError("Candidate B/sample ID alignment mismatch")
    np.save(output / "B1024_v1_2.npy", candidate_b, allow_pickle=False)
    np.save(output / "sample_ids_v1_2.npy", candidate_ids, allow_pickle=False)

    master_fields = _fields(master)
    candidate_fields = _fields(candidates)
    _write_csv(output / "samples_master_v1_2.csv", master, master_fields)
    _write_csv(output / "samples_qc_valid_v1_2.csv", candidates, candidate_fields)

    forbidden = [
        "core_id", "temperature_core_c", "T_core_K", "log_LSP_raw", "LSP_raw",
        "LSP", "LSP_z", "P_loss", "efficiency_percent",
    ]
    schema = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "processing_version": PROCESSING_VERSION,
        "tabular_features": list(FINETUNE_TABULAR_FEATURES),
        "tabular_width": len(FINETUNE_TABULAR_FEATURES),
        "waveform": {"name": "B(t)_1024", "shape": [len(candidates), 1024], "source": "Scope #2 CH1 primary Vin"},
        "targets_raw": ["efficiency_percent", "P_loss", "LSP_raw"],
        "optimization_lsp_target": "LSP_z",
        "metadata_only": ["core_id"],
        "forbidden_predictive_inputs": forbidden,
        "temperature_core_is_target_construction_only": True,
        "normalization_file": "train_normalization_v1_2.json",
    }
    _write_json(output / "feature_schema_v1_2.json", schema)

    transfer = audit_xlstm_transfer_checkpoint(project_root, 8)
    row_counts = Counter(str(row["split"]) for row in candidates)
    core_counts = Counter(str(row["core_id"]) for row in candidates)
    statuses = {
        "B_READY": True,
        "TEMPERATURE_MAPPING_READY": True,
        "PRIMARY_DATASET_READY": True,
        "ELECTRICAL_QC_READY": True,
        "LSP_DEFINITION_READY": True,
        "LSP_TARGET_READY": True,
        "SPLIT_READY": True,
        "NORMALIZATION_READY": True,
        "LEAKAGE_GUARD_READY": True,
        "TRAIN_READY": True,
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "processing_version": PROCESSING_VERSION,
        "master_rows": len(master),
        "primary_measured_rows": len(primary),
        "supplementary_excluded_rows": len(supplementary),
        "qc_valid_rows": len(candidates),
        "candidate_core_counts": dict(core_counts),
        "group_counts": dict(group_counts),
        "qc_valid_group_count": len(set(group_ids)),
        "groups_with_zero_qc_valid_rows": zero_valid_groups,
        "row_counts": {name: row_counts[name] for name in ("train", "validation", "test")},
        "split_seed": SPLIT_SEED,
        "split_hash": split_hash,
        "B1024_shape": list(candidate_b.shape),
        "LSP_raw": {
            "min": float(np.min(lsp_raw)),
            "median": float(np.median(lsp_raw)),
            "max": float(np.max(lsp_raw)),
        },
        "mu_LSP_train": mu_lsp,
        "sigma_LSP_train": sigma_lsp,
        "pretrained_checkpoint": CHECKPOINT_RELATIVE,
        "pretrained_checkpoint_sha256": transfer["checkpoint_sha256"],
        "intended_finetuning_command": INTENDED_COMMAND,
        "test_accessed_for_model_selection": False,
        "test_evaluations": 0,
        "training_run": False,
        "statuses": statuses,
    }
    _write_json(output / "dataset_summary_v1_2.json", summary)

    audit = f"""# LSP v1.2 Audit

## Frozen definition

`T_core_K = temperature_core_c + 273.15`; `log_LSP_raw = (125000.0 / 8.314462618) * (1/T_core_K - 1/298.15)`; `LSP_raw = exp(log_LSP_raw)`. Construction used float64 and no epsilon. LSP is a dimensionless relative thermal-stress proxy, not lifetime, RUL, time-to-failure, or service hours. `Ea_eff` is a literature-informed effective thermal-aging sensitivity parameter, not a dataset-identified or experimentally established material constant.

## Eligibility and temperature provenance

LSP was generated for {len(candidates)} QC-valid primary rows only ({core_counts['FE']} FE, {core_counts['COMMERCIAL']} COMMERCIAL). HARD_FAIL and supplementary rows retain blank LSP targets. The physically verified one-time channel swap was asserted for every source row; no temperature offset or slope was fitted.

## Numerical checks

- `LSP_raw` min/median/max: {np.min(lsp_raw):.12g} / {np.median(lsp_raw):.12g} / {np.max(lsp_raw):.12g}
- `LSP_raw(T_ref=298.15 K)`: {float(arrhenius_lsp_v1_2(25.0)):.12g}
- Strictly decreasing with increasing core temperature: PASS
- Finite and positive: PASS

## Split and normalization

Seed {SPLIT_SEED} assigned exactly 72/9/9 identities across all 90 measured operating-condition groups. QC-valid candidates occur in {len(set(group_ids))} groups; `{', '.join(zero_valid_groups)}` has no QC-valid row but remains explicitly assigned in the frozen group manifest. Candidate row counts are train/validation/test = {row_counts['train']}/{row_counts['validation']}/{row_counts['test']}. Population mean/std (`numpy`, `ddof=0`) were fitted on {len(train)} training rows only: `mu_LSP_train={mu_lsp:.17g}`, `sigma_LSP_train={sigma_lsp:.17g}`. Validation, test, the Arrhenius reference, and the PIRL Arrhenius residual all use this same frozen scaler; no residual scaler exists. Split hash: `{split_hash}`.

No test result or test metric was used for model selection. No training was run.
"""
    (reports / "LSP_v1_2_audit.md").write_text(audit, encoding="utf-8")

    feature_text = ", ".join(FINETUNE_TABULAR_FEATURES)
    report = f"""# MEPI v1.2 Final Train Readiness

## Protocol

Both frozen v1.2 copies are byte-identical at SHA-256 `{PROTOCOL_SHA256}`. Historical v1.1 remains unchanged at `{V1_1_PROTOCOL_SHA256}`.

## Dataset and QC

Primary measured rows: {len(primary)}. QC-valid rows: {len(candidates)} ({core_counts['FE']} FE, {core_counts['COMMERCIAL']} COMMERCIAL). The old Commercial 6.x session is preserved as {len(supplementary)} supplementary rows and excluded from all active subsets and normalization. Every active primary row uses Scope #2 CH1 primary Vin for B; `B1024_v1_2.npy` shape is {tuple(candidate_b.shape)}. No demo or duplicate sample entered the dataset.

## LSP

The exact v1.2 float64/no-epsilon definition is implemented. `LSP_raw` min/median/max = {np.min(lsp_raw):.12g}/{np.median(lsp_raw):.12g}/{np.max(lsp_raw):.12g}. It equals one at 298.15 K, is finite and positive, and decreases monotonically with increasing core temperature.

## Frozen split and normalization

Group counts train/validation/test = 72/9/9; row counts = {row_counts['train']}/{row_counts['validation']}/{row_counts['test']}. No group crosses subsets. QC-valid rows occur in {len(set(group_ids))}/90 identities; `{', '.join(zero_valid_groups)}` is retained in the group manifest with zero eligible rows. Training-only LSP population mean/std (`ddof=0`) = {mu_lsp:.17g}/{sigma_lsp:.17g}. The target, Arrhenius reference, and PIRL Arrhenius residual use this identical scaler. Validation/test were not refitted, and test was not used for model-development decisions.

## Leakage and downstream boundary

Exactly nine tabular inputs: `{feature_text}`. Core identity, corrected core temperature, absolute core temperature, LSP construction fields, targets, and power quantities are excluded from predictive inputs. Transfer audit passed for `{CHECKPOINT_RELATIVE}` at SHA-256 `{transfer['checkpoint_sha256']}`; only `waveform_encoder` and matching depth-8 xLSTM `backbone` will transfer.

Intended command (prepared, not executed):

```bash
{INTENDED_COMMAND}
```

Training remains disabled in the config pending explicit execution authorization. `test_evaluations = 0`; `training_run = false`.

""" + "\n".join(f"{name} = {'TRUE' if value else 'FALSE'}" for name, value in statuses.items()) + "\n"
    (reports / "MEPI_V1_2_FINAL_TRAIN_READINESS.md").write_text(report, encoding="utf-8")

    checksum_names = [
        "samples_master_v1_2.csv",
        "samples_qc_valid_v1_2.csv",
        "B1024_v1_2.npy",
        "sample_ids_v1_2.npy",
        "split_manifest_v1_2.csv",
        "split_manifest_v1_2.json",
        "train_normalization_v1_2.json",
        "feature_schema_v1_2.json",
        "dataset_summary_v1_2.json",
    ]
    _write_checksums(output, checksum_names)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summary = build(args.project_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
