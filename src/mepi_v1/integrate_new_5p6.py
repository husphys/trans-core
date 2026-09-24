"""Integrate the verified new COMMERCIAL 5.6-V session into MEPI v1.1 v4.

The command reads immutable acquisition evidence, reprocesses every finetune
waveform from raw Scope #2 CH1, applies the verified temperature-channel map
exactly once, and stops before split construction, normalization, test access,
or training because LSP remains undefined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .build_finetune_dataset import (
    EFFECTIVE_AREA_M2,
    EXPECTED_FREQUENCIES,
    EXPECTED_VOLTAGES,
    N_PRIMARY,
    THD_LIMIT_PERCENT,
    Session,
    _write_csv,
    _write_json,
    discover_sessions,
    process_sample,
    sha256_file,
)
from .constants import EXPECTED_PROTOCOL_SHA256, FINETUNE_TABULAR_FEATURES, PROTOCOL_VERSION, WAVEFORM_LENGTH
from .remediate_finetune_dataset import MAPPING_CORRECTION, MAPPING_VERIFIED_DATE

PROCESSING_VERSION_V4 = "primary-B-reprocess-v4-new-commercial-5p6"
TEMPERATURE_MAPPING_VERSION = "physical-core-room-swap-v1-2026-09-23"
NEW_MAPPING_BASIS = "NEW_COMMERCIAL_5P6_REMEASUREMENT_CONFIRMED"
OLD_HIGH_ROLE = "excluded_legacy_high_voltage"
PRIMARY_ROLE = "primary_paired"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_mapping(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != 13 or len({row["session_id"] for row in rows}) != 13:
        raise AssertionError("v4 mapping must contain exactly 13 unique finetune sessions")
    if any(row["confirmed_by_experimenter"] != "1" for row in rows):
        raise AssertionError("Every v4 inclusion/exclusion decision must be experimenter-confirmed")
    return {row["session_id"]: row for row in rows}


def _median(rows: list[dict[str, str]], field: str) -> float:
    return float(statistics.median(float(row[field]) for row in rows))


def _discover_new_and_old(sessions: list[Session]) -> tuple[Session, Session]:
    finetune = [session for session in sessions if session.mode == "finetune"]
    new_candidates = [
        session
        for session in finetune
        if session.config.get("core_id") == "COMMERCIAL"
        and 5.5 <= float(session.config.get("vin_reference_v", 0.0)) <= 5.8
        and len(session.rows) == 75
        and len({int(round(float(row["frequency_set_hz"]))) for row in session.rows}) == 15
        and len({int(row["repeat_id"]) for row in session.rows}) == 5
    ]
    old_candidates = [
        session
        for session in finetune
        if session.config.get("core_id") == "COMMERCIAL"
        and float(session.config.get("vin_reference_v", 0.0)) >= 6.0
    ]
    if len(new_candidates) != 1:
        raise AssertionError(f"Expected exactly one new COMMERCIAL 5.6-V candidate, got {len(new_candidates)}")
    if len(old_candidates) != 1:
        raise AssertionError(f"Expected exactly one old COMMERCIAL 6.x-V session, got {len(old_candidates)}")
    return new_candidates[0], old_candidates[0]


def _condition_id(frequency: int, voltage: float) -> str:
    voltage_text = f"{voltage:.1f}".replace(".", "p")
    return f"F{frequency}_V{voltage_text}_RL49p6025_SINE"


def _correct_and_map(
    processed: dict[str, Any],
    source: dict[str, str],
    mapping: dict[str, str],
) -> dict[str, Any]:
    if source.get("temperature_mapping_correction"):
        raise AssertionError("DOUBLE_TEMPERATURE_SWAP_BLOCKED: raw source already claims correction")
    original_ambient = float(source["temperature_ambient_c"])
    original_core = float(source["temperature_core_c"])
    raw_recomputed_vin = float(processed["vin_rms_v"])
    source_vin = float(source["vin_rms_v"])
    if not np.isclose(raw_recomputed_vin, source_vin, rtol=0.0, atol=1e-10):
        raise AssertionError("Raw Scope #2 CH1 Vin does not reproduce stored measured vin_rms_v")

    role = mapping["condition_role"]
    canonical_text = mapping["canonical_nominal_v"].strip()
    canonical: float | str = float(canonical_text) if canonical_text else ""
    if role == PRIMARY_ROLE and canonical not in EXPECTED_VOLTAGES:
        raise AssertionError("Primary session lacks one frozen nominal voltage")
    if role == OLD_HIGH_ROLE and canonical != "":
        raise AssertionError("Old 6.x session must not receive a primary canonical voltage")

    warnings = set(filter(None, str(processed["qc_warning_reasons"]).split(";")))
    if role == PRIMARY_ROLE:
        warnings.discard("UNEXPECTED_VOLTAGE_LEVEL")
    warnings.add("TEMPERATURE_CALIBRATION_PENDING")
    hard = set(filter(None, str(processed["qc_hard_fail_reasons"]).split(";")))
    qc_status = "HARD_FAIL" if hard else ("WARNING" if warnings else "PASS")
    processed.update(
        {
            "processing_version": PROCESSING_VERSION_V4,
            "original_condition_group_id": source["condition_group_id"],
            "original_vin_set_group_v": float(source["vin_set_group_v"]),
            "canonical_vin_set_group_v": canonical,
            "mapping_basis": mapping["mapping_basis"],
            "confirmed_by_experimenter": int(mapping["confirmed_by_experimenter"]),
            "condition_role": role,
            "condition_group_id": (
                _condition_id(int(round(float(source["frequency_set_hz"]))), float(canonical))
                if role == PRIMARY_ROLE
                else source["condition_group_id"]
            ),
            "vin_rms_v_raw_recomputed_check": raw_recomputed_vin,
            # Preserve the acquisition record exactly as the measured model input.
            "vin_rms_v": source_vin,
            "temperature_ambient_raw_legacy_c": original_ambient,
            "temperature_core_raw_legacy_c": original_core,
            "temperature_ambient_c": original_core,
            "temperature_core_c": original_ambient,
            "temperature_core_raw": original_ambient,
            "temperature_core_corrected": original_ambient,
            "temp_rise_c": original_ambient - original_core,
            "temperature_mapping_correction": MAPPING_CORRECTION,
            "temperature_mapping_verified_date": MAPPING_VERIFIED_DATE,
            "temperature_mapping_version": TEMPERATURE_MAPPING_VERSION,
            "temperature_calibration_applied": 0,
            "temperature_calibration_status": "REFERENCE_BASED_CALIBRATION_NOT_FOUND",
            "B_source": "scope2_ch1_primary_vin",
            "B1024_source": "raw Scope #2 CH1 primary Vin",
            "qc_status": qc_status,
            "qc_hard_fail_reasons": ";".join(sorted(hard)),
            "qc_warning_reasons": ";".join(sorted(warnings)),
            "qc_pass": int(qc_status != "HARD_FAIL"),
            "qc_reason": ";".join(sorted(hard)),
            "qc_warning": ";".join(sorted(warnings)),
            "LSP": "",
            "lsp_pending": 1,
            "LSP_processing_version": "NOT_GENERATED_LSP_DEFINITION_INCOMPLETE",
        }
    )
    return processed


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
    }


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _duplicate_audit(raw_root: Path, sessions: list[Session], new_session: Session, project_root: Path) -> dict[str, Any]:
    sample_owners: dict[str, str] = {}
    duplicate_ids = []
    samples_hashes: dict[str, list[str]] = defaultdict(list)
    for session in sessions:
        samples_path = session.path / "samples.csv"
        samples_hashes[_file_sha(samples_path)].append(str(samples_path))
        for row in session.rows:
            sample_id = row["sample_id"]
            if sample_id in sample_owners:
                duplicate_ids.append({"sample_id": sample_id, "first": sample_owners[sample_id], "second": str(samples_path)})
            else:
                sample_owners[sample_id] = str(samples_path)

    old_hashes: set[str] = set()
    manifest = project_root / "archive_manifest/raw_file_manifest.csv"
    if manifest.is_file():
        old_hashes = {row["sha256"] for row in _read_csv(manifest)}
    new_raw_hashes: dict[str, list[str]] = defaultdict(list)
    for path in sorted((new_session.path / "raw").glob("*.csv")):
        new_raw_hashes[_file_sha(path)].append(str(path))
    return {
        "session_directory_matches": [str(path) for path in raw_root.rglob(new_session.path.name) if path.is_dir()],
        "all_sample_id_count": len(sample_owners),
        "duplicate_sample_ids": duplicate_ids,
        "duplicate_samples_csv_hash_groups": [paths for paths in samples_hashes.values() if len(paths) > 1],
        "new_raw_file_count": sum(len(paths) for paths in new_raw_hashes.values()),
        "new_internal_duplicate_raw_hash_groups": [paths for paths in new_raw_hashes.values() if len(paths) > 1],
        "new_raw_hashes_matching_prior_archive": sorted(digest for digest in new_raw_hashes if digest in old_hashes),
    }


def _write_new_raw_manifest(raw_root: Path, new_session: Session, project_root: Path) -> None:
    paths = [raw_root / "read_temp.ino", new_session.path / "config.json", new_session.path / "samples.csv"]
    paths.extend(sorted((new_session.path / "raw").glob("*.csv")))
    rows = [
        {
            "relative_path": path.relative_to(raw_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": (
                "arduino_temperature_source"
                if path.suffix.lower() == ".ino"
                else "raw_waveform"
                if "/raw/" in path.as_posix()
                else "session_metadata"
            ),
        }
        for path in paths
    ]
    _write_csv(
        project_root / "archive_manifest/new_5p6_raw_manifest.csv",
        rows,
        ["relative_path", "size_bytes", "sha256", "role"],
    )


def _grid(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for voltage in EXPECTED_VOLTAGES:
        for frequency in EXPECTED_FREQUENCIES:
            members = [
                row
                for row in primary
                if float(row["canonical_vin_set_group_v"]) == voltage
                and int(round(float(row["frequency_set_hz"]))) == frequency
            ]
            fe = [row for row in members if row["core_id"] == "FE"]
            commercial = [row for row in members if row["core_id"] == "COMMERCIAL"]
            fe_valid = sum(row["qc_status"] != "HARD_FAIL" for row in fe)
            commercial_valid = sum(row["qc_status"] != "HARD_FAIL" for row in commercial)
            if len(fe) == len(commercial) == 5:
                status = "COMPLETE_ALL_VALID" if fe_valid == commercial_valid == 5 else "COMPLETE_WITH_QC_FAILURES"
            elif members:
                status = "PARTIAL"
            else:
                status = "MISSING"
            rows.append(
                {
                    "canonical_vin_set_group_v": voltage,
                    "frequency_set_hz": frequency,
                    "FE_total": len(fe),
                    "FE_valid": fe_valid,
                    "COMMERCIAL_total": len(commercial),
                    "COMMERCIAL_valid": commercial_valid,
                    "FE_hardfail": len(fe) - fe_valid,
                    "COMMERCIAL_hardfail": len(commercial) - commercial_valid,
                    "group_status": status,
                }
            )
    return rows


def _write_checksums(output: Path, names: list[str]) -> None:
    (output / "checksums_v4.sha256").write_text(
        "".join(f"{sha256_file(output / name)}  {name}\n" for name in names), encoding="utf-8"
    )


def build(raw_root: Path, project_root: Path, protocol: Path, mapping_path: Path) -> dict[str, Any]:
    if sha256_file(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Authoritative MEPI v1.1 protocol hash mismatch")
    sessions = discover_sessions(raw_root)
    finetune_sessions = [session for session in sessions if session.mode == "finetune"]
    new_session, old_high_session = _discover_new_and_old(sessions)
    mapping = _read_mapping(mapping_path)
    if {session.path.name for session in finetune_sessions} != set(mapping):
        raise AssertionError("v4 mapping does not exactly cover discovered finetune sessions")
    if mapping[new_session.path.name]["mapping_basis"] != NEW_MAPPING_BASIS:
        raise AssertionError("New session is not explicitly confirmed as the 5.6-V remeasurement")
    if mapping[old_high_session.path.name]["condition_role"] != OLD_HIGH_ROLE:
        raise AssertionError("Old high-voltage session is not explicitly excluded")

    duplicate_audit = _duplicate_audit(raw_root, sessions, new_session, project_root)
    if (
        len(duplicate_audit["session_directory_matches"]) != 1
        or duplicate_audit["duplicate_sample_ids"]
        or duplicate_audit["duplicate_samples_csv_hash_groups"]
        or duplicate_audit["new_internal_duplicate_raw_hash_groups"]
        or duplicate_audit["new_raw_hashes_matching_prior_archive"]
    ):
        raise AssertionError("Duplicate acquisition evidence detected")

    prior_v2 = {
        row["sample_id"]: row
        for row in _read_csv(project_root / "data/MEPI/v1_1/samples_master_reprocessed_v2.csv")
    }
    master: list[dict[str, Any]] = []
    waveforms: dict[str, np.ndarray] = {}
    for session in finetune_sessions:
        decision = mapping[session.path.name]
        for source in session.rows:
            processed, waveform, _ = process_sample(session, source)
            row = _correct_and_map(processed, source, decision)
            if row["sample_id"] in prior_v2:
                previous = prior_v2[row["sample_id"]]
                if not (
                    float(row["temperature_ambient_c"]) == float(previous["temperature_ambient_c"])
                    and float(row["temperature_core_c"]) == float(previous["temperature_core_c"])
                ):
                    raise AssertionError("Historical temperature mapping differs from v2; possible double-swap")
            master.append(row)
            waveforms[row["sample_id"]] = waveform

    if len(master) != 975 or len({row["sample_id"] for row in master}) != 975:
        raise AssertionError("Expected 975 unique finetune rows including the excluded old 6.x session")
    primary = [row for row in master if row["condition_role"] == PRIMARY_ROLE]
    excluded = [row for row in master if row["condition_role"] == OLD_HIGH_ROLE]
    candidate = [row for row in primary if int(row["qc_pass"]) == 1]
    if len(primary) != 900 or len(excluded) != 75:
        raise AssertionError("Primary/excluded role counts are not 900/75")
    if any(row["source_session_id"] == old_high_session.path.name for row in primary + candidate):
        raise AssertionError("Old 6.x session entered primary or candidate data")
    if any(row["dataset_mode"] == "demo" for row in master + primary + candidate):
        raise AssertionError("Demo row entered a finetune v4 artifact")
    if any(row["B_source"] != "scope2_ch1_primary_vin" for row in primary):
        raise AssertionError("Mixed B source detected in active primary rows")
    if any(row["qc_status"] == "HARD_FAIL" for row in candidate):
        raise AssertionError("HARD_FAIL row entered candidate fine-tune data")

    grid = _grid(primary)
    if len(grid) != 90 or any(row["group_status"] in {"PARTIAL", "MISSING"} for row in grid):
        raise AssertionError("The paired measurement grid is not complete")

    output = project_root / "data/MEPI/v1_1"
    reports = project_root / "reports"
    fields = list(master[0])
    _write_csv(output / "samples_master_reprocessed_v4.csv", master, fields)
    _write_csv(output / "samples_primary_paired_v4.csv", primary, fields)
    _write_csv(output / "samples_candidate_finetune_v4.csv", candidate, fields)

    def save_arrays(prefix: str, rows: list[dict[str, Any]]) -> None:
        np.save(output / f"B1024{prefix}_v4.npy", np.stack([waveforms[row["sample_id"]] for row in rows]).astype(np.float32), allow_pickle=False)
        np.save(output / f"sample_ids{prefix}_v4.npy", np.asarray([row["sample_id"] for row in rows]), allow_pickle=False)

    save_arrays("", master)
    save_arrays("_primary_paired", primary)
    save_arrays("_candidate_finetune", candidate)
    _write_csv(reports / "paired_grid_completeness_v4.csv", grid, list(grid[0]))

    new_rows = [row for row in master if row["source_session_id"] == new_session.path.name]
    old_rows = [row for row in master if row["source_session_id"] == old_high_session.path.name]
    new_qc_fields = [
        "sample_id", "source_session_id", "frequency_set_hz", "frequency_hz", "repeat_id",
        "original_vin_set_group_v", "canonical_vin_set_group_v", "vin_rms_v", "vout_rms_v",
        "vin_thd_percent", "vout_thd_percent", "phase_shift_deg", "input_vi_phase_deg",
        "pin_w", "pout_w", "Pcu_w", "P_loss", "efficiency_percent",
        "vshunt_rms_v_keithley", "vshunt_rms_v_scope", "vshunt_scope_vs_keithley_percent",
        "B_peak_t", "B_rms", "B_thd_percent", "dBdt_max", "form_factor",
        "qc_status", "qc_hard_fail_reasons", "qc_warning_reasons",
    ]
    _write_csv(reports / "new_commercial_5p6_qc_reaudit.csv", new_rows, new_qc_fields)

    source_rows = new_session.rows
    raw_scope1 = list((new_session.path / "raw").glob("*_scope1.csv"))
    raw_scope2 = list((new_session.path / "raw").glob("*_scope2.csv"))
    legacy_b = list((new_session.path / "processed").glob("*.csv"))
    legacy_pass = sum(str(row["qc_pass"]).strip() in {"1", "1.0", "True", "true"} for row in source_rows)
    discovery = f"""# New Commercial 5.6-V Session Discovery

- session_id: `{new_session.path.name}`
- source path: `{new_session.path}`
- core_id: `COMMERCIAL`
- dataset_mode: `finetune`
- row count: {len(source_rows)}
- frequency levels: {len({int(round(float(row['frequency_set_hz']))) for row in source_rows})} ({', '.join(str(value) for value in EXPECTED_FREQUENCIES)} Hz)
- repeat counts: {len({int(row['repeat_id']) for row in source_rows})} identifiers; five rows per frequency
- vin_reference_v min/median/max: {_stats([float(row['vin_reference_v']) for row in source_rows])['min']:.15g} / {_median(source_rows, 'vin_reference_v'):.15g} / {_stats([float(row['vin_reference_v']) for row in source_rows])['max']:.15g} V
- vin_rms_v min/median/max: {_stats([float(row['vin_rms_v']) for row in source_rows])['min']:.15g} / {_median(source_rows, 'vin_rms_v'):.15g} / {_stats([float(row['vin_rms_v']) for row in source_rows])['max']:.15g} V
- raw Scope1 file count: {len(raw_scope1)}
- raw Scope2 file count: {len(raw_scope2)}
- legacy B file count: {len(legacy_b)}
- legacy QC PASS/FAIL counts: {legacy_pass}/{len(source_rows) - legacy_pass}

Identification used row content and coverage, not the directory name alone. Duplicate audit: one matching session directory, zero duplicate sample IDs, zero duplicate `samples.csv` hashes, zero duplicate new raw hashes, and zero new raw hashes matching the prior archive manifest.
"""
    (reports / "new_5p6_discovery.md").write_text(discovery, encoding="utf-8")

    arduino_source = raw_root / "read_temp.ino"
    arduino_audit = f"""# Arduino Temperature Mapping Audit

Source: `{arduino_source}`  
SHA-256: `{sha256_file(arduino_source)}`  
Project copy: `hardware/temperature_reader/read_temp.ino`

## Source behavior

- Lines 3-7 name the D4/D5/D9 MAX6675 channel `roomTemp`.
- Lines 9-13 name the D13/D12/D10 MAX6675 channel `coreTemp`.
- Line 16 configures 9600 baud.
- Lines 22-25 accept the single ASCII command `T`.
- Lines 26-32 perform one Celsius read per channel and transmit `roomTemp,coreTemp\\n`.
- There is no averaging and no slope/offset calibration. The 200-ms delay at line 35 is not calibration.

## Evidence reconciliation

The Arduino identifiers and comment claim Room first/Core second. They conflict with the prior physical tracing and experimenter verification that the installed first channel is TempCore and the installed second channel is TempRoom. Variable names cannot prove physical placement. Because hardware/configuration was unchanged and the new CSV shows the same legacy negative core-minus-ambient pattern before correction, v4 retains the physically verified mapping:

```text
board field 1 = TempCore
board field 2 = TempRoom
temperature_ambient_c = board field 2
temperature_core_c = board field 1
```

This is classified as a legacy Arduino channel-label mismatch, not evidence to reverse the physically verified interpretation. Absolute calibration remains unavailable.
"""
    (reports / "arduino_temperature_mapping_audit.md").write_text(arduino_audit, encoding="utf-8")
    _write_json(reports / "new_5p6_duplicate_audit.json", duplicate_audit)
    _write_new_raw_manifest(raw_root, new_session, project_root)

    primary_qc = Counter(row["qc_status"] for row in primary)
    new_qc = Counter(row["qc_status"] for row in new_rows)
    new_hard = Counter(
        (int(round(float(row["frequency_set_hz"]))), reason)
        for row in new_rows
        for reason in row["qc_hard_fail_reasons"].split(";")
        if reason
    )
    grid_status = Counter(row["group_status"] for row in grid)
    primary_hard = sum(row["qc_status"] == "HARD_FAIL" for row in primary)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "processing_version": PROCESSING_VERSION_V4,
        "new_session_id": new_session.path.name,
        "new_session_rows": len(new_rows),
        "old_high_session_id": old_high_session.path.name,
        "old_high_rows_preserved_excluded": len(old_rows),
        "master_rows": len(master),
        "primary_paired_rows": len(primary),
        "candidate_finetune_rows": len(candidate),
        "candidate_FE_rows": sum(row["core_id"] == "FE" for row in candidate),
        "candidate_COMMERCIAL_rows": sum(row["core_id"] == "COMMERCIAL" for row in candidate),
        "primary_hardfail_rows": primary_hard,
        "primary_qc_counts": dict(primary_qc),
        "new_qc_counts": dict(new_qc),
        "new_hardfail_by_frequency_reason": {f"{frequency}:{reason}": count for (frequency, reason), count in sorted(new_hard.items())},
        "grid_status_counts": dict(grid_status),
        "B1024_v4_shape": [len(master), WAVEFORM_LENGTH],
        "B1024_primary_paired_v4_shape": [len(primary), WAVEFORM_LENGTH],
        "B1024_candidate_finetune_v4_shape": [len(candidate), WAVEFORM_LENGTH],
        "test_accessed": False,
        "split_created": False,
        "normalization_fitted": False,
        "training_run": False,
        "statuses": {
            "NEW_COMMERCIAL_5P6_READY": True,
            "OLD_COMMERCIAL_6X_EXCLUDED": True,
            "B_READY": True,
            "TEMPERATURE_MAPPING_READY": True,
            "TEMPERATURE_CALIBRATION_READY": False,
            "PRIMARY_PAIRED_GRID_MEASURED": True,
            "ELECTRICAL_QC_READY": True,
            "CANDIDATE_FINETUNE_DATA_READY": True,
            "LSP_DEFINITION_READY": False,
            "LSP_READY": False,
            "TRAIN_READY": False,
        },
    }
    _write_json(output / "dataset_summary_v4.json", summary)
    _write_json(
        output / "feature_schema_v4.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "processing_version": PROCESSING_VERSION_V4,
            "tabular_features": list(FINETUNE_TABULAR_FEATURES),
            "waveform": {
                "name": "B(t)_1024",
                "shape": [None, WAVEFORM_LENGTH],
                "source": "raw Scope #2 CH1 primary Vin",
            },
            "targets": ["efficiency_percent", "P_loss", "LSP"],
            "metadata_only": ["core_id"],
            "forbidden_predictive_inputs": [
                "core_id",
                "temperature_core_c",
                "iin_rms_a",
                "vout_rms_v",
                "pin_w",
                "pout_w",
                "Pcu_w",
                "P_loss",
                "efficiency_percent",
                "LSP",
            ],
            "normalization": {
                "status": "NOT_FITTED",
                "reason": "LSP_DEFINITION_INCOMPLETE_AND_NO_SPLIT_CREATED",
                "fit_scope": "train_only_after_group_split",
            },
            "split_status": "NOT_CREATED",
            "test_accessed": False,
        },
    )

    before_ambient = [float(row["temperature_ambient_raw_legacy_c"]) for row in new_rows]
    before_core = [float(row["temperature_core_raw_legacy_c"]) for row in new_rows]
    after_ambient = [float(row["temperature_ambient_c"]) for row in new_rows]
    after_core = [float(row["temperature_core_c"]) for row in new_rows]
    rise = [float(row["temp_rise_c"]) for row in new_rows]
    frequency_lines = []
    for frequency in EXPECTED_FREQUENCIES:
        members = [row for row in new_rows if int(round(float(row["frequency_set_hz"]))) == frequency]
        frequency_lines.append(
            f"| {frequency} | {sum(row['qc_status'] == 'PASS' for row in members)} | "
            f"{sum(row['qc_status'] == 'WARNING' for row in members)} | "
            f"{sum(row['qc_status'] == 'HARD_FAIL' for row in members)} | "
            f"{min(float(row['vout_thd_percent']) for row in members):.6g} | "
            f"{statistics.median(float(row['vout_thd_percent']) for row in members):.6g} | "
            f"{max(float(row['vout_thd_percent']) for row in members):.6g} |"
        )
    report = f"""# MEPI v1.1 New 5.6-V Final Integration

## New Commercial 5.6 session

Path: `{new_session.path}`. Session `{new_session.path.name}` contains 75 rows covering 15 frequencies and five repeats per frequency. `vin_reference_v` min/median/max is 5.65/5.65/5.65 V; preserved measured `vin_rms_v` min/median/max is {min(float(row['vin_rms_v']) for row in new_rows):.6g}/{statistics.median(float(row['vin_rms_v']) for row in new_rows):.6g}/{max(float(row['vin_rms_v']) for row in new_rows):.6g} V. Discovery and duplicate checks passed.

## Arduino temperature mapping

`/mnt/e/MEPI/read_temp.ino` transmits the channel named `roomTemp` first and `coreTemp` second, comma-separated at 9600 baud after command `T`, with no averaging/calibration. Those software identifiers conflict with the physically traced installed sensor order. Physical tracing plus experimenter verification remain authoritative: field 1 is TempCore and field 2 is TempRoom. See `reports/arduino_temperature_mapping_audit.md`.

## Temperature correction

The raw legacy labels were preserved and the swap was applied exactly once. Before correction, ambient-label/core-label means were {statistics.mean(before_ambient):.6g}/{statistics.mean(before_core):.6g} C. After correction, ambient/core means are {statistics.mean(after_ambient):.6g}/{statistics.mean(after_core):.6g} C. Corrected rise min/median/max/mean is {min(rise):.6g}/{statistics.median(rise):.6g}/{max(rise):.6g}/{statistics.mean(rise):.6g} C; negative/zero/positive counts are {sum(value < 0 for value in rise)}/{sum(value == 0 for value in rise)}/{sum(value > 0 for value in rise)}. No affine calibration was inferred.

## B reprocessing

Every v4 finetune row was deterministically rebuilt from raw Scope #2 CH1 primary Vin with Np={N_PRIMARY} and Ae={EFFECTIVE_AREA_M2:.9g} m2, complete-cycle integration/drift correction and 1024-point resampling. All active primary rows assert `B_source = scope2_ch1_primary_vin`; legacy acquisition B is excluded. New-session B peak range is {min(float(row['B_peak_t']) for row in new_rows):.6g}-{max(float(row['B_peak_t']) for row in new_rows):.6g} T and B THD range is {min(float(row['B_thd_percent']) for row in new_rows):.6g}-{max(float(row['B_thd_percent']) for row in new_rows):.6g}%.

## Electrical QC

New-session PASS/WARNING/HARD_FAIL = {new_qc['PASS']}/{new_qc['WARNING']}/{new_qc['HARD_FAIL']}. All ten hard failures are `VOUT_THD > {THD_LIMIT_PERCENT:g}%`: five at 1000 Hz and five at 1250 Hz. Thresholds were not changed.

| Frequency (Hz) | PASS | WARNING | HARD_FAIL | Vout THD min (%) | median (%) | max (%) |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(frequency_lines)}

## Commercial 6.x legacy session

Session `{old_high_session.path.name}` remains preserved with {len(old_rows)} rows and original raw evidence. All rows are marked `{OLD_HIGH_ROLE}` and are absent from the primary paired dataset, candidate fine-tune rows, split candidates and normalization candidates. It is not counted as 5.6 V.

## Primary paired grid

Measurement coverage is 900 rows across all 90 intended voltage-frequency groups: {grid_status['COMPLETE_ALL_VALID']} `COMPLETE_ALL_VALID` and {grid_status['COMPLETE_WITH_QC_FAILURES']} `COMPLETE_WITH_QC_FAILURES`; no group is PARTIAL or MISSING. Primary PASS/WARNING/HARD_FAIL = {primary_qc['PASS']}/{primary_qc['WARNING']}/{primary_qc['HARD_FAIL']}.

## Candidate fine-tune dataset

Candidate QC-valid rows: {len(candidate)} total, with {summary['candidate_FE_rows']} FE and {summary['candidate_COMMERCIAL_rows']} COMMERCIAL. It excludes {primary_hard} primary HARD_FAIL rows, all demo rows and the old 6.x session. The feature schema remains exactly the frozen nine inputs plus `B(t)_1024`; core ID is metadata only. LSP is still blank, so these are candidate rows rather than final train-ready rows.

## Remaining blockers

No independent reference-thermometer calibration exists. The new Arduino/acquisition files supply no LSP constants or rules. Numeric Ea, epsilon, the train-only Tref estimator, LSP normalization/scaling/bounds and PIRL residual normalization remain unresolved. No split was created, normalization was not fitted, the test subset was not accessed and training was not run.

NEW_COMMERCIAL_5P6_READY = TRUE
OLD_COMMERCIAL_6X_EXCLUDED = TRUE
B_READY = TRUE
TEMPERATURE_MAPPING_READY = TRUE
TEMPERATURE_CALIBRATION_READY = FALSE
PRIMARY_PAIRED_GRID_MEASURED = TRUE
ELECTRICAL_QC_READY = TRUE
CANDIDATE_FINETUNE_DATA_READY = TRUE
LSP_DEFINITION_READY = FALSE
LSP_READY = FALSE
TRAIN_READY = FALSE
"""
    (reports / "MEPI_V1_1_NEW_5P6_FINAL_INTEGRATION.md").write_text(report, encoding="utf-8")

    names = [
        "samples_master_reprocessed_v4.csv",
        "samples_primary_paired_v4.csv",
        "samples_candidate_finetune_v4.csv",
        "B1024_v4.npy",
        "sample_ids_v4.npy",
        "B1024_primary_paired_v4.npy",
        "sample_ids_primary_paired_v4.npy",
        "B1024_candidate_finetune_v4.npy",
        "sample_ids_candidate_finetune_v4.npy",
        "dataset_summary_v4.json",
        "feature_schema_v4.json",
    ]
    _write_checksums(output, names)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/e/MEPI"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=Path("MEPI-FROZEN-PROTOCOL v1.1.md"))
    parser.add_argument("--mapping", type=Path, default=Path("configs/finetune_session_voltage_mapping_v4.csv"))
    args = parser.parse_args()
    summary = build(
        args.raw_root.resolve(),
        args.project_root.resolve(),
        args.protocol.resolve(),
        args.mapping.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
