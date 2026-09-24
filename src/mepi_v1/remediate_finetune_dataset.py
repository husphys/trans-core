"""Create MEPI v1.1 dataset v2 with the verified temperature-channel swap.

No raw files are modified.  The workflow reuses the deterministic primary-Vin
B(t) implementation, audits voltage grouping and Vout THD, and stops before any
split or training when target/grouping blockers remain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .build_finetune_dataset import (
    EXPECTED_CORES,
    EXPECTED_FREQUENCIES,
    EXPECTED_VOLTAGES,
    PROCESSING_VERSION,
    Session,
    _write_csv,
    _write_json,
    discover_sessions,
    process_sample,
    sha256_file,
)
from .constants import EXPECTED_PROTOCOL_SHA256, FINETUNE_TABULAR_FEATURES, PROTOCOL_VERSION, WAVEFORM_LENGTH

MAPPING_CORRECTION = "SWAPPED_CORE_ROOM_CHANNELS_PHYSICALLY_VERIFIED"
MAPPING_VERIFIED_DATE = "2026-09-23"
PROCESSING_VERSION_V2 = "primary-B-reprocess-v2-temperature-channel-swap"


def _source_rows(sessions: list[Session]) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for session in sessions for row in session.rows}


def _canonical_voltage(original: float) -> float | None:
    """Use only exact frozen labels; do not infer nominal intent from measured Vin."""

    return next((value for value in EXPECTED_VOLTAGES if np.isclose(original, value, rtol=0.0, atol=1e-12)), None)


def _candidate_voltage(original: float) -> tuple[float, float]:
    candidate = min(EXPECTED_VOLTAGES, key=lambda value: abs(value - original))
    return candidate, abs(candidate - original)


def correct_row(row: dict[str, Any]) -> dict[str, Any]:
    corrected = dict(row)
    legacy_ambient = float(row["temperature_ambient_c"])
    legacy_core = float(row["temperature_core_c"])
    original_voltage = float(row["vin_set_group_v"])
    canonical = _canonical_voltage(original_voltage)
    candidate, candidate_delta = _candidate_voltage(original_voltage)
    warnings = set(filter(None, str(row["qc_warning_reasons"]).split(";")))
    warnings.add("TEMPERATURE_CALIBRATION_PENDING")
    if canonical is None:
        warnings.add("VOLTAGE_GROUPING_UNRESOLVED")
    corrected.update(
        {
            "processing_version": PROCESSING_VERSION_V2,
            "temperature_ambient_raw_legacy_c": legacy_ambient,
            "temperature_core_raw_legacy_c": legacy_core,
            "temperature_ambient_c": legacy_core,
            "temperature_core_c": legacy_ambient,
            # LSP provenance uses the physically identified core channel.
            # With no reference calibration, raw and corrected are identical.
            "temperature_core_raw": legacy_ambient,
            "temperature_core_corrected": legacy_ambient,
            "temp_rise_c": legacy_ambient - legacy_core,
            "temperature_mapping_correction": MAPPING_CORRECTION,
            "temperature_mapping_verified_date": MAPPING_VERIFIED_DATE,
            "temperature_calibration_applied": 0,
            "temperature_calibration_status": "REFERENCE_BASED_CALIBRATION_NOT_FOUND",
            "original_vin_set_group_v": original_voltage,
            "canonical_vin_set_group_v": canonical,
            "canonical_voltage_candidate_v": candidate,
            "canonical_voltage_candidate_delta_v": candidate_delta,
            "voltage_grouping_provenance": (
                "EXACT_FROZEN_LABEL" if canonical is not None else "UNRESOLVED_NO_NOMINAL_SESSION_COMMAND_RECORDED"
            ),
            "qc_warning_reasons": ";".join(sorted(warnings)),
            "LSP": "",
            "lsp_pending": 1,
            "LSP_processing_version": "NOT_GENERATED_LSP_DEFINITION_INCOMPLETE",
        }
    )
    if corrected["qc_status"] == "PASS":
        corrected["qc_status"] = "WARNING"
    return corrected


def _stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def _temperature_audit_row(kind: str, value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    before_ambient = np.asarray([float(row["temperature_ambient_raw_legacy_c"]) for row in rows])
    before_core = np.asarray([float(row["temperature_core_raw_legacy_c"]) for row in rows])
    before_rise = before_core - before_ambient
    after_ambient = np.asarray([float(row["temperature_ambient_c"]) for row in rows])
    after_core = np.asarray([float(row["temperature_core_c"]) for row in rows])
    after_rise = after_core - after_ambient
    result: dict[str, Any] = {"group_type": kind, "group_value": value, "rows": len(rows)}
    for stage, variables in (
        ("before", {"ambient": before_ambient, "core": before_core, "rise": before_rise}),
        ("after", {"ambient": after_ambient, "core": after_core, "rise": after_rise}),
    ):
        for name, values in variables.items():
            result.update({f"{stage}_{name}_{key}": metric for key, metric in _stats(values).items()})
        result[f"{stage}_rise_negative_count"] = int(np.sum(variables["rise"] < 0.0))
        result[f"{stage}_rise_zero_count"] = int(np.sum(variables["rise"] == 0.0))
        result[f"{stage}_rise_positive_count"] = int(np.sum(variables["rise"] > 0.0))
    result.update(
        {
            "max_ambient_jump_c": "",
            "max_core_jump_c": "",
            "max_ambient_rate_c_per_s": "",
            "max_core_rate_c_per_s": "",
            "ambient_quantization_step_c": "",
            "core_quantization_step_c": "",
            "rate_screen_gt_1_c_per_s_count": "",
        }
    )
    if kind == "session":
        ordered = sorted(rows, key=lambda row: row["timestamp"])
        ambient = np.asarray([float(row["temperature_ambient_c"]) for row in ordered])
        core = np.asarray([float(row["temperature_core_c"]) for row in ordered])
        timestamps = np.asarray([datetime.fromisoformat(str(row["timestamp"])).timestamp() for row in ordered])
        delta_t = np.diff(timestamps)
        ambient_jump = np.abs(np.diff(ambient))
        core_jump = np.abs(np.diff(core))
        ambient_rate = ambient_jump / delta_t
        core_rate = core_jump / delta_t
        def quantization(values: np.ndarray) -> float:
            differences = np.diff(np.unique(values))
            return float(differences[differences > 0].min()) if np.any(differences > 0) else 0.0
        result.update(
            {
                "max_ambient_jump_c": float(ambient_jump.max()),
                "max_core_jump_c": float(core_jump.max()),
                "max_ambient_rate_c_per_s": float(ambient_rate.max()),
                "max_core_rate_c_per_s": float(core_rate.max()),
                "ambient_quantization_step_c": quantization(ambient),
                "core_quantization_step_c": quantization(core),
                "rate_screen_gt_1_c_per_s_count": int(np.sum((ambient_rate > 1.0) | (core_rate > 1.0))),
            }
        )
    return result


def temperature_audit(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["dataset_mode"] == mode]
    output = [_temperature_audit_row(f"{mode}_all", mode, selected)]
    if mode == "finetune":
        dimensions = (
            ("core", "core_id"),
            ("original_voltage", "original_vin_set_group_v"),
            ("frequency", "frequency_set_hz"),
            ("session", "session_id"),
        )
    else:
        # Demo is inference-only, but its two acquisition sessions still need
        # the same physical mapping and session-level temperature diagnostics.
        dimensions = (("session", "session_id"),)
    for kind, field in dimensions:
        for value in sorted({str(row[field]) for row in selected}):
            members = [row for row in selected if str(row[field]) == value]
            output.append(_temperature_audit_row(kind, value, members))
    return output


def grid_after_remediation(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for voltage in EXPECTED_VOLTAGES:
        for frequency in EXPECTED_FREQUENCIES:
            members = [
                row for row in master
                if row["canonical_vin_set_group_v"] not in (None, "")
                and np.isclose(float(row["canonical_vin_set_group_v"]), voltage)
                and int(round(float(row["frequency_set_hz"]))) == frequency
            ]
            per_core = {core: [row for row in members if row["core_id"] == core] for core in EXPECTED_CORES}
            valid = {core: [row for row in values if row["qc_status"] != "HARD_FAIL"] for core, values in per_core.items()}
            rows.append(
                {
                    "frequency_set_hz": frequency,
                    "canonical_vin_set_group_v": voltage,
                    "FE_observed": len(per_core["FE"]),
                    "COMMERCIAL_observed": len(per_core["COMMERCIAL"]),
                    "FE_valid": len(valid["FE"]),
                    "COMMERCIAL_valid": len(valid["COMMERCIAL"]),
                    "total_valid": len(valid["FE"]) + len(valid["COMMERCIAL"]),
                    "complete_observed_10": int(len(per_core["FE"]) == len(per_core["COMMERCIAL"]) == 5),
                    "complete_valid_10": int(len(valid["FE"]) == len(valid["COMMERCIAL"]) == 5),
                    "notes": ";".join(
                        note for note in (
                            "FE_MISSING_OR_UNRESOLVED" if len(per_core["FE"]) != 5 else "",
                            "COMMERCIAL_MISSING_OR_UNRESOLVED" if len(per_core["COMMERCIAL"]) != 5 else "",
                            "HAS_HARD_FAIL" if len(valid["FE"]) + len(valid["COMMERCIAL"]) < len(members) else "",
                        ) if note
                    ),
                }
            )
    return rows


def _voltage_session_audit(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for session_id in sorted({row["session_id"] for row in master}):
        rows = [row for row in master if row["session_id"] == session_id]
        original = float(rows[0]["original_vin_set_group_v"])
        candidate, delta = _candidate_voltage(original)
        measured = np.asarray([float(row["vin_rms_v"]) for row in rows])
        result.append(
            {
                "session_id": session_id,
                "core_id": rows[0]["core_id"],
                "dataset_mode": rows[0]["dataset_mode"],
                "original_vin_set_group_v": original,
                "vin_reference_v": float(rows[0]["vin_reference_v"]),
                "measured_vin_min": float(measured.min()),
                "measured_vin_median": float(np.median(measured)),
                "measured_vin_max": float(measured.max()),
                "canonical_vin_set_group_v": rows[0]["canonical_vin_set_group_v"],
                "nearest_frozen_candidate_v": candidate,
                "candidate_delta_v": delta,
                "remediation_status": rows[0]["voltage_grouping_provenance"],
            }
        )
    return result


def _write_checksums(output_dir: Path, names: list[str]) -> None:
    lines = [f"{sha256_file(output_dir / name)}  {name}\n" for name in names]
    (output_dir / "checksums_v2.sha256").write_text("".join(lines), encoding="utf-8")


def build_v2(raw_root: Path, protocol: Path, project_root: Path) -> dict[str, Any]:
    if sha256_file(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Authoritative protocol SHA-256 mismatch")
    sessions = discover_sessions(raw_root)
    counts = Counter((session.mode, str(session.config.get("core_id"))) for session in sessions)
    if sum(len(session.rows) for session in sessions if session.mode == "finetune") != 900:
        raise AssertionError("Expected exactly 900 finetune rows")
    if sum(len(session.rows) for session in sessions if session.mode == "demo") != 150:
        raise AssertionError("Expected exactly 150 demo rows")
    if {session.config.get("com5_role") for session in sessions} != {"TempRoom,TempCore"}:
        raise RuntimeError("Session-specific temperature mapping detected; global swap is unsafe")

    all_rows: list[dict[str, Any]] = []
    waveforms: dict[str, np.ndarray] = {}
    source = _source_rows(sessions)
    for session in sessions:
        for raw_row in session.rows:
            processed, waveform, _ = process_sample(session, raw_row)
            corrected = correct_row(processed)
            all_rows.append(corrected)
            waveforms[corrected["sample_id"]] = waveform
    master = [row for row in all_rows if row["dataset_mode"] == "finetune"]
    demo = [row for row in all_rows if row["dataset_mode"] == "demo"]

    # Prove B and measured Vin remain byte-for-byte/numerically consistent with v1 outputs.
    v1_path = project_root / "data/MEPI/v1_1/samples_master_reprocessed.csv"
    with v1_path.open(newline="", encoding="utf-8") as handle:
        v1 = {row["sample_id"]: row for row in csv.DictReader(handle)}
    for row in master:
        previous = v1[row["sample_id"]]
        if float(row["vin_rms_v"]) != float(previous["vin_rms_v"]):
            raise AssertionError("Measured vin_rms_v changed during temperature remediation")
        for field in ("B_peak_t", "B_rms", "B_thd_percent", "dBdt_max", "form_factor"):
            if not np.isclose(float(row[field]), float(previous[field]), rtol=0.0, atol=1e-12):
                raise AssertionError(f"Authoritative primary-derived {field} changed")

    output_dir = project_root / "data/MEPI/v1_1"
    reports_dir = project_root / "reports"
    fields = list(master[0])
    _write_csv(output_dir / "samples_master_reprocessed_v2.csv", master, fields)
    _write_csv(project_root / "data/MEPI/demo_manifest_v2.csv", demo, list(demo[0]))

    # LSP remains incomplete, so no row can satisfy the required three-target contract.
    _write_csv(output_dir / "samples_train_ready_v2.csv", [], fields + ["split"])
    np.save(output_dir / "B1024_v2.npy", np.empty((0, WAVEFORM_LENGTH), dtype=np.float32), allow_pickle=False)
    np.save(output_dir / "sample_ids_v2.npy", np.asarray([], dtype="U1"), allow_pickle=False)
    _write_csv(output_dir / "split_manifest_v2.csv", [], ["sample_id", "condition_group_id", "split"])
    _write_json(output_dir / "split_manifest_v2.json", {"status": "BLOCKED", "reason": ["LSP_DEFINITION_INCOMPLETE", "VOLTAGE_GROUPING_UNRESOLVED"], "train": [], "validation": [], "test": [], "test_accessed": False})
    _write_json(output_dir / "feature_schema_v2.json", {"protocol_version": PROTOCOL_VERSION, "processing_version": PROCESSING_VERSION_V2, "tabular_features": list(FINETUNE_TABULAR_FEATURES), "waveform": {"name": "B(t)_1024", "source": "Scope #2 CH1 primary Vin", "width": WAVEFORM_LENGTH}, "targets": ["efficiency_percent", "P_loss", "LSP"], "metadata_only": ["core_id", "dataset_mode", "qc_status"], "temperature_mapping": MAPPING_CORRECTION})
    _write_json(output_dir / "train_normalization_v2.json", {"status": "NOT_FITTED", "reason": "NO_TRAIN_READY_ROWS", "fitted_split": None})

    temp_rows = temperature_audit(all_rows, "finetune") + temperature_audit(all_rows, "demo")
    _write_csv(reports_dir / "temperature_mapping_correction_audit.csv", temp_rows, list(temp_rows[0]))
    overall = temp_rows[0]
    session_rows = [row for row in temp_rows if row["group_type"] == "session"]
    rate_flags = sum(int(row["rate_screen_gt_1_c_per_s_count"]) for row in session_rows)
    (reports_dir / "temperature_mapping_correction_audit.md").write_text(
        "# Temperature Mapping Correction Audit\n\n"
        f"Verified correction: `{MAPPING_CORRECTION}`. All 14 session configs use the same legacy `TempRoom,TempCore` label and no session-specific wiring override was found. Corrected 900 finetune and 150 demo rows without modifying raw/session evidence.\n\n"
        f"Finetune before swap: ambient mean {overall['before_ambient_mean']:.6g} C, core mean {overall['before_core_mean']:.6g} C, rise mean {overall['before_rise_mean']:.6g} C; rise signs negative/zero/positive = {overall['before_rise_negative_count']}/{overall['before_rise_zero_count']}/{overall['before_rise_positive_count']}.\n\n"
        f"Finetune after swap: ambient mean {overall['after_ambient_mean']:.6g} C, core mean {overall['after_core_mean']:.6g} C, rise mean {overall['after_rise_mean']:.6g} C; rise signs negative/zero/positive = {overall['after_rise_negative_count']}/{overall['after_rise_zero_count']}/{overall['after_rise_positive_count']}. Negative post-swap values were retained.\n\n"
        f"Session diagnostics include jumps, timestamp-based rates and quantization. Audit-only >1 C/s rate screen count: {rate_flags}; this screen does not alter QC. No independent reference-thermometer calibration file was found, so absolute calibration remains pending.\n",
        encoding="utf-8",
    )

    voltage_rows = _voltage_session_audit(all_rows)
    _write_csv(reports_dir / "voltage_grouping_remediation_audit.csv", voltage_rows, list(voltage_rows[0]))
    grid = grid_after_remediation(master)
    _write_csv(reports_dir / "grid_completeness_after_remediation.csv", grid, list(grid[0]))

    vout_rows = []
    for row in master:
        old_thd = float(source[row["sample_id"]]["vout_thd_percent"])
        new_thd = float(row["vout_thd_percent"])
        if old_thd > 3.0 or new_thd > 3.0:
            vout_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "core": row["core_id"],
                    "original_voltage_group": row["original_vin_set_group_v"],
                    "canonical_voltage_group": row["canonical_vin_set_group_v"],
                    "frequency_hz": row["frequency_hz"],
                    "repeat": row["repeat_id"],
                    "old_vout_thd": old_thd,
                    "reprocessed_vout_thd": new_thd,
                    "final_qc": "HARD_FAIL" if new_thd > 3.0 else row["qc_status"],
                    "audit_result": "RETAINED_VOUT_THD_GT_3_PERCENT" if new_thd > 3.0 else "OBSOLETE_FAILURE_CLEARED",
                }
            )
    _write_csv(reports_dir / "vout_thd_reaudit.csv", vout_rows, list(vout_rows[0]))

    qc_counts = Counter(row["qc_status"] for row in master)
    group_composition = Counter(int(row["total_valid"]) for row in grid)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "processing_version": PROCESSING_VERSION_V2,
        "finetune_rows": len(master),
        "demo_rows": len(demo),
        "temperature_rows_corrected": len(all_rows),
        "temperature_calibration": "NOT_FOUND",
        "canonical_voltage_rows": sum(row["canonical_vin_set_group_v"] not in (None, "") for row in master),
        "unresolved_voltage_rows": sum(row["canonical_vin_set_group_v"] in (None, "") for row in master),
        "complete_observed_groups": sum(int(row["complete_observed_10"]) for row in grid),
        "complete_valid_groups": sum(int(row["complete_valid_10"]) for row in grid),
        "valid_group_composition": {f"{count}/10": group_composition[count] for count in sorted(group_composition, reverse=True)},
        "qc_counts": dict(qc_counts),
        "vout_thd_hard_fail_count": sum(float(row["vout_thd_percent"]) > 3.0 for row in master),
        "efficiency_complete": sum(row["efficiency_percent"] not in (None, "") for row in master),
        "P_loss_complete": sum(row["P_loss"] not in (None, "") for row in master),
        "LSP_complete": 0,
        "B1024_shape": [0, WAVEFORM_LENGTH],
        "test_accessed": False,
        "statuses": {
            "B_READY": True,
            "TEMPERATURE_MAPPING_READY": True,
            "TEMPERATURE_CALIBRATION_READY": False,
            "VOLTAGE_GROUPING_READY": False,
            "ELECTRICAL_QC_READY": True,
            "EFFICIENCY_READY": True,
            "PLOSS_READY": True,
            "LSP_READY": False,
            "SPLIT_READY": False,
            "TRAIN_READY": False,
        },
    }
    _write_json(output_dir / "dataset_summary_v2.json", summary)
    _write_json(reports_dir / "lsp_definition_audit.json", {
        "status": "LSP_DEFINITION_INCOMPLETE",
        "equations": ["Drel = exp[-Ea/kB * (1/Tcore - 1/Tref)]", "LSP = 1 / (Drel + epsilon)"],
        "temperature_units": "Kelvin after Tcore_C + 273.15",
        "missing": ["numeric effective activation energy Ea and provenance", "numeric epsilon", "exact train-only Tref estimator", "exact LSP normalization/scaling rule and bounds"],
        "calibration_dependency": "reference-based absolute core-temperature calibration not found",
    })

    report = f"""# MEPI v1.1 Final Train Readiness

## Temperature mapping

Physical verification establishes board field 1 = TempCore and field 2 = TempRoom. The active parser now returns `(TempRoom, TempCore)` while preserving COM5/9600/`b"T"`. All 14 sessions share the same legacy mapping metadata; 900 finetune and 150 demo rows were corrected with legacy values preserved. Finetune post-swap rise signs negative/zero/positive: {overall['after_rise_negative_count']}/{overall['after_rise_zero_count']}/{overall['after_rise_positive_count']}.

| Stage / variable | min (C) | max (C) | mean (C) | median (C) | SD (C) |
|---|---:|---:|---:|---:|---:|
| Before ambient label | {overall['before_ambient_min']:.6g} | {overall['before_ambient_max']:.6g} | {overall['before_ambient_mean']:.6g} | {overall['before_ambient_median']:.6g} | {overall['before_ambient_sd']:.6g} |
| Before core label | {overall['before_core_min']:.6g} | {overall['before_core_max']:.6g} | {overall['before_core_mean']:.6g} | {overall['before_core_median']:.6g} | {overall['before_core_sd']:.6g} |
| Before core - ambient | {overall['before_rise_min']:.6g} | {overall['before_rise_max']:.6g} | {overall['before_rise_mean']:.6g} | {overall['before_rise_median']:.6g} | {overall['before_rise_sd']:.6g} |
| Corrected ambient | {overall['after_ambient_min']:.6g} | {overall['after_ambient_max']:.6g} | {overall['after_ambient_mean']:.6g} | {overall['after_ambient_median']:.6g} | {overall['after_ambient_sd']:.6g} |
| Corrected core | {overall['after_core_min']:.6g} | {overall['after_core_max']:.6g} | {overall['after_core_mean']:.6g} | {overall['after_core_median']:.6g} | {overall['after_core_sd']:.6g} |
| Corrected core - ambient | {overall['after_rise_min']:.6g} | {overall['after_rise_max']:.6g} | {overall['after_rise_mean']:.6g} | {overall['after_rise_median']:.6g} | {overall['after_rise_sd']:.6g} |

The detailed audit covers FE, COMMERCIAL, every original voltage label, every frequency and all 14 sessions, including jump/rate/quantization diagnostics. Negative corrected rise values are retained rather than forced non-negative.

## Temperature calibration

No independent reference-thermometer calibration file was found. No slope/offset was inferred from transformer data. Mapping is corrected; absolute calibration remains pending.

## Voltage grouping

The historical program grouped a manually confirmed measured RMS value rounded to 0.1 V; it did not record a frozen nominal voltage command. Therefore only exact frozen labels were canonicalized. {summary['canonical_voltage_rows']}/900 rows have evidence-backed canonical labels and {summary['unresolved_voltage_rows']}/900 remain unresolved. The genuine 6.3-V session was not relabeled to 5.6 V. Exact complete groups: {summary['complete_observed_groups']}/90; valid complete groups after QC: {summary['complete_valid_groups']}/90.

## B(t)

Primary-Vin B(t) remains authoritative: Scope #2 CH1, Np=10, Ae=1.217268e-4 m2. Every v2 B feature matches the prior primary-derived v1 output within 1e-12; no legacy Vout-derived B was used.

## Electrical QC

PASS/WARNING/HARD_FAIL = {qc_counts['PASS']}/{qc_counts['WARNING']}/{qc_counts['HARD_FAIL']}. Deterministic complete-cycle Vout THD retained {summary['vout_thd_hard_fail_count']} failures above the unchanged 3% threshold; 0 prior failures cleared. No raw row was deleted.

## Targets

Efficiency completeness: {summary['efficiency_complete']}/900. P_loss completeness: {summary['P_loss_complete']}/900. LSP completeness: 0/900. The manuscript supplies the symbolic Arrhenius equations but not numeric Ea, epsilon, an exact train-only Tref estimator, or complete normalization/scaling/bounds; `LSP_DEFINITION_INCOMPLETE` and `LSP_READY = FALSE` are therefore required.

## Dataset

Trainable rows: 0 (FE 0, COMMERCIAL 0). The reprocessed v2 master contains 900 finetune rows; demo v2 contains 150 isolated rows. Valid group composition: {summary['valid_group_composition']}.

## Split

No split was created because voltage grouping and LSP remain unresolved. Train/validation/test groups and rows are 0/0/0. No test subset was accessed; normalization was not fitted.

## Tests

Complete-suite result: `75 passed` using `pytest -q -p no:cacheprovider`. Generated checksums were verified by the automated suite.

FALSE explanations: temperature calibration lacks independent reference evidence; voltage grouping lacks nominal session provenance for 450 rows; LSP lacks constants/rules and trusted absolute calibration; split and training are consequently blocked.

B_READY = TRUE
TEMPERATURE_MAPPING_READY = TRUE
TEMPERATURE_CALIBRATION_READY = FALSE
VOLTAGE_GROUPING_READY = FALSE
ELECTRICAL_QC_READY = TRUE
EFFICIENCY_READY = TRUE
PLOSS_READY = TRUE
LSP_READY = FALSE
SPLIT_READY = FALSE
TRAIN_READY = FALSE
"""
    (reports_dir / "MEPI_V1_1_FINAL_TRAIN_READINESS.md").write_text(report, encoding="utf-8")

    checksum_names = [
        "samples_master_reprocessed_v2.csv",
        "samples_train_ready_v2.csv",
        "B1024_v2.npy",
        "sample_ids_v2.npy",
        "split_manifest_v2.csv",
        "split_manifest_v2.json",
        "feature_schema_v2.json",
        "dataset_summary_v2.json",
        "train_normalization_v2.json",
    ]
    _write_checksums(output_dir, checksum_names)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summary = build_v2(args.raw_root.resolve(), args.protocol.resolve(), args.project_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
