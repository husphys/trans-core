"""Build the MEPI v1.1 experimental dataset from immutable raw acquisitions.

This module performs no hardware control and no model training.  Legacy B1024
files are read only for comparison; every scientific waveform is reconstructed
from Scope #2 CH1 primary voltage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    EXPECTED_PROTOCOL_SHA256,
    FINETUNE_TABULAR_FEATURES,
    PROTOCOL_VERSION,
    R_SHUNT_TOTAL_OHM,
    WAVEFORM_LENGTH,
)
from .waveform import TransformerConstants, voltage_to_flux_density

EXPECTED_VOLTAGES = (2.8, 3.3, 3.9, 4.5, 5.0, 5.6)
EXPECTED_FREQUENCIES = tuple(range(1000, 4501, 250))
EXPECTED_CORES = ("FE", "COMMERCIAL")
PROCESSING_VERSION = "primary-B-reprocess-v1"
N_PRIMARY = 10
EFFECTIVE_AREA_M2 = 1.217268e-4
THD_LIMIT_PERCENT = 3.0
SHUNT_SCOPE_FAIL_PERCENT = 25.0
PEAK_DISPLAY_LIMIT = 0.90


@dataclass(frozen=True)
class Session:
    path: Path
    mode: str
    config: dict[str, Any]
    rows: list[dict[str, str]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def discover_sessions(raw_root: Path) -> list[Session]:
    sessions: list[Session] = []
    data_root = raw_root / "MEPI_DATA"
    for mode in ("finetune", "demo"):
        for path in sorted((data_root / mode).glob("session_*")):
            config_path = path / "config.json"
            samples_path = path / "samples.csv"
            if not config_path.is_file() or not samples_path.is_file():
                raise FileNotFoundError(f"Incomplete session metadata: {path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            rows = _read_csv(samples_path)
            sessions.append(Session(path=path, mode=mode, config=config, rows=rows))
    if not sessions:
        raise FileNotFoundError(f"No MEPI sessions found under {data_root}")
    return sessions


def _resolve_session_file(session: Session, value: str) -> Path:
    return session.path / value.replace("\\", "/")


def build_inventory(raw_root: Path, sessions: list[Session]) -> dict[str, Any]:
    file_paths = sorted(path for path in raw_root.rglob("*") if path.is_file())
    session_lookup = {session.path: session for session in sessions}
    file_inventory: list[dict[str, Any]] = []
    for path in file_paths:
        owner = next((session for root, session in session_lookup.items() if path.is_relative_to(root)), None)
        if "/raw/" in path.as_posix():
            category = "raw_waveform"
        elif "/processed/" in path.as_posix():
            category = "legacy_processed_B1024"
        elif "/preview/" in path.as_posix():
            category = "preview"
        elif path.suffix.lower() == ".py":
            category = "acquisition_code"
        elif path.name == "samples.csv":
            category = "sample_index"
        elif path.name.endswith("config.json"):
            category = "config"
        else:
            category = "supporting"
        file_inventory.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "category": category,
            "core": owner.config.get("core_id") if owner else None,
            "dataset_mode": owner.mode if owner else None,
            "session_id": owner.path.name if owner else None,
        })
    session_rows: list[dict[str, Any]] = []
    for session in sessions:
        sample_ids = [row.get("sample_id", "") for row in session.rows]
        condition_keys = [
            (row.get("condition_group_id", ""), row.get("core_id", ""), row.get("repeat_id", ""))
            for row in session.rows
        ]
        missing: list[str] = []
        for row in session.rows:
            for field in ("raw_scope1_file", "raw_scope2_file", "processed_B1024_file"):
                value = row.get(field, "")
                if not value or not _resolve_session_file(session, value).is_file():
                    missing.append(f"{row.get('sample_id')}:{field}")
        raw_files = list((session.path / "raw").glob("*.csv"))
        b_files = list((session.path / "processed").glob("*_B1024.csv"))
        session_rows.append(
            {
                "path": str(session.path),
                "size_bytes": sum(p.stat().st_size for p in session.path.rglob("*") if p.is_file()),
                "core": session.config.get("core_id"),
                "dataset_mode": session.mode,
                "voltage_level": session.config.get("vin_set_group_v"),
                "vin_reference_v": session.config.get("vin_reference_v"),
                "number_of_frequencies": len({row.get("frequency_set_hz") for row in session.rows}),
                "repeats": sorted({int(row["repeat_id"]) for row in session.rows}),
                "csv_rows": len(session.rows),
                "raw_waveform_files": len(raw_files),
                "raw_scope1_files": sum(p.name.endswith("_scope1.csv") for p in raw_files),
                "raw_scope2_files": sum(p.name.endswith("_scope2.csv") for p in raw_files),
                "B1024_files": len(b_files),
                "preview_files": len(list((session.path / "preview").glob("*"))),
                "legacy_qc_pass": sum(row.get("qc_pass") == "1" for row in session.rows),
                "legacy_qc_fail": sum(row.get("qc_pass") != "1" for row in session.rows),
                "missing_files": missing,
                "duplicate_sample_ids": sorted(k for k, v in Counter(sample_ids).items() if v > 1),
                "duplicate_condition_rows": sum(v - 1 for v in Counter(condition_keys).values() if v > 1),
            }
        )
    return {
        "raw_root": str(raw_root),
        "total_size_bytes": sum(path.stat().st_size for path in file_paths),
        "total_files": len(file_paths),
        "file_type_counts": dict(sorted(Counter(path.suffix.lower() or "NO_SUFFIX" for path in file_paths).items())),
        "acquisition_scripts": [str(path) for path in raw_root.rglob("*.py")],
        "config_files": [str(path) for path in raw_root.rglob("*.json")],
        "file_inventory": file_inventory,
        "sessions": session_rows,
    }


def _load_numeric_csv(path: Path, expected_header: tuple[str, ...]) -> np.ndarray:
    with path.open(encoding="utf-8-sig") as handle:
        header = tuple(handle.readline().strip().split(","))
    if header != expected_header:
        raise ValueError(f"Unexpected columns in {path}: {header}")
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(expected_header) or values.shape[0] < 10:
        raise ValueError(f"Incomplete waveform table: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"NaN/Inf in raw waveform: {path}")
    return values


def _rms_ac(values: np.ndarray) -> float:
    centered = values - float(np.mean(values))
    return float(np.sqrt(np.mean(centered**2)))


def _cycle_pair(time: np.ndarray, first: np.ndarray, second: np.ndarray, frequency_hz: float) -> tuple[np.ndarray, np.ndarray]:
    period = 1.0 / frequency_hz
    if time[-1] - time[0] < period or np.any(np.diff(time) <= 0):
        raise ValueError("Waveform does not contain a strictly sampled complete cycle")
    start = 0.5 * (time[0] + time[-1] - period)
    start = min(max(start, time[0]), time[-1] - period)
    query = start + np.arange(WAVEFORM_LENGTH, dtype=np.float64) * period / WAVEFORM_LENGTH
    return np.interp(query, time, first), np.interp(query, time, second)


def _phase_deg(reference: np.ndarray, comparison: np.ndarray) -> float:
    def phase(values: np.ndarray) -> float:
        centered = values - float(np.mean(values))
        coefficient = np.sum(centered * np.exp(-2j * np.pi * np.arange(values.size) / values.size))
        if abs(coefficient) <= np.finfo(np.float64).eps:
            raise ValueError("Degenerate waveform phase")
        return float(np.angle(coefficient))
    return float(np.rad2deg((phase(reference) - phase(comparison) + np.pi) % (2 * np.pi) - np.pi))


def waveform_features(waveform: np.ndarray, frequency_hz: float) -> dict[str, float]:
    values = np.asarray(waveform, dtype=np.float64)
    values = values - float(values.mean())
    spectrum = np.abs(np.fft.rfft(values))
    fundamental = float(spectrum[1])
    harmonics = float(np.sqrt(np.sum(spectrum[2:6] ** 2)))
    absmean = float(np.mean(np.abs(values)))
    dt = 1.0 / (frequency_hz * values.size)
    periodic_delta = np.diff(np.concatenate((values, values[:1]))) / dt
    return {
        "B_peak_t": float(np.max(np.abs(values))),
        "B_rms": float(np.sqrt(np.mean(values**2))),
        "B_thd_percent": 100.0 * harmonics / fundamental if fundamental > 0 else math.nan,
        "dBdt_max": float(np.max(np.abs(periodic_delta))),
        "form_factor": float(np.sqrt(np.mean(values**2)) / absmean) if absmean > 0 else math.nan,
    }


def _legacy_b(path: Path) -> np.ndarray:
    values = _load_numeric_csv(path, ("index", "time_rel_s", "phase_fraction", "B_t"))
    return values[:, 3]


def _float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not np.isfinite(value):
        raise ValueError(f"{field} is not finite")
    return value


def process_sample(session: Session, source: dict[str, str]) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    scope1_path = _resolve_session_file(session, source["raw_scope1_file"])
    scope2_path = _resolve_session_file(session, source["raw_scope2_file"])
    legacy_path = _resolve_session_file(session, source["processed_B1024_file"])
    before = {path: sha256_file(path) for path in (scope1_path, scope2_path)}
    scope1 = _load_numeric_csv(
        scope1_path,
        ("time_ch1_s", "vin_primary_v", "time_ch2_s", "vamp_before_shunt_v", "time_math_s", "vshunt_math_ch2_minus_ch1_v", "iin_math_diag_a"),
    )
    scope2 = _load_numeric_csv(scope2_path, ("time_vin_s", "vin_primary_v", "time_vout_s", "vout_v"))
    if not np.allclose(scope2[:, 0], scope2[:, 2], rtol=0.0, atol=1e-12):
        raise ValueError("Scope #2 CH1/CH2 time bases differ")
    frequency = _float(source, "frequency_hz")
    vin_cycle, vout_cycle = _cycle_pair(scope2[:, 0], scope2[:, 1], scope2[:, 3], frequency)
    shunt_on_primary = np.interp(scope1[:, 0], scope1[:, 4], scope1[:, 5])
    primary_cycle, shunt_cycle = _cycle_pair(scope1[:, 0], scope1[:, 1], shunt_on_primary, frequency)
    b1024 = voltage_to_flux_density(
        scope2[:, 0], scope2[:, 1], frequency_hz=frequency,
        constants=TransformerConstants(N_PRIMARY, EFFECTIVE_AREA_M2),
    )
    features = waveform_features(b1024, frequency)
    vin_rms = _rms_ac(scope2[:, 1])
    vout_rms = _rms_ac(scope2[:, 3])
    scope1_vin_rms = _rms_ac(scope1[:, 1])
    vshunt_scope_rms = _rms_ac(scope1[:, 5])
    vshunt_keithley = abs(_float(source, "vshunt_rms_v_keithley"))
    iin = vshunt_keithley / R_SHUNT_TOTAL_OHM
    phase_shift = _phase_deg(vin_cycle, vout_cycle)
    input_vi_phase = _phase_deg(primary_cycle, shunt_cycle)
    rload = _float(source, "r_load_ohm")
    rp = _float(source, "r_primary_ohm")
    rs = _float(source, "r_secondary_ohm")
    iout = vout_rms / rload
    pin = scope1_vin_rms * iin * math.cos(math.radians(input_vi_phase))
    pout = vout_rms * iout
    pcu = iin**2 * rp + iout**2 * rs
    ploss = pin - pout - pcu
    efficiency = 100.0 * pout / pin if pin > 0 else math.nan
    vin_thd = waveform_features(vin_cycle, frequency)["B_thd_percent"]
    vout_thd = waveform_features(vout_cycle, frequency)["B_thd_percent"]
    shunt_mismatch = 100.0 * abs(vshunt_scope_rms - vshunt_keithley) / max(vshunt_scope_rms, vshunt_keithley)
    hard: list[str] = []
    warnings: list[str] = []
    if int(round(_float(source, "frequency_set_hz"))) not in EXPECTED_FREQUENCIES or frequency <= 0:
        hard.append("FREQUENCY_INVALID")
    if vin_thd > THD_LIMIT_PERCENT:
        hard.append("VIN_THD")
    if vout_thd > THD_LIMIT_PERCENT:
        hard.append("VOUT_THD")
    if shunt_mismatch > SHUNT_SCOPE_FAIL_PERCENT:
        hard.append("SHUNT_SCOPE_MISMATCH")
    if not 0.0 < efficiency <= 100.0:
        hard.append("EFFICIENCY_INVALID")
    if ploss < -1e-6:
        hard.append("PLOSS_NEGATIVE")
    for field in ("vin_peak_display_ratio", "vout_peak_display_ratio"):
        if source.get(field) not in (None, "") and _float(source, field) >= PEAK_DISPLAY_LIMIT:
            hard.append(field.upper().replace("_DISPLAY_RATIO", "_CLIP_RISK"))
    if abs(scope1_vin_rms - vin_rms) / max(scope1_vin_rms, vin_rms) * 100.0 > 3.0:
        warnings.append("VIN_SCOPE1_SCOPE2_MISMATCH")
    if abs(vin_rms - _float(source, "vin_reference_v")) / _float(source, "vin_reference_v") * 100.0 > 2.0:
        warnings.append("VIN_REFERENCE_DRIFT")
    ambient = _float(source, "temperature_ambient_c")
    core_temp = _float(source, "temperature_core_c")
    if not (-40.0 <= ambient <= 100.0 and -40.0 <= core_temp <= 150.0):
        warnings.append("TEMPERATURE_IMPLAUSIBLE")
    voltage_group = _float(source, "vin_set_group_v")
    if voltage_group not in EXPECTED_VOLTAGES:
        warnings.append("UNEXPECTED_VOLTAGE_LEVEL")
    legacy = _legacy_b(legacy_path)
    legacy_features = waveform_features(legacy, frequency)
    comparison = {
        f"{name}_old": legacy_features[name] for name in features
    } | {
        f"{name}_new": features[name] for name in features
    }
    status = "HARD_FAIL" if hard else ("WARNING" if warnings else "PASS")
    row: dict[str, Any] = dict(source)
    row.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "processing_version": PROCESSING_VERSION,
            "source_session_id": source["session_id"],
            "source_sample_id": source["sample_id"],
            "source_samples_csv": str(session.path / "samples.csv"),
            "source_scope1_path": str(scope1_path),
            "source_scope2_path": str(scope2_path),
            "source_scope1_sha256": before[scope1_path],
            "source_scope2_sha256": before[scope2_path],
            "vin_rms_v": vin_rms,
            "phase_shift_deg": phase_shift,
            "temperature_ambient_c": ambient,
            **features,
            "vin_rms_v_scope1_power": scope1_vin_rms,
            "vshunt_rms_v_scope": vshunt_scope_rms,
            "iin_rms_a": iin,
            "input_vi_phase_deg": input_vi_phase,
            "vout_rms_v": vout_rms,
            "iout_rms_a_calc": iout,
            "pin_w": pin,
            "pout_w": pout,
            "Pcu_w": pcu,
            "P_loss": ploss,
            "efficiency_percent": efficiency,
            "LSP": "",
            "lsp_pending": 1,
            "vin_thd_percent": vin_thd,
            "vout_thd_percent": vout_thd,
            "vshunt_scope_vs_keithley_percent": shunt_mismatch,
            "qc_status": status,
            "qc_hard_fail_reasons": ";".join(sorted(set(hard))),
            "qc_warning_reasons": ";".join(sorted(set(warnings))),
            "legacy_B1024_role": "LEGACY_DATA_NOT_GROUND_TRUTH",
            "B1024_source": "raw Scope #2 CH1 primary Vin",
        }
    )
    if any(sha256_file(path) != digest for path, digest in before.items()):
        raise AssertionError("Raw waveform changed during processing")
    return row, b1024, comparison


def _grid_rows(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((float(r["vin_set_group_v"]), int(round(float(r["frequency_set_hz"]))), r["core_id"]) for r in master)
    pass_counts = Counter((float(r["vin_set_group_v"]), int(round(float(r["frequency_set_hz"]))), r["core_id"]) for r in master if r["qc_status"] != "HARD_FAIL")
    voltages = list(EXPECTED_VOLTAGES) + sorted({key[0] for key in counts if key[0] not in EXPECTED_VOLTAGES})
    rows = []
    for voltage in voltages:
        for frequency in EXPECTED_FREQUENCIES:
            fe = counts[voltage, frequency, "FE"]
            commercial = counts[voltage, frequency, "COMMERCIAL"]
            expected_voltage = voltage in EXPECTED_VOLTAGES
            complete = expected_voltage and fe == commercial == 5
            notes = []
            if not expected_voltage:
                notes.append("UNEXPECTED_VOLTAGE_LEVEL")
            if fe != 5:
                notes.append(f"FE_EXPECTED_5_OBSERVED_{fe}")
            if commercial != 5:
                notes.append(f"COMMERCIAL_EXPECTED_5_OBSERVED_{commercial}")
            rows.append({
                "frequency_set_hz": frequency,
                "vin_set_group_v": voltage,
                "FE_repeat_count": fe,
                "COMMERCIAL_repeat_count": commercial,
                "FE_qc_pass_count": pass_counts[voltage, frequency, "FE"],
                "COMMERCIAL_qc_pass_count": pass_counts[voltage, frequency, "COMMERCIAL"],
                "complete_group": int(complete),
                "notes": ";".join(notes),
            })
    return rows


def _summary_stats(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[field] = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    return result


def _qc_breakdown(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in master:
        grouped[(row["core_id"], float(row["vin_set_group_v"]), int(round(float(row["frequency_set_hz"]))))].append(row)
    output = []
    for (core, voltage, frequency), rows in sorted(grouped.items()):
        statuses = Counter(row["qc_status"] for row in rows)
        hard = Counter(reason for row in rows for reason in row["qc_hard_fail_reasons"].split(";") if reason)
        warning = Counter(reason for row in rows for reason in row["qc_warning_reasons"].split(";") if reason)
        output.append({
            "core_id": core,
            "vin_set_group_v": voltage,
            "frequency_set_hz": frequency,
            "rows": len(rows),
            "PASS": statuses["PASS"],
            "WARNING": statuses["WARNING"],
            "HARD_FAIL": statuses["HARD_FAIL"],
            "hard_fail_reasons": ";".join(f"{key}:{value}" for key, value in sorted(hard.items())),
            "warning_reasons": ";".join(f"{key}:{value}" for key, value in sorted(warning.items())),
        })
    return output


def _temperature_audit(master: list[dict[str, Any]]) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    for session_id, rows in sorted(defaultdict(list, {key: list(value) for key, value in __import__("itertools").groupby(sorted(master, key=lambda row: row["session_id"]), key=lambda row: row["session_id"])}).items()):
        ordered = sorted(rows, key=lambda row: row["timestamp"])
        ambient = np.asarray([float(row["temperature_ambient_c"]) for row in ordered])
        core = np.asarray([float(row["temperature_core_c"]) for row in ordered])
        rise = core - ambient
        x = np.arange(len(rows), dtype=np.float64)
        sessions.append({
            "session_id": session_id,
            "core_id": rows[0]["core_id"],
            "vin_set_group_v": float(rows[0]["vin_set_group_v"]),
            "rows": len(rows),
            "ambient_min": float(ambient.min()), "ambient_max": float(ambient.max()), "ambient_mean": float(ambient.mean()), "ambient_sd": float(ambient.std(ddof=1)),
            "core_min": float(core.min()), "core_max": float(core.max()), "core_mean": float(core.mean()), "core_sd": float(core.std(ddof=1)),
            "rise_min": float(rise.min()), "rise_max": float(rise.max()), "rise_mean": float(rise.mean()), "rise_sd": float(rise.std(ddof=1)),
            "ambient_max_consecutive_jump": float(np.max(np.abs(np.diff(ambient)))),
            "core_max_consecutive_jump": float(np.max(np.abs(np.diff(core)))),
            "ambient_trend_c_per_sample": float(np.polyfit(x, ambient, 1)[0]),
            "core_trend_c_per_sample": float(np.polyfit(x, core, 1)[0]),
            "negative_temp_rise_rows": int(np.sum(rise < 0.0)),
        })
    return {
        "sessions": sessions,
        "by_core": {
            core_id: _summary_stats(rows, ("temperature_ambient_c", "temperature_core_c", "temp_rise_c"))
            for core_id, rows in ((core, [row for row in master if row["core_id"] == core]) for core in EXPECTED_CORES)
        },
        "by_voltage": {
            str(voltage): _summary_stats(rows, ("temperature_ambient_c", "temperature_core_c", "temp_rise_c"))
            for voltage, rows in ((voltage, [row for row in master if float(row["vin_set_group_v"]) == voltage]) for voltage in sorted({float(row["vin_set_group_v"]) for row in master}))
        },
        "sensor_anomalies": {
            "negative_temp_rise_rows": sum(float(row["temp_rise_c"]) < 0.0 for row in master),
            "comment": "Negative core-minus-ambient readings are retained and flagged for LSP provenance review; electrical validity is evaluated independently.",
        },
    }


def _b_sanity_and_plot(master: list[dict[str, Any]], waveforms: dict[str, np.ndarray], comparisons: list[dict[str, Any]], reports_dir: Path) -> dict[str, Any]:
    directional_voltage = []
    for (core, frequency), rows in __import__("itertools").groupby(sorted(master, key=lambda row: (row["core_id"], float(row["frequency_set_hz"]), float(row["vin_rms_v"]))), key=lambda row: (row["core_id"], float(row["frequency_set_hz"]))):
        by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_session[row["session_id"]].append(row)
        medians = sorted(
            (float(np.median([float(row["vin_rms_v"]) for row in values])), float(np.median([float(row["B_peak_t"]) for row in values])))
            for values in by_session.values()
        )
        directional_voltage.extend(np.diff([value[1] for value in medians]) >= 0.0)
    directional_frequency = []
    for (core, voltage), rows in __import__("itertools").groupby(sorted(master, key=lambda row: (row["core_id"], float(row["vin_set_group_v"]), float(row["frequency_hz"]))), key=lambda row: (row["core_id"], float(row["vin_set_group_v"]))):
        by_frequency: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_frequency[int(round(float(row["frequency_set_hz"])))].append(float(row["B_peak_t"]))
        medians = [float(np.median(by_frequency[f])) for f in sorted(by_frequency)]
        directional_frequency.extend(np.diff(medians) <= 0.0)
    discontinuity = {row["sample_id"]: float(np.max(np.abs(np.diff(np.concatenate((waveforms[row["sample_id"]], waveforms[row["sample_id"]][:1]))))) / max(float(row["B_peak_t"]), 1e-12)) for row in master}
    repeat_cvs = []
    for key in sorted({(row["condition_group_id"], row["core_id"]) for row in master}):
        values = np.asarray([float(row["B_peak_t"]) for row in master if (row["condition_group_id"], row["core_id"]) == key])
        if values.size > 1:
            repeat_cvs.append(float(values.std(ddof=1) / abs(values.mean())))
    result = {
        "B_peak_increases_with_measured_vin_pair_fraction": float(np.mean(directional_voltage)),
        "B_peak_decreases_with_frequency_group_fraction": float(np.mean(directional_frequency)),
        "max_periodic_step_over_peak": max(discontinuity.values()),
        "samples_with_periodic_step_over_peak_gt_0_10": sum(value > 0.10 for value in discontinuity.values()),
        "max_repeat_B_peak_cv": max(repeat_cvs),
        "groups_with_repeat_B_peak_cv_gt_0_10": sum(value > 0.10 for value in repeat_cvs),
        "B_peak_above_1_t_count": sum(float(row["B_peak_t"]) > 1.0 for row in master),
        "B_peak_range_t": [min(float(row["B_peak_t"]) for row in master), max(float(row["B_peak_t"]) for row in master)],
    }
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mepi-matplotlib")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for core, color in (("FE", "tab:blue"), ("COMMERCIAL", "tab:orange")):
            rows = [row for row in master if row["core_id"] == core]
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                grouped[int(round(float(row["frequency_set_hz"])))].append(float(row["B_peak_t"]))
            axes[0].plot(sorted(grouped), [np.median(grouped[key]) for key in sorted(grouped)], marker="o", ms=3, label=core, color=color)
        axes[0].set(xlabel="Frequency setpoint (Hz)", ylabel="Median B peak (T)", title="Frequency sanity")
        axes[0].legend()
        for core, color in (("FE", "tab:blue"), ("COMMERCIAL", "tab:orange")):
            row = next(row for row in master if row["core_id"] == core and int(round(float(row["frequency_set_hz"]))) == 2500)
            axes[1].plot(np.arange(WAVEFORM_LENGTH) / WAVEFORM_LENGTH, waveforms[row["sample_id"]], label=core, color=color)
        axes[1].set(xlabel="Cycle fraction", ylabel="B (T)", title="Representative primary-derived B(t)")
        axes[1].legend()
        old = np.asarray([row["B_peak_t_old"] for row in comparisons if row["dataset_mode"] == "finetune"])
        new = np.asarray([row["B_peak_t_new"] for row in comparisons if row["dataset_mode"] == "finetune"])
        axes[2].scatter(old, new, s=5, alpha=0.45)
        axes[2].set(xlabel="Legacy Vout-derived B peak (T)", ylabel="Primary-derived B peak (T)", title="Old vs new")
        figure.tight_layout()
        figure.savefig(reports_dir / "B_reprocessing_diagnostics.png", dpi=140)
        plt.close(figure)
        result["diagnostic_plot"] = str(reports_dir / "B_reprocessing_diagnostics.png")
    except ImportError:
        result["diagnostic_plot"] = "NOT_CREATED_MATPLOTLIB_UNAVAILABLE"
    return result


def _inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = ["# MEPI Dataset Inventory", "", f"- Raw root: `{inventory['raw_root']}`", f"- Total: {inventory['total_files']} files, {inventory['total_size_bytes']} bytes", "", "| Session | Mode | Core | Voltage | Rows | Scope1 | Scope2 | B1024 | Legacy QC pass/fail | Missing |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in inventory["sessions"]:
        lines.append(f"| {Path(row['path']).name} | {row['dataset_mode']} | {row['core']} | {row['voltage_level']} | {row['csv_rows']} | {row['raw_scope1_files']} | {row['raw_scope2_files']} | {row['B1024_files']} | {row['legacy_qc_pass']}/{row['legacy_qc_fail']} | {len(row['missing_files'])} |")
    lines += ["", "All legacy processed B1024 files are inventory-only artifacts and are not ground truth.", ""]
    return "\n".join(lines)


def _scan_legacy_occurrences(project_root: Path, raw_root: Path) -> list[dict[str, str]]:
    patterns = {
        "build_B1024": re.compile(r"\bbuild_B1024\b", re.I),
        "N_SECONDARY": re.compile(r"\bN_SECONDARY\b", re.I),
        "Vout used for B": re.compile(r"Vout.{0,40}(?:B\(t\)|B1024|flux)|(?:B\(t\)|B1024|flux).{0,40}Vout", re.I),
        "secondary Vout": re.compile(r"secondary\s+Vout", re.I),
        "0.2508": re.compile(r"(?<![\d.])0\.2508(?!\d)"),
        "0.25": re.compile(r"(?<![\d.])0\.25(?!\d)"),
        "Rshunt": re.compile(r"\bR_?shunt\b", re.I),
        "MEPI-FROZEN-PROTOCOL v1.0": re.compile(r"MEPI[-_ ]FROZEN[-_ ]PROTOCOL\s+v1\.0", re.I),
        "held-out": re.compile(r"\bheld-out\b", re.I),
        "12 features": re.compile(r"\b12\s+features\b", re.I),
        "22 features": re.compile(r"\b22\s+features\b", re.I),
    }
    rows: list[dict[str, str]] = []
    candidates = [raw_root / "mepi_data_acquisition.py", raw_root / "README.txt"]
    candidates += [
        p for p in project_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".py", ".md", ".yaml", ".json"}
        and not ({".git", ".tools", "data", "archive_manifest"} & set(p.relative_to(project_root).parts))
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            matched = [name for name, pattern in patterns.items() if pattern.search(line)]
            if not matched:
                continue
            if path.is_relative_to(raw_root):
                classification = "HISTORICAL" if path.name == "README.txt" else "LEGACY_DATA"
            elif "test" in path.name.lower():
                classification = "NEGATIVE TEST"
            elif "build_finetune_dataset.py" == path.name and ("patterns" in line or "re.compile" in line):
                classification = "NEGATIVE TEST"
            elif "v1.0" in path.name.lower() or "experiments" in path.parts or "legacy" in path.parts or "protocol_compatibility_audit" in path.name:
                classification = "HISTORICAL"
            elif "legacy" in line.lower() or "old method" in line.lower() or "comparison-only" in line.lower():
                classification = "COMMENT"
            elif "reports" in path.parts or "docs" in path.parts or "protocol" in path.name.lower() or line.lstrip().startswith(("#", "//")):
                classification = "COMMENT"
            else:
                classification = "ACTIVE"
            rows.append({"path": str(path), "line": str(number), "matches": ";".join(matched), "classification": classification, "text": line.strip()[:300]})
    return rows


def _write_raw_manifest(raw_root: Path, protocol: Path, archive_dir: Path) -> None:
    roles: list[tuple[Path, str]] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or "/preview/" in path.as_posix() or "/processed/" in path.as_posix():
            continue
        role = "raw_waveform" if "/raw/" in path.as_posix() else "supporting_metadata_or_code"
        roles.append((path, role))
    manifest_rows = []
    checksum_lines = []
    for path, role in roles:
        digest = sha256_file(path)
        relative = path.relative_to(raw_root).as_posix()
        manifest_rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": digest, "role": role})
        checksum_lines.append(f"{digest}  {relative}\n")
    _write_csv(archive_dir / "raw_file_manifest.csv", manifest_rows, ["relative_path", "size_bytes", "sha256", "role"])
    (archive_dir / "raw_checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
    (archive_dir / "RAW_DATA_README.md").write_text(
        "# MEPI v1.1 raw archive\n\n"
        "Archive `/mnt/e/MEPI` without changing source files. The manifest intentionally includes raw Scope #1/#2, original `samples.csv`, session configs, and acquisition code; regenerable preview and legacy processed directories are excluded.\n\n"
        f"Authoritative protocol: `{protocol.name}`\n\nProtocol SHA-256: `{sha256_file(protocol)}`\n",
        encoding="utf-8",
    )


def build_dataset(raw_root: Path, protocol: Path, project_root: Path, output_dir: Path, reports_dir: Path, archive_dir: Path) -> dict[str, Any]:
    if sha256_file(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Authoritative protocol SHA-256 mismatch")
    sessions = discover_sessions(raw_root)
    inventory = build_inventory(raw_root, sessions)
    _write_json(reports_dir / "mepi_dataset_inventory.json", inventory)
    (reports_dir / "mepi_dataset_inventory.md").write_text(_inventory_markdown(inventory), encoding="utf-8")
    master: list[dict[str, Any]] = []
    demo: list[dict[str, Any]] = []
    comparisons: list[dict[str, float]] = []
    waveforms: dict[str, np.ndarray] = {}
    for session in sessions:
        for source in session.rows:
            row, waveform, comparison = process_sample(session, source)
            (master if session.mode == "finetune" else demo).append(row)
            comparisons.append(comparison | {"dataset_mode": session.mode, "core_id": row["core_id"], "sample_id": row["sample_id"]})
            waveforms[row["sample_id"]] = waveform
    if len(master) != 900 or len(demo) != 150:
        raise AssertionError(f"Unexpected row counts: finetune={len(master)}, demo={len(demo)}")
    # Repeat-level B variation is diagnostic only and never target/model-error driven.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in master:
        groups[(row["condition_group_id"], row["core_id"])].append(row)
    for members in groups.values():
        values = np.asarray([float(row["B_peak_t"]) for row in members])
        cv = float(values.std(ddof=1) / abs(values.mean())) if len(values) > 1 and values.mean() else 0.0
        if cv > 0.10:
            for row in members:
                warnings = set(filter(None, row["qc_warning_reasons"].split(";"))) | {"B_PEAK_REPEAT_CV_GT_10_PERCENT"}
                row["qc_warning_reasons"] = ";".join(sorted(warnings))
                if row["qc_status"] == "PASS":
                    row["qc_status"] = "WARNING"
    grid = _grid_rows(master)
    _write_csv(reports_dir / "grid_completeness.csv", grid, ["frequency_set_hz", "vin_set_group_v", "FE_repeat_count", "COMMERCIAL_repeat_count", "FE_qc_pass_count", "COMMERCIAL_qc_pass_count", "complete_group", "notes"])
    qc_breakdown = _qc_breakdown(master)
    _write_csv(reports_dir / "qc_breakdown.csv", qc_breakdown, ["core_id", "vin_set_group_v", "frequency_set_hz", "rows", "PASS", "WARNING", "HARD_FAIL", "hard_fail_reasons", "warning_reasons"])
    temperature_audit = _temperature_audit(master)
    _write_json(reports_dir / "temperature_audit.json", temperature_audit)
    _write_csv(reports_dir / "temperature_session_audit.csv", temperature_audit["sessions"], list(temperature_audit["sessions"][0]))
    b_sanity = _b_sanity_and_plot(master, waveforms, comparisons, reports_dir)
    fields = list(master[0])
    _write_csv(output_dir / "samples_master_reprocessed.csv", master, fields)
    _write_csv(project_root / "data/MEPI/demo_manifest.csv", demo, list(demo[0]))
    # Required three-target contract is fail-closed: all LSP values are pending.
    train_ready: list[dict[str, Any]] = []
    _write_csv(output_dir / "samples_train_ready.csv", train_ready, fields + ["split"])
    np.save(output_dir / "B1024.npy", np.empty((0, WAVEFORM_LENGTH), dtype=np.float32), allow_pickle=False)
    np.save(output_dir / "sample_ids.npy", np.asarray([], dtype="U1"), allow_pickle=False)
    split_fields = ["sample_id", "condition_group_id", "split"]
    _write_csv(output_dir / "split_manifest.csv", [], split_fields)
    _write_json(output_dir / "split_manifest.json", {"status": "BLOCKED", "reason": "LSP_PENDING_AND_GRID_LABEL_MISMATCH", "seed": 42, "train": [], "validation": [], "test": [], "test_accessed": False})
    _write_json(output_dir / "feature_schema.json", {"protocol_version": PROTOCOL_VERSION, "processing_version": PROCESSING_VERSION, "tabular_features": list(FINETUNE_TABULAR_FEATURES), "waveform": {"name": "B(t)_1024", "shape": [None, WAVEFORM_LENGTH], "source": "Scope #2 CH1 primary Vin"}, "targets": ["efficiency_percent", "P_loss", "LSP"], "metadata_only": ["core_id"], "forbidden_predictive_inputs": ["core_id", "temperature_core_c", "iin_rms_a", "vout_rms_v", "pin_w", "pout_w", "Pcu_w", "P_loss", "efficiency_percent", "LSP"]})
    qc_counts = Counter(row["qc_status"] for row in master)
    hard_reasons = Counter(reason for row in master for reason in row["qc_hard_fail_reasons"].split(";") if reason)
    warning_reasons = Counter(reason for row in master for reason in row["qc_warning_reasons"].split(";") if reason)
    complete_expected = sum(int(row["complete_group"]) for row in grid)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "processing_version": PROCESSING_VERSION,
        "master_rows": len(master),
        "demo_rows": len(demo),
        "train_ready_rows": 0,
        "B1024_shape": [0, WAVEFORM_LENGTH],
        "qc_status_counts": dict(qc_counts),
        "qc_hard_fail_reasons": dict(hard_reasons),
        "qc_warning_reasons": dict(warning_reasons),
        "expected_complete_groups": complete_expected,
        "target_complete_counts": {target: sum(row.get(target) not in (None, "") for row in master) for target in ("efficiency_percent", "P_loss", "LSP")},
        "test_accessed": False,
        "train_ready": False,
        "blockers": ["LSP_PENDING_FOR_ALL_900_FINETUNE_ROWS", "OBSERVED_VOLTAGE_LABELS_DO_NOT_MATCH_FROZEN_SIX_LEVEL_GRID", "NEGATIVE_CORE_MINUS_AMBIENT_TEMPERATURE_IN_852_OF_900_ROWS_REQUIRES_LSP_SENSOR_PROVENANCE_REVIEW"],
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "train_normalization.json", {"status": "NOT_FITTED", "reason": "NO_TRAIN_READY_ROWS", "fitted_split": None})
    b_differences = {}
    for feature in ("B_peak_t", "B_rms", "B_thd_percent", "dBdt_max", "form_factor"):
        old = np.asarray([row[f"{feature}_old"] for row in comparisons], dtype=np.float64)
        new = np.asarray([row[f"{feature}_new"] for row in comparisons], dtype=np.float64)
        b_differences[feature] = {"old_mean": float(old.mean()), "new_mean": float(new.mean()), "mean_absolute_change": float(np.mean(np.abs(new - old))), "changed_count_at_1e-9": int(np.sum(~np.isclose(old, new, rtol=0.0, atol=1e-9)))}
    _write_json(reports_dir / "B_reprocessing_audit.json", {"samples_reprocessed": len(comparisons), "source": "raw Scope #2 CH1 primary Vin", "legacy_source": "Scope #2 CH2 Vout", "feature_differences": b_differences, "sanity": b_sanity})
    (reports_dir / "B_reprocessing_audit.md").write_text(
        "# B(t) Reprocessing Audit\n\n"
        f"Reprocessed {len(comparisons)} samples (900 finetune + 150 demo) from raw Scope #2 CH1 using Np={N_PRIMARY} and Ae={EFFECTIVE_AREA_M2} m^2. Raw primary waveforms were available for every sample. Legacy CH2/Vout-derived B1024 files were comparison-only and never used as ground truth.\n\n"
        "Algorithm: validate synchronous finite raw channels; least-squares remove DC/linear baseline; trapezoidal integration; remove integration drift; extract one centered complete cycle; resample periodically to 1024 points; remove residual B offset; compute peak, RMS, harmonics 2-5 THD, periodic maximum derivative, and RMS/mean-absolute form factor.\n\n"
        + "\n".join(f"- {name}: old mean {stats['old_mean']:.8g}, new mean {stats['new_mean']:.8g}, mean absolute change {stats['mean_absolute_change']:.8g}, changed {stats['changed_count_at_1e-9']}/{len(comparisons)}" for name, stats in b_differences.items())
        + f"\n\nSanity: Bpeak-up-with-measured-Vin pair fraction={b_sanity['B_peak_increases_with_measured_vin_pair_fraction']:.4f}; Bpeak-down-with-frequency group fraction={b_sanity['B_peak_decreases_with_frequency_group_fraction']:.4f}; max repeat CV={b_sanity['max_repeat_B_peak_cv']:.4%}; abrupt periodic discontinuities (>10% peak)={b_sanity['samples_with_periodic_step_over_peak_gt_0_10']}; Bpeak >1 T rows={b_sanity['B_peak_above_1_t_count']}.\n\nNo remeasurement is indicated solely for B(t): every sample retains usable raw Scope #2 CH1. Dataset-level blockers are documented separately.\n",
        encoding="utf-8",
    )
    occurrences = _scan_legacy_occurrences(project_root, raw_root)
    _write_csv(reports_dir / "legacy_inconsistency_audit.csv", occurrences, ["path", "line", "matches", "classification", "text"])
    electrical = _summary_stats(master, ("vin_rms_v", "vout_rms_v", "iin_rms_a", "pin_w", "pout_w", "Pcu_w", "P_loss", "efficiency_percent", "phase_shift_deg", "input_vi_phase_deg"))
    temperatures = _summary_stats(master, ("temperature_ambient_c", "temperature_core_c", "temp_rise_c"))
    _write_json(reports_dir / "electrical_temperature_summary.json", {"electrical": electrical, "temperature": temperatures})
    session_counts = Counter((s.mode, str(s.config.get("core_id"))) for s in sessions)
    final = f"""# MEPI v1.1 Dataset Final Audit

## 1. Raw inventory

- Total: {inventory['total_size_bytes']} bytes in {inventory['total_files']} files; {len(sessions)} sessions.
- FE finetune/demo sessions: {session_counts['finetune', 'FE']}/{session_counts['demo', 'FE']}.
- COMMERCIAL finetune/demo sessions: {session_counts['finetune', 'COMMERCIAL']}/{session_counts['demo', 'COMMERCIAL']}.

## 2. Grid completeness

- Expected and observed finetune rows: 900/900.
- Frozen expected groups complete under exact stored voltage labels: {complete_expected}/90.
- The observed session labels contain 3.2, 4.6, 5.1, 5.7, and 6.3 V outside the frozen grid; no relabeling was inferred. See `grid_completeness.csv` for every missing/extra condition.

## 3. B(t) correction

- Old method: legacy acquisition integrated Scope #2 CH2 Vout with `N_SECONDARY`.
- Correct method: raw Scope #2 CH1 primary Vin integrated with Np={N_PRIMARY}, Ae={EFFECTIVE_AREA_M2} m^2.
- Reprocessed: {len(comparisons)} samples; raw primary waveform available: yes for all.
- Exact feature differences are in `B_reprocessing_audit.md/json`.

## 4. QC

- PASS: {qc_counts['PASS']}; WARNING: {qc_counts['WARNING']}; HARD_FAIL: {qc_counts['HARD_FAIL']}.
- Hard-fail reasons: {dict(hard_reasons)}.
- Warning reasons: {dict(warning_reasons)}.
- Per-core/voltage/frequency counts and reasons are explicit in `qc_breakdown.csv`.

## 5. Electrical sanity

Recomputed from raw synchronized waveforms and authoritative Keithley shunt voltage. Summary: `{json.dumps(electrical, sort_keys=True)}`.

The legacy acquisition implementation numerically divides Keithley voltage by `R_SHUNT_OHM = 1.5152`, but its banner/session `iin_dataset_rule` text incorrectly says `/ 0.2508`; the source README also says 0.25 ohm. These retained source files were not rewritten. All searched occurrences and their ACTIVE/HISTORICAL/COMMENT/NEGATIVE TEST/LEGACY DATA classification are in `legacy_inconsistency_audit.csv`.

## 6. Temperature sanity

Ambient/core/rise summary: `{json.dumps(temperatures, sort_keys=True)}`. There are {temperature_audit['sensor_anomalies']['negative_temp_rise_rows']} negative core-minus-ambient readings; session min/max/mean/SD, consecutive jumps and time trends are in `temperature_session_audit.csv`, with core/voltage summaries in `temperature_audit.json`. Temperature quality was not used alone to delete electrically valid samples.

## 7. Train-ready dataset

- Master rows: {len(master)} (FE {sum(r['core_id']=='FE' for r in master)}, COMMERCIAL {sum(r['core_id']=='COMMERCIAL' for r in master)}).
- Train-ready rows: 0; B1024 shape: (0, 1024); tabular schema: exactly 9 frozen features.
- Target completeness: efficiency {summary['target_complete_counts']['efficiency_percent']}/900, P_loss {summary['target_complete_counts']['P_loss']}/900, LSP 0/900.

## 8. Split

No train/validation/test assignment was frozen because cleaning did not yield target-complete rows and the voltage-label grid is unresolved. Test rows were not accessed for model development. Normalization was not fitted.

## 9. Demo

Two demo sessions / {len(demo)} rows are isolated in `data/MEPI/demo_manifest.csv`; zero demo rows entered train-ready data.

## 10. Git integration

Added processing code, tests, schemas, reports, manifests, checksums, and compact metadata. Raw waveforms, preview bulk, legacy processed bulk, and archives remain excluded. Git CLI status is unavailable in this workspace because `.git/HEAD` and `.git/config` are not exposed.

## 11. Raw archive

`archive_manifest/` contains a README, file manifest, and SHA-256 list for raw/supporting source files. Regenerable previews and legacy processed B are not required for scientific reconstruction.

## 12. Blockers

1. Exact LSP Arrhenius formulation/constants remain unfrozen/unimplemented; all 900 rows retain `lsp_pending=1`.
2. Stored voltage group labels do not match the frozen six-level cross-core grid. An evidence-backed mapping/correction is required; this build does not infer one.
3. Core-minus-ambient temperature is negative in {temperature_audit['sensor_anomalies']['negative_temp_rise_rows']}/900 rows; resolve sensor identity/calibration/provenance before using core temperature to derive LSP.

## 13. Final verdict

`TRAIN_READY = FALSE`

Raw data are sufficient for deterministic B/electrical reconstruction, but the required three-target dataset and frozen 90-group split cannot yet be produced without resolving all listed blockers.
"""
    (reports_dir / "MEPI_V1_1_DATASET_FINAL_AUDIT.md").write_text(final, encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(
        "# MEPI experimental dataset\n\nProtocol: v1.1  \nProcessing: primary-B reprocess v1\n\nThis build is fail-closed (`TRAIN_READY = FALSE`). `samples_master_reprocessed.csv` is the complete reprocessed finetune master. Empty train-ready arrays/manifests prove that no pending-LSP or grid-ambiguous sample was silently admitted. Rebuild with:\n\n```bash\npython -m src.mepi_v1.build_finetune_dataset --raw-root /mnt/e/MEPI --protocol \"MEPI-FROZEN-PROTOCOL v1.1.md\"\n```\n",
        encoding="utf-8",
    )
    _write_raw_manifest(raw_root, protocol, archive_dir)
    checksum_targets = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "checksums.sha256")
    (output_dir / "checksums.sha256").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_targets), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / "data/MEPI/v1_1").resolve()
    reports_dir = (args.reports_dir or project_root / "reports").resolve()
    archive_dir = (args.archive_dir or project_root / "archive_manifest").resolve()
    summary = build_dataset(args.raw_root.resolve(), args.protocol.resolve(), project_root, output_dir, reports_dir, archive_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
