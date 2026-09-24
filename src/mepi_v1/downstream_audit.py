"""Fail-closed audit and deterministic preparation for the real downstream dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition import (
    derive_input_vi_phase_deg,
    derive_phase_shift_deg,
    input_current_rms_a,
    input_power_w,
    validate_protocol_version,
)
from .config import load_config, resolve_path
from .constants import (
    DOWNSTREAM_GROUP_KEY_FIELDS,
    FINETUNE_TABULAR_FEATURES,
    FORBIDDEN_FINETUNE_INPUTS,
    PROTOCOL_VERSION,
    validate_finetune_feature_names,
)
from .splitting import assert_disjoint_splits, deterministic_group_split
from .targets import ArrheniusConfig, build_targets
from .waveform import TransformerConstants, voltage_to_flux_density

REQUIRED_MEASURED_FIELDS = (
    "sample_id",
    "protocol_version",
    "replicate_id",
    "core_type",
    "geometry_id",
    "frequency_hz",
    "f_set",
    "vin_rms_v",
    "Vin_set",
    "Rload",
    "waveform",
    "vshunt_rms_v",
    "scope1_vin_rms_v",
    "vout_rms_v",
    "iout_rms_a",
    "temperature_core_c",
    "temperature_ambient_c",
    "scope1_waveform_file",
    "scope2_waveform_file",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return format(float(value), ".17g") if isinstance(value, (float, int)) else str(value).strip()


def operating_group_id(record: Mapping[str, Any]) -> str:
    missing = [field for field in DOWNSTREAM_GROUP_KEY_FIELDS if record.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Missing downstream group fields: {missing}")
    key = "|".join(_canonical(record[field]) for field in DOWNSTREAM_GROUP_KEY_FIELDS)
    return "op_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def validate_predictive_schema(feature_names: Sequence[str]) -> None:
    names = tuple(feature_names)
    leakage = sorted(set(names) & FORBIDDEN_FINETUNE_INPUTS)
    if leakage:
        raise ValueError(f"LEAKAGE_DENYLIST_VIOLATION: {leakage}")
    validate_finetune_feature_names(names)


def _float(record: Mapping[str, Any], name: str) -> float:
    try:
        value = float(record[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be present and numeric") from error
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _load_scope_csv(path: str | Path, channel_fields: tuple[str, str]) -> tuple[np.ndarray, ...]:
    """Read one synchronous two-channel scope record without altering it."""

    raw_path = Path(path)
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = ("time_s", *channel_fields)
    if not rows or not set(required).issubset(rows[0]):
        raise ValueError(f"Raw waveform {raw_path} requires columns {sorted(required)}")
    return tuple(
        np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in required
    )


def load_scope1_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Scope #1 CH1 primary and CH2 before-shunt from one record."""

    return _load_scope_csv(
        path, ("scope1_ch1_primary_v", "scope1_ch2_before_shunt_v")
    )


def load_scope2_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Scope #2 CH1 Vin primary and CH2 Vout from one record."""

    return _load_scope_csv(path, ("scope2_ch1_vin_primary_v", "scope2_ch2_vout_v"))


def waveform_features(waveform: np.ndarray, frequency_hz: float) -> dict[str, float]:
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    spectrum = np.abs(np.fft.rfft(values - values.mean()))
    fundamental = float(spectrum[1]) if spectrum.size > 1 else 0.0
    harmonics = float(np.sqrt(np.sum(spectrum[2:] ** 2))) if spectrum.size > 2 else 0.0
    mean_absolute = float(np.mean(np.abs(values)))
    dt = 1.0 / (float(frequency_hz) * values.size)
    return {
        "B_peak_t": float(np.max(np.abs(values))),
        "B_rms": float(np.sqrt(np.mean(values**2))),
        "B_thd_percent": float(100.0 * harmonics / fundamental) if fundamental > 0 else 0.0,
        "dBdt_max": float(np.max(np.abs(np.gradient(values, dt)))),
        "form_factor": float(np.sqrt(np.mean(values**2)) / mean_absolute)
        if mean_absolute > 0
        else 0.0,
    }


def _validate_optional_derived(
    record: Mapping[str, Any], name: str, derived: float, *, atol: float = 1e-9
) -> None:
    """Reject conflicting precomputed quantities instead of silently overwriting them."""

    if record.get(name) in (None, ""):
        return
    recorded = _float(record, name)
    if not np.isclose(recorded, derived, rtol=1e-9, atol=atol):
        raise ValueError(
            f"{name} conflicts with the MEPI v1.1 derivation: "
            f"recorded={recorded}, derived={derived}"
        )


def preprocess_record(
    record: Mapping[str, Any], *, dataset_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    validate_protocol_version(record)
    transformer = config.get("transformer", {})
    required_constants = ("primary_turns", "effective_area_m2")
    missing = [name for name in required_constants if transformer.get(name) is None]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_METADATA: transformer.{', transformer.'.join(missing)}")

    scope1_path = Path(str(record["scope1_waveform_file"]))
    scope2_path = Path(str(record["scope2_waveform_file"]))
    if not scope1_path.is_absolute():
        scope1_path = (dataset_dir / scope1_path).resolve()
    if not scope2_path.is_absolute():
        scope2_path = (dataset_dir / scope2_path).resolve()
    scope1_hash_before = _sha256_file(scope1_path)
    scope2_hash_before = _sha256_file(scope2_path)
    scope1_time_s, scope1_primary_v, scope1_before_shunt_v = load_scope1_csv(scope1_path)
    scope2_time_s, scope2_primary_v, scope2_vout_v = load_scope2_csv(scope2_path)
    frequency = _float(record, "frequency_hz")

    # Scope #1 current phase is target construction only.  It is never aliased to
    # the Scope #2 Vin-Vout phase_shift_deg model feature.
    input_vi_phase = derive_input_vi_phase_deg(
        scope1_time_s,
        scope1_primary_v,
        scope1_before_shunt_v,
        frequency_hz=frequency,
    )
    phase_shift = derive_phase_shift_deg(
        scope2_time_s,
        scope2_primary_v,
        scope2_vout_v,
        frequency_hz=frequency,
    )
    iin_rms = input_current_rms_a(_float(record, "vshunt_rms_v"))
    pin = input_power_w(_float(record, "scope1_vin_rms_v"), iin_rms, input_vi_phase)
    _validate_optional_derived(record, "iin_rms_a", iin_rms)
    _validate_optional_derived(record, "input_vi_phase_deg", input_vi_phase, atol=1e-6)
    _validate_optional_derived(record, "phase_shift_deg", phase_shift, atol=1e-6)
    _validate_optional_derived(record, "input_power_w", pin, atol=1e-9)

    vin_rms = _float(record, "vin_rms_v")
    if vin_rms < 0.0:
        raise ValueError("vin_rms_v must be an actual non-negative Scope #2 CH1 measurement")
    b_waveform = voltage_to_flux_density(
        scope2_time_s,
        scope2_primary_v,
        frequency_hz=frequency,
        constants=TransformerConstants(
            primary_turns=int(transformer["primary_turns"]),
            effective_area_m2=float(transformer["effective_area_m2"]),
        ),
    )
    if (
        _sha256_file(scope1_path) != scope1_hash_before
        or _sha256_file(scope2_path) != scope2_hash_before
    ):
        raise AssertionError("Raw scope waveform changed during preprocessing")
    processed = dict(record)
    processed.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "iin_rms_a": iin_rms,
            "input_vi_phase_deg": input_vi_phase,
            "phase_shift_deg": phase_shift,
            "input_power_w": pin,
        }
    )
    pout = _float(record, "vout_rms_v") * _float(record, "iout_rms_a")
    _validate_optional_derived(record, "output_power_w", pout, atol=1e-9)
    processed["output_power_w"] = pout
    processed.update(waveform_features(b_waveform, frequency))
    processed["B_waveform"] = b_waveform.tolist()
    processed["scope1_waveform_sha256"] = scope1_hash_before
    processed["scope2_waveform_sha256"] = scope2_hash_before
    return processed


def _load_records(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        raise ValueError("The audited dataset index must be a CSV file")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("The real fine-tuning dataset is empty")
    if "protocol_version" not in records[0]:
        raise ValueError(
            "LEGACY_OR_UNKNOWN_PROTOCOL: dataset has no protocol_version; "
            "v1.0 data must not be silently reinterpreted as v1.1"
        )
    missing = sorted(set(REQUIRED_MEASURED_FIELDS) - set(records[0]))
    if missing:
        raise ValueError(f"Missing required measured fields: {missing}")
    for record in records:
        validate_protocol_version(record)
    return records


def _validate_physical_design(records: list[dict[str, Any]], config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    expected_cores = tuple(config.get("dataset", {}).get("core_types", ()))
    cores = sorted({str(row["core_type"]).strip() for row in records})
    if len(expected_cores) != 2 or sorted(expected_cores) != cores:
        blockers.append(f"expected exactly configured two core types {expected_cores}, found {cores}")
    geometries = {str(row["geometry_id"]).strip() for row in records}
    if len(geometries) != 1:
        blockers.append(f"cores do not share one geometry_id: {sorted(geometries)}")
    if any(str(row["waveform"]).strip().lower() not in {"sinusoidal", "sine"} for row in records):
        blockers.append("excitation is not sinusoidal-only")
    if any(not np.isclose(_float(row, "Rload"), 50.0, atol=1e-9) for row in records):
        blockers.append("Rload is not exactly 50 ohm for every record")

    per_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        per_group[operating_group_id(row)].append(row)
    for group, members in per_group.items():
        counts = Counter(str(row["core_type"]).strip() for row in members)
        if set(counts) != set(expected_cores) or any(counts[core] != 5 for core in expected_cores):
            blockers.append(f"{group} does not contain 5 repetitions for each core: {dict(counts)}")
        for core in expected_cores:
            repetitions = [str(row["replicate_id"]).strip() for row in members if row["core_type"] == core]
            if len(set(repetitions)) != 5:
                blockers.append(f"{group}/{core} repetition IDs are not five unique values")
    return blockers


def _dataset_fingerprint(index_path: Path, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(index_path.read_bytes())
    for row in sorted(records, key=lambda item: str(item["sample_id"])):
        digest.update(str(row["sample_id"]).encode("utf-8"))
        for key in ("_scope1_path", "_scope2_path"):
            digest.update(bytes.fromhex(_sha256_file(Path(str(row[key])))))
    return digest.hexdigest()


def _write_manifests(
    output_root: Path, records: list[dict[str, Any]], splits: np.ndarray, dataset_fingerprint: str
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        destination = output_root / f"{split}_manifest.csv"
        rows = [row for row, assigned in zip(records, splits, strict=True) if assigned == split]
        lines = [
            (
                "sample_id,group_id,core_type,replicate_id,"
                "scope1_waveform_sha256,scope2_waveform_sha256\n"
            )
        ]
        lines.extend(
            f"{row['sample_id']},{row['group_id']},{row['core_type']},{row['replicate_id']},"
            f"{row['scope1_waveform_sha256']},{row['scope2_waveform_sha256']}\n"
            for row in sorted(rows, key=lambda item: str(item["sample_id"]))
        )
        content = "".join(lines)
        if destination.exists() and destination.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Frozen manifest already exists with different content: {destination}")
        destination.write_text(content, encoding="utf-8")
        manifests[split] = _sha256_file(destination)
    split_digest = hashlib.sha256(
        "".join(f"{name}:{manifests[name]}\n" for name in ("train", "validation", "test")).encode()
    ).hexdigest()
    fingerprint_path = output_root / "fingerprints.json"
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "manifest_sha256": manifests,
        "split_fingerprint": split_digest,
    }
    if fingerprint_path.exists() and json.loads(fingerprint_path.read_text()) != payload:
        raise RuntimeError("Frozen downstream fingerprints already exist with different content")
    fingerprint_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def audit_finetune_dataset(config_path: str | Path) -> dict[str, Any]:
    """Audit real data and freeze manifests only after every fail-closed check passes."""

    config = load_config(config_path)
    dataset_value = config.get("dataset", {}).get("path")
    if not dataset_value:
        return {
            "status": "WAITING_FOR_REAL_FINETUNE_DATA",
            "display": "DATASET STATUS:\nWAITING FOR REAL FINETUNE DATA",
            "blockers": ["dataset.path is not configured"],
        }
    dataset_path = resolve_path(config, dataset_value)
    if not dataset_path.is_file():
        return {
            "status": "WAITING_FOR_REAL_FINETUNE_DATA",
            "display": "DATASET STATUS:\nWAITING FOR REAL FINETUNE DATA",
            "blockers": [f"configured dataset does not exist: {dataset_path}"],
        }
    validate_predictive_schema(FINETUNE_TABULAR_FEATURES)
    records = _load_records(dataset_path)
    blockers = _validate_physical_design(records, config)
    transformer = config.get("transformer", {})
    arrhenius = config.get("arrhenius", {})
    required_metadata = {
        "primary_turns": transformer.get("primary_turns"),
        "effective_area_m2": transformer.get("effective_area_m2"),
        "primary_resistance_ohm": transformer.get("primary_resistance_ohm"),
        "secondary_resistance_ohm": transformer.get("secondary_resistance_ohm"),
        "activation_energy_ev": arrhenius.get("activation_energy_ev"),
        "reference_temperature_k": arrhenius.get("reference_temperature_k"),
        "epsilon": arrhenius.get("epsilon"),
    }
    missing_metadata = [name for name, value in required_metadata.items() if value is None]
    if missing_metadata:
        blockers.append("MISSING_REQUIRED_METADATA: " + ", ".join(missing_metadata))
    if blockers:
        return {"status": "FAIL", "blockers": blockers, "sample_count": len(records)}

    processed: list[dict[str, Any]] = []
    arrhenius_config = ArrheniusConfig(
        activation_energy_ev=float(arrhenius["activation_energy_ev"]),
        reference_temperature_k=float(arrhenius["reference_temperature_k"]),
        epsilon=float(arrhenius["epsilon"]),
    )
    for row in records:
        scope1_path = Path(row["scope1_waveform_file"])
        scope2_path = Path(row["scope2_waveform_file"])
        if not scope1_path.is_absolute():
            scope1_path = (dataset_path.parent / scope1_path).resolve()
        if not scope2_path.is_absolute():
            scope2_path = (dataset_path.parent / scope2_path).resolve()
        row["_scope1_path"] = str(scope1_path)
        row["_scope2_path"] = str(scope2_path)
        missing_scope_paths = [path for path in (scope1_path, scope2_path) if not path.is_file()]
        if missing_scope_paths:
            blockers.extend(f"raw scope waveform missing: {path}" for path in missing_scope_paths)
            continue
        item = preprocess_record(row, dataset_dir=dataset_path.parent, config=config)
        item["group_id"] = operating_group_id(row)
        target = build_targets(
            item,
            primary_resistance_ohm=float(transformer["primary_resistance_ohm"]),
            secondary_resistance_ohm=float(transformer["secondary_resistance_ohm"]),
            arrhenius=arrhenius_config,
        )
        if not target.qc_pass:
            blockers.append(f"{row['sample_id']} target QC failed: {target.qc_reason}")
        item.update(
            {
                "efficiency_percent": target.efficiency_percent,
                "Pcu_w": target.copper_loss_w,
                "P_loss": target.loss_w,
                "LSP": target.lsp,
            }
        )
        processed.append(item)
    if blockers:
        return {"status": "FAIL", "blockers": blockers, "sample_count": len(records)}

    group_ids = [row["group_id"] for row in processed]
    splits = deterministic_group_split(group_ids, seed=int(config.get("seed", 42)))
    hashes = [
        f"{row['scope1_waveform_sha256']}:{row['scope2_waveform_sha256']}"
        for row in processed
    ]
    sample_ids = [str(row["sample_id"]) for row in processed]
    assert_disjoint_splits(sample_ids, group_ids, splits, hashes)
    for group in set(group_ids):
        if len(set(splits[np.asarray(group_ids) == group])) != 1:
            raise AssertionError("Operating-condition group crossed splits")
    dataset_fingerprint = _dataset_fingerprint(dataset_path, records)
    output_root = resolve_path(config, config["outputs"]["manifests_dir"])
    evidence = _write_manifests(output_root, processed, splits, dataset_fingerprint)
    return {
        "status": "PASS",
        "sample_count": len(processed),
        "core_types": sorted({row["core_type"] for row in processed}),
        "split_counts": dict(Counter(splits)),
        "test_accessed": False,
        **evidence,
    }
