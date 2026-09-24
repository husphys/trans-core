"""Legacy pre-v1.1 dataset inventory retained as historical audit evidence.

Active acquisition and downstream validation live in :mod:`src.mepi_v1.downstream_audit`.
This module must not be used to reinterpret legacy rows as v1.1 measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from src.utils.config import load_yaml, resolve_path


REQUIRED_MAGNET_KEYS = {"B", "f", "T", "P", "material"}
FINE_TUNE_REQUIRED = {
    "core_type",
    "frequency_hz",
    "waveform_type",
    "B_waveform",
    "temperature_core_c",
    "temperature_ambient_c",
    "efficiency_percent",
    "P_loss",
    "input_power_w",
    "output_power_w",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "missing_or_non_numeric": int(np.isnan(values).sum()),
        "inf": int(np.isinf(values).sum()),
        "zero": int((finite == 0).sum()),
        "negative": int((finite < 0).sum()),
        "min": float(finite.min()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
    }


def _parse_waveform(value: Any) -> np.ndarray | None:
    if isinstance(value, str) and value.lstrip().startswith("["):
        try:
            parsed = json.loads(value)
            array = np.asarray(parsed, dtype=np.float32).reshape(-1)
            return array
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return np.asarray(value, dtype=np.float32).reshape(-1)
        except (ValueError, TypeError):
            return None
    return None


def audit_csv(path: Path, waveform_fingerprints: bool = True) -> dict[str, Any]:
    frame = pd.read_csv(path)
    report: dict[str, Any] = {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {column: int(count) for column, count in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric": {
            column: _numeric_summary(frame[column])
            for column in frame.columns
            if column != "B_waveform" and pd.api.types.is_numeric_dtype(frame[column])
        },
    }

    if "B_waveform" in frame:
        lengths: Counter[int] = Counter()
        invalid = nonfinite = 0
        exact_hashes: Counter[str] = Counter()
        shape_hashes: Counter[str] = Counter()
        for raw in frame["B_waveform"]:
            waveform = _parse_waveform(raw)
            if waveform is None:
                invalid += 1
                continue
            lengths[len(waveform)] += 1
            nonfinite += int((~np.isfinite(waveform)).sum())
            exact_hashes[hashlib.sha256(waveform.tobytes()).hexdigest()] += 1
            if waveform_fingerprints and waveform.size:
                centered = waveform - np.nanmean(waveform)
                scale = np.nanmax(np.abs(centered))
                normalized = centered / scale if np.isfinite(scale) and scale > 0 else centered
                fingerprint = np.round(normalized, 3).astype(np.float32).tobytes()
                shape_hashes[hashlib.sha256(fingerprint).hexdigest()] += 1
        report["waveform"] = {
            "numeric_rows": int(sum(lengths.values())),
            "non_numeric_rows": invalid,
            "length_counts": dict(sorted(lengths.items())),
            "nonfinite_values": nonfinite,
            "exact_duplicate_rows": int(sum(count - 1 for count in exact_hashes.values() if count > 1)),
            "normalized_fingerprint_duplicate_rows": int(
                sum(count - 1 for count in shape_hashes.values() if count > 1)
            ),
        }

    columns = set(frame.columns)
    report["fine_tune_required_columns_present"] = sorted(FINE_TUNE_REQUIRED & columns)
    report["fine_tune_required_columns_missing"] = sorted(FINE_TUNE_REQUIRED - columns)
    if {"input_power_w", "output_power_w", "efficiency_percent", "P_loss"} <= columns:
        pin = pd.to_numeric(frame["input_power_w"], errors="coerce")
        pout = pd.to_numeric(frame["output_power_w"], errors="coerce")
        reported_eff = pd.to_numeric(frame["efficiency_percent"], errors="coerce")
        reported_loss = pd.to_numeric(frame["P_loss"], errors="coerce")
        derived_eff = 100.0 * pout / pin
        derived_loss = pin - pout
        eff_error = (derived_eff - reported_eff).abs()
        loss_error = (derived_loss - reported_loss).abs()
        report["target_consistency"] = {
            "efficiency_abs_error_percentiles": {
                str(q): float(eff_error.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "loss_abs_error_w_percentiles": {
                str(q): float(loss_error.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "efficiency_exact_rows_tolerance_1e-6": int((eff_error <= 1e-6).sum()),
            "loss_exact_rows_tolerance_1e-6": int((loss_error <= 1e-6).sum()),
            "derived_targets_available": bool((pin > 0).all() and np.isfinite(pin).all() and np.isfinite(pout).all()),
            "copper_loss_supported": False,
            "copper_loss_reason": "No winding-current and winding-resistance fields are present.",
        }
    return report


def _dataset_keys(handle: h5py.File) -> list[str]:
    keys: list[str] = []
    handle.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None)
    return keys


def audit_hdf5(path: Path, deep_duplicates: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "datasets": {}, "root_attributes": {}}
    with h5py.File(path, "r") as handle:
        report["root_attributes"] = {key: _jsonable(value) for key, value in handle.attrs.items()}
        keys = _dataset_keys(handle)
        report["keys"] = keys
        report["required_keys_missing"] = sorted(REQUIRED_MAGNET_KEYS - set(keys))
        for key in keys:
            dataset = handle[key]
            item: dict[str, Any] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "attributes": {name: _jsonable(value) for name, value in dataset.attrs.items()},
            }
            if np.issubdtype(dataset.dtype, np.number):
                missing = inf = 0
                minimum = math.inf
                maximum = -math.inf
                batch = 2048
                for start in range(0, len(dataset), batch):
                    values = np.asarray(dataset[start : start + batch])
                    missing += int(np.isnan(values).sum())
                    inf += int(np.isinf(values).sum())
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        minimum = min(minimum, float(finite.min()))
                        maximum = max(maximum, float(finite.max()))
                item.update(
                    {
                        "nan": missing,
                        "inf": inf,
                        "min": minimum if minimum != math.inf else None,
                        "max": maximum if maximum != -math.inf else None,
                    }
                )
            report["datasets"][key] = item

        if "material" in handle:
            materials = [_jsonable(value) for value in handle["material"][:]]
            report["material_counts"] = dict(Counter(materials))
        if "B" in handle:
            waveforms = handle["B"]
            report["sample_count"] = int(len(waveforms))
            report["waveform_length"] = int(waveforms.shape[1]) if waveforms.ndim > 1 else None
            if deep_duplicates:
                exact: Counter[str] = Counter()
                normalized: Counter[str] = Counter()
                for start in range(0, len(waveforms), 1024):
                    block = np.asarray(waveforms[start : start + 1024], dtype=np.float32)
                    for row in block:
                        exact[hashlib.sha256(row.tobytes()).hexdigest()] += 1
                        centered = row - np.mean(row)
                        scale = np.max(np.abs(centered))
                        shaped = centered / scale if scale > 0 else centered
                        normalized[hashlib.sha256(np.round(shaped, 3).astype(np.float32).tobytes()).hexdigest()] += 1
                report["waveform_duplicates"] = {
                    "exact_duplicate_rows": int(sum(v - 1 for v in exact.values() if v > 1)),
                    "normalized_fingerprint_duplicate_rows": int(
                        sum(v - 1 for v in normalized.values() if v > 1)
                    ),
                    "near_duplicate_definition": "mean-centered, peak-normalized, rounded to 0.001",
                }
    return report


def _glob(config: dict[str, Any], pattern: str) -> list[Path]:
    root = resolve_path(config, ".")
    return sorted(path for path in root.glob(pattern) if path.is_file() and not path.name.endswith(".part"))


def run_audit(config_path: str, deep_duplicates: bool = True) -> dict[str, Any]:
    config = load_yaml(config_path)
    magnet = _glob(config, config["data"]["magnet_glob"])
    legacy = _glob(config, config["data"]["legacy_finetune_glob"])
    source_csvs = sorted(resolve_path(config, config["sources"]["datasets"]["path"]).rglob("*.csv"))
    payload = {
        "magnet": [audit_hdf5(path, deep_duplicates) for path in magnet],
        "csv": [audit_csv(path, waveform_fingerprints=deep_duplicates) for path in source_csvs],
        "legacy_finetune_files": [str(path) for path in legacy],
    }
    output = resolve_path(config, "data/manifests/data_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--no-deep-duplicates", action="store_true")
    args = parser.parse_args()
    payload = run_audit(args.config, not args.no_deep_duplicates)
    print(json.dumps({"magnet_files": len(payload["magnet"]), "csv_files": len(payload["csv"])}, indent=2))


if __name__ == "__main__":
    main()
