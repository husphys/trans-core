"""Offline input adapters with explicit provenance boundaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.mepi_v1.constants import FINETUNE_TABULAR_FEATURES, WAVEFORM_LENGTH


def load_repository_screening(project_root: str | Path) -> list[dict[str, Any]]:
    """Load already-generated real demo predictions without rerunning inference."""

    root = Path(project_root).resolve()
    path = root / "reports/MEPI_V1_5_FREQUENCY_SCREENING_PREDICTIONS.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 146:
        raise AssertionError("Expected 146 usable 3.9-V demo predictions")
    if {row["core_id"] for row in rows} != {"FE", "COMMERCIAL"}:
        raise AssertionError("Repository demonstration must include FE and Commercial")
    if {float(row["candidate_frequency_hz"]) for row in rows} != set(range(1000, 4501, 250)):
        raise AssertionError("Repository demonstration frequency grid changed")
    manifest_path = root / "data/MEPI/demo_manifest_v2.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if any(row["sample_id"] not in source_rows for row in rows):
        raise AssertionError("Screening predictions do not map to the demo manifest")
    return [{**source_rows[row["sample_id"]], **row} for row in rows]


def _load_waveform(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        table = np.loadtxt(path, delimiter=",", skiprows=1)
        value = table[:, -1] if table.ndim == 2 else table
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.shape != (WAVEFORM_LENGTH,) or not np.isfinite(value).all():
        raise ValueError(f"Waveform must contain exactly 1024 finite values: {path}")
    return value


def load_compatible_csv(path: str | Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load rows containing nine features and a relative ``b_waveform_file``."""

    source = Path(path).resolve()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Compatible CSV is empty")
    required = {*FINETUNE_TABULAR_FEATURES, "b_waveform_file"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Compatible CSV is missing columns: {missing}")
    projected, waveforms = [], []
    for row in rows:
        features = {name: float(row[name]) for name in FINETUNE_TABULAR_FEATURES}
        if not np.isfinite(list(features.values())).all():
            raise ValueError("Compatible CSV contains non-finite features")
        waveform_path = (source.parent / row["b_waveform_file"]).resolve()
        waveforms.append(_load_waveform(waveform_path))
        projected.append({**row, "features": features, "waveform_path": str(waveform_path)})
    return projected, np.stack(waveforms)
