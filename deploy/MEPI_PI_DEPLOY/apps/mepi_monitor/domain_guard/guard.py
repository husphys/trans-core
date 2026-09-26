"""Auditable warning-only prediction-domain guard."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.mepi_v1.constants import FINETUNE_TABULAR_FEATURES

UNITS = {
    "frequency_hz": "Hz",
    "vin_rms_v": "V RMS",
    "phase_shift_deg": "degree",
    "temperature_ambient_c": "degC",
    "B_peak_t": "T",
    "B_rms": "T",
    "B_thd_percent": "%",
    "dBdt_max": "T/s",
    "form_factor": "dimensionless",
}


@dataclass(frozen=True)
class DomainAssessment:
    status: str
    core_status: str
    feature_status: dict[str, str]
    warnings: tuple[str, ...]


def build_prediction_domain(
    project_root: str | Path, output_path: str | Path | None = None
) -> dict[str, object]:
    root = Path(project_root).resolve()
    source = root / "data/MEPI/v1_2/samples_qc_valid_v1_2.csv"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        train = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    if len(train) != 687:
        raise AssertionError(f"Frozen training-domain row count changed: {len(train)}")
    variables: dict[str, dict[str, object]] = {}
    for name in FINETUNE_TABULAR_FEATURES:
        values = np.asarray([float(row[name]) for row in train], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training-domain values for {name}")
        variables[name] = {
            "training_min": float(values.min()),
            "training_max": float(values.max()),
            "robust_central_range": {
                "method": "empirical_quantile",
                "lower_quantile": 0.05,
                "upper_quantile": 0.95,
                "lower": float(np.quantile(values, 0.05)),
                "upper": float(np.quantile(values, 0.95)),
            },
            "units": UNITS[name],
            "source_column": name,
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "FROZEN_TRAINING_DATA_DERIVED",
        "purpose": "warning_and_interpretation_only_no_clamping",
        "source_artifact": "data/MEPI/v1_2/samples_qc_valid_v1_2.csv",
        "source_split": "train",
        "training_rows": len(train),
        "known_core_metadata": sorted({row["core_id"] for row in train}),
        "feature_order": list(FINETUNE_TABULAR_FEATURES),
        "classification": {
            "GREEN": "all features inside the empirical training 5th-95th percentile central range",
            "YELLOW": "all features inside observed training min-max, with at least one outside the central range",
            "RED": "at least one feature outside observed training min-max",
        },
        "variables": variables,
    }
    if output_path is not None:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = root / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return payload


class PredictionDomain:
    def __init__(self, artifact: str | Path) -> None:
        self.path = Path(artifact)
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        if tuple(self.payload["feature_order"]) != FINETUNE_TABULAR_FEATURES:
            raise AssertionError("Prediction-domain feature order changed")

    def assess(self, features: Mapping[str, float], *, known_core: bool) -> DomainAssessment:
        statuses: dict[str, str] = {}
        warnings: list[str] = []
        for name in FINETUNE_TABULAR_FEATURES:
            value = float(features[name])
            rule = self.payload["variables"][name]
            if value < rule["training_min"] or value > rule["training_max"]:
                statuses[name] = "RED"
                warnings.append(f"{name} is outside observed training min-max")
            else:
                central = rule["robust_central_range"]
                if value < central["lower"] or value > central["upper"]:
                    statuses[name] = "YELLOW"
                    warnings.append(f"{name} is near the investigated-domain boundary")
                else:
                    statuses[name] = "GREEN"
        if "RED" in statuses.values():
            overall = "RED — OUT OF INVESTIGATED DOMAIN"
        elif "YELLOW" in statuses.values():
            overall = "YELLOW — NEAR DOMAIN BOUNDARY"
        else:
            overall = "GREEN — IN DOMAIN"
        core_status = (
            "KNOWN CORE"
            if known_core
            else "UNSEEN CORE — EXPLORATORY PREDICTION"
        )
        if not known_core:
            warnings.insert(0, core_status)
        return DomainAssessment(overall, core_status, statuses, tuple(warnings))
