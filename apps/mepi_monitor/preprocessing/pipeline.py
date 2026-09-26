"""Live capture preprocessing assembled from frozen repository functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apps.mepi_monitor.profiles.models import TransformerProfile
from src.mepi_v1.build_finetune_dataset import _cycle_pair, _phase_deg, _rms_ac, waveform_features
from src.mepi_v1.constants import FINETUNE_TABULAR_FEATURES, WAVEFORM_LENGTH
from src.mepi_v1.waveform import TransformerConstants, voltage_to_flux_density


@dataclass(frozen=True)
class ProcessedCapture:
    time_s: np.ndarray
    vin_v: np.ndarray
    vout_v: np.ndarray
    vin_cycle_v: np.ndarray
    vout_cycle_v: np.ndarray
    b_waveform_t: np.ndarray
    measured: dict[str, float]
    derived: dict[str, float]
    features: dict[str, float]


def process_scope_capture(
    time_s: np.ndarray,
    vin_v: np.ndarray,
    vout_v: np.ndarray,
    *,
    frequency_hz: float,
    ambient_temperature_c: float | None,
    profile: TransformerProfile,
) -> ProcessedCapture:
    """Process synchronized CH1/CH2 data through the frozen live path.

    Ambient temperature is deliberately required and is never defaulted.
    """

    profile.validate()
    if ambient_temperature_c is None or not np.isfinite(ambient_temperature_c):
        raise ValueError("Ambient temperature is required (source: Manual input)")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("Measured frequency must be positive and finite")
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    vin = np.asarray(vin_v, dtype=np.float64).reshape(-1)
    vout = np.asarray(vout_v, dtype=np.float64).reshape(-1)
    if time.shape != vin.shape or time.shape != vout.shape or time.size < 100:
        raise ValueError("CH1 and CH2 must share a synchronized time base with >=100 points")
    if not np.isfinite(np.column_stack((time, vin, vout))).all():
        raise ValueError("Scope capture contains non-finite values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Scope time base must be strictly increasing")

    vin_cycle, vout_cycle = _cycle_pair(time, vin, vout, float(frequency_hz))
    b1024 = voltage_to_flux_density(
        time,
        vin,
        frequency_hz=float(frequency_hz),
        constants=TransformerConstants(
            profile.primary_turns, profile.effective_area_m2
        ),
    )
    if b1024.shape != (WAVEFORM_LENGTH,):
        raise AssertionError("Frozen B waveform is not exactly 1024 points")
    magnetic = waveform_features(b1024, float(frequency_hz))
    measured = {
        "frequency_hz": float(frequency_hz),
        "vin_rms_v": _rms_ac(vin),
        "vout_rms_v": _rms_ac(vout),
    }
    derived = {
        "phase_shift_deg": _phase_deg(vin_cycle, vout_cycle),
        **magnetic,
    }
    features = {
        "frequency_hz": measured["frequency_hz"],
        "vin_rms_v": measured["vin_rms_v"],
        "phase_shift_deg": derived["phase_shift_deg"],
        "temperature_ambient_c": float(ambient_temperature_c),
        **magnetic,
    }
    if tuple(features) != FINETUNE_TABULAR_FEATURES:
        raise AssertionError("Frozen nine-feature order changed")
    return ProcessedCapture(
        time_s=time.copy(),
        vin_v=vin.copy(),
        vout_v=vout.copy(),
        vin_cycle_v=vin_cycle,
        vout_cycle_v=vout_cycle,
        b_waveform_t=b1024,
        measured=measured,
        derived=derived,
        features=features,
    )
