"""MEPI v1.1 acquisition-boundary calculations without hardware I/O.

Keithley COM3 supplies AC Vrms across the complete series shunt pair.  Scope #1
supplies the primary-to-current phase used for input power.  Scope #2 supplies
the distinct Vin-to-Vout phase feature and the primary waveform used for B(t).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .constants import PROTOCOL_VERSION, R_SHUNT_TOTAL_OHM


def _finite_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain at least two finite values")
    return array


def validate_synchronous_record(time_s: Any, **channels: Any) -> tuple[np.ndarray, ...]:
    """Validate one time base shared by all channels in an acquisition record."""

    time = _finite_1d(time_s, "time_s")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    arrays = tuple(_finite_1d(values, name) for name, values in channels.items())
    if any(array.shape != time.shape for array in arrays):
        raise ValueError("Synchronous scope channels must share one time base and length")
    return (time, *arrays)


def _phasor_phase_deg(time_s: np.ndarray, waveform_v: np.ndarray, frequency_hz: float) -> float:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    centered = waveform_v - float(np.mean(waveform_v))
    kernel = centered * np.exp(-2j * np.pi * frequency_hz * time_s)
    phasor = np.trapezoid(kernel, time_s)
    scale = float(np.trapezoid(np.abs(centered), time_s))
    if scale == 0.0 or abs(phasor) <= np.finfo(np.float64).eps * scale:
        raise ValueError("Cannot determine phase from a zero/degenerate waveform")
    return float(np.rad2deg(np.angle(phasor)))


def _wrapped_phase_difference_deg(reference_deg: float, comparison_deg: float) -> float:
    difference = (reference_deg - comparison_deg + 180.0) % 360.0 - 180.0
    return float(difference)


def derive_input_vi_phase_deg(
    time_s: Any,
    scope1_ch1_primary_v: Any,
    scope1_ch2_before_shunt_v: Any,
    *,
    frequency_hz: float,
) -> float:
    """Return phase(CH1 primary) - phase(CH2-CH1 shunt/current)."""

    time, primary, before_shunt = validate_synchronous_record(
        time_s,
        scope1_ch1_primary_v=scope1_ch1_primary_v,
        scope1_ch2_before_shunt_v=scope1_ch2_before_shunt_v,
    )
    shunt = before_shunt - primary
    return _wrapped_phase_difference_deg(
        _phasor_phase_deg(time, primary, frequency_hz),
        _phasor_phase_deg(time, shunt, frequency_hz),
    )


def derive_phase_shift_deg(
    time_s: Any,
    scope2_ch1_vin_primary_v: Any,
    scope2_ch2_vout_v: Any,
    *,
    frequency_hz: float,
) -> float:
    """Return Scope #2 phase(Vin primary) - phase(Vout)."""

    time, vin, vout = validate_synchronous_record(
        time_s,
        scope2_ch1_vin_primary_v=scope2_ch1_vin_primary_v,
        scope2_ch2_vout_v=scope2_ch2_vout_v,
    )
    return _wrapped_phase_difference_deg(
        _phasor_phase_deg(time, vin, frequency_hz),
        _phasor_phase_deg(time, vout, frequency_hz),
    )


def input_current_rms_a(
    vshunt_rms_v: float, *, shunt_total_ohm: float = R_SHUNT_TOTAL_OHM
) -> float:
    """Convert Keithley COM3 shunt-pair AC Vrms to input RMS current."""

    if not np.isfinite(shunt_total_ohm) or not np.isclose(
        shunt_total_ohm, R_SHUNT_TOTAL_OHM, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"MEPI v1.1 requires the complete {R_SHUNT_TOTAL_OHM} ohm shunt pair"
        )
    if not np.isfinite(vshunt_rms_v) or vshunt_rms_v < 0.0:
        raise ValueError("vshunt_rms_v must be non-negative and finite")
    return float(vshunt_rms_v / R_SHUNT_TOTAL_OHM)


def input_power_w(
    scope1_vin_rms_v: float, iin_rms_a: float, input_vi_phase_deg: float
) -> float:
    """Calculate Pin using Scope #1 Vin-current phase, never Scope #2 phase shift."""

    values = (scope1_vin_rms_v, iin_rms_a, input_vi_phase_deg)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Input-power quantities must be finite")
    if scope1_vin_rms_v < 0.0 or iin_rms_a < 0.0:
        raise ValueError("RMS voltage and current must be non-negative")
    power = float(scope1_vin_rms_v * iin_rms_a * np.cos(np.deg2rad(input_vi_phase_deg)))
    if not np.isfinite(power):
        raise ValueError("input_power_w must be finite")
    return power


def validate_protocol_version(record: Mapping[str, Any]) -> None:
    version = str(record.get("protocol_version", "")).strip()
    if version != PROTOCOL_VERSION:
        actual = version or "MISSING"
        raise ValueError(
            f"LEGACY_OR_UNKNOWN_PROTOCOL: expected {PROTOCOL_VERSION!r}, got {actual!r}; "
            "v1.0 data must not be silently reinterpreted as v1.1"
        )
