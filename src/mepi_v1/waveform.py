"""Deterministic voltage-to-flux waveform processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import WAVEFORM_LENGTH


@dataclass(frozen=True)
class TransformerConstants:
    primary_turns: int
    effective_area_m2: float

    def validate(self) -> None:
        if self.primary_turns <= 0:
            raise ValueError("primary_turns must be positive")
        if not np.isfinite(self.effective_area_m2) or self.effective_area_m2 <= 0.0:
            raise ValueError("effective_area_m2 must be positive and finite")


def correct_offset_and_drift(voltage: np.ndarray) -> np.ndarray:
    """Remove the least-squares linear trend without modifying the input array."""

    signal = np.array(voltage, dtype=np.float64, copy=True).reshape(-1)
    if signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError("Voltage waveform must contain at least two finite values")
    coordinate = np.linspace(-1.0, 1.0, signal.size)
    design = np.column_stack((coordinate, np.ones(signal.size)))
    trend = design @ np.linalg.lstsq(design, signal, rcond=None)[0]
    return (signal - trend).astype(np.float64)


def numerical_integral(time_s: np.ndarray, voltage_v: np.ndarray) -> np.ndarray:
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    voltage = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    if time.shape != voltage.shape or time.size < 2:
        raise ValueError("time_s and voltage_v must have the same length of at least two")
    if not np.isfinite(time).all() or not np.isfinite(voltage).all():
        raise ValueError("Waveform time and voltage must be finite")
    delta = np.diff(time)
    if np.any(delta <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    increments = 0.5 * (voltage[1:] + voltage[:-1]) * delta
    return np.concatenate(([0.0], np.cumsum(increments)))


def extract_complete_cycles(
    time_s: np.ndarray, values: np.ndarray, frequency_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return one complete, centered cycle on its original sample grid.

    A single cycle is intentional: downstream spectral features assume that the
    1024 values represent exactly one period.  Returning several cycles and then
    compressing them to 1024 points changes that meaning.
    """

    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if time.shape != signal.shape or time.size < 2:
        raise ValueError("time_s and values must have matching lengths")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    period = 1.0 / float(frequency_hz)
    duration = float(time[-1] - time[0])
    if duration < period:
        raise ValueError("Raw waveform does not contain one complete cycle")
    start_time = 0.5 * (time[0] + time[-1] - period)
    start_time = min(max(start_time, float(time[0])), float(time[-1] - period))
    end_time = start_time + period
    start_index = int(np.searchsorted(time, start_time, side="left"))
    end_index = int(np.searchsorted(time, end_time, side="right"))
    selected_time = time[start_index:end_index]
    selected_signal = signal[start_index:end_index]
    if selected_time.size == 0 or selected_time[0] > start_time:
        selected_time = np.insert(selected_time, 0, start_time)
        selected_signal = np.insert(selected_signal, 0, np.interp(start_time, time, signal))
    elif selected_time[0] < start_time:
        selected_time[0] = start_time
        selected_signal[0] = np.interp(start_time, time, signal)
    if selected_time[-1] < end_time:
        selected_time = np.append(selected_time, end_time)
        selected_signal = np.append(selected_signal, np.interp(end_time, time, signal))
    elif selected_time[-1] > end_time:
        selected_time[-1] = end_time
        selected_signal[-1] = np.interp(end_time, time, signal)
    return selected_time, selected_signal


def resample_waveform(values: np.ndarray, length: int = WAVEFORM_LENGTH) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError("Waveform must contain at least two finite values")
    if length != WAVEFORM_LENGTH:
        raise ValueError(f"MEPI waveform output must be exactly {WAVEFORM_LENGTH} values")
    old_coordinate = np.linspace(0.0, 1.0, signal.size)
    # Periodic model input excludes the duplicated cycle endpoint.
    new_coordinate = np.linspace(0.0, 1.0, length, endpoint=False)
    return np.interp(new_coordinate, old_coordinate, signal).astype(np.float32)


def voltage_to_flux_density(
    time_s: np.ndarray,
    primary_voltage_v: np.ndarray,
    *,
    frequency_hz: float,
    constants: TransformerConstants,
) -> np.ndarray:
    """Convert raw Scope #2 CH1 primary voltage to a 1024-point B(t) waveform."""

    constants.validate()
    # Copies in correct_offset_and_drift guarantee the caller's raw waveform is preserved.
    corrected = correct_offset_and_drift(primary_voltage_v)
    integrated = numerical_integral(time_s, corrected)
    flux_density = integrated / (constants.primary_turns * constants.effective_area_m2)
    # Remove accumulated endpoint drift without fitting away the physical
    # fundamental (a least-squares trend on an integrated sine is biased).
    endpoint_drift = np.linspace(
        float(flux_density[0]), float(flux_density[-1]), flux_density.size
    )
    flux_density = flux_density - endpoint_drift
    _, complete = extract_complete_cycles(time_s, flux_density, frequency_hz)
    waveform = resample_waveform(complete, WAVEFORM_LENGTH).astype(np.float64)
    waveform -= float(waveform.mean())
    return waveform.astype(np.float32)
