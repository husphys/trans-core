"""Deterministic downstream target construction with fail-closed QC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

BOLTZMANN_EV_PER_K = 8.617333262145e-5
GAS_CONSTANT_J_PER_MOL_K = 8.314462618
LSP_V1_2_EFFECTIVE_ACTIVATION_ENERGY_J_PER_MOL = 125000.0
LSP_V1_2_REFERENCE_TEMPERATURE_K = 298.15


@dataclass(frozen=True)
class ArrheniusConfig:
    activation_energy_ev: float
    reference_temperature_k: float
    epsilon: float
    boltzmann_ev_per_k: float = BOLTZMANN_EV_PER_K

    def validate(self) -> None:
        values = (
            self.activation_energy_ev,
            self.reference_temperature_k,
            self.epsilon,
            self.boltzmann_ev_per_k,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("All Arrhenius constants must be positive and finite")


@dataclass(frozen=True)
class TargetResult:
    efficiency_percent: float | None
    copper_loss_w: float | None
    loss_w: float | None
    lsp: float | None
    qc_pass: bool
    qc_reason: str


def efficiency_percent(input_power_w: float, output_power_w: float) -> float:
    if not np.isfinite(input_power_w) or input_power_w <= 0.0:
        raise ValueError("input_power_w must be positive and finite")
    if not np.isfinite(output_power_w):
        raise ValueError("output_power_w must be finite")
    return float(100.0 * output_power_w / input_power_w)


def copper_loss_w(
    iin_rms_a: float, iout_rms_a: float, primary_resistance_ohm: float, secondary_resistance_ohm: float
) -> float:
    values = (iin_rms_a, iout_rms_a, primary_resistance_ohm, secondary_resistance_ohm)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Copper-loss inputs must be finite")
    if iin_rms_a < 0.0 or iout_rms_a < 0.0:
        raise ValueError("RMS currents must be non-negative")
    if primary_resistance_ohm < 0.0 or secondary_resistance_ohm < 0.0:
        raise ValueError("Winding resistances must be non-negative")
    return float(
        iin_rms_a**2 * primary_resistance_ohm
        + iout_rms_a**2 * secondary_resistance_ohm
    )


def loss_target_w(input_power_w: float, output_power_w: float, copper_loss: float) -> float:
    values = (input_power_w, output_power_w, copper_loss)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Loss-target inputs must be finite")
    return float(input_power_w - output_power_w - copper_loss)


def arrhenius_lsp(core_temperature_c: float, config: ArrheniusConfig) -> float:
    config.validate()
    if not np.isfinite(core_temperature_c):
        raise ValueError("temperature_core_c must be finite")
    core_temperature_k = float(core_temperature_c + 273.15)
    if core_temperature_k <= 0.0:
        raise ValueError("Absolute core temperature must be positive")
    exponent = -config.activation_energy_ev / config.boltzmann_ev_per_k * (
        1.0 / core_temperature_k - 1.0 / config.reference_temperature_k
    )
    relative_stress = float(np.exp(exponent))
    return float(1.0 / (relative_stress + config.epsilon))


def arrhenius_log_lsp_v1_2(core_temperature_c: float | np.ndarray) -> np.ndarray:
    """Return the frozen v1.2 log relative thermal-stress proxy in float64."""

    core_c = np.asarray(core_temperature_c, dtype=np.float64)
    if not np.isfinite(core_c).all():
        raise ValueError("temperature_core_c must be finite")
    core_k = core_c + 273.15
    if np.any(core_k <= 0.0):
        raise ValueError("Absolute core temperature must be positive")
    return (
        LSP_V1_2_EFFECTIVE_ACTIVATION_ENERGY_J_PER_MOL
        / GAS_CONSTANT_J_PER_MOL_K
        * (1.0 / core_k - 1.0 / LSP_V1_2_REFERENCE_TEMPERATURE_K)
    ).astype(np.float64)


def arrhenius_lsp_v1_2(core_temperature_c: float | np.ndarray) -> np.ndarray:
    """Return the v1.2 dimensionless relative proxy; no epsilon is applied."""

    result = np.exp(arrhenius_log_lsp_v1_2(core_temperature_c)).astype(np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("v1.2 LSP must be finite and positive")
    return result


def normalize_lsp_v1_2(
    lsp_raw: float | np.ndarray, *, mean: float, scale: float
) -> np.ndarray:
    """Apply the single train-fitted v1.2 LSP scaler in float64."""

    values = np.asarray(lsp_raw, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(mean) or not np.isfinite(scale):
        raise ValueError("LSP normalization inputs must be finite")
    if scale <= 0.0:
        raise ValueError("LSP normalization scale must be positive")
    return ((values - np.float64(mean)) / np.float64(scale)).astype(np.float64)


def arrhenius_residual_v1_2(
    lsp_aux_z: float | np.ndarray,
    lsp_arr_raw: float | np.ndarray,
    *,
    mean: float,
    scale: float,
) -> np.ndarray:
    """Build PIRL's Arrhenius residual using the same train-only LSP scaler."""

    auxiliary = np.asarray(lsp_aux_z, dtype=np.float64)
    reference_z = normalize_lsp_v1_2(lsp_arr_raw, mean=mean, scale=scale)
    if auxiliary.shape != reference_z.shape:
        raise ValueError("Auxiliary LSP and Arrhenius reference shapes must match")
    return (auxiliary - reference_z).astype(np.float64)


def build_targets(
    record: Mapping[str, Any],
    *,
    primary_resistance_ohm: float | None,
    secondary_resistance_ohm: float | None,
    arrhenius: ArrheniusConfig | None,
) -> TargetResult:
    """Build all frozen targets, returning a QC failure if any input is unavailable."""

    required = (
        "input_power_w",
        "output_power_w",
        "iin_rms_a",
        "iout_rms_a",
        "temperature_core_c",
    )
    missing = [name for name in required if record.get(name) is None]
    if primary_resistance_ohm is None:
        missing.append("primary_resistance_ohm")
    if secondary_resistance_ohm is None:
        missing.append("secondary_resistance_ohm")
    if arrhenius is None:
        missing.append("arrhenius_config")
    if missing:
        return TargetResult(None, None, None, None, False, "missing: " + ", ".join(missing))
    try:
        efficiency = efficiency_percent(float(record["input_power_w"]), float(record["output_power_w"]))
        copper = copper_loss_w(
            float(record["iin_rms_a"]),
            float(record["iout_rms_a"]),
            float(primary_resistance_ohm),
            float(secondary_resistance_ohm),
        )
        loss = loss_target_w(float(record["input_power_w"]), float(record["output_power_w"]), copper)
        lsp = arrhenius_lsp(float(record["temperature_core_c"]), arrhenius)
        if loss < 0.0:
            raise ValueError("P_loss is negative")
    except (TypeError, ValueError, OverflowError) as error:
        return TargetResult(None, None, None, None, False, str(error))
    return TargetResult(efficiency, copper, loss, lsp, True, "")
