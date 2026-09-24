import numpy as np

from src.mepi_v1.targets import (
    ArrheniusConfig,
    arrhenius_lsp,
    build_targets,
    copper_loss_w,
    efficiency_percent,
    loss_target_w,
)


def test_target_formulas() -> None:
    assert efficiency_percent(10.0, 8.0) == 80.0
    copper = copper_loss_w(2.0, 1.0, 0.1, 0.2)
    assert np.isclose(copper, 0.6)
    assert np.isclose(loss_target_w(10.0, 8.0, copper), 1.4)


def test_arrhenius_lsp_uses_core_temperature_only_for_target() -> None:
    config = ArrheniusConfig(activation_energy_ev=0.5, reference_temperature_k=300.0, epsilon=1e-8)
    assert arrhenius_lsp(26.85, config) > 0.0


def test_missing_target_measurements_fail_qc() -> None:
    result = build_targets(
        {"input_power_w": 10.0, "output_power_w": 8.0},
        primary_resistance_ohm=None,
        secondary_resistance_ohm=None,
        arrhenius=None,
    )
    assert result.qc_pass is False
    assert result.efficiency_percent is None
