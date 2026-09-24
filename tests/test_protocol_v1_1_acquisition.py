from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.mepi_v1.acquisition import (
    derive_input_vi_phase_deg,
    derive_phase_shift_deg,
    input_current_rms_a,
    input_power_w,
    validate_protocol_version,
)
from src.mepi_v1.constants import (
    EXPECTED_PROTOCOL_SHA256,
    FINETUNE_TABULAR_FEATURES,
    FORBIDDEN_FINETUNE_INPUTS,
    PROTOCOL_RELATIVE_PATHS,
    PROTOCOL_VERSION,
    R_SHUNT_1_OHM,
    R_SHUNT_2_OHM,
    R_SHUNT_TOTAL_OHM,
)
from src.mepi_v1.downstream_audit import preprocess_record

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_v1_1_authoritative_copies_have_expected_sha256() -> None:
    for relative in PROTOCOL_RELATIVE_PATHS:
        path = ROOT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_SHA256


def test_shunt_pair_and_keithley_voltage_to_current() -> None:
    assert R_SHUNT_1_OHM + R_SHUNT_2_OHM == pytest.approx(1.5152)
    assert R_SHUNT_TOTAL_OHM == pytest.approx(1.5152)
    assert input_current_rms_a(1.5152) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="complete 1.5152 ohm shunt pair"):
        input_current_rms_a(1.5152, shunt_total_ohm=0.2508)
    with pytest.raises(ValueError, match="non-negative"):
        input_current_rms_a(-0.1)


def test_scope1_power_phase_and_scope2_model_phase_are_distinct() -> None:
    frequency = 1_000.0
    time_s = np.linspace(0.0, 5.0 / frequency, 5001)
    omega_t = 2.0 * np.pi * frequency * time_s
    primary = np.sin(omega_t + np.deg2rad(30.0))
    shunt = 0.1 * np.sin(omega_t - np.deg2rad(10.0))
    before_shunt = primary + shunt
    vout = 0.5 * np.sin(omega_t - np.deg2rad(20.0))

    input_phase = derive_input_vi_phase_deg(
        time_s, primary, before_shunt, frequency_hz=frequency
    )
    model_phase = derive_phase_shift_deg(time_s, primary, vout, frequency_hz=frequency)

    assert input_phase == pytest.approx(40.0, abs=0.05)
    assert model_phase == pytest.approx(50.0, abs=0.05)
    assert input_phase != pytest.approx(model_phase)
    assert input_power_w(10.0, 1.0, input_phase) == pytest.approx(
        10.0 * np.cos(np.deg2rad(40.0))
    )


def test_frozen_feature_order_and_target_leakage_denylist() -> None:
    assert FINETUNE_TABULAR_FEATURES == (
        "frequency_hz",
        "vin_rms_v",
        "phase_shift_deg",
        "temperature_ambient_c",
        "B_peak_t",
        "B_rms",
        "B_thd_percent",
        "dBdt_max",
        "form_factor",
    )
    assert len(FINETUNE_TABULAR_FEATURES) == 9
    excluded = {
        "iin_rms_a",
        "vshunt_rms_v",
        "input_vi_phase_deg",
        "input_power_w",
        "output_power_w",
        "Pcu_w",
        "P_loss",
        "temperature_core_c",
    }
    assert excluded <= FORBIDDEN_FINETUNE_INPUTS
    assert excluded.isdisjoint(FINETUNE_TABULAR_FEATURES)


def test_legacy_protocol_is_rejected_explicitly() -> None:
    validate_protocol_version({"protocol_version": PROTOCOL_VERSION})
    with pytest.raises(ValueError, match="must not be silently reinterpreted"):
        validate_protocol_version({"protocol_version": "MEPI-FROZEN-PROTOCOL v1.0"})
    with pytest.raises(ValueError, match="MISSING"):
        validate_protocol_version({})


def test_preprocess_record_binds_each_scope_to_its_frozen_role(tmp_path: Path) -> None:
    frequency = 1_000.0
    time_s = np.linspace(0.0, 5.0 / frequency, 5001)
    omega_t = 2.0 * np.pi * frequency * time_s
    primary = np.sin(omega_t + np.deg2rad(30.0))
    shunt = 0.1 * np.sin(omega_t - np.deg2rad(10.0))
    before_shunt = primary + shunt
    vout = 0.5 * np.sin(omega_t - np.deg2rad(20.0))

    scope1_path = tmp_path / "scope1.csv"
    scope2_path = tmp_path / "scope2.csv"
    scope1_path.write_text(
        "time_s,scope1_ch1_primary_v,scope1_ch2_before_shunt_v\n"
        + "".join(
            f"{time},{ch1},{ch2}\n"
            for time, ch1, ch2 in zip(time_s, primary, before_shunt, strict=True)
        ),
        encoding="utf-8",
    )
    scope2_path.write_text(
        "time_s,scope2_ch1_vin_primary_v,scope2_ch2_vout_v\n"
        + "".join(
            f"{time},{ch1},{ch2}\n"
            for time, ch1, ch2 in zip(time_s, primary, vout, strict=True)
        ),
        encoding="utf-8",
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "frequency_hz": frequency,
        "vin_rms_v": 0.70710678,
        "Vin_set": 99.0,
        "vshunt_rms_v": 1.5152,
        "scope1_vin_rms_v": 10.0,
        "vout_rms_v": 5.0,
        "iout_rms_a": 0.2,
        "scope1_waveform_file": scope1_path.name,
        "scope2_waveform_file": scope2_path.name,
    }
    processed = preprocess_record(
        record,
        dataset_dir=tmp_path,
        config={"transformer": {"primary_turns": 10, "effective_area_m2": 1e-4}},
    )

    assert processed["vin_rms_v"] == record["vin_rms_v"]
    assert processed["vin_rms_v"] != record["Vin_set"]
    assert processed["iin_rms_a"] == pytest.approx(1.0)
    assert processed["input_vi_phase_deg"] == pytest.approx(40.0, abs=0.05)
    assert processed["phase_shift_deg"] == pytest.approx(50.0, abs=0.05)
    assert processed["input_power_w"] == pytest.approx(10.0 * np.cos(np.deg2rad(40.0)), rel=1e-4)
    assert processed["output_power_w"] == pytest.approx(1.0)
    assert np.asarray(processed["B_waveform"]).shape == (1024,)
    assert np.max(np.abs(processed["B_waveform"])) > 0.0
    assert len(processed["scope1_waveform_sha256"]) == 64
    assert len(processed["scope2_waveform_sha256"]) == 64
