"""Frozen scientific input, output, and acquisition contracts for MEPI v1.1."""

from __future__ import annotations

WAVEFORM_LENGTH = 1024

PROTOCOL_VERSION = "MEPI-FROZEN-PROTOCOL v1.1"
EXPECTED_PROTOCOL_SHA256 = "a515052cd2f8cf2970b731cfb48dee055c316a5d4acc0f0ca90313436c344c0f"
PROTOCOL_RELATIVE_PATHS = (
    "MEPI-FROZEN-PROTOCOL v1.1.md",
    "docs/MEPI-FROZEN-PROTOCOL v1.1.md",
)

R_SHUNT_1_OHM = 0.7576
R_SHUNT_2_OHM = 0.7576
R_SHUNT_TOTAL_OHM = R_SHUNT_1_OHM + R_SHUNT_2_OHM
if abs(R_SHUNT_TOTAL_OHM - 1.5152) >= 1e-12:  # pragma: no cover - import-time invariant
    raise AssertionError("MEPI v1.1 total shunt resistance must be 1.5152 ohm")

PRETRAIN_TABULAR_FEATURES = (
    "frequency_hz",
    "temperature_c",
)

FINETUNE_TABULAR_FEATURES = (
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

# These fields are metadata, targets, or participate in target construction.  They must
# never enter the predictive fine-tuning tensor.
FORBIDDEN_FINETUNE_INPUTS = frozenset(
    {
        "core_type",
        "waveform_type",
        "waveform",
        "temperature_core_c",
        "T_core_K",
        "iin_rms_a",
        "vshunt_rms_v",
        "input_vi_phase_deg",
        "scope1_vin_rms_v",
        "vout_rms_v",
        "iout_rms_a",
        "input_power_w",
        "output_power_w",
        "Pcu_w",
        "P_loss",
        "efficiency_percent",
        "LSP",
        "LSP_raw",
        "log_LSP_raw",
        "LSP_z",
        "LSP_Arr_raw",
        "LSP_Arr_z",
        "temp_rise_c",
    }
)

# Canonical names used by the new physical transformer dataset.  The grouping
# deliberately excludes core identity and repetition so both cores and every
# repeat of one operating condition are indivisible.
DOWNSTREAM_GROUP_KEY_FIELDS = ("f_set", "Vin_set", "Rload", "waveform")

FINETUNE_RAW_FIELDS = (
    "sample_id",
    "protocol_version",
    "group_id",
    "replicate_id",
    "core_type",
    "geometry_id",
    "frequency_hz",
    "f_set",
    "Vin_set",
    "Rload",
    "waveform",
    "vin_rms_v",
    "vshunt_rms_v",
    "iin_rms_a",
    "scope1_vin_rms_v",
    "input_vi_phase_deg",
    "vout_rms_v",
    "iout_rms_a",
    "phase_shift_deg",
    "temperature_core_c",
    "temperature_ambient_c",
    "scope1_waveform_file",
    "scope2_waveform_file",
    "scope1_waveform_sha256",
    "scope2_waveform_sha256",
    "B_waveform",
    "B_peak_t",
    "B_rms",
    "B_thd_percent",
    "dBdt_max",
    "form_factor",
    "vin_peak_v",
    "temp_rise_c",
    "input_power_w",
    "output_power_w",
    "Pcu_w",
    "P_loss",
    "efficiency_percent",
    "LSP",
    "qc_pass",
    "qc_reason",
)

GROUP_KEY_FIELDS = (
    "frequency_setpoint",
    "vin_setpoint",
    "load_resistance",
    "waveform_type",
)

CANDIDATE_BACKBONES = (
    "TCN",
    "LSTM",
    "BiLSTM",
    "LSTM-Attention",
    "GRU",
    "BiGRU",
    "RWKV",
    "xLSTM",
)

FINETUNE_TARGETS = ("efficiency_percent", "P_loss", "LSP")


def validate_finetune_feature_names(feature_names: tuple[str, ...] | list[str]) -> None:
    """Reject any deviation from the ordered nine-feature frozen contract."""

    names = tuple(feature_names)
    forbidden = sorted(set(names) & FORBIDDEN_FINETUNE_INPUTS)
    if forbidden:
        raise ValueError(f"Forbidden leakage fields in fine-tuning inputs: {forbidden}")
    if names != FINETUNE_TABULAR_FEATURES:
        raise ValueError(
            "Fine-tuning feature order must exactly match FINETUNE_TABULAR_FEATURES; "
            f"received {names!r}"
        )
