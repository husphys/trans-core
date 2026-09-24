from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.mepi_v1.backbones import build_backbone
from src.mepi_v1.checkpointing import (
    load_transfer_weights,
    make_pretraining_checkpoint,
    save_checkpoint,
)
from src.mepi_v1.config import load_pretraining_config
from src.mepi_v1.constants import (
    CANDIDATE_BACKBONES,
    FINETUNE_TABULAR_FEATURES,
    PRETRAIN_TABULAR_FEATURES,
    WAVEFORM_LENGTH,
)
from src.mepi_v1.data import MagNetDataset, finetune_tabular, pretrain_tabular, validate_waveform
from src.mepi_v1.models import DownstreamMEPI, PretrainingModel
from src.mepi_v1.scaling import TrainOnlyStandardizer
from src.mepi_v1.splitting import assert_disjoint_splits, deterministic_group_split
from src.mepi_v1.splitting import stable_group_id
from src.mepi_v1.waveform import (
    TransformerConstants,
    correct_offset_and_drift,
    numerical_integral,
    voltage_to_flux_density,
)


def test_frozen_feature_dimensions_and_order() -> None:
    assert PRETRAIN_TABULAR_FEATURES == ("frequency_hz", "temperature_c")
    assert len(PRETRAIN_TABULAR_FEATURES) == 2
    assert len(FINETUNE_TABULAR_FEATURES) == 9
    assert FINETUNE_TABULAR_FEATURES[0] == "frequency_hz"
    assert FINETUNE_TABULAR_FEATURES[-1] == "form_factor"


def test_pretraining_tabular_has_no_material_one_hot() -> None:
    values = pretrain_tabular(100_000.0, 25.0)
    assert values.shape == (2,)
    model = PretrainingModel("BiGRU", material_count=10, latent_dim=32, backbone_layers=1)
    assert model.operating_encoder_pretrain.input_dim == 2


def test_pretraining_waveform_is_exactly_1024() -> None:
    assert WAVEFORM_LENGTH == 1024
    assert validate_waveform(np.zeros(1024, dtype=np.float32)).shape == (1024,)
    with pytest.raises(ValueError):
        validate_waveform(np.zeros(1000, dtype=np.float32))


def test_all_pretraining_configs_consume_shared_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    for backbone in CANDIDATE_BACKBONES:
        name = backbone.lower().replace("-", "_")
        config = load_pretraining_config(root / "configs" / f"pretrain_{name}.yaml")
        assert config["training"]["batch_size"] == 256
        assert config["training"]["num_workers"] == 4
        assert config["training"]["persistent_workers"] is True
        assert config["runtime"]["deterministic_algorithms"] is False
        assert config["runtime"]["allow_tf32"] is True


def test_ram_cache_matches_authoritative_hdf5_rows(tmp_path) -> None:
    path = tmp_path / "tiny.h5"
    waveforms = np.arange(4 * WAVEFORM_LENGTH, dtype=np.float32).reshape(4, WAVEFORM_LENGTH)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("B", data=waveforms)
        handle.create_dataset("f", data=np.asarray([1, 2, 3, 4], dtype=np.float32))
        handle.create_dataset("T", data=np.asarray([10, 20, 30, 40], dtype=np.float32))
        handle.create_dataset("P", data=np.asarray([5, 6, 7, 8], dtype=np.float32))
        handle.create_dataset(
            "material", data=np.asarray(["a", "b", "a", "b"], dtype=h5py.string_dtype())
        )
    kwargs = {
        "material_to_index": {"a": 0, "b": 1},
        "waveform_mean": 2.0,
        "waveform_scale": 4.0,
        "operating_mean": [1.0, 10.0],
        "operating_scale": [2.0, 5.0],
    }
    indices = [3, 0, 2]
    direct = MagNetDataset(path, indices, cache_mode="hdf5", **kwargs)
    cached = MagNetDataset(path, indices, cache_mode="ram", **kwargs)
    for position in range(len(indices)):
        for key in ("waveform", "tabular", "target", "material_index"):
            torch.testing.assert_close(cached[position][key], direct[position][key])
        assert cached[position]["sample_index"] == direct[position]["sample_index"]
        assert cached[position]["material"] == direct[position]["material"]


def test_finetune_leakage_guard_rejects_forbidden_fields() -> None:
    record = {name: float(index) for index, name in enumerate(FINETUNE_TABULAR_FEATURES)}
    assert finetune_tabular(record).shape == (9,)
    with pytest.raises(ValueError, match="Forbidden leakage"):
        finetune_tabular(record | {"temperature_core_c": 50.0}, [*FINETUNE_TABULAR_FEATURES[:-1], "temperature_core_c"])


def test_downstream_encoder_is_new_nine_feature_encoder() -> None:
    model = DownstreamMEPI("BiGRU", latent_dim=32, backbone_layers=1)
    assert model.tabular_encoder_finetune.input_dim == 9


def test_group_key_does_not_use_core_identity() -> None:
    base = {
        "frequency_setpoint": 10_000,
        "vin_setpoint": 5.0,
        "load_resistance": 50.0,
        "waveform_type": "sinusoidal",
    }
    assert stable_group_id(base | {"core_type": "fabricated"}) == stable_group_id(
        base | {"core_type": "commercial"}
    )


def test_group_split_has_zero_sample_group_and_hash_overlap() -> None:
    groups = [f"g{index}" for index in range(20) for _ in range(2)]
    splits = deterministic_group_split(groups, seed=42)
    sample_ids = [f"s{index}" for index in range(len(groups))]
    hashes = [f"h{index}" for index in range(len(groups))]
    assert_disjoint_splits(sample_ids, groups, splits, hashes)


def test_scaler_can_only_fit_training_subset() -> None:
    scaler = TrainOnlyStandardizer()
    with pytest.raises(ValueError, match="only"):
        scaler.fit(np.asarray([[1.0], [2.0]]), split="validation")
    scaler.fit(np.asarray([[1.0], [3.0]]), split="train")
    assert scaler.state is not None and scaler.state.fitted_split == "train"


def test_waveform_processing_is_deterministic_and_preserves_raw() -> None:
    frequency = 1_000.0
    time = np.linspace(0.0, 3.0 / frequency, 3001)
    raw = 2.0 * np.pi * frequency * np.cos(2.0 * np.pi * frequency * time) + 0.3
    original = raw.copy()
    constants = TransformerConstants(primary_turns=10, effective_area_m2=1e-4)
    first = voltage_to_flux_density(time, raw, frequency_hz=frequency, constants=constants)
    second = voltage_to_flux_density(time, raw, frequency_hz=frequency, constants=constants)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1024,)


def test_zero_offset_correction_and_numerical_integration() -> None:
    time = np.linspace(0.0, 1.0, 101)
    corrected = correct_offset_and_drift(np.ones_like(time) * 5.0)
    assert abs(float(corrected.mean())) < 1e-12
    integrated = numerical_integral(time, np.ones_like(time))
    np.testing.assert_allclose(integrated, time, atol=1e-12)


def test_all_and_only_frozen_backbones_run() -> None:
    sequence = torch.zeros(2, 8, 32)
    for name in CANDIDATE_BACKBONES:
        output = build_backbone(name, width=32, layers=1)(sequence)
        assert output.shape == sequence.shape
    with pytest.raises(ValueError):
        build_backbone("Transformer", width=32, layers=1)


def test_tcn_training_step_routes_multiple_materials_with_amp() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = PretrainingModel(
        "TCN", material_count=3, latent_dim=32, backbone_layers=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    waveform = torch.randn(4, 1, WAVEFORM_LENGTH, device=device)
    operating = torch.randn(4, len(PRETRAIN_TABULAR_FEATURES), device=device)
    material_index = torch.tensor([0, 1, 2, 1], device=device)
    target = torch.randn(4, device=device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        prediction = model(waveform, operating, material_index)
        loss = torch.nn.functional.mse_loss(prediction, target)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert all(
        model.temporary_heads.heads[index][0].weight.grad is not None
        for index in range(3)
    )


def test_transfer_loads_waveform_and_backbone_only(tmp_path) -> None:
    pretrain = PretrainingModel("BiGRU", material_count=2, latent_dim=32, backbone_layers=1)
    checkpoint = make_pretraining_checkpoint(
        pretrain,
        preprocessing_metadata={"fitted_split": "train"},
        config={},
        git_commit=None,
        random_seed=42,
        material_names=["a", "b"],
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, checkpoint_path)
    downstream = DownstreamMEPI("BiGRU", latent_dim=32, backbone_layers=1)
    tabular_before = copy.deepcopy(downstream.tabular_encoder_finetune.state_dict())
    report = load_transfer_weights(downstream, checkpoint_path)
    assert report["transferred"] == ["waveform_encoder", "backbone"]
    for key, value in downstream.waveform_encoder.state_dict().items():
        torch.testing.assert_close(value, checkpoint["waveform_encoder"][key])
    for key, value in downstream.tabular_encoder_finetune.state_dict().items():
        torch.testing.assert_close(value, tabular_before[key])
