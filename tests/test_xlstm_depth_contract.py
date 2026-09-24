from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest
import torch

from src.mepi_v1.backbones import XLSTMBackbone
from src.mepi_v1.models import PretrainingModel
from src.mepi_v1.xlstm_depth import DEPTHS, XLSTMDepthRun


ROOT = Path(__file__).resolve().parents[1]


def test_xlstm_depth_changes_real_block_count_and_parameter_count() -> None:
    counts = []
    for depth in DEPTHS:
        model = PretrainingModel(
            "xLSTM", material_count=10, latent_dim=32, backbone_layers=depth
        )
        assert isinstance(model.backbone, XLSTMBackbone)
        assert len(model.backbone.blocks) == depth
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert counts == sorted(counts)
    assert len(set(counts)) == len(DEPTHS)


def test_every_xlstm_depth_preserves_forward_shape() -> None:
    sequence = torch.randn(2, 8, 32)
    for depth in DEPTHS:
        assert XLSTMBackbone(32, depth)(sequence).shape == sequence.shape


def test_xlstm_depth_amp_forward_backward() -> None:
    # Keep the public software smoke independent of host CUDA state.
    device = torch.device("cpu")
    model = PretrainingModel(
        "xLSTM", material_count=2, latent_dim=32, backbone_layers=2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    waveform = torch.randn(2, 1, 1024, device=device)
    operating = torch.randn(2, 2, device=device)
    material = torch.tensor([0, 1], device=device)
    target = torch.randn(2, device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        prediction = model(waveform, operating, material)
        loss = torch.nn.functional.mse_loss(prediction, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(loss)


def test_depth_runner_isolated_from_official_root_and_test_split() -> None:
    run = XLSTMDepthRun(ROOT, 2)
    assert run.experiment_dir == ROOT / "experiments/xlstm_depth_v1/depth_2"
    assert ROOT / "experiments/pretrain_v1" not in run.experiment_dir.parents
    assert set(run._manifests()) == {"train", "validation"}
    source = textwrap.dedent(inspect.getsource(XLSTMDepthRun.finalize_validation_only))
    ast.parse(source)
    assert 'manifests["test"]' not in source
    assert "magnet_test" not in source


def test_depth_preprocessing_reuse_is_fingerprint_gated() -> None:
    source = textwrap.dedent(inspect.getsource(XLSTMDepthRun._prepare))
    assert "dataset_fingerprint" in source
    assert "fitted_split" in source
    assert "train_sample_count" in source
    assert "official_path" in source
    assert 'official.get("config", {}).get("backbone")' in source


def test_depth_config_and_manual_notebook_contract() -> None:
    config = (ROOT / "configs/xlstm_depth_v1.yaml").read_text(encoding="utf-8")
    assert "depths: [2, 4, 6, 8, 10]" in config
    assert "epochs: 10" in config
    name = "21_xlstm_depth_experiment.ipynb"
    payload = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    text = json.dumps(payload)
    for section in "ABCDEFGHIJKLMNOPQRS":
        assert f"# {section}." in text
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=name)
    assert "RUN_DEPTHS = [2, 4, 6, 8, 10]" in text
    assert "FORCE_RETRAIN = False" in text
    assert "MEPI_REQUIRED_INTERPRETER" in text
    assert "/home/" not in text
    assert "run_manual_depth(2)" in text
    assert "run_manual_depth(10)" in text
    assert "minimum validation MAE_norm" in text
    assert "WAITING FOR REAL FINETUNE DATA" in text
    assert "magnet_test" not in text
    assert "run_xlstm_depth_experiment" not in text


def test_no_terminal_full_depth_runner_exists() -> None:
    assert not (ROOT / "scripts/run_xlstm_depth_experiment.py").exists()
    source = (ROOT / "src/mepi_v1/xlstm_depth.py").read_text(encoding="utf-8")
    assert "def run_xlstm_depth_experiment" not in source


def test_invalid_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="Depth must be one of"):
        XLSTMDepthRun(ROOT, 3)
