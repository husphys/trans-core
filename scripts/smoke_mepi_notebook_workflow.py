"""Non-training smoke check for the real MagNet batch and all frozen models."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mepi_v1.checkpointing import save_checkpoint
from src.mepi_v1.constants import CANDIDATE_BACKBONES
from src.mepi_v1.data import MagNetDataset
from src.mepi_v1.models import PretrainingModel


H5_PATH = ROOT / "data/raw/source_datasets/Dataset/magnet_pretrain_186k.h5"
TRAIN_MANIFEST = ROOT / "data/splits/mepi_v1/magnet_train.csv"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MEPI notebook smoke check")
    device = torch.device("cuda")
    print(
        f"[PASS] CUDA visible: torch={torch.__version__}, CUDA={torch.version.cuda}, "
        f"GPU={torch.cuda.get_device_name(0)}"
    )
    with TRAIN_MANIFEST.open("r", newline="", encoding="utf-8") as handle:
        sample_index = int(next(csv.DictReader(handle))["sample_index"])
    with h5py.File(H5_PATH, "r") as handle:
        materials = sorted(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in set(handle["material"][:])
        )
    dataset = MagNetDataset(
        H5_PATH,
        [sample_index],
        material_to_index={name: index for index, name in enumerate(materials)},
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch["waveform"].shape == (1, 1, 1024)
    assert batch["tabular"].shape == (1, 2)
    print(f"[PASS] One real MagNet batch: waveform={tuple(batch['waveform'].shape)}, tabular={tuple(batch['tabular'].shape)}, target={tuple(batch['target'].shape)}")
    for backbone in CANDIDATE_BACKBONES:
        model = PretrainingModel(
            backbone, material_count=len(materials), latent_dim=256, backbone_layers=4
        ).to(device)
        model.eval()
        with torch.no_grad():
            output = model(
                batch["waveform"].to(device),
                batch["tabular"].to(device),
                batch["material_index"].to(device),
            )
        assert output.shape == (1,)
        print(f"[PASS] {backbone}: output={tuple(output.shape)}, parameters={sum(p.numel() for p in model.parameters())}")
    with tempfile.TemporaryDirectory(prefix="mepi-smoke-") as directory:
        path = Path(directory) / "last_checkpoint.pt"
        save_checkpoint({"model_state": model.state_dict(), "step": 1}, path)
        save_checkpoint({"model_state": model.state_dict(), "step": 2}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["step"] == 2
        assert list(Path(directory).glob("*.pt")) == [path]
        assert not list(Path(directory).glob("*.tmp"))
    print("[PASS] Atomic checkpoint replacement and load")
    print("MEPI NOTEBOOK WORKFLOW SMOKE: PASS (no training performed)")


if __name__ == "__main__":
    main()
