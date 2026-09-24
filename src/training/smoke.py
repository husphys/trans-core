"""Legacy material-one-hot smoke path retained as audit evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.models.pretrain import PretrainCoreLossModel, pretrain_loss
from src.utils.config import load_yaml, resolve_path
from src.utils.provenance import runtime_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    args = parser.parse_args()
    train_config = load_yaml(args.config)
    paths_config = load_yaml(resolve_path(train_config, train_config["paths_config"]))
    root = resolve_path(paths_config, ".")
    h5_files = sorted(root.glob(paths_config["data"]["magnet_glob"]))
    split_file = resolve_path(paths_config, "data/splits/magnet_train.csv")
    if len(h5_files) != 1 or not split_file.exists():
        raise RuntimeError("Real-data smoke test requires the audited HDF5 and train split manifest.")
    import pandas as pd

    train_frame = pd.read_csv(split_file)
    train_indices = train_frame["sample_index"].head(4).to_numpy(dtype=int)
    # Build the categorical vocabulary from the training manifest only.  The
    # smoke path must not inspect validation/test rows, even for preprocessing.
    all_materials = sorted(train_frame["material"].astype(str).unique())
    with h5py.File(h5_files[0], "r") as handle:
        waveforms = np.asarray(handle["B"][train_indices], dtype=np.float32)
        frequency = np.asarray(handle["f"][train_indices], dtype=np.float32).reshape(-1, 1)
        temperature = np.asarray(handle["T"][train_indices], dtype=np.float32).reshape(-1, 1)
        losses = np.log1p(np.asarray(handle["P"][train_indices], dtype=np.float32).reshape(-1))
        material_index = {name: index for index, name in enumerate(all_materials)}
        onehot = np.zeros((len(train_indices), len(all_materials)), dtype=np.float32)
        for row, value in enumerate(handle["material"][train_indices]):
            onehot[row, material_index[_decode(value)]] = 1.0
    waveform_tensor = torch.from_numpy(waveforms).unsqueeze(1)
    tabular_tensor = torch.from_numpy(np.concatenate([frequency, temperature, onehot], axis=1))
    target = torch.from_numpy((losses - losses.mean()) / (losses.std() + 1e-8))
    model = PretrainCoreLossModel(tabular_dim=tabular_tensor.shape[1], latent_dim=64, layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(waveform_tensor, tabular_tensor)
    loss = pretrain_loss(prediction, target)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss is non-finite")
    report = runtime_record(root)
    report.update(
        {
            "status": "PASS",
            "scope": "one CPU optimization step on four real MagNet training samples",
            "test_split_accessed": False,
            "loss": float(loss.detach()),
            "waveform_shape": list(waveform_tensor.shape),
            "tabular_shape": list(tabular_tensor.shape),
        }
    )
    output = resolve_path(paths_config, "results/reports/smoke_test.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _decode(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


if __name__ == "__main__":
    main()
