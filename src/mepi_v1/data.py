"""Dataset schemas and tensor construction for frozen MEPI inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import (
    FINETUNE_RAW_FIELDS,
    FINETUNE_TABULAR_FEATURES,
    PRETRAIN_TABULAR_FEATURES,
    WAVEFORM_LENGTH,
    validate_finetune_feature_names,
)


def stable_magnet_sample_id(
    *, material: str, frequency_hz: float, temperature_c: float, waveform: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(material.encode("utf-8"))
    digest.update(np.asarray([frequency_hz, temperature_c], dtype="<f8").tobytes())
    digest.update(np.asarray(waveform, dtype="<f4").reshape(-1).tobytes())
    return "mag_" + digest.hexdigest()[:24]


def validate_waveform(waveform: Any, *, allow_resample: bool = False) -> np.ndarray:
    if isinstance(waveform, str):
        try:
            waveform = json.loads(waveform)
        except json.JSONDecodeError as error:
            raise ValueError("B_waveform is not a numeric JSON array") from error
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("B_waveform must contain at least two finite numeric values")
    if values.size != WAVEFORM_LENGTH:
        if not allow_resample:
            raise ValueError(
                f"B_waveform must contain exactly {WAVEFORM_LENGTH} values, got {values.size}"
            )
        values = np.interp(
            np.linspace(0.0, 1.0, WAVEFORM_LENGTH),
            np.linspace(0.0, 1.0, values.size),
            values,
        ).astype(np.float32)
    return values


def pretrain_tabular(frequency_hz: float, temperature_c: float) -> np.ndarray:
    values = np.asarray([frequency_hz, temperature_c], dtype=np.float32)
    if values.shape != (len(PRETRAIN_TABULAR_FEATURES),) or not np.isfinite(values).all():
        raise ValueError("Pretraining tabular input must be two finite values [frequency_hz, temperature_c]")
    return values


def finetune_tabular(
    record: Mapping[str, Any], feature_names: Sequence[str] = FINETUNE_TABULAR_FEATURES
) -> np.ndarray:
    validate_finetune_feature_names(tuple(feature_names))
    missing = [name for name in feature_names if record.get(name) is None]
    if missing:
        raise ValueError(f"Missing fine-tuning predictive fields: {missing}")
    values = np.asarray([record[name] for name in feature_names], dtype=np.float32)
    if values.shape != (len(FINETUNE_TABULAR_FEATURES),) or not np.isfinite(values).all():
        raise ValueError("Fine-tuning tabular input must contain nine finite values")
    return values


def validate_finetune_schema(record: Mapping[str, Any]) -> list[str]:
    """Return absent raw/metadata fields without fabricating defaults."""

    return [name for name in FINETUNE_RAW_FIELDS if name not in record]


class MagNetDataset(Dataset[dict[str, Any]]):
    """Lazy HDF5 dataset using material only as metadata/head-routing index."""

    def __init__(
        self,
        h5_path: str | Path,
        indices: Sequence[int],
        *,
        material_to_index: Mapping[str, int],
        waveform_mean: float = 0.0,
        waveform_scale: float = 1.0,
        operating_mean: Sequence[float] = (0.0, 0.0),
        operating_scale: Sequence[float] = (1.0, 1.0),
        target_transform: Any | None = None,
        cache_mode: str = "hdf5",
    ) -> None:
        self.h5_path = str(Path(h5_path))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.material_to_index = dict(material_to_index)
        self.waveform_mean = float(waveform_mean)
        self.waveform_scale = float(waveform_scale)
        self.operating_mean = np.asarray(operating_mean, dtype=np.float32)
        self.operating_scale = np.asarray(operating_scale, dtype=np.float32)
        self.target_transform = target_transform
        if cache_mode not in {"hdf5", "ram"}:
            raise ValueError("cache_mode must be 'hdf5' or 'ram'")
        self.cache_mode = cache_mode
        self._handle: Any | None = None
        self._waveforms: np.ndarray | None = None
        self._operating: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._material_indices: np.ndarray | None = None
        if self.operating_mean.shape != (2,) or self.operating_scale.shape != (2,):
            raise ValueError("Pretraining operating-condition scaler must have dimension 2")
        if self.cache_mode == "ram":
            self._preload()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Each DataLoader worker opens its own read-only handle after deserialization.
        state["_handle"] = None
        return state

    def _h5(self) -> Any:
        if self._handle is None:
            try:
                import h5py
            except ImportError as error:  # pragma: no cover - environment-specific message
                raise RuntimeError("h5py is required to read MagNet HDF5 files") from error
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    @staticmethod
    def _read_rows(dataset: Any, indices: np.ndarray) -> np.ndarray:
        order = np.argsort(indices)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        sorted_indices = indices[order]
        output = np.empty(
            (len(indices), *dataset.shape[1:]), dtype=dataset.dtype
        )
        block_size = 8192
        for start in range(0, len(dataset), block_size):
            stop = min(start + block_size, len(dataset))
            left = int(np.searchsorted(sorted_indices, start, side="left"))
            right = int(np.searchsorted(sorted_indices, stop, side="left"))
            if right > left:
                block = np.asarray(dataset[start:stop])
                output[left:right] = block[sorted_indices[left:right] - start]
        return output[inverse]

    def _preload(self) -> None:
        handle = self._h5()
        waveforms = self._read_rows(handle["B"], self.indices).astype(np.float32, copy=False)
        waveforms -= self.waveform_mean
        waveforms /= self.waveform_scale
        frequency = self._read_rows(handle["f"], self.indices).astype(np.float32).reshape(-1)
        temperature = self._read_rows(handle["T"], self.indices).astype(np.float32).reshape(-1)
        operating = np.column_stack((frequency, temperature)).astype(np.float32, copy=False)
        operating -= self.operating_mean
        operating /= self.operating_scale
        losses = self._read_rows(handle["P"], self.indices).astype(np.float32).reshape(-1)
        targets = (
            np.asarray(self.target_transform.transform(losses), dtype=np.float32).reshape(-1)
            if self.target_transform is not None
            else losses
        )
        raw_materials = self._read_rows(handle["material"], self.indices)
        materials = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_materials
        ]
        unknown = sorted(set(materials) - self.material_to_index.keys())
        if unknown:
            raise ValueError(f"Unknown material metadata {unknown[0]!r}")
        self._waveforms = waveforms
        self._operating = operating
        self._targets = targets
        self._material_indices = np.asarray(
            [self.material_to_index[value] for value in materials], dtype=np.int64
        )
        self._handle.close()
        self._handle = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        if self.cache_mode == "ram":
            assert self._waveforms is not None
            assert self._operating is not None
            assert self._targets is not None
            assert self._material_indices is not None
            material_index = int(self._material_indices[position])
            return {
                "waveform": torch.from_numpy(self._waveforms[position]).unsqueeze(0),
                "tabular": torch.from_numpy(self._operating[position]),
                "target": torch.tensor(self._targets[position], dtype=torch.float32),
                "material_index": torch.tensor(material_index, dtype=torch.long),
                "material": next(
                    name for name, index in self.material_to_index.items() if index == material_index
                ),
                "sample_index": int(self.indices[position]),
            }
        index = int(self.indices[position])
        handle = self._h5()
        waveform = validate_waveform(handle["B"][index])
        frequency = float(np.asarray(handle["f"][index]).reshape(-1)[0])
        temperature = float(np.asarray(handle["T"][index]).reshape(-1)[0])
        loss = float(np.asarray(handle["P"][index]).reshape(-1)[0])
        raw_material = handle["material"][index]
        material = raw_material.decode("utf-8") if isinstance(raw_material, bytes) else str(raw_material)
        if material not in self.material_to_index:
            raise ValueError(f"Unknown material metadata {material!r}")
        operating = (pretrain_tabular(frequency, temperature) - self.operating_mean) / self.operating_scale
        normalized_waveform = (waveform - self.waveform_mean) / self.waveform_scale
        target = (
            float(np.asarray(self.target_transform.transform(np.asarray([loss]))).reshape(-1)[0])
            if self.target_transform is not None
            else loss
        )
        return {
            "waveform": torch.from_numpy(normalized_waveform.astype(np.float32)).unsqueeze(0),
            "tabular": torch.from_numpy(operating.astype(np.float32)),
            "target": torch.tensor(target, dtype=torch.float32),
            "material_index": torch.tensor(self.material_to_index[material], dtype=torch.long),
            "material": material,
            "sample_index": index,
        }


class FineTuneRecordsDataset(Dataset[dict[str, Any]]):
    """In-memory interface for future verified experimental records."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = validate_waveform(record["B_waveform"])
        tabular = finetune_tabular(record)
        return {
            "sample_id": str(record["sample_id"]),
            "group_id": str(record["group_id"]),
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "tabular": torch.from_numpy(tabular),
        }
