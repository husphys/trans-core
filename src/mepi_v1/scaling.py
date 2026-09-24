"""Small serializable train-only standardizers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StandardizationState:
    mean: list[float]
    scale: list[float]
    fitted_split: str
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainOnlyStandardizer:
    """Mean/std standardizer that can only be fitted on a named training subset."""

    def __init__(self) -> None:
        self.state: StandardizationState | None = None

    def fit(self, values: np.ndarray, *, split: str) -> "TrainOnlyStandardizer":
        if split != "train":
            raise ValueError(f"Scalers may only be fitted on split='train', received {split!r}")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[0] == 0 or not np.isfinite(array).all():
            raise ValueError("Scaler input must be non-empty and finite")
        mean = array.mean(axis=0)
        scale = array.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        self.state = StandardizationState(
            mean=mean.tolist(),
            scale=scale.tolist(),
            fitted_split="train",
            sample_count=int(array.shape[0]),
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Standardizer has not been fitted")
        array = np.asarray(values, dtype=np.float64)
        return ((array - np.asarray(self.state.mean)) / np.asarray(self.state.scale)).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Standardizer has not been fitted")
        array = np.asarray(values, dtype=np.float64)
        return (array * np.asarray(self.state.scale) + np.asarray(self.state.mean)).astype(np.float32)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TrainOnlyStandardizer":
        scaler = cls()
        scaler.state = StandardizationState(**state)
        if scaler.state.fitted_split != "train":
            raise ValueError("Refusing scaler metadata not fitted on the training subset")
        return scaler


class CoreLossTransform:
    """Legacy-compatible log1p plus train-only standardization transform."""

    name = "log1p_standardize"

    def __init__(self) -> None:
        self.standardizer = TrainOnlyStandardizer()

    def fit(self, core_loss: np.ndarray, *, split: str) -> "CoreLossTransform":
        values = np.asarray(core_loss, dtype=np.float64)
        if np.any(values < 0.0):
            raise ValueError("Core loss must be non-negative for log1p")
        self.standardizer.fit(np.log1p(values), split=split)
        return self

    def transform(self, core_loss: np.ndarray) -> np.ndarray:
        values = np.asarray(core_loss, dtype=np.float64)
        if np.any(values < 0.0):
            raise ValueError("Core loss must be non-negative for log1p")
        return self.standardizer.transform(np.log1p(values)).reshape(values.shape)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        logged = self.standardizer.inverse_transform(normalized)
        return np.expm1(logged).astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        if self.standardizer.state is None:
            raise RuntimeError("Target transform has not been fitted")
        return {
            "transform": self.name,
            "scaler": "population mean and standard deviation",
            "inverse": "expm1(z * scale + mean)",
            "state": self.standardizer.state.to_dict(),
        }

