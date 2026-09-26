"""GUI-independent, inference-only loader for the frozen MEPI v1.5 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.mepi_v1.config import load_config, resolve_path
from src.mepi_v1.constants import FINETUNE_TABULAR_FEATURES, WAVEFORM_LENGTH
from src.mepi_v1.final_model_manifest_v1_5 import (
    MANIFEST_RELATIVE_PATH,
    sha256_file,
    validate_final_model_manifest,
)
from src.mepi_v1.finetune_v1_4 import (
    WAVEFORM_MEAN,
    WAVEFORM_SCALE,
    load_steinmetz_prior,
    steinmetz_reference_z,
)
from src.mepi_v1.finetune_v1_5 import construct_v1_5_model


@dataclass(frozen=True)
class Prediction:
    efficiency_percent: float
    core_loss_w: float
    lsp: float
    lsp_sigma: float


class FrozenMEPIEngine:
    """Strictly load the selected checkpoint and perform raw-space inference."""

    def __init__(self, project_root: str | Path, *, device: str = "cpu") -> None:
        self.root = Path(project_root).resolve()
        self.manifest = validate_final_model_manifest(
            self.root, self.root / MANIFEST_RELATIVE_PATH
        )
        self.config_path = self.root / self.manifest["source_model_config_path"]
        if sha256_file(self.config_path) != self.manifest["source_model_config_sha256"]:
            raise AssertionError("Frozen source-model config SHA256 mismatch")
        self.config = load_config(self.config_path)
        normalization_path = resolve_path(
            self.config, self.config["dataset"]["normalization"]
        )
        if sha256_file(normalization_path) != self.config["dataset"]["normalization_sha256"]:
            raise AssertionError("Frozen train-only normalization SHA256 mismatch")
        import json

        self.normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        if self.normalization.get("fitted_split") != "train":
            raise AssertionError("Only frozen train-fitted normalization is permitted")
        checkpoint_path = self.root / self.manifest["checkpoint_path"]
        if sha256_file(checkpoint_path) != self.manifest["checkpoint_sha256"]:
            raise AssertionError("Selected checkpoint SHA256 mismatch")
        model, audit = construct_v1_5_model(self.config_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError("Selected checkpoint did not strict-load")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model = model.to(self.device).eval()
        self.model_audit = audit
        self.prior = load_steinmetz_prior(self.config)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = self.manifest["checkpoint_sha256"]

    def _feature_tensor(self, features: Mapping[str, float]) -> torch.Tensor:
        if set(features) != set(FINETUNE_TABULAR_FEATURES):
            missing = sorted(set(FINETUNE_TABULAR_FEATURES) - set(features))
            extra = sorted(set(features) - set(FINETUNE_TABULAR_FEATURES))
            raise ValueError(f"Frozen feature schema mismatch; missing={missing}, extra={extra}")
        values = []
        for name in FINETUNE_TABULAR_FEATURES:
            value = float(features[name])
            scaler = self.normalization["feature_scalers"][name]
            values.append((value - float(scaler["mean"][0])) / float(scaler["scale"][0]))
        if not np.isfinite(values).all():
            raise ValueError("Predictive features must be finite")
        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def predict(self, b_waveform_t: np.ndarray, features: Mapping[str, float]) -> Prediction:
        waveform = np.asarray(b_waveform_t, dtype=np.float64).reshape(-1)
        if waveform.shape != (WAVEFORM_LENGTH,) or not np.isfinite(waveform).all():
            raise ValueError("B waveform must contain exactly 1024 finite values")
        waveform_z = ((waveform - WAVEFORM_MEAN) / WAVEFORM_SCALE).astype(np.float32)
        waveform_tensor = torch.from_numpy(waveform_z).view(1, 1, -1).to(self.device)
        tabular = self._feature_tensor(features)
        frequency = torch.tensor([float(features["frequency_hz"])], device=self.device)
        b_peak = torch.tensor([float(features["B_peak_t"])], device=self.device)
        with torch.inference_mode():
            p_st_z = steinmetz_reference_z(frequency, b_peak, self.prior)
            outputs = self.model(waveform_tensor, tabular, p_st_z)

        scalers = self.normalization["target_scalers"]

        def inverse(name: str, value: torch.Tensor) -> float:
            scaler = scalers[name]
            return float(value.detach().cpu().item() * float(scaler["scale"][0]) + float(scaler["mean"][0]))

        lsp_scale = float(scalers["LSP_raw"]["scale"][0])
        return Prediction(
            efficiency_percent=inverse("efficiency_percent", outputs["efficiency_z"]),
            core_loss_w=inverse("P_loss", outputs["P_loss_z"]),
            lsp=inverse("LSP_raw", outputs["mu_LSP_z"]),
            lsp_sigma=float(torch.sqrt(outputs["var_LSP_z"].double()).cpu().item() * lsp_scale),
        )

    def status(self) -> dict[str, object]:
        return {
            "model": "MEPI v1.5 xLSTM depth 8",
            "checkpoint": str(self.checkpoint_path.relative_to(self.root)),
            "checkpoint_sha256": self.checkpoint_sha256,
            "strict_load": True,
            "training_performed": False,
            "parameter_count": self.model_audit["parameter_count"],
            "device": str(self.device),
        }
