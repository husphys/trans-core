"""Protocol-compliant MagNet pretraining and downstream MEPI models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import build_backbone
from .constants import (
    FINETUNE_TABULAR_FEATURES,
    PRETRAIN_TABULAR_FEATURES,
    WAVEFORM_LENGTH,
)


def _require_shape(tensor: torch.Tensor, *, last: int, label: str) -> None:
    if tensor.ndim != 2 or tensor.shape[-1] != last:
        raise ValueError(f"{label} must have shape [batch, {last}], got {tuple(tensor.shape)}")


class WaveformEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 7, padding=3),
            nn.GELU(),
        )
        self.projection = nn.Linear(129, latent_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1:] != (1, WAVEFORM_LENGTH):
            raise ValueError(
                f"waveform must have shape [batch, 1, {WAVEFORM_LENGTH}], got {tuple(waveform.shape)}"
            )
        temporal = self.temporal(waveform)
        spectrum = torch.abs(torch.fft.rfft(waveform.float(), dim=-1))
        spectrum = F.adaptive_avg_pool1d(spectrum, temporal.shape[-1])
        return self.projection(torch.cat((temporal, spectrum), dim=1).transpose(1, 2))


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        _require_shape(tabular, last=self.input_dim, label="tabular")
        return self.network(tabular).unsqueeze(1)


class MultimodalFusion(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            latent_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, waveform_tokens: torch.Tensor, tabular_token: torch.Tensor) -> torch.Tensor:
        fused, _ = self.cross_attention(
            waveform_tokens, tabular_token, tabular_token, need_weights=False
        )
        return self.norm(waveform_tokens + fused)


class MaterialSpecificRegressionHeads(nn.Module):
    """Temporary heads routed by material metadata, never by representation input."""

    def __init__(self, latent_dim: int, material_count: int) -> None:
        super().__init__()
        if material_count <= 0:
            raise ValueError("material_count must be positive")
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))
            for _ in range(material_count)
        )

    def forward(self, representation: torch.Tensor, material_index: torch.Tensor) -> torch.Tensor:
        indices = material_index.reshape(-1).long()
        if indices.shape[0] != representation.shape[0]:
            raise ValueError("One material routing index is required per sample")
        if torch.any(indices < 0) or torch.any(indices >= len(self.heads)):
            raise ValueError("Material routing index is out of range")
        prediction = torch.empty(
            representation.shape[0], device=representation.device, dtype=torch.float32
        )
        for index, head in enumerate(self.heads):
            mask = indices == index
            if torch.any(mask):
                prediction[mask] = head(representation[mask]).squeeze(-1).float()
        return prediction


class PretrainingModel(nn.Module):
    """Shared representation plus material-routed temporary core-loss heads.

    Material identity is accepted only as a routing index after representation learning.
    There is intentionally no PIRL or MTPH in this model.
    """

    def __init__(
        self,
        backbone_name: str,
        *,
        material_count: int,
        latent_dim: int = 256,
        backbone_layers: int = 4,
    ) -> None:
        super().__init__()
        self.waveform_encoder = WaveformEncoder(latent_dim)
        self.operating_encoder_pretrain = TabularEncoder(
            len(PRETRAIN_TABULAR_FEATURES), latent_dim
        )
        self.fusion = MultimodalFusion(latent_dim)
        self.backbone = build_backbone(
            backbone_name, width=latent_dim, layers=backbone_layers
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.temporary_heads = MaterialSpecificRegressionHeads(latent_dim, material_count)

    def shared_representation(self, waveform: torch.Tensor, operating: torch.Tensor) -> torch.Tensor:
        waveform_tokens = self.waveform_encoder(waveform)
        operating_token = self.operating_encoder_pretrain(operating)
        sequence = self.fusion(waveform_tokens, operating_token)
        return self.norm(self.backbone(sequence).mean(dim=1))

    def forward(
        self, waveform: torch.Tensor, operating: torch.Tensor, material_index: torch.Tensor
    ) -> torch.Tensor:
        representation = self.shared_representation(waveform, operating)
        return self.temporary_heads(representation, material_index)


class PhysicsInformedResidualLayer(nn.Module):
    """Project externally constructed Steinmetz/Arrhenius residuals into latent space."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(2, latent_dim), nn.Tanh())

    def forward(self, representation: torch.Tensor, physics_residuals: torch.Tensor) -> torch.Tensor:
        _require_shape(physics_residuals, last=2, label="physics_residuals")
        if physics_residuals.shape[0] != representation.shape[0]:
            raise ValueError("Physics residual batch must match representation batch")
        return representation + self.projection(physics_residuals)


class MultiTaskPredictionHeads(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.efficiency = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.loss_related = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.lsp = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        lsp = self.lsp(representation)
        return {
            "efficiency_percent": F.softplus(self.efficiency(representation).squeeze(-1)),
            "P_loss": F.softplus(self.loss_related(representation).squeeze(-1)),
            "LSP": F.softplus(lsp[:, 0]),
            "LSP_variance": F.softplus(lsp[:, 1]) + 1e-6,
        }


class DownstreamMEPI(nn.Module):
    """Downstream skeleton with a newly initialized nine-feature tabular encoder."""

    def __init__(
        self,
        backbone_name: str = "xLSTM",
        *,
        latent_dim: int = 256,
        backbone_layers: int = 8,
    ) -> None:
        super().__init__()
        self.waveform_encoder = WaveformEncoder(latent_dim)
        self.tabular_encoder_finetune = TabularEncoder(
            len(FINETUNE_TABULAR_FEATURES), latent_dim
        )
        self.fusion = MultimodalFusion(latent_dim)
        self.backbone = build_backbone(
            backbone_name, width=latent_dim, layers=backbone_layers
        )
        self.pirl = PhysicsInformedResidualLayer(latent_dim)
        self.mtph = MultiTaskPredictionHeads(latent_dim)

    def forward(
        self,
        waveform: torch.Tensor,
        tabular: torch.Tensor,
        physics_residuals: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        waveform_tokens = self.waveform_encoder(waveform)
        tabular_token = self.tabular_encoder_finetune(tabular)
        sequence = self.fusion(waveform_tokens, tabular_token)
        representation = self.backbone(sequence).mean(dim=1)
        refined = self.pirl(representation, physics_residuals)
        return self.mtph(refined)


class SteinmetzResidualLayerV14(nn.Module):
    """Project the single inference-safe normalized Steinmetz residual."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(1, latent_dim), nn.Tanh())

    def forward(self, representation: torch.Tensor, r_st: torch.Tensor) -> torch.Tensor:
        _require_shape(r_st, last=1, label="r_st")
        if r_st.shape[0] != representation.shape[0]:
            raise ValueError("Steinmetz residual batch must match representation batch")
        return representation + self.projection(r_st)


class MultiTaskPredictionHeadsV14(nn.Module):
    """Normalized-space v1.4 heads with an unconstrained LSP mean."""

    variance_floor = 1e-6

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.efficiency_z = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.p_loss_z = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.mu_lsp_z = nn.Linear(latent_dim, 1)
        self.raw_var_lsp_z = nn.Linear(latent_dim, 1)

    def forward(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_lsp_z = self.mu_lsp_z(representation).squeeze(-1)
        raw_var_lsp_z = self.raw_var_lsp_z(representation).squeeze(-1)
        return {
            "efficiency_z": self.efficiency_z(representation).squeeze(-1),
            "P_loss_z": self.p_loss_z(representation).squeeze(-1),
            "mu_LSP_z": mu_lsp_z,
            "raw_var_LSP_z": raw_var_lsp_z,
            "var_LSP_z": F.softplus(raw_var_lsp_z) + self.variance_floor,
        }


class DownstreamMEPIV14(nn.Module):
    """Inference-safe MEPI v1.4 model with Steinmetz-only latent PIRL.

    The caller supplies only the inference-computable ``P_St_z`` reference.
    The auxiliary predictor and the exact residual used by both PIRL and the
    physics loss are produced inside this forward pass. Targets, ``core_id``,
    and measured core temperature are deliberately absent from the signature.
    """

    def __init__(
        self,
        backbone_name: str = "xLSTM",
        *,
        latent_dim: int = 256,
        backbone_layers: int = 8,
    ) -> None:
        super().__init__()
        self.waveform_encoder = WaveformEncoder(latent_dim)
        self.tabular_encoder_finetune = TabularEncoder(
            len(FINETUNE_TABULAR_FEATURES), latent_dim
        )
        self.fusion = MultimodalFusion(latent_dim)
        self.backbone = build_backbone(
            backbone_name, width=latent_dim, layers=backbone_layers
        )
        self.p_loss_aux_z = nn.Linear(latent_dim, 1)
        self.pirl = SteinmetzResidualLayerV14(latent_dim)
        self.mtph = MultiTaskPredictionHeadsV14(latent_dim)

    def forward(
        self,
        waveform: torch.Tensor,
        tabular: torch.Tensor,
        p_st_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require_shape(p_st_z, last=1, label="p_st_z")
        waveform_tokens = self.waveform_encoder(waveform)
        tabular_token = self.tabular_encoder_finetune(tabular)
        sequence = self.fusion(waveform_tokens, tabular_token)
        representation = self.backbone(sequence).mean(dim=1)
        p_loss_aux_z = self.p_loss_aux_z(representation)
        r_st = p_loss_aux_z - p_st_z
        s_phys = self.pirl(representation, r_st)
        outputs = self.mtph(s_phys)
        outputs.update(
            {
                "P_loss_aux_z": p_loss_aux_z.squeeze(-1),
                "P_St_z": p_st_z.squeeze(-1),
                "r_St": r_st.squeeze(-1),
            }
        )
        return outputs
