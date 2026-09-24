"""Legacy 12-input model retained for reproducibility evidence only.

Use :mod:`src.mepi_v1.models` for frozen-protocol experiments.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultimodalInputEncoder(nn.Module):
    """Encode B(t) and [f, T, onehot(material)] without synthetic features."""

    def __init__(self, tabular_dim: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.waveform = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 7, padding=3),
            nn.GELU(),
        )
        self.wave_projection = nn.Linear(129, latent_dim)
        self.tabular = nn.Sequential(nn.Linear(tabular_dim, 256), nn.GELU(), nn.Linear(256, latent_dim))
        self.attention = nn.MultiheadAttention(latent_dim, 8, batch_first=True)

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        temporal = self.waveform(waveform)
        spectrum = torch.abs(torch.fft.rfft(waveform.float(), dim=-1))
        spectrum = F.adaptive_avg_pool1d(spectrum, temporal.shape[-1])
        query = self.wave_projection(torch.cat([temporal, spectrum], dim=1).transpose(1, 2))
        context = self.tabular(tabular).unsqueeze(1)
        fused, _ = self.attention(query, context, context, need_weights=False)
        return fused


class BiGRUBackbone(nn.Module):
    def __init__(self, latent_dim: int = 256, layers: int = 4) -> None:
        super().__init__()
        self.network = nn.GRU(latent_dim, latent_dim // 2, layers, batch_first=True, bidirectional=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.network(sequence)
        return output


class PretrainCoreLossModel(nn.Module):
    """MagNet-only model: encoder, selected backbone, and temporary P head.

    PIRL, LSP, RUL, and uncertainty heads are deliberately absent.
    """

    def __init__(self, tabular_dim: int, latent_dim: int = 256, layers: int = 4) -> None:
        super().__init__()
        self.encoder = MultimodalInputEncoder(tabular_dim, latent_dim)
        self.backbone = BiGRUBackbone(latent_dim, layers)
        self.norm = nn.LayerNorm(latent_dim)
        self.core_loss_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(waveform, tabular)
        encoded = self.backbone(encoded)
        return self.core_loss_head(self.norm(encoded[:, -1])).squeeze(-1)


def pretrain_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target) + 0.3 * F.l1_loss(prediction, target)
