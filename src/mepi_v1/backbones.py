"""The eight frozen manuscript candidate sequence backbones."""

from __future__ import annotations

import torch
from torch import nn

from .constants import CANDIDATE_BACKBONES


class TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        channels_first = sequence.transpose(1, 2)
        return self.norm(sequence + self.network(channels_first).transpose(1, 2))


class TCNBackbone(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TemporalBlock(width, 2**index) for index in range(layers))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            sequence = block(sequence)
        return sequence


class RecurrentBackbone(nn.Module):
    def __init__(self, kind: str, width: int, layers: int, bidirectional: bool) -> None:
        super().__init__()
        hidden = width // 2 if bidirectional else width
        recurrent = nn.LSTM if kind == "LSTM" else nn.GRU
        self.network = recurrent(
            width,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if layers > 1 else 0.0,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.network(sequence)
        return output


class LSTMAttentionBackbone(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.lstm = RecurrentBackbone("LSTM", width, layers, bidirectional=True)
        self.attention = nn.MultiheadAttention(width, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        recurrent = self.lstm(sequence)
        attended, _ = self.attention(recurrent, recurrent, recurrent, need_weights=False)
        return self.norm(recurrent + attended)


class RWKVBlock(nn.Module):
    """Compact RWKV-style time/channel mixing block used in the manuscript candidate set."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(width)
        self.channel_norm = nn.LayerNorm(width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.receptance = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.channel = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = self.time_norm(sequence)
        shifted = torch.cat((normalized[:, :1], normalized[:, :-1]), dim=1)
        mixed = 0.5 * normalized + 0.5 * shifted
        time_output = self.output(
            torch.sigmoid(self.receptance(mixed)) * self.value(mixed) * torch.tanh(self.key(mixed))
        )
        sequence = sequence + time_output
        return sequence + self.channel(self.channel_norm(sequence))


class RWKVBackbone(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(RWKVBlock(width) for _ in range(layers)))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.blocks(sequence)


class XLSTMBlock(nn.Module):
    """Extended-LSTM candidate matching the legacy evaluated model family."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.memory = nn.LSTM(width, width, num_layers=1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(width, width), nn.Sigmoid())
        self.projection = nn.Linear(width, width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(sequence)
        memory, _ = self.memory(normalized)
        return sequence + self.gate(normalized) * self.projection(memory)


class XLSTMBackbone(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(XLSTMBlock(width) for _ in range(layers)))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.blocks(sequence)


def build_backbone(name: str, *, width: int = 256, layers: int = 4) -> nn.Module:
    if name not in CANDIDATE_BACKBONES:
        raise ValueError(f"Backbone must be one of {CANDIDATE_BACKBONES}, received {name!r}")
    if width <= 0 or layers <= 0:
        raise ValueError("Backbone width and layers must be positive")
    if name == "TCN":
        return TCNBackbone(width, layers)
    if name in {"LSTM", "BiLSTM"}:
        return RecurrentBackbone("LSTM", width, layers, name == "BiLSTM")
    if name == "LSTM-Attention":
        return LSTMAttentionBackbone(width, layers)
    if name in {"GRU", "BiGRU"}:
        return RecurrentBackbone("GRU", width, layers, name == "BiGRU")
    if name == "RWKV":
        return RWKVBackbone(width, layers)
    return XLSTMBackbone(width, layers)

