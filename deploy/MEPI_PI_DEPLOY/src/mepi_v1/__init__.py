"""Protocol-compliant MEPI v1 research pipeline.

The legacy implementation remains outside this package.  Public constants are exported
here so every loader and model shares the same frozen feature contract.
"""

from .constants import (
    CANDIDATE_BACKBONES,
    FINETUNE_TABULAR_FEATURES,
    PRETRAIN_TABULAR_FEATURES,
    WAVEFORM_LENGTH,
)

__all__ = [
    "CANDIDATE_BACKBONES",
    "FINETUNE_TABULAR_FEATURES",
    "PRETRAIN_TABULAR_FEATURES",
    "WAVEFORM_LENGTH",
]

