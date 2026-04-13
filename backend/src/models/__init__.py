"""Neural models for Bach harmonic sequence prediction."""

from .baseline_lstm import BaroqueHarmonyLSTM
from .music_transformer import (
    CHECKPOINT_SCHEMA_VERSION,
    MultiStreamEmbedding,
    MusicTransformer,
    MusicTransformerBlock,
)

__all__ = [
    "BaroqueHarmonyLSTM",
    "CHECKPOINT_SCHEMA_VERSION",
    "MultiStreamEmbedding",
    "MusicTransformer",
    "MusicTransformerBlock",
]
