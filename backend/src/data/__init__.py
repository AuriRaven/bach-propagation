"""Data extraction, encoding, and dataset utilities for the Bach harmonisation model."""

from .sequence_extractor import ExtractionConfig, extract_sequence, extract_corpus, augment_sequences
from .encoder import Encoder, TOTAL_VOCAB, PAD, START, END
from .dataset import HarmonyDataset, build_datasets

__all__ = [
    'ExtractionConfig',
    'extract_sequence',
    'extract_corpus',
    'augment_sequences',
    'Encoder',
    'TOTAL_VOCAB',
    'PAD',
    'START',
    'END',
    'HarmonyDataset',
    'build_datasets',
]
