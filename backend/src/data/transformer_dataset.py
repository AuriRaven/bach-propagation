"""
PyTorch :class:`Dataset` wrapping the Stage-5 encoded sequence file
(``data/encoded/encoded_sequences.jsonl``) for the Layer-7 transformer.

Responsibilities
----------------

1. **Deterministic 80/10/10 split by ``score_id`` hash.** Splits happen
   *before* windowing so that no BWV piece leaks between train / val /
   test. Uses a stable hash of the ``score_id`` string so reruns
   produce the same partition without a persisted manifest.

2. **Sliding windows over each score.** A single chorale sequence is
   ~200 timesteps; a window size of 128 with stride 64 gives the model
   overlapping context without duplicating each chord too aggressively.
   Sequences shorter than the window are padded with ``PAD=0``; longer
   ones produce multiple windows. Sequences shorter than 2 timesteps
   (nothing to predict) are dropped.

3. **Aligned three-stream output.** Each item is a dict of equally-shaped
   LongTensors:

   * ``chord_input``    — input chord ids,       shape ``(window_size,)``
   * ``rn_input``       — input Roman-numeral ids, shape ``(window_size,)``
   * ``cadence_input``  — input cadence ids,     shape ``(window_size,)``
   * ``chord_target``   — next-token targets,    shape ``(window_size,)``
   * ``attention_mask`` — 1 for real tokens, 0 for PAD, shape ``(window_size,)``

   The model's objective is next-chord prediction: ``chord_target[t]``
   is the chord that followed ``chord_input[t]`` in the original
   sequence.

Notes
-----
Windowing reads the vocabulary file only to recover the ``PAD`` id —
the special-token indices are constants in :mod:`src.data.harmonic_encoder`,
but reading the file is cheap and keeps the dataset self-describing if a
vocab format bump ever shifts the ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.harmonic_encoder import PAD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: 80/10/10 split thresholds on the score-hash.
_SPLIT_THRESHOLDS: Dict[str, Tuple[int, int]] = {
    # (lo, hi): bucket in [lo, hi) out of 100
    "train": (0, 80),
    "val":   (80, 90),
    "test":  (90, 100),
}


# ---------------------------------------------------------------------------
# Window descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Window:
    """Pointer to (score index, start offset) — concrete tensors are
    materialised lazily in ``__getitem__`` to keep the dataset cheap."""

    score_idx: int
    start: int


# ---------------------------------------------------------------------------
# HarmonicSequenceDataset
# ---------------------------------------------------------------------------


class HarmonicSequenceDataset(Dataset):
    """Windowed three-stream sequence dataset for next-chord prediction.

    Args:
        jsonl_path: Path to ``encoded_sequences.jsonl`` (one JSON object
            per line, as written by :mod:`scripts.build_harmonic_vocab`).
        vocab_path: Path to ``vocab.json``. Currently read only to
            expose the PAD id (``HarmonicVocabulary`` always uses
            ``PAD=0`` but we honour whatever the file says).
        split: ``"train"`` / ``"val"`` / ``"test"``. Splits are
            deterministic by hash of ``score_id``.
        window_size: Length of each input / target tensor. Sequences
            shorter than this are padded to length; longer ones are
            tiled by ``stride``.
        stride: Hop between successive windows from the same score.
            Defaults to ``window_size // 2`` when ``None``.
        pad_id: Override for the pad id. Usually left as the vocab default.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        vocab_path: str | Path,
        split: str = "train",
        window_size: int = 128,
        stride: int | None = 64,
        pad_id: int | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
        if window_size <= 1:
            raise ValueError(f"window_size must be > 1, got {window_size}")
        if stride is None:
            stride = max(1, window_size // 2)
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        self._split = split
        self._window_size = int(window_size)
        self._stride = int(stride)
        self._pad_id = int(pad_id) if pad_id is not None else PAD

        # Allow callers to hand us a non-existent vocab file iff they've
        # already pinned the pad_id. Useful for tests.
        if Path(vocab_path).exists():
            with Path(vocab_path).open("r", encoding="utf-8") as f:
                vocab = json.load(f)
            specials = vocab.get("specials") or {}
            vocab_pad = specials.get("PAD")
            if pad_id is None and isinstance(vocab_pad, int):
                self._pad_id = vocab_pad

        # Load, split, then window.
        records = _load_jsonl(Path(jsonl_path))
        records = [r for r in records if _is_usable(r)]
        selected = [r for r in records if _bucket_of(r["score_id"]) == split]

        self._score_ids: List[str] = [r["score_id"] for r in selected]
        self._chord_seqs: List[List[int]] = [list(r["chord_tokens"]) for r in selected]
        self._rn_seqs: List[List[int]] = [list(r["rn_tokens"]) for r in selected]
        self._cadence_seqs: List[List[int]] = [list(r["cadence_tokens"]) for r in selected]

        self._windows: List[_Window] = list(self._build_windows())

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        w = self._windows[idx]
        W = self._window_size
        # We need W+1 raw tokens to build aligned input/target of length W.
        chord = self._slice(self._chord_seqs[w.score_idx], w.start, W + 1)
        rn = self._slice(self._rn_seqs[w.score_idx], w.start, W + 1)
        cad = self._slice(self._cadence_seqs[w.score_idx], w.start, W + 1)

        chord_input = chord[:-1]
        chord_target = chord[1:]
        rn_input = rn[:-1]
        cadence_input = cad[:-1]

        attention_mask = [
            1 if tok != self._pad_id else 0 for tok in chord_input
        ]

        return {
            "chord_input":    torch.tensor(chord_input, dtype=torch.long),
            "rn_input":       torch.tensor(rn_input, dtype=torch.long),
            "cadence_input":  torch.tensor(cadence_input, dtype=torch.long),
            "chord_target":   torch.tensor(chord_target, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Introspection helpers (used by tests + training logs)
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def split(self) -> str:
        return self._split

    @property
    def score_ids(self) -> Tuple[str, ...]:
        """Unique score_ids present in this split."""
        return tuple(dict.fromkeys(self._score_ids))

    # ------------------------------------------------------------------
    # Window + slicing internals
    # ------------------------------------------------------------------

    def _build_windows(self):
        W = self._window_size
        for score_idx, seq in enumerate(self._chord_seqs):
            L = len(seq)
            if L < 2:
                # Not enough to form a single (input, target) pair.
                continue
            # Start offsets 0, stride, 2*stride, … while the window has
            # at least one real transition inside it (i.e. start + 1 < L).
            # Last start is chosen so that the window still covers new
            # material; if the sequence is shorter than W+1 we still emit
            # one padded window.
            last_start = max(0, L - (W + 1))
            if last_start == 0:
                yield _Window(score_idx=score_idx, start=0)
                continue
            start = 0
            while start <= last_start:
                yield _Window(score_idx=score_idx, start=start)
                start += self._stride
            # Ensure the tail is covered exactly once.
            if start - self._stride < last_start:
                yield _Window(score_idx=score_idx, start=last_start)

    def _slice(self, seq: Sequence[int], start: int, length: int) -> List[int]:
        """Return ``length`` tokens from ``seq`` starting at ``start``,
        right-padded with ``pad_id`` if the source runs out."""
        end = start + length
        chunk = list(seq[start:end])
        if len(chunk) < length:
            chunk.extend([self._pad_id] * (length - len(chunk)))
        return chunk


# ---------------------------------------------------------------------------
# File + split helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    """Eagerly load all records so ``len`` is cheap and splits can scan
    every score_id before windowing."""
    if not path.exists():
        raise FileNotFoundError(path)
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"malformed JSONL at {path}:{lineno}: {e}"
                ) from e
    return records


def _is_usable(rec: Dict[str, object]) -> bool:
    """Drop records that are missing the streams we need."""
    return (
        isinstance(rec, dict)
        and "score_id" in rec
        and "chord_tokens" in rec
        and "rn_tokens" in rec
        and "cadence_tokens" in rec
    )


def _bucket_of(score_id: str) -> str:
    """Hash ``score_id`` into one of the 80/10/10 buckets."""
    h = int(hashlib.sha1(str(score_id).encode("utf-8")).hexdigest(), 16)
    bucket = h % 100
    for split, (lo, hi) in _SPLIT_THRESHOLDS.items():
        if lo <= bucket < hi:
            return split
    return "test"  # unreachable but keeps the type checker happy


__all__ = ["HarmonicSequenceDataset"]
