"""
Multi-stream harmonic tokeniser for the Bach transformer.

Stage 4 writes a JSON cache of per-score harmonic analyses
(``data/analysis_cache/corpus/*.json``). Stage 5 turns each of those files
into three aligned integer streams — **chord**, **roman numeral**,
**cadence** — that a multi-head transformer can consume directly.

Why three streams?
    A chord is not just one thing. The model needs to learn:

      * what the sonority is          → **chord** stream
      * what it means functionally    → **roman numeral** stream
      * when it closes a phrase       → **cadence** stream

    Flattening these into a single-stream vocabulary explodes the token
    count (root × quality × inversion × secondary × degree ≈ 10k) and
    discards the parallel structure.  Parallel streams keep each vocab
    small and let the transformer learn joint embeddings through separate
    heads.

Token-label conventions
-----------------------

    * **chord**    ``"<pitch-class-name>:<quality>"`` — ``"G:major"``,
      ``"A:minor7"``.  Pitch-class names use the project-wide 12-element
      table (``C``, ``C#``, ``D``, ``Eb``, ``E`` …).
    * **roman**    the existing :pyattr:`RomanNumeral.label` string —
      ``"V7"``, ``"ii6"``, ``"V/V"``, ``"iiø65"``.  Degree, quality,
      inversion, and secondary target are all already encoded there.
    * **cadence**  the :class:`CadenceType` value — ``"authentic_perfect"``,
      ``"phrygian_half"``, ``"plagal"`` — plus the sentinel ``"none"`` for
      timesteps with no cadence arrival.

Specials (reserved, identical indices across all three streams)::

    PAD = 0   BOS = 1   EOS = 2   UNK = 3

The cadence stream also reserves::

    NONE = 4   — timestep has no cadence arrival

Alignment
---------

Each chord event produces exactly one timestep in each of the three
streams.  Cadence tokens are attached at the chord index pointed to by
``cadence.final_index`` in the Stage 4 JSON; every other timestep gets
``NONE``.  Sequences are wrapped with ``BOS`` / ``EOS`` in all three
streams so the transformer sees identical boundary positions.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Pitch-class names shared with :pyattr:`src.core.data_structures`.
_PITCH_CLASS_NAMES: Tuple[str, ...] = (
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
)

#: Bumped when the on-disk format changes. Consumers refuse to read a
#: vocabulary file whose ``schema_version`` they don't recognise.
VOCAB_SCHEMA_VERSION: str = "1.0.0"

#: Special-token identifiers. Identical across every stream so positional
#: embeddings and padding masks line up.
PAD: int = 0
BOS: int = 1
EOS: int = 2
UNK: int = 3

#: Extra cadence-only sentinel: "no cadence at this timestep".
CAD_NONE: int = 4

_SHARED_SPECIALS: Tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
_CAD_SPECIALS: Tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>", "<none>")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HarmonicEncoderError(RuntimeError):
    """Base class for harmonic-encoder errors."""


class SchemaVersionMismatch(HarmonicEncoderError):
    """Raised when a vocabulary file's ``schema_version`` isn't supported."""


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


@dataclass
class HarmonicVocabulary:
    """Token↔id maps for the three harmonic streams.

    Attributes:
        chord_to_id: Maps chord labels (``"G:major"``) to integer ids.
        rn_to_id:    Maps Roman-numeral labels (``"V7"``, ``"V/V"``) to ids.
        cadence_to_id: Maps cadence labels (``"authentic_perfect"``,
            ``"none"``) to ids. Shorter than the other two vocabularies.

    The dataclass is mutable so :meth:`HarmonicEncoder.build` can fill it
    during construction; callers should treat it as immutable once built.
    """

    chord_to_id: Dict[str, int] = field(default_factory=dict)
    rn_to_id: Dict[str, int] = field(default_factory=dict)
    cadence_to_id: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------

    @property
    def chord_vocab_size(self) -> int:
        return len(self.chord_to_id)

    @property
    def rn_vocab_size(self) -> int:
        return len(self.rn_to_id)

    @property
    def cadence_vocab_size(self) -> int:
        return len(self.cadence_to_id)

    # ------------------------------------------------------------------
    # Lookups (encode direction)
    # ------------------------------------------------------------------

    def chord_id(self, label: str) -> int:
        """Return the chord id for ``label``; falls back to UNK."""
        return self.chord_to_id.get(label, UNK)

    def rn_id(self, label: str) -> int:
        """Return the Roman-numeral id for ``label``; falls back to UNK."""
        return self.rn_to_id.get(label, UNK)

    def cadence_id(self, label: str) -> int:
        """Return the cadence id for ``label``; falls back to UNK."""
        return self.cadence_to_id.get(label, UNK)

    # ------------------------------------------------------------------
    # Lookups (decode direction)
    # ------------------------------------------------------------------

    def chord_label(self, idx: int) -> str:
        return _reverse(self.chord_to_id)[idx]

    def rn_label(self, idx: int) -> str:
        return _reverse(self.rn_to_id)[idx]

    def cadence_label(self, idx: int) -> str:
        return _reverse(self.cadence_to_id)[idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": VOCAB_SCHEMA_VERSION,
            "specials": {
                "PAD": PAD,
                "BOS": BOS,
                "EOS": EOS,
                "UNK": UNK,
                "CAD_NONE": CAD_NONE,
            },
            "chord_to_id": dict(self.chord_to_id),
            "rn_to_id": dict(self.rn_to_id),
            "cadence_to_id": dict(self.cadence_to_id),
        }

    def save(self, path: str | Path) -> None:
        """Write the vocabulary to a JSON file."""
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HarmonicVocabulary":
        """Load a vocabulary from the JSON file written by :meth:`save`.

        Raises:
            SchemaVersionMismatch: if the file's ``schema_version`` differs
                from :data:`VOCAB_SCHEMA_VERSION`.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("schema_version")
        if version != VOCAB_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"vocabulary {path} has schema_version={version!r}, "
                f"expected {VOCAB_SCHEMA_VERSION!r}"
            )
        return cls(
            chord_to_id=dict(data["chord_to_id"]),
            rn_to_id=dict(data["rn_to_id"]),
            cadence_to_id=dict(data["cadence_to_id"]),
        )


def _reverse(d: Mapping[str, int]) -> Dict[int, str]:
    """Return the inverse mapping. Defined as a free helper so it caches
    naturally via a per-instance Python binding if callers memoize."""
    return {idx: label for label, idx in d.items()}


# ---------------------------------------------------------------------------
# Encoded sequence container
# ---------------------------------------------------------------------------


@dataclass
class EncodedSequence:
    """Three aligned token streams for one score.

    Attributes:
        score_id: Matches the ``score_id`` field of the Stage 4 JSON.
        chord_tokens:   Chord-stream ids, ``[BOS, …, EOS]``.
        rn_tokens:      Roman-numeral-stream ids, same length.
        cadence_tokens: Cadence-stream ids, same length, filled with
            ``CAD_NONE`` at timesteps without a cadence arrival.
        onsets: Onset string per timestep (``"3/2"``) or ``""`` for specials.
            Preserved for debugging and for downstream tooling that wants
            to align tokens with the score timeline.
    """

    score_id: str
    chord_tokens: List[int]
    rn_tokens: List[int]
    cadence_tokens: List[int]
    onsets: List[str]

    def __post_init__(self) -> None:
        if not (
            len(self.chord_tokens)
            == len(self.rn_tokens)
            == len(self.cadence_tokens)
            == len(self.onsets)
        ):
            raise ValueError(
                "chord_tokens, rn_tokens, cadence_tokens, onsets must all "
                "have the same length"
            )

    @property
    def length(self) -> int:
        return len(self.chord_tokens)

    def to_dict(self) -> Dict[str, object]:
        return {
            "score_id": self.score_id,
            "chord_tokens": list(self.chord_tokens),
            "rn_tokens": list(self.rn_tokens),
            "cadence_tokens": list(self.cadence_tokens),
            "onsets": list(self.onsets),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EncodedSequence":
        return cls(
            score_id=str(data["score_id"]),
            chord_tokens=list(data["chord_tokens"]),  # type: ignore[arg-type]
            rn_tokens=list(data["rn_tokens"]),        # type: ignore[arg-type]
            cadence_tokens=list(data["cadence_tokens"]),  # type: ignore[arg-type]
            onsets=list(data["onsets"]),              # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Label extraction from Stage 4 JSON
# ---------------------------------------------------------------------------


def chord_label(chord: Mapping[str, object]) -> str:
    """Render a chord dict from the Stage 4 JSON as its token label.

    Args:
        chord: dict with integer ``root`` (0-11) and string ``quality``.
    """
    root = int(chord["root"])  # type: ignore[arg-type]
    quality = str(chord["quality"])
    return f"{_PITCH_CLASS_NAMES[root]}:{quality}"


def rn_label(rn: Mapping[str, object]) -> str:
    """Return the canonical Roman-numeral label from a Stage 4 RN dict.

    The Stage 4 JSON already carries ``label`` (e.g. ``"V7"``, ``"ii6"``)
    — produced by :pymeth:`RomanNumeral.label`. We use it verbatim so the
    round-trip to musicological notation is preserved.
    """
    return str(rn["label"])


def cadence_label(cadence: Mapping[str, object]) -> str:
    """Return the cadence type string (``"authentic_perfect"`` etc.)."""
    return str(cadence["cadence_type"])


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


@dataclass
class HarmonicEncoderConfig:
    """Tunable parameters for vocabulary construction.

    Attributes:
        min_chord_count: Chord labels observed fewer than this many times
            across the corpus are dropped from the vocabulary (they will
            encode as ``UNK``). Guards against long-tail template artefacts.
        min_rn_count: Same for Roman-numeral labels. Roman numerals have a
            long tail of secondary-dominant permutations; the default
            threshold is slightly higher than for chords.
    """

    min_chord_count: int = 1
    min_rn_count: int = 1


class HarmonicEncoder:
    """Converts Stage 4 analysis JSONs into three aligned token streams.

    Usage::

        enc = HarmonicEncoder.from_cache_dir("data/analysis_cache")
        seq = enc.encode_file("data/analysis_cache/corpus/bwv66.6.json")

    The encoder is stateless apart from its :attr:`vocabulary`. Rebuild
    the vocabulary whenever the underlying corpus changes; persistent
    vocabularies should be loaded with :pymeth:`HarmonicVocabulary.load`
    to keep ids stable across training runs.
    """

    def __init__(self, vocabulary: HarmonicVocabulary) -> None:
        self.vocabulary = vocabulary

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_cache_dir(
        cls,
        cache_dir: str | Path,
        config: Optional[HarmonicEncoderConfig] = None,
    ) -> "HarmonicEncoder":
        """Build an encoder from every analysis JSON under ``cache_dir``.

        ``manifest.json`` and ``failures.jsonl`` are skipped — only files
        that look like per-score analyses (have a ``chords`` key) are
        consumed.
        """
        vocab = build_vocabulary(
            _iter_analysis_files(Path(cache_dir)),
            config or HarmonicEncoderConfig(),
        )
        return cls(vocab)

    @classmethod
    def from_analyses(
        cls,
        analyses: Iterable[Mapping[str, object]],
        config: Optional[HarmonicEncoderConfig] = None,
    ) -> "HarmonicEncoder":
        """Build an encoder directly from in-memory analysis dicts.

        Handy for tests — avoids touching the filesystem.
        """
        vocab = build_vocabulary(analyses, config or HarmonicEncoderConfig())
        return cls(vocab)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, analysis: Mapping[str, object]) -> EncodedSequence:
        """Encode one Stage 4 analysis dict into aligned token streams."""
        chords = list(analysis.get("chords", []))  # type: ignore[arg-type]
        rns = list(analysis.get("roman_numerals", []))  # type: ignore[arg-type]
        cadences = list(analysis.get("cadences", []))  # type: ignore[arg-type]

        if len(chords) != len(rns):
            raise HarmonicEncoderError(
                f"chord/rn length mismatch for "
                f"{analysis.get('score_id', '<unknown>')}: "
                f"{len(chords)} vs {len(rns)}"
            )

        # Build cadence-by-final-index lookup.
        cadence_at: Dict[int, str] = {}
        for cad in cadences:
            idx = int(cad.get("final_index", -1))  # type: ignore[arg-type]
            if 0 <= idx < len(chords):
                cadence_at[idx] = cadence_label(cad)

        chord_ids: List[int] = [BOS]
        rn_ids: List[int] = [BOS]
        cad_ids: List[int] = [BOS]
        onsets: List[str] = [""]

        for i, (chord, rn) in enumerate(zip(chords, rns)):
            chord_ids.append(self.vocabulary.chord_id(chord_label(chord)))
            rn_ids.append(self.vocabulary.rn_id(rn_label(rn)))
            if i in cadence_at:
                cad_ids.append(self.vocabulary.cadence_id(cadence_at[i]))
            else:
                cad_ids.append(CAD_NONE)
            onsets.append(str(chord.get("onset", "")))

        chord_ids.append(EOS)
        rn_ids.append(EOS)
        cad_ids.append(EOS)
        onsets.append("")

        return EncodedSequence(
            score_id=str(analysis.get("score_id", "<unknown>")),
            chord_tokens=chord_ids,
            rn_tokens=rn_ids,
            cadence_tokens=cad_ids,
            onsets=onsets,
        )

    def encode_file(self, path: str | Path) -> EncodedSequence:
        """Load a Stage 4 JSON from disk and encode it."""
        with Path(path).open("r", encoding="utf-8") as f:
            analysis = json.load(f)
        return self.encode(analysis)

    # ------------------------------------------------------------------
    # Decoding (debug)
    # ------------------------------------------------------------------

    def decode(self, seq: EncodedSequence) -> Dict[str, List[str]]:
        """Decode each stream back to human-readable token labels.

        Specials render as ``"<bos>"`` / ``"<eos>"`` / ``"<pad>"`` /
        ``"<unk>"``; cadence ``CAD_NONE`` renders as ``"<none>"``.
        """
        return {
            "chord": [_decode_token(t, self.vocabulary.chord_to_id, is_cadence=False)
                      for t in seq.chord_tokens],
            "rn": [_decode_token(t, self.vocabulary.rn_to_id, is_cadence=False)
                   for t in seq.rn_tokens],
            "cadence": [_decode_token(t, self.vocabulary.cadence_to_id, is_cadence=True)
                        for t in seq.cadence_tokens],
        }


def _decode_token(
    token_id: int,
    vocab: Mapping[str, int],
    *,
    is_cadence: bool,
) -> str:
    """Render one token id as its string form, covering specials."""
    if token_id == PAD:
        return "<pad>"
    if token_id == BOS:
        return "<bos>"
    if token_id == EOS:
        return "<eos>"
    if token_id == UNK:
        return "<unk>"
    if is_cadence and token_id == CAD_NONE:
        return "<none>"
    rev = _reverse(vocab)
    return rev.get(token_id, "<unk>")


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------


def build_vocabulary(
    analyses: Iterable[Mapping[str, object]],
    config: HarmonicEncoderConfig,
) -> HarmonicVocabulary:
    """Count labels across ``analyses`` and assign deterministic ids.

    Tokens are assigned in **descending frequency order** after the
    specials. Ties are broken alphabetically so vocab ids are fully
    reproducible given the same corpus.
    """
    chord_counter: Counter[str] = Counter()
    rn_counter: Counter[str] = Counter()
    cadence_counter: Counter[str] = Counter()

    for analysis in analyses:
        for chord in analysis.get("chords", []) or []:   # type: ignore[union-attr]
            chord_counter[chord_label(chord)] += 1
        for rn in analysis.get("roman_numerals", []) or []:  # type: ignore[union-attr]
            rn_counter[rn_label(rn)] += 1
        for cad in analysis.get("cadences", []) or []:   # type: ignore[union-attr]
            cadence_counter[cadence_label(cad)] += 1

    # Filter by minimum count, then sort.
    chord_sorted = _sorted_by_count(chord_counter, config.min_chord_count)
    rn_sorted = _sorted_by_count(rn_counter, config.min_rn_count)
    cadence_sorted = _sorted_by_count(cadence_counter, 1)

    return HarmonicVocabulary(
        chord_to_id=_assign_ids(chord_sorted, _SHARED_SPECIALS),
        rn_to_id=_assign_ids(rn_sorted, _SHARED_SPECIALS),
        cadence_to_id=_assign_ids(cadence_sorted, _CAD_SPECIALS),
    )


def _sorted_by_count(counter: Counter[str], min_count: int) -> List[str]:
    """Return labels sorted by descending count, alphabetical tiebreak."""
    return sorted(
        (label for label, c in counter.items() if c >= min_count),
        key=lambda label: (-counter[label], label),
    )


def _assign_ids(labels: Sequence[str], specials: Sequence[str]) -> Dict[str, int]:
    """Assign ids ``0..len(specials)-1`` to specials, then the rest."""
    mapping: Dict[str, int] = {name: idx for idx, name in enumerate(specials)}
    next_id = len(specials)
    for label in labels:
        if label in mapping:
            continue
        mapping[label] = next_id
        next_id += 1
    return mapping


# ---------------------------------------------------------------------------
# File enumeration
# ---------------------------------------------------------------------------


def _iter_analysis_files(cache_dir: Path) -> Iterable[Mapping[str, object]]:
    """Yield every parseable analysis JSON under ``cache_dir``.

    Skips ``manifest.json`` / ``failures.jsonl`` and anything lacking the
    per-score ``chords`` key (defensive against partial caches).
    """
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    for path in sorted(cache_dir.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "chords" in data:
            yield data


__all__ = [
    "BOS",
    "CAD_NONE",
    "EOS",
    "EncodedSequence",
    "HarmonicEncoder",
    "HarmonicEncoderConfig",
    "HarmonicEncoderError",
    "HarmonicVocabulary",
    "PAD",
    "SchemaVersionMismatch",
    "UNK",
    "VOCAB_SCHEMA_VERSION",
    "build_vocabulary",
    "cadence_label",
    "chord_label",
    "rn_label",
]
