"""
backend/src/models/generator.py

Grammar-constrained autoregressive music generation.

Pipeline per decoding step:
  1. Forward pass → logits (chord_vocab_size,)
  2. Temperature scaling
  3. BaroqueGrammar hard constraints → set forbidden logits to -inf
  4. Cadence boost at required positions
  5. top-k or nucleus (top-p) sampling

After generation: GrammarValidator scores the full RN sequence.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from src.corpus.transition_matrix import TransitionMatrix
from src.data.harmonic_encoder import HarmonicVocabulary, PAD, BOS, EOS, UNK, CAD_NONE
from src.grammar.rules import BaroqueGrammar
from src.grammar.validator import GrammarReport, GrammarValidator
from src.models.music_transformer import MusicTransformer


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class GeneratedSequence:
    """
    Complete output of one generation call.

    Attributes:
        chord_tokens:   Decoded chord token IDs (excluding BOS/EOS).
        rn_sequence:    Human-readable Roman numeral labels aligned
                        to chord_tokens. Derived via transition_matrix
                        top-1 lookup for each chord token.
        cadence_tokens: Cadence stream token IDs (length == chord_tokens).
        onsets:         Beat position of each chord event.
                        Assumes quarter-note = 1 beat per token.
        grammar_report: GrammarValidator output for this sequence.
        metadata:       Generation parameters stored for reproducibility.
    """
    chord_tokens:   list[int]
    rn_sequence:    list[str]
    cadence_tokens: list[int]
    onsets:         list[float]
    grammar_report: GrammarReport
    metadata:       dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MusicGenerator:
    """
    Loads a trained MusicTransformer checkpoint and generates
    Bach-style chord sequences with grammar constraints.

    Grammar constraints are ALWAYS active during generate().
    They cannot be disabled at inference time.

    Usage::

        gen = MusicGenerator.from_checkpoint(
            "checkpoints/transformer-v1/best.pt",
            "data/encoded/vocab.json",
            "data/encoded/transition_matrix.json",
        )
        seq = gen.generate(key_mode="minor", n_tokens=64)
        score = gen.decode_to_music21(seq)
        score.write("musicxml", "output.xml")
    """

    # Logit boost applied to dominant-function chords when a cadence
    # is required at the current position.
    _CADENCE_BOOST = 4.0

    def __init__(
        self,
        model: MusicTransformer,
        vocab: HarmonicVocabulary,
        transition_matrix: TransitionMatrix,
        grammar: BaroqueGrammar,
        validator: GrammarValidator,
        device: torch.device,
    ) -> None:
        self._model     = model.to(device).eval()
        self._vocab     = vocab
        self._matrix    = transition_matrix
        self._grammar   = grammar
        self._validator = validator
        self._device    = device

        # Pre-build chord_id → most-likely RN label cache for constraint lookup.
        # Uses transition_matrix top-1 in minor (conservative default).
        self._chord_to_rn: dict[int, str] = self._build_chord_to_rn_cache()

        # Identify chord IDs that imply dominant function (V, V7, viio).
        # Used for cadence boost.
        self._dominant_chord_ids: set[int] = self._find_dominant_chord_ids()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        vocab_path: str,
        transition_matrix_path: str,
        device: str = "auto",
    ) -> "MusicGenerator":
        """
        Load a self-describing checkpoint and wire up all dependencies.

        Args:
            checkpoint_path:        Path to best.pt produced by TransformerTrainer.
            vocab_path:             Path to vocab.json from Stage 5.
            transition_matrix_path: Path to transition_matrix.json from Stage 6.
            device:                 "auto" | "mps" | "cuda" | "cpu".

        Returns:
            Ready-to-use MusicGenerator.

        Raises:
            FileNotFoundError: if any input path does not exist.
        """
        for p in (checkpoint_path, vocab_path, transition_matrix_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        resolved = _resolve_device(device)
        model, _ = MusicTransformer.load_checkpoint(checkpoint_path, device=str(resolved))

        vocab  = HarmonicVocabulary.load(vocab_path)
        matrix = TransitionMatrix.from_json(transition_matrix_path)

        return cls(
            model=model,
            vocab=vocab,
            transition_matrix=matrix,
            grammar=BaroqueGrammar(),
            validator=GrammarValidator(),
            device=resolved,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        key_mode:      str            = "minor",
        n_tokens:      int            = 128,
        temperature:   float          = 0.9,
        top_k:         int            = 10,
        top_p:         Optional[float] = None,
        prompt_tokens: Optional[dict]  = None,
        seed:          Optional[int]   = None,
    ) -> GeneratedSequence:
        """
        Generate n_tokens chord tokens autoregressively.

        Args:
            key_mode:      "minor" or "major" — used for grammar constraints.
            n_tokens:      Number of chord tokens to generate (excl. BOS/EOS).
            temperature:   Softmax temperature. Lower = more deterministic.
            top_k:         Keep only top-k logits before sampling. 0 = disabled.
            top_p:         Nucleus sampling threshold. None = disabled.
                           If both top_k and top_p are set, top_k is applied
                           first, then top_p.
            prompt_tokens: Optional dict with keys "chord_tokens", "rn_tokens",
                           "cadence_tokens" (each list[int]) to prime generation.
                           If None, starts from [BOS] on all three streams.
            seed:          Random seed for reproducibility. None = non-deterministic.

        Returns:
            GeneratedSequence with grammar_report attached.
        """
        if seed is not None:
            torch.manual_seed(seed)

        bos  = BOS
        pad  = PAD  # noqa: F841
        none = CAD_NONE

        # Initialise token buffers
        if prompt_tokens is not None:
            chord_buf   = list(prompt_tokens.get("chord_tokens",   [bos]))
            rn_buf      = list(prompt_tokens.get("rn_tokens",      [bos]))
            cadence_buf = list(prompt_tokens.get("cadence_tokens",  [none]))
        else:
            chord_buf   = [bos]
            rn_buf      = [bos]
            cadence_buf = [none]

        beats_per_token = 1.0          # quarter note per chord event
        tokens_per_measure = 4         # 4/4 assumed; cadence scheduler uses measures

        generated_chord_tokens: list[int] = []

        # Initialise prev_rn from the last rn token in the buffer so that
        # grammar constraints correctly reflect any prompt context.
        _rn_id_to_label = {v: k for k, v in self._vocab.rn_to_id.items()}
        last_rn_id = rn_buf[-1]
        if last_rn_id in _rn_id_to_label:
            prev_rn = _rn_id_to_label[last_rn_id]
        else:
            prev_rn = "i" if key_mode == "minor" else "I"

        with torch.no_grad():
            for step in range(n_tokens):
                # ── Build input tensors ───────────────────────────────────
                c_t = torch.tensor([chord_buf],   dtype=torch.long, device=self._device)
                r_t = torch.tensor([rn_buf],      dtype=torch.long, device=self._device)
                a_t = torch.tensor([cadence_buf], dtype=torch.long, device=self._device)

                # ── Forward pass ──────────────────────────────────────────
                logits = self._model(c_t, r_t, a_t)   # (1, T, chord_vocab_size)
                logits = logits[0, -1, :]              # (chord_vocab_size,)

                # ── Temperature ───────────────────────────────────────────
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-8)

                # ── Grammar hard constraints ──────────────────────────────
                logits = self._apply_grammar_constraints(logits, prev_rn, key_mode)

                # ── Cadence boost ─────────────────────────────────────────
                current_measure = step // tokens_per_measure
                total_measures  = n_tokens // tokens_per_measure + 1
                if self._grammar.requires_cadence_at(current_measure, total_measures):
                    logits = self._apply_cadence_boost(logits)

                # ── Sampling ──────────────────────────────────────────────
                next_id = self._sample(logits, top_k=top_k, top_p=top_p)

                # ── Update buffers ────────────────────────────────────────
                chord_buf.append(next_id)
                rn_label = self._chord_to_rn.get(next_id, prev_rn)
                rn_id    = self._vocab.rn_id(rn_label)
                rn_buf.append(rn_id)
                cadence_buf.append(none)     # cadence stream: none during body

                generated_chord_tokens.append(next_id)
                prev_rn = rn_label

        # ── Decode RN sequence ────────────────────────────────────────────
        rn_sequence = [
            self._chord_to_rn.get(t, "?") for t in generated_chord_tokens
        ]

        # ── Grammar validation ────────────────────────────────────────────
        cadence_measures = [
            m for m in range(total_measures)
            if self._grammar.requires_cadence_at(m, total_measures)
        ]
        grammar_report = self._validator.validate(
            rn_sequence=rn_sequence,
            key_mode=key_mode,
            matrix=self._matrix,
            grammar=self._grammar,
            cadence_measures=cadence_measures,
            total_measures=total_measures,
        )

        onsets = [i * beats_per_token for i in range(len(generated_chord_tokens))]

        return GeneratedSequence(
            chord_tokens=generated_chord_tokens,
            rn_sequence=rn_sequence,
            cadence_tokens=[none] * len(generated_chord_tokens),
            onsets=onsets,
            grammar_report=grammar_report,
            metadata={
                "key_mode":    key_mode,
                "n_tokens":    n_tokens,
                "temperature": temperature,
                "top_k":       top_k,
                "top_p":       top_p,
                "seed":        seed,
            },
        )

    # ------------------------------------------------------------------
    # Music21 export
    # ------------------------------------------------------------------

    def decode_to_music21(
        self,
        sequence: GeneratedSequence,
        instrument: str = "violoncello",
    ):
        """
        Convert generated chord token sequence → music21 Score.

        Each token becomes a quarter-note chord in root position.
        Key signature is inferred from sequence.metadata["key_mode"].
        Instrument is set to violoncello (thesis target).

        Returns:
            music21.stream.Score — MusicXML-exportable via
            score.write("musicxml", "output.xml").
        """
        import music21 as m21

        key_mode = sequence.metadata.get("key_mode", "minor")
        score    = m21.stream.Score()
        part     = m21.stream.Part()

        # Instrument
        instr = m21.instrument.fromString(instrument)
        part.append(instr)

        # Key signature
        ks = m21.key.Key("c", key_mode) if key_mode == "minor" \
             else m21.key.Key("C", key_mode)
        part.append(ks)

        # Time signature
        part.append(m21.meter.TimeSignature("4/4"))

        for token_id in sequence.chord_tokens:
            label = self._vocab.chord_label(token_id)
            m21_element = self._chord_label_to_music21(label)
            part.append(m21_element)

        score.append(part)
        return score

    def generate_musicxml(
        self,
        key_mode:    str   = "minor",
        n_tokens:    int   = 128,
        temperature: float = 0.9,
    ) -> str:
        """
        Convenience method: generate → decode → return base64 MusicXML string.
        Used directly by the FastAPI /api/generate endpoint.
        """
        seq   = self.generate(key_mode=key_mode, n_tokens=n_tokens,
                              temperature=temperature)
        score = self.decode_to_music21(seq)

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            score.write("musicxml", fp=tmp_path)
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_grammar_constraints(
        self,
        logits:   torch.Tensor,
        prev_rn:  str,
        key_mode: str,
    ) -> torch.Tensor:
        """
        Set logits for chord tokens that would produce a forbidden
        RN transition to -inf.

        Forbidden transitions are defined on RN labels. We map each
        candidate chord_id to its most likely RN via _chord_to_rn,
        then call BaroqueGrammar.is_valid_transition().
        """
        logits = logits.clone()
        for chord_id, rn_label in self._chord_to_rn.items():
            if chord_id >= logits.shape[0]:
                continue
            if not self._grammar.is_valid_transition(prev_rn, rn_label, key_mode):
                logits[chord_id] = float("-inf")
        return logits

    def _apply_cadence_boost(self, logits: torch.Tensor) -> torch.Tensor:
        """Boost logits for dominant-function chord IDs by _CADENCE_BOOST."""
        logits = logits.clone()
        for chord_id in self._dominant_chord_ids:
            if chord_id < logits.shape[0]:
                logits[chord_id] += self._CADENCE_BOOST
        return logits

    def _sample(
        self,
        logits: torch.Tensor,
        top_k:  int,
        top_p:  Optional[float],
    ) -> int:
        """Apply top-k and/or nucleus filtering, then sample."""
        # top-k
        if top_k > 0:
            k = min(top_k, logits.shape[-1])
            top_vals, _ = torch.topk(logits, k)
            threshold = top_vals[..., -1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # top-p (nucleus)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)

        # If all logits were -inf (should not happen with Laplace smoothing,
        # but guard against it), fall back to uniform.
        if torch.isnan(probs).any() or probs.sum() == 0:
            probs = torch.ones_like(probs) / probs.shape[0]

        return int(torch.multinomial(probs, num_samples=1).item())

    def _build_chord_to_rn_cache(self) -> dict[int, str]:
        """
        Map each chord_id to its most semantically consistent RN label.

        Strategy: parse the chord label ("G:minor" → root + quality)
        and infer the RN degree from the root interval relative to C
        (key-agnostic approximation). Falls back to top-1 from
        transition_matrix in minor if parsing fails.
        """
        chord_to_rn: dict[int, str] = {}
        specials = {PAD, BOS, EOS, UNK, CAD_NONE}

        # Degree lookup: semitone interval from C → diatonic degree (minor)
        _semitone_to_degree_minor = {
            0: "i", 2: "ii", 3: "III", 5: "iv",
            7: "v",  8: "VI", 10: "VII",
        }
        _quality_to_suffix = {
            "major":      "",
            "minor":      "",
            "diminished": "°",
            "augmented":  "+",
            "dominant7":  "7",
            "major7":     "M7",
            "minor7":     "m7",
            "dim7":       "°7",
            "half-dim7":  "ø7",
        }
        _note_to_semitone = {
            "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
        }

        for label, idx in self._vocab.chord_to_id.items():
            if idx in specials:
                continue
            try:
                root_str, quality = label.split(":", 1)
                root_name = root_str[0].upper()
                semitone  = _note_to_semitone.get(root_name, 0)
                # Handle sharps/flats in root
                for ch in root_str[1:]:
                    if ch == "#": semitone += 1
                    elif ch in ("b", "-"): semitone -= 1
                semitone %= 12
                degree = _semitone_to_degree_minor.get(semitone, "i")
                suffix = _quality_to_suffix.get(quality, "")
                chord_to_rn[idx] = f"{degree}{suffix}"
            except (ValueError, AttributeError):
                chord_to_rn[idx] = "i"   # safe fallback

        return chord_to_rn

    def _find_dominant_chord_ids(self) -> set[int]:
        """
        Return chord IDs whose _chord_to_rn label starts with "V"
        (dominant function — candidates for cadence boost).
        """
        return {
            cid for cid, rn in self._chord_to_rn.items()
            if rn.startswith("V") or rn.startswith("v")
        }

    @staticmethod
    def _chord_label_to_music21(label: str):
        """
        Convert a chord label string ("G:minor", "C:major") to a
        music21 element (Chord or Rest).
        """
        import music21 as m21

        specials = {"PAD", "BOS", "EOS", "UNK", "CAD_NONE", "none", ""}
        if label in specials or ":" not in label:
            r = m21.note.Rest()
            r.quarterLength = 1.0
            return r

        try:
            root_str, quality = label.split(":", 1)
            root = m21.pitch.Pitch(root_str)
            quality_map = {
                "major":      [0, 4, 7],
                "minor":      [0, 3, 7],
                "diminished": [0, 3, 6],
                "augmented":  [0, 4, 8],
                "dominant7":  [0, 4, 7, 10],
                "major7":     [0, 4, 7, 11],
                "minor7":     [0, 3, 7, 10],
                "dim7":       [0, 3, 6, 9],
                "half-dim7":  [0, 3, 6, 10],
            }
            intervals = quality_map.get(quality, [0, 4, 7])
            pitches   = [root.transpose(i) for i in intervals]
            chord = m21.chord.Chord(pitches)
            chord.quarterLength = 1.0
            return chord
        except Exception:
            r = m21.note.Rest()
            r.quarterLength = 1.0
            return r