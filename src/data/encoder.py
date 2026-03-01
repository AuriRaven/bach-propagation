"""
Integer token vocabulary for the Bach harmonic sequence model.

Vocabulary layout (90 tokens total)
-------------------------------------
  0          PAD
  1          START
  2          END
  3 – 62     Chord tokens  (60 = 12 roots × 5 canonical qualities)
  63 – 65    Function tokens  (0=T, 1=PD, 2=D → offset by FUNCTION_OFFSET)
  66 – 89    Key tokens  (24 = 12 tonics × 2 modes)

Non-standard qualities are collapsed before encoding:
  major7   → major     (idx 0)
  minor7   → minor     (idx 1)
  dim7     → diminished (idx 2)
  half-dim7→ diminished (idx 2)
  augmented→ augmented  (idx 3)
  dominant7→ dominant7  (idx 4)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

PAD: int = 0
START: int = 1
END: int = 2
N_SPECIAL: int = 3

CHORD_OFFSET: int = N_SPECIAL        # 3..62   (60 tokens)
FUNCTION_OFFSET: int = 63            # 63..65  (3 tokens)
KEY_OFFSET: int = 66                 # 66..89  (24 tokens)
TOTAL_VOCAB: int = 90

CHORD_VOCAB_SIZE: int = 60
FUNCTION_VOCAB_SIZE: int = 3
KEY_VOCAB_SIZE: int = 24

# Quality → canonical quality index
_QUALITY_TO_IDX: Dict[str, int] = {
    'major':      0,
    'major7':     0,
    'minor':      1,
    'minor7':     1,
    'diminished': 2,
    'dim7':       2,
    'half-dim7':  2,
    'augmented':  3,
    'dominant7':  4,
}

_IDX_TO_QUALITY: List[str] = ['major', 'minor', 'diminished', 'augmented', 'dominant7']

# Mode → index for key tokens
_MODE_TO_IDX: Dict[str, int] = {'major': 0, 'minor': 1}
_IDX_TO_MODE: List[str] = ['major', 'minor']


# ---------------------------------------------------------------------------
# Encoder class
# ---------------------------------------------------------------------------

class Encoder:
    """
    Converts event dicts ↔ integer tokens.

    Stateless except for an optional chord→function majority-vote mapping
    built from training data via ``build_chord_function_map``.
    """

    def __init__(self) -> None:
        self._chord_fn_map: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Chord tokens
    # ------------------------------------------------------------------

    def chord_to_token(self, root: int, quality: str) -> int:
        """
        Return the chord token for the given root (0-11) and quality string.

        Unknown qualities map to idx=0 (major) as a safe fallback.
        """
        quality_idx = _QUALITY_TO_IDX.get(quality, 0)
        return CHORD_OFFSET + root * 5 + quality_idx

    def token_to_chord(self, token: int) -> Tuple[int, str]:
        """
        Decode a chord token back to (root, canonical_quality).

        Raises ValueError for tokens outside the chord range.
        """
        if not (CHORD_OFFSET <= token < CHORD_OFFSET + CHORD_VOCAB_SIZE):
            raise ValueError(f"Token {token} is not a chord token (expected {CHORD_OFFSET}..{CHORD_OFFSET + CHORD_VOCAB_SIZE - 1})")
        offset = token - CHORD_OFFSET
        root = offset // 5
        quality_idx = offset % 5
        return root, _IDX_TO_QUALITY[quality_idx]

    # ------------------------------------------------------------------
    # Function tokens
    # ------------------------------------------------------------------

    def function_to_token(self, function_int: int) -> int:
        """Convert harmonic function int (0/1/2) to vocabulary token."""
        return FUNCTION_OFFSET + function_int

    def token_to_function(self, token: int) -> int:
        """Convert a function vocabulary token back to int (0/1/2)."""
        if not (FUNCTION_OFFSET <= token < FUNCTION_OFFSET + FUNCTION_VOCAB_SIZE):
            raise ValueError(f"Token {token} is not a function token")
        return token - FUNCTION_OFFSET

    # ------------------------------------------------------------------
    # Key tokens
    # ------------------------------------------------------------------

    def key_to_token(self, tonic: int, mode: str) -> int:
        """Convert (tonic 0-11, mode 'major'/'minor') to a key token."""
        mode_idx = _MODE_TO_IDX.get(mode, 0)
        return KEY_OFFSET + tonic * 2 + mode_idx

    def token_to_key(self, token: int) -> Tuple[int, str]:
        """Decode a key token back to (tonic, mode)."""
        if not (KEY_OFFSET <= token < KEY_OFFSET + KEY_VOCAB_SIZE):
            raise ValueError(f"Token {token} is not a key token")
        offset = token - KEY_OFFSET
        tonic = offset // 2
        mode_idx = offset % 2
        return tonic, _IDX_TO_MODE[mode_idx]

    # ------------------------------------------------------------------
    # Sequence encoding
    # ------------------------------------------------------------------

    def encode_sequence(self, sequence: List[dict]) -> List[int]:
        """
        Encode an event dict list to ``[START] + [chord_token, ...] + [END]``.

        Empty sequences produce ``[START, END]``.
        """
        tokens = [START]
        for ev in sequence:
            tokens.append(self.chord_to_token(ev['chord_root'], ev['chord_quality']))
        tokens.append(END)
        return tokens

    def encode_functions(self, sequence: List[dict]) -> List[int]:
        """
        Return function-int per event (parallel to body of encode_sequence;
        no START/END tokens).
        """
        return [ev['harmonic_function'] for ev in sequence]

    # ------------------------------------------------------------------
    # Chord → function majority vote
    # ------------------------------------------------------------------

    def build_chord_function_map(
        self,
        sequences: List[List[dict]],
    ) -> Dict[int, int]:
        """
        For each chord token observed across *sequences*, compute the
        majority harmonic function. Stores the result internally and returns
        the mapping dict.
        """
        counts: Dict[int, Counter] = {}
        for seq in sequences:
            for ev in seq:
                tok = self.chord_to_token(ev['chord_root'], ev['chord_quality'])
                fn = ev['harmonic_function']
                if tok not in counts:
                    counts[tok] = Counter()
                counts[tok][fn] += 1

        self._chord_fn_map = {
            tok: counter.most_common(1)[0][0]
            for tok, counter in counts.items()
        }
        return self._chord_fn_map

    def token_function(self, chord_token: int) -> int:
        """
        Look up the majority harmonic function for a chord token.

        Falls back to 0 (TONIC) for unseen tokens.
        """
        return self._chord_fn_map.get(chord_token, 0)
