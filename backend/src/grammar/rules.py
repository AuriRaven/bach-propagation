"""
Hard constraints encoding Baroque tonal rules.

These rules are applied at **inference time** to prune the beam-search
frontier. A corpus-only model (a transformer trained on the analysis
cache, or the Layer-6.1 Markov prior) will still occasionally propose
transitions that are textbook errors — ``V7 → IV`` is a classic example,
as is a root-position Neapolitan without preparation. A 266-chorale
corpus is not large enough to drive those probabilities to zero on its
own, so we encode the rule explicitly and veto the move at decode time.

Design
------
Two tiers:

* **Forbidden** progressions receive weight ``0.0`` and are deleted from
  the beam.
* **Discouraged** progressions receive ``probability × 0.01`` — they are
  allowed when they are overwhelmingly supported by the corpus, but are
  severely penalised relative to stronger alternatives.

The forbidden set is intentionally small. Adding too many rules here
biases the output away from genuine Bach idioms (he *does* write
retrogressions, he *does* use Neapolitans). We encode only transitions
that are reliably wrong across the whole Baroque literature.

Cadence scheduling
------------------
Baroque phrases are typically 4-8 bars long and every major formal
boundary is articulated by a cadence. ``requires_cadence_at`` enforces
a cadence every 8 measures at minimum, plus at the final two measures
(so the piece ends on a proper close rather than drifting into
harmonic prose).

Key-frame rule
--------------
Tonal pieces begin and end on the tonic. ``validate_key_frame`` checks
that the first and last Roman numerals are ``I`` (major) or ``i``
(minor). Anything else — opening on ``V``, ending on ``IV`` — is
recorded as a violation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Tuple

if TYPE_CHECKING:
    from src.corpus.transition_matrix import TransitionMatrix


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------


#: Progressions that are *never* valid in Bach's harmonic practice.
#: Encoded as literal string pairs so they can be looked up in O(1).
FORBIDDEN_PROGRESSIONS: FrozenSet[Tuple[str, str]] = frozenset({
    ("I",  "bII"),   # Neapolitan without preparation (major)
    ("i",  "bII"),   # Neapolitan without preparation (minor)
    ("V7", "IV"),    # Dominant seventh resolving up (parallel 5ths)
    ("V7", "iv"),    # Dominant seventh resolving up (minor)
})

#: Progressions that are idiomatically weak but not impossible. A
#: transformer will occasionally suggest them; we penalise but don't
#: forbid.
DISCOURAGED_PROGRESSIONS: FrozenSet[Tuple[str, str]] = frozenset({
    ("V", "IV"),    # Retrogression (V→IV) — permitted in sequences only
    ("V", "iv"),
    ("vii", "IV"),
    ("viio", "IV"),
})

_TONIC_LABELS_BY_MODE = {"major": ("I",), "minor": ("i",)}


# ---------------------------------------------------------------------------
# Cadence scheduling
# ---------------------------------------------------------------------------


#: Phrases longer than this without a cadence are considered
#: structurally weak. Baroque chorales generally land every 4 bars; 8 is
#: the loosest acceptable bound.
CADENCE_INTERVAL_MEASURES: int = 8

#: The final closing cadence lands in the last two bars of the piece.
FINAL_CADENCE_WINDOW: int = 2


# ---------------------------------------------------------------------------
# BaroqueGrammar
# ---------------------------------------------------------------------------


class BaroqueGrammar:
    """Stateless container of Baroque tonal constraints.

    All methods are pure; the object carries no corpus data. Pass the
    :class:`TransitionMatrix` in explicitly whenever a probability is
    needed — callers may have multiple matrices (one per corpus slice,
    for example) and we don't want the grammar to own any.
    """

    # ------------------------------------------------------------------
    # Transition validity
    # ------------------------------------------------------------------

    def is_valid_transition(
        self,
        from_rn: str,
        to_rn: str,
        key_mode: str,
    ) -> bool:
        """Return ``True`` iff ``(from_rn → to_rn)`` is not explicitly
        forbidden.

        The ``key_mode`` argument is accepted for future per-mode rule
        tables (minor-specific augmented-sixth behaviours, for example).
        For the current forbidden set it's unused, but keeping it in the
        signature avoids a breaking change later.
        """
        _ = key_mode  # reserved for future use
        return (from_rn, to_rn) not in FORBIDDEN_PROGRESSIONS

    def is_discouraged_transition(
        self,
        from_rn: str,
        to_rn: str,
        key_mode: str,
    ) -> bool:
        """Return ``True`` iff the transition is in the discouraged set."""
        _ = key_mode
        return (from_rn, to_rn) in DISCOURAGED_PROGRESSIONS

    def transition_weight(
        self,
        from_rn: str,
        to_rn: str,
        key_mode: str,
        matrix: "TransitionMatrix",
    ) -> float:
        """Combined weight for beam-search scoring.

        ::

            forbidden   → 0.0
            discouraged → matrix.probability(...) * 0.01
            valid       → matrix.probability(...)

        The multiplicative 0.01 penalty is an order-of-magnitude hit —
        large enough to suppress the discouraged edge except when it
        is the *only* high-probability candidate.
        """
        if not self.is_valid_transition(from_rn, to_rn, key_mode):
            return 0.0
        prob = matrix.probability(key_mode, from_rn, to_rn)
        if self.is_discouraged_transition(from_rn, to_rn, key_mode):
            return prob * 0.01
        return prob

    # ------------------------------------------------------------------
    # Cadence scheduling
    # ------------------------------------------------------------------

    def requires_cadence_at(self, measure: int, total_measures: int) -> bool:
        """True if a cadence is structurally required at ``measure``.

        Requirements:
          * Every ``CADENCE_INTERVAL_MEASURES``-th measure (8, 16, 24, …).
          * Each of the final ``FINAL_CADENCE_WINDOW`` measures — we
            accept either one, but at least one must carry a cadence.

        Measures are 1-indexed to match conventional musical numbering.
        """
        if measure <= 0 or total_measures <= 0:
            return False
        if measure > total_measures:
            return False

        # Final window: the last FINAL_CADENCE_WINDOW measures are
        # mandatory landing zones.
        if measure > total_measures - FINAL_CADENCE_WINDOW:
            return True

        # Periodic cadence: every 8 bars.
        if measure % CADENCE_INTERVAL_MEASURES == 0:
            return True

        return False

    def required_cadence_measures(self, total_measures: int) -> List[int]:
        """Enumerate every measure that demands a cadence.

        Useful to both the validator (for coverage scoring) and any
        planner that wants to schedule cadence targets in advance.
        """
        return [
            m for m in range(1, total_measures + 1)
            if self.requires_cadence_at(m, total_measures)
        ]

    # ------------------------------------------------------------------
    # Key-frame (start / end on tonic)
    # ------------------------------------------------------------------

    def validate_key_frame(
        self,
        rn_sequence: Iterable[str],
        key_mode: str,
    ) -> List[str]:
        """Verify that ``rn_sequence`` opens and closes on the tonic.

        Args:
            rn_sequence: Sequence of Roman-numeral label strings.
            key_mode: ``"major"`` or ``"minor"``. Determines whether the
                tonic we look for is ``I`` or ``i``.

        Returns:
            List of violation descriptions. Empty when the sequence is
            well-formed. Multiple violations (e.g. bad opening *and* bad
            closing) are reported separately.
        """
        rns = list(rn_sequence)
        if not rns:
            return ["empty sequence: cannot validate key frame"]

        tonics = _TONIC_LABELS_BY_MODE.get(key_mode)
        if tonics is None:
            return [f"unknown key_mode {key_mode!r}"]

        violations: List[str] = []
        if not _matches_tonic(rns[0], tonics):
            violations.append(
                f"opening chord {rns[0]!r} is not tonic "
                f"(expected one of {list(tonics)})"
            )
        if not _matches_tonic(rns[-1], tonics):
            violations.append(
                f"closing chord {rns[-1]!r} is not tonic "
                f"(expected one of {list(tonics)})"
            )
        return violations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_tonic(label: str, tonics: Tuple[str, ...]) -> bool:
    """Whether ``label`` is a plain tonic or a tonic inversion (I6, i6)."""
    for t in tonics:
        if label == t:
            return True
        # Inversions of the tonic are still tonic statements for
        # key-frame purposes (though a piece that ends on I6 is
        # pedagogically weaker, that's a soft complaint the validator
        # can raise separately).
        if label.startswith(t) and len(label) > len(t) and label[len(t):].isdigit():
            return True
    return False


__all__ = [
    "BaroqueGrammar",
    "CADENCE_INTERVAL_MEASURES",
    "DISCOURAGED_PROGRESSIONS",
    "FINAL_CADENCE_WINDOW",
    "FORBIDDEN_PROGRESSIONS",
]
