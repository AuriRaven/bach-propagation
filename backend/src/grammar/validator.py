"""
Post-generation quality scorer for Roman-numeral sequences.

Where :mod:`src.grammar.rules` is applied *during* generation to prune
the beam, :class:`GrammarValidator` is applied *after* generation to
grade a finished sequence. The report it returns drives:

  * UI badges for "tonally valid" vs. "needs work";
  * thesis comparison tables between model variants;
  * regression tests that assert generated chorales score above a
    chosen tonal-quality floor.

Scoring components
------------------

``tonal_score`` is a convex combination of:

  * **Forbidden-progression density** — the fraction of transitions
    that are *not* in ``FORBIDDEN_PROGRESSIONS``.
  * **Transition-probability geometric mean** — a soft measure of how
    "Bach-like" the sequence is on average. Mapped onto [0, 1] with a
    stretched sigmoid so a median-probability sequence sits around 0.5
    rather than the geometric-mean raw value.
  * **Cadence coverage** — the fraction of structurally required
    cadence slots that the sequence actually fills.
  * **Key-frame validity** — 1.0 if the sequence opens and closes on
    tonic, 0.0 otherwise. This is a heavy penalty: a chorale that
    doesn't end on I is always wrong.

Weights were chosen to keep ``is_valid`` aligned with "no *hard*
violations present" while still penalising soft drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from src.corpus.transition_matrix import TransitionMatrix
from src.grammar.rules import BaroqueGrammar


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GrammarViolation:
    """One grammar failure located at a specific sequence position.

    Attributes:
        position: Index in the input RN sequence (0-based). For the
            missing-cadence violation this is the *measure* number of
            the expected cadence, not a sequence index — see
            ``violation_type``.
        from_rn: The source Roman numeral involved, or "" for
            non-transition violations.
        to_rn: The target Roman numeral involved, or "" for
            non-transition violations.
        violation_type: One of ``"forbidden"``, ``"low_probability"``,
            ``"missing_cadence"``, ``"key_frame"``.
        severity: 1.0 = hard violation (invalidates the piece); values
            in (0, 1) are soft. Used as a weight when aggregating.
    """

    position: int
    from_rn: str
    to_rn: str
    violation_type: str
    severity: float


@dataclass
class GrammarReport:
    """Aggregate scorecard for a complete RN sequence."""

    is_valid: bool
    violations: List[GrammarViolation]
    tonal_score: float          # 0..1, 1.0 = fully valid
    avg_transition_prob: float  # geometric mean over valid transitions
    cadence_coverage: float     # fraction of required cadence slots filled

    def summary(self) -> str:
        """One-line human-readable summary for logs / UI tooltips."""
        return (
            f"valid={self.is_valid} "
            f"score={self.tonal_score:.3f} "
            f"avg_p={self.avg_transition_prob:.3f} "
            f"cad_cov={self.cadence_coverage:.2f} "
            f"violations={len(self.violations)}"
        )


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


#: Transition probabilities below this are flagged as
#: ``"low_probability"``. Tuned against the empirical Bach corpus: 1 %
#: of the Laplace-smoothed mass sits roughly at the long tail.
LOW_PROBABILITY_THRESHOLD: float = 0.01

#: Weights used when combining sub-scores into ``tonal_score``. Must
#: sum to 1.0.
_SCORE_WEIGHTS = {
    "forbidden": 0.40,
    "avg_prob":  0.20,
    "cadence":   0.20,
    "key_frame": 0.20,
}


# ---------------------------------------------------------------------------
# GrammarValidator
# ---------------------------------------------------------------------------


class GrammarValidator:
    """Score a complete RN sequence against all Baroque grammar rules.

    The validator is stateless; everything it needs is passed in on
    each :meth:`validate` call. Callers supply:

      * the RN sequence under test,
      * its ``key_mode`` ("major" or "minor"),
      * a populated :class:`TransitionMatrix` for probability lookups,
      * a :class:`BaroqueGrammar` for forbidden / cadence rules,
      * optional ``cadence_measures`` and ``total_measures`` if the
        sequence has a known metric framing.

    Passing no cadence information skips the cadence-coverage
    component entirely (coverage = 1.0) — useful when the caller is
    scoring raw harmonic progressions without a metric plan.
    """

    def validate(
        self,
        rn_sequence: Sequence[str],
        key_mode: str,
        matrix: TransitionMatrix,
        grammar: BaroqueGrammar,
        *,
        cadence_measures: Optional[Iterable[int]] = None,
        total_measures: Optional[int] = None,
    ) -> GrammarReport:
        rns = list(rn_sequence)
        violations: List[GrammarViolation] = []

        # 1. Transition-level checks -----------------------------------------
        n_forbidden, avg_prob, transition_violations = self._check_transitions(
            rns, key_mode, matrix, grammar
        )
        violations.extend(transition_violations)

        # 2. Key-frame (open/close on tonic) --------------------------------
        key_violations = grammar.validate_key_frame(rns, key_mode)
        for msg in key_violations:
            violations.append(GrammarViolation(
                position=0 if "opening" in msg else max(len(rns) - 1, 0),
                from_rn="",
                to_rn="",
                violation_type="key_frame",
                severity=1.0,
            ))
        key_frame_score = 1.0 if not key_violations else 0.0

        # 3. Cadence coverage -----------------------------------------------
        cadence_coverage, cadence_violations = self._check_cadences(
            cadence_measures, total_measures, grammar
        )
        violations.extend(cadence_violations)

        # 4. Aggregate -------------------------------------------------------
        forbidden_score = self._forbidden_score(rns, n_forbidden)
        avg_prob_score = self._probability_score(avg_prob)
        tonal_score = (
            _SCORE_WEIGHTS["forbidden"] * forbidden_score
            + _SCORE_WEIGHTS["avg_prob"] * avg_prob_score
            + _SCORE_WEIGHTS["cadence"] * cadence_coverage
            + _SCORE_WEIGHTS["key_frame"] * key_frame_score
        )
        # is_valid := no hard violations.
        is_valid = all(v.severity < 1.0 for v in violations)

        return GrammarReport(
            is_valid=is_valid,
            violations=violations,
            tonal_score=round(tonal_score, 6),
            avg_transition_prob=round(avg_prob, 6),
            cadence_coverage=round(cadence_coverage, 6),
        )

    # ------------------------------------------------------------------
    # Transition pass
    # ------------------------------------------------------------------

    def _check_transitions(
        self,
        rns: Sequence[str],
        key_mode: str,
        matrix: TransitionMatrix,
        grammar: BaroqueGrammar,
    ) -> Tuple[int, float, List[GrammarViolation]]:
        """Walk the sequence pair-by-pair, logging forbidden / low-
        probability transitions and accumulating the geometric mean of
        valid transition probabilities."""
        n_forbidden = 0
        log_sum = 0.0
        n_valid = 0
        violations: List[GrammarViolation] = []

        for i in range(len(rns) - 1):
            a, b = rns[i], rns[i + 1]

            if not grammar.is_valid_transition(a, b, key_mode):
                n_forbidden += 1
                violations.append(GrammarViolation(
                    position=i,
                    from_rn=a,
                    to_rn=b,
                    violation_type="forbidden",
                    severity=1.0,
                ))
                continue

            prob = matrix.probability(key_mode, a, b)
            if prob < LOW_PROBABILITY_THRESHOLD:
                violations.append(GrammarViolation(
                    position=i,
                    from_rn=a,
                    to_rn=b,
                    violation_type="low_probability",
                    severity=_soft_severity(prob),
                ))
            if prob > 0:
                log_sum += math.log(prob)
                n_valid += 1

        avg_prob = math.exp(log_sum / n_valid) if n_valid else 0.0
        return n_forbidden, avg_prob, violations

    # ------------------------------------------------------------------
    # Cadence coverage
    # ------------------------------------------------------------------

    def _check_cadences(
        self,
        cadence_measures: Optional[Iterable[int]],
        total_measures: Optional[int],
        grammar: BaroqueGrammar,
    ) -> Tuple[float, List[GrammarViolation]]:
        """Return (coverage, violations). Coverage is the fraction of
        required measures that appear in ``cadence_measures``.

        If the caller didn't supply metric information, we assume the
        caller is scoring a purely harmonic progression with no formal
        plan — coverage defaults to 1.0 and no cadence violations are
        raised.
        """
        if total_measures is None or cadence_measures is None:
            return 1.0, []

        required = grammar.required_cadence_measures(int(total_measures))
        if not required:
            return 1.0, []

        supplied = {int(m) for m in cadence_measures}
        missing = [m for m in required if m not in supplied]

        violations: List[GrammarViolation] = [
            GrammarViolation(
                position=m,                     # measure number, not index
                from_rn="",
                to_rn="",
                violation_type="missing_cadence",
                severity=1.0,
            )
            for m in missing
        ]

        coverage = 1.0 - (len(missing) / len(required))
        return coverage, violations

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _forbidden_score(self, rns: Sequence[str], n_forbidden: int) -> float:
        """Fraction of transitions that are **not** forbidden."""
        n_transitions = max(len(rns) - 1, 0)
        if n_transitions == 0:
            return 1.0
        return 1.0 - (n_forbidden / n_transitions)

    def _probability_score(self, avg_prob: float) -> float:
        """Map a geometric-mean probability onto [0, 1].

        Smoothed corpus probabilities sit in a narrow band well below
        1.0, so a linear mapping would make even good sequences look
        bad. A logistic curve centred near the empirical median maps
        the realistic range onto a reasonable grade.
        """
        if avg_prob <= 0:
            return 0.0
        # log-space midpoint — tuned to 0.1 so that a row-typical
        # probability of ~0.1 produces a ~0.5 score.
        midpoint = math.log(0.1)
        steepness = 1.5
        return 1.0 / (1.0 + math.exp(-steepness * (math.log(avg_prob) - midpoint)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _soft_severity(prob: float) -> float:
    """Severity in (0, 1) for a low-probability transition.

    A probability well below the threshold is treated as more severe
    (closer to 1.0) but never exactly 1.0 — low probability on its own
    is a *soft* violation.
    """
    if prob <= 0:
        return 0.99
    return max(0.0, min(0.99, 1.0 - prob / LOW_PROBABILITY_THRESHOLD))


__all__ = [
    "GrammarReport",
    "GrammarValidator",
    "GrammarViolation",
    "LOW_PROBABILITY_THRESHOLD",
]
