"""
Empirical Roman-numeral transition probabilities extracted from the
Stage 4 analysis cache.

A :class:`TransitionMatrix` answers one question: *given that the current
chord is Roman numeral ``X`` in a piece in mode ``M``, what is the
probability the next chord is ``Y``?*  Everything we do downstream —
beam-search scoring, grammar validation, thesis commentary — bottoms out
in these counts.

Two independent matrices are kept, one per mode, because Bach's harmonic
language in minor (with its iv → V phrygian, augmented sixths, melodic
minor V → i, …) is structurally different from his major-mode
progressions.  Merging them would blur both.

Smoothing
---------
Raw counts contain zeros for plausible-but-unobserved transitions
(``I → VI``, ``iv → V7``, etc.). A beam decoder that multiplies
probabilities will collapse the moment it encounters one of those, which
is exactly wrong: the model should be *discouraged* from taking that
edge, not mathematically forbidden. Laplace (add-1) smoothing over the
union of labels observed in each mode gives every pair a non-zero
probability without distorting the common cases.

JSON format (human-readable, thesis-friendly)
---------------------------------------------
::

    {
      "schema_version": "1.0.0",
      "modes": {
        "major": {
          "labels": ["I", "V", "IV", ...],            # sorted rows
          "transitions": {
            "I":  [["V", 0.412], ["IV", 0.235], ...], # sorted desc
            "V":  [["I", 0.506], ["vi", 0.184], ...],
            ...
          },
          "row_counts": {"I": 1243, "V": 982, ...}    # raw totals (pre-smoothing)
        },
        "minor": { ... }
      }
    }
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

#: Bumped when the on-disk format changes.
MATRIX_SCHEMA_VERSION: str = "1.0.0"

_VALID_MODES: Tuple[str, ...] = ("major", "minor")


# ---------------------------------------------------------------------------
# Internal per-mode table
# ---------------------------------------------------------------------------


@dataclass
class _ModeTable:
    """Counts and derived probabilities for one key mode.

    Attributes:
        row_counts: Raw observation count for each ``from_rn`` label.
        pair_counts: Raw observation count for each ``(from_rn, to_rn)``
            ordered pair.
        labels: Union of every label seen as a ``from`` or a ``to`` — the
            smoothing alphabet.
    """

    row_counts: Counter = field(default_factory=Counter)
    pair_counts: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    labels: Set[str] = field(default_factory=set)

    def add(self, from_rn: str, to_rn: str) -> None:
        self.row_counts[from_rn] += 1
        self.pair_counts[from_rn][to_rn] += 1
        self.labels.add(from_rn)
        self.labels.add(to_rn)

    # ------------------------------------------------------------------
    # Laplace-smoothed probability
    # ------------------------------------------------------------------

    def probability(self, from_rn: str, to_rn: str) -> float:
        """Return ``P(to_rn | from_rn)`` with add-1 smoothing.

        If ``from_rn`` has never been observed we return a uniform
        distribution over the mode's label alphabet — the most uninformed
        prior we can write without inventing structure.
        """
        alphabet = self.labels | {from_rn, to_rn}
        V = len(alphabet)
        if V == 0:
            # Completely empty table; fall back to a degenerate uniform.
            return 0.0
        numerator = self.pair_counts.get(from_rn, Counter()).get(to_rn, 0) + 1
        denominator = self.row_counts.get(from_rn, 0) + V
        return numerator / denominator

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def sorted_labels(self) -> List[str]:
        """Labels sorted by descending row count, alphabetical tiebreak."""
        return sorted(
            self.labels,
            key=lambda lbl: (-self.row_counts.get(lbl, 0), lbl),
        )

    def successor_rows(self) -> Dict[str, List[Tuple[str, float]]]:
        """``from_rn → [(to_rn, P), …]`` sorted by descending probability."""
        rows: Dict[str, List[Tuple[str, float]]] = {}
        for from_rn in self.sorted_labels():
            row = [
                (to_rn, self.probability(from_rn, to_rn))
                for to_rn in self.labels
            ]
            row.sort(key=lambda pair: (-pair[1], pair[0]))
            rows[from_rn] = row
        return rows


# ---------------------------------------------------------------------------
# TransitionMatrix
# ---------------------------------------------------------------------------


class TransitionMatrix:
    """Empirical RN transition probabilities segmented by key mode.

    ``matrix[mode][from_rn][to_rn] = probability``.

    The matrix is **mutable while being built** (:meth:`build_from_cache`
    populates it) then frozen by convention. All probability queries go
    through :meth:`probability`, which applies Laplace smoothing so zero
    probabilities never leak into the beam search.
    """

    def __init__(self) -> None:
        self._tables: Dict[str, _ModeTable] = {m: _ModeTable() for m in _VALID_MODES}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_from_cache(self, cache_dir: str | Path) -> None:
        """Walk ``cache_dir`` and count every adjacent RN pair.

        Each Stage-4 analysis file contributes adjacent pairs of
        ``roman_numerals``. The ``local_key_mode`` of the *source* RN
        decides which mode table the pair belongs to; if it's missing we
        fall back to the analysis's global key mode. Modulations across
        a transition (different local modes on either side) are assigned
        to the source's mode — they are real chord-to-chord observations
        regardless of tonicisation.
        """
        path = Path(cache_dir)
        if not path.exists():
            raise FileNotFoundError(cache_dir)

        for analysis in _iter_analyses(path):
            self._ingest(analysis)

    def build_from_analyses(self, analyses: Iterable[Mapping[str, object]]) -> None:
        """In-memory construction path — used by tests and callers that
        have already loaded analyses into RAM."""
        for analysis in analyses:
            self._ingest(analysis)

    def _ingest(self, analysis: Mapping[str, object]) -> None:
        rns = list(analysis.get("roman_numerals", []) or [])  # type: ignore[arg-type]
        if len(rns) < 2:
            return

        global_mode = _extract_global_mode(analysis)

        for a, b in zip(rns, rns[1:]):
            from_label = str(a.get("label"))
            to_label = str(b.get("label"))
            if not from_label or not to_label:
                continue
            mode = _extract_rn_mode(a) or global_mode
            if mode not in self._tables:
                continue
            self._tables[mode].add(from_label, to_label)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def probability(self, mode: str, from_rn: str, to_rn: str) -> float:
        """Return ``P(to_rn | from_rn)`` in ``mode``. Never zero."""
        self._check_mode(mode)
        return self._tables[mode].probability(from_rn, to_rn)

    def top_k(
        self,
        mode: str,
        from_rn: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Top-``k`` most likely successors of ``from_rn`` in ``mode``.

        Returns ``[(label, probability), …]`` sorted by descending
        probability. If no observations exist at all, returns an empty
        list (rather than fabricating labels).
        """
        self._check_mode(mode)
        table = self._tables[mode]
        if not table.labels:
            return []
        scored = [
            (label, table.probability(from_rn, label))
            for label in table.labels
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def labels(self, mode: str) -> List[str]:
        """All RN labels observed in ``mode``."""
        self._check_mode(mode)
        return self._tables[mode].sorted_labels()

    def row_count(self, mode: str, from_rn: str) -> int:
        """Raw observation count for ``from_rn`` in ``mode``."""
        self._check_mode(mode)
        return self._tables[mode].row_counts.get(from_rn, 0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        """Produce the human-readable JSON payload."""
        payload: Dict[str, object] = {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "modes": {},
        }
        for mode, table in self._tables.items():
            rows = table.successor_rows()
            payload["modes"][mode] = {  # type: ignore[index]
                "labels": table.sorted_labels(),
                "transitions": {
                    from_rn: [[to_rn, round(p, 6)] for to_rn, p in row]
                    for from_rn, row in rows.items()
                },
                "row_counts": {
                    lbl: int(table.row_counts.get(lbl, 0))
                    for lbl in table.sorted_labels()
                },
            }
        return payload

    def to_json(self, path: str | Path) -> None:
        """Write the matrix to a JSON file — thesis deliverable."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "TransitionMatrix":
        """Load a matrix previously written by :meth:`to_json`.

        Only the raw counts are rebuilt from the file; the object's
        smoothed probabilities are recomputed from those counts on every
        query.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("schema_version")
        if version != MATRIX_SCHEMA_VERSION:
            raise ValueError(
                f"transition_matrix {path} has schema_version={version!r}, "
                f"expected {MATRIX_SCHEMA_VERSION!r}"
            )
        tm = cls()
        modes = data.get("modes", {})
        for mode, mode_payload in modes.items():
            if mode not in tm._tables:
                continue
            table = tm._tables[mode]
            # Rebuild raw counts from the serialised probabilities + row
            # totals. We can't recover integer pair-counts exactly, but
            # we preserve the (from_rn, to_rn, P) signal needed for
            # queries by back-computing a count that reproduces the
            # stored probability when Laplace-smoothed.
            row_counts: Mapping[str, int] = mode_payload.get("row_counts", {})
            labels: Sequence[str] = mode_payload.get("labels", [])
            table.labels.update(labels)
            for lbl, count in row_counts.items():
                table.row_counts[lbl] = int(count)
                table.labels.add(lbl)
            transitions: Mapping[str, Sequence[Sequence[object]]] = mode_payload.get(
                "transitions", {}
            )
            V = len(table.labels)
            for from_rn, row in transitions.items():
                from_total = table.row_counts.get(from_rn, 0)
                for pair in row:
                    to_rn = str(pair[0])
                    probability = float(pair[1])
                    # Invert the Laplace formula: count = P*(N+V) - 1.
                    recovered = round(probability * (from_total + V)) - 1
                    if recovered > 0:
                        table.pair_counts[from_rn][to_rn] = recovered
                    table.labels.add(to_rn)
        return tm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_mode(self, mode: str) -> None:
        if mode not in self._tables:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {_VALID_MODES}"
            )


# ---------------------------------------------------------------------------
# File / analysis helpers
# ---------------------------------------------------------------------------


def _iter_analyses(cache_dir: Path) -> Iterable[Mapping[str, object]]:
    """Yield every Stage-4 per-score analysis under ``cache_dir``.

    ``manifest.json`` and anything lacking ``roman_numerals`` is skipped.
    Parse errors are silently ignored — a partial cache shouldn't block
    the whole build.
    """
    for path in sorted(cache_dir.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "roman_numerals" in data:
            yield data


def _extract_rn_mode(rn: Mapping[str, object]) -> Optional[str]:
    """Return the RN's local-key mode if present and recognised."""
    mode = rn.get("local_key_mode")
    if isinstance(mode, str) and mode in _VALID_MODES:
        return mode
    return None


def _extract_global_mode(analysis: Mapping[str, object]) -> Optional[str]:
    """Fallback: the analysis's global-key mode."""
    gk = analysis.get("global_key")
    if isinstance(gk, Mapping):
        mode = gk.get("mode")
        if isinstance(mode, str) and mode in _VALID_MODES:
            return mode
    return None


__all__ = [
    "MATRIX_SCHEMA_VERSION",
    "TransitionMatrix",
]
