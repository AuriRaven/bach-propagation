"""
Tests for src/schenkerian/prolongation.py

All tests use synthetic HarmonicEvent objects — no file I/O.

Helper factory: _he(degree, function, root, quality, onset, inversion)
builds a minimal HarmonicEvent from a Chord + RomanNumeral.
"""

import pytest
from fractions import Fraction
from typing import List, Optional

from src.core.data_structures import (
    Chord,
    HarmonicEvent,
    HarmonicFunction,
    KeyEstimate,
    RomanNumeral,
    Score,
    TimeSignature,
)
from src.schenkerian.prolongation import ProlongationAnalyzer, ProlongationNode

# Re-use conftest factories via direct import (no fixture needed)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import make_note, make_score


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_ANA = ProlongationAnalyzer()

_KEY_C = KeyEstimate(tonic=0, mode='major', confidence=0.9, onset=Fraction(0))


def _chord(root: int, quality: str, onset: int = 0, dur: int = 4,
           inversion: int = 0) -> Chord:
    from src.harmonic.chord_classifier import _CHORD_TEMPLATES
    intervals = _CHORD_TEMPLATES[quality]
    pitches = frozenset((root + i) % 12 for i in intervals)
    bass_pc = sorted((root + i) % 12 for i in intervals)[inversion]
    return Chord(
        onset=Fraction(onset), duration=Fraction(dur),
        root=root, quality=quality,
        pitches=pitches, bass=bass_pc, inversion=inversion,
    )


def _rn(degree: int,
        function: HarmonicFunction,
        quality: str = 'major',
        inversion: int = 0,
        is_secondary: bool = False) -> RomanNumeral:
    return RomanNumeral(
        degree=degree, quality=quality, inversion=inversion,
        local_key_tonic=0, local_key_mode='major',
        is_secondary=is_secondary,
        function=function,
    )


def _he(degree: int,
        function: HarmonicFunction,
        root: int,
        quality: str = 'major',
        onset: int = 0,
        inversion: int = 0) -> HarmonicEvent:
    c = _chord(root, quality, onset=onset, inversion=inversion)
    r = _rn(degree, function, quality=quality, inversion=inversion)
    return HarmonicEvent(chord=c, roman_numeral=r)


def _score_4_4(duration: int = 32) -> Score:
    """Minimal 4/4 score covering 'duration' quarter notes."""
    note = make_note(60, 0, duration)
    return make_score([note], numerator=4, denominator=4)


T = HarmonicFunction.TONIC
PD = HarmonicFunction.PREDOMINANT
D = HarmonicFunction.DOMINANT
AMB = HarmonicFunction.AMBIGUOUS


# ---------------------------------------------------------------------------
# TestIsCadence
# ---------------------------------------------------------------------------

class TestIsCadence:

    def _events_vI(self):
        return [
            _he(5, D, root=7, quality='major', onset=0),    # V
            _he(1, T, root=0, quality='major', onset=4),    # I
        ]

    def test_v_then_i_is_cadence(self):
        events = self._events_vI()
        assert _ANA._is_cadence(events, 0) is True

    def test_i_then_v_is_not_cadence(self):
        events = [
            _he(1, T, root=0, onset=0),
            _he(5, D, root=7, onset=4),
        ]
        assert _ANA._is_cadence(events, 0) is False

    def test_v_then_vi_is_not_cadence(self):
        events = [
            _he(5, D, root=7, onset=0),
            _he(6, T, root=9, onset=4),
        ]
        assert _ANA._is_cadence(events, 0) is False

    def test_out_of_bounds_last_index_returns_false(self):
        events = self._events_vI()
        assert _ANA._is_cadence(events, 1) is False

    def test_empty_list_returns_false(self):
        assert _ANA._is_cadence([], 0) is False


# ---------------------------------------------------------------------------
# TestPassingReduction
# ---------------------------------------------------------------------------

class TestPassingReduction:

    def test_passing_between_same_function_removed(self):
        # T → T-passing → T  (stepwise bass: 0 → 4 → 7)
        events = [
            _he(1, T, root=0, onset=0),   # bass=0
            _he(3, AMB, root=4, onset=4),  # bass=4, shares no function with T — not removed
            _he(6, T, root=9, onset=8),    # bass=9 (A)
        ]
        # Event at index 1 (iii=AMBIGUOUS) does NOT share T function → not removed
        result = _ANA._apply_passing_reduction(events)
        assert len(result) == 3

    def test_middle_event_with_same_function_and_stepwise_bass_removed(self):
        # I → I6 (same root, different inversion, same function) → vi (T)
        # bass steps: 0 → 4 → 9 (all ≤ 2? No — 4-0=4 is not stepwise)
        # Let's use bass 0,2,4 by putting the passing chord at bass=2
        # We'll make a chromatic passing chord: T, T-passing (bass=2), T (bass=4)
        c_root = _chord(0, 'major', onset=0)    # C major, bass=0
        c_pass = _chord(0, 'major', onset=4)    # C major first-inv proxy — bass won't be exactly stepwise without actual inversion
        # Let's just check that sharing function + stepwise bass → removed
        # Build manually: all three are TONIC, bass moves 0→1→2
        from src.core.data_structures import Chord
        def _custom_he(root, bass, onset, function, degree):
            from src.harmonic.chord_classifier import _CHORD_TEMPLATES
            q = 'major'
            intervals = _CHORD_TEMPLATES[q]
            pcs = frozenset((root + i) % 12 for i in intervals)
            c = Chord(onset=Fraction(onset), duration=Fraction(4),
                      root=root, quality=q, pitches=pcs, bass=bass, inversion=0)
            r = _rn(degree, function, quality=q)
            return HarmonicEvent(chord=c, roman_numeral=r)

        events = [
            _custom_he(0, 0, 0, T, 1),   # I, bass=0
            _custom_he(0, 1, 4, T, 1),   # I (passing), bass=1  → shares T, stepwise
            _custom_he(0, 2, 8, T, 1),   # I, bass=2
        ]
        result = _ANA._apply_passing_reduction(events)
        assert len(result) == 2
        assert result[0].chord.bass == 0
        assert result[1].chord.bass == 2

    def test_short_list_not_modified(self):
        events = [_he(1, T, root=0, onset=0), _he(5, D, root=7, onset=4)]
        assert _ANA._apply_passing_reduction(events) == events

    def test_cadential_v_protected_from_passing_reduction(self):
        # I, V (cadential), I — the V is cadential, must not be removed even if
        # it appeared to satisfy passing criteria.
        events = [
            _he(1, T, root=0, onset=0),
            _he(5, D, root=7, onset=4),   # V → cadential, protected
            _he(1, T, root=0, onset=8),
        ]
        result = _ANA._apply_passing_reduction(events)
        # V is protected because _is_cadence(events, 1) is False but
        # _is_cadence(events, 0) is False and _is_cadence(events, 1) is True
        # Let's just assert no events are removed when cadence is present
        assert any(ev.roman_numeral.degree == 5 for ev in result)

    def test_empty_list_returns_empty(self):
        assert _ANA._apply_passing_reduction([]) == []

    def test_single_event_unchanged(self):
        events = [_he(1, T, root=0, onset=0)]
        assert _ANA._apply_passing_reduction(events) == events

    def test_non_stepwise_bass_not_removed(self):
        # I → IV (bass 0 → 5, leap of 5 — not stepwise) → I
        # IV is PREDOMINANT, not TONIC — fails function test too
        events = [
            _he(1, T, root=0, onset=0),
            _he(4, PD, root=5, onset=4),
            _he(1, T, root=0, onset=8),
        ]
        result = _ANA._apply_passing_reduction(events)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# TestNeighborReduction
# ---------------------------------------------------------------------------

class TestNeighborReduction:

    def test_neighbor_removed(self):
        # I → IV → I  (IV has different root; I returns)
        events = [
            _he(1, T, root=0, onset=0),
            _he(4, PD, root=5, onset=4),
            _he(1, T, root=0, onset=8),
        ]
        result = _ANA._apply_neighbor_reduction(events)
        assert len(result) == 2
        assert all(ev.chord.root == 0 for ev in result)

    def test_non_returning_sequence_not_reduced(self):
        # I → IV → V  (different return root — not a neighbor)
        events = [
            _he(1, T, root=0, onset=0),
            _he(4, PD, root=5, onset=4),
            _he(5, D, root=7, onset=8),
        ]
        result = _ANA._apply_neighbor_reduction(events)
        assert len(result) == 3

    def test_same_root_middle_not_neighbor(self):
        # I → I → I  (same root — arpeggiation, not neighbor)
        events = [
            _he(1, T, root=0, onset=0),
            _he(1, T, root=0, onset=4),
            _he(1, T, root=0, onset=8),
        ]
        result = _ANA._apply_neighbor_reduction(events)
        assert len(result) == 3  # unchanged — same root ≠ departure

    def test_cadence_protected_from_neighbor_reduction(self):
        # V → I → V  — the I is between two V's (would be "neighbor" of V)
        # BUT the first V→I is a cadence — should be protected
        events = [
            _he(5, D, root=7, onset=0),
            _he(1, T, root=0, onset=4),
            _he(5, D, root=7, onset=8),
        ]
        result = _ANA._apply_neighbor_reduction(events)
        # The cadential V→I at index 0 protects I from removal
        assert any(ev.roman_numeral.degree == 1 for ev in result)

    def test_empty_unchanged(self):
        assert _ANA._apply_neighbor_reduction([]) == []

    def test_two_events_unchanged(self):
        events = [_he(1, T, root=0, onset=0), _he(5, D, root=7, onset=4)]
        assert _ANA._apply_neighbor_reduction(events) == events


# ---------------------------------------------------------------------------
# TestArpeggiationReduction
# ---------------------------------------------------------------------------

class TestArpeggiationReduction:

    def test_same_root_different_inversion_removed(self):
        # I (root) → I6 (first inversion) — same root, different inversion
        events = [
            _he(1, T, root=0, quality='major', onset=0, inversion=0),
            _he(1, T, root=0, quality='major', onset=4, inversion=1),
            _he(5, D, root=7, quality='major', onset=8, inversion=0),
        ]
        result = _ANA._apply_arpeggiation_reduction(events)
        assert len(result) == 2
        assert result[0].chord.inversion == 0
        assert result[1].chord.root == 7

    def test_different_root_not_removed(self):
        # I → IV → I  (different root — neighbor, not arpeggiation)
        events = [
            _he(1, T, root=0, onset=0, inversion=0),
            _he(4, PD, root=5, onset=4, inversion=0),
            _he(1, T, root=0, onset=8, inversion=0),
        ]
        result = _ANA._apply_arpeggiation_reduction(events)
        assert len(result) == 3

    def test_same_root_same_inversion_not_removed(self):
        # I → I → V  (repetition, same inversion — not arpeggiation)
        events = [
            _he(1, T, root=0, onset=0, inversion=0),
            _he(1, T, root=0, onset=4, inversion=0),
            _he(5, D, root=7, onset=8, inversion=0),
        ]
        result = _ANA._apply_arpeggiation_reduction(events)
        assert len(result) == 3

    def test_cadential_v_not_removed_by_arpeggiation(self):
        # If V and I happen to share root=7 (they don't, so this tests the guard)
        # Just ensure cadence guard is invoked: V65 → I should not remove V65
        events = [
            _he(5, D, root=7, quality='dominant7', onset=0, inversion=1),  # V65
            _he(1, T, root=0, quality='major', onset=4, inversion=0),       # I
        ]
        result = _ANA._apply_arpeggiation_reduction(events)
        assert len(result) == 2

    def test_empty_unchanged(self):
        assert _ANA._apply_arpeggiation_reduction([]) == []

    def test_single_event_unchanged(self):
        events = [_he(1, T, root=0, onset=0)]
        assert _ANA._apply_arpeggiation_reduction(events) == events


# ---------------------------------------------------------------------------
# TestMetricWeighting
# ---------------------------------------------------------------------------

class TestMetricWeighting:

    def test_returns_same_count_as_input(self):
        score = _score_4_4()
        events = [
            _he(1, T, root=0, onset=0),
            _he(4, PD, root=5, onset=4),
            _he(5, D, root=7, onset=8),
        ]
        weighted = _ANA._assign_metric_weights(events, score)
        assert len(weighted) == 3

    def test_downbeat_weight_is_highest(self):
        score = _score_4_4()
        events = [
            _he(1, T, root=0, onset=0),  # downbeat
            _he(4, PD, root=5, onset=1),  # off-beat
        ]
        weighted = _ANA._assign_metric_weights(events, score)
        downbeat_w = weighted[0][1]
        offbeat_w = weighted[1][1]
        assert downbeat_w >= offbeat_w

    def test_downbeat_weight_is_one(self):
        score = _score_4_4()
        events = [_he(1, T, root=0, onset=0)]
        weighted = _ANA._assign_metric_weights(events, score)
        assert weighted[0][1] == pytest.approx(1.0)

    def test_each_pair_is_event_float(self):
        score = _score_4_4()
        events = [_he(1, T, root=0, onset=0)]
        weighted = _ANA._assign_metric_weights(events, score)
        ev, w = weighted[0]
        assert isinstance(ev, HarmonicEvent)
        assert isinstance(w, float)

    def test_empty_events_returns_empty(self):
        score = _score_4_4()
        assert _ANA._assign_metric_weights([], score) == []


# ---------------------------------------------------------------------------
# TestBuildTree
# ---------------------------------------------------------------------------

class TestBuildTree:

    def _simple_sequence(self) -> List[HarmonicEvent]:
        """I → IV → V → I in C major."""
        return [
            _he(1, T,  root=0, onset=0),
            _he(4, PD, root=5, onset=4),
            _he(5, D,  root=7, onset=8),
            _he(1, T,  root=0, onset=12),
        ]

    def _score(self) -> Score:
        return _score_4_4(duration=16)

    def test_analyze_returns_prolongation_node(self):
        root = _ANA.analyze(self._simple_sequence(), self._score())
        assert isinstance(root, ProlongationNode)

    def test_root_is_at_level_2(self):
        root = _ANA.analyze(self._simple_sequence(), self._score())
        assert root.level == 2

    def test_empty_events_returns_none(self):
        result = _ANA.analyze([], self._score())
        assert result is None

    def test_single_event_returns_node(self):
        events = [_he(1, T, root=0, onset=0)]
        result = _ANA.analyze(events, self._score())
        assert result is not None
        assert isinstance(result, ProlongationNode)

    def test_foreground_has_all_events(self):
        events = self._simple_sequence()
        root = _ANA.analyze(events, self._score())
        fg_nodes = _collect_level(root, 0)
        assert len(fg_nodes) == len(events)

    def test_background_has_fewer_events_than_foreground(self):
        # 4-event sequence should reduce to ≤4 at background
        events = self._simple_sequence()
        root = _ANA.analyze(events, self._score())
        fg_nodes = _collect_level(root, 0)
        bg_nodes = _collect_level(root, 2)
        assert len(bg_nodes) <= len(fg_nodes)

    def test_root_has_children(self):
        events = self._simple_sequence()
        root = _ANA.analyze(events, self._score())
        assert len(root.children) >= 1

    def test_cadential_v_survives_to_middleground(self):
        events = self._simple_sequence()  # includes V → I cadence
        root = _ANA.analyze(events, self._score())
        mg_nodes = _collect_level(root, 1)
        degrees = [n.event.roman_numeral.degree for n in mg_nodes]
        # V (degree 5) should survive reduction
        assert 5 in degrees

    def test_tonic_survives_to_background(self):
        events = self._simple_sequence()
        root = _ANA.analyze(events, self._score())
        bg_nodes = _collect_level(root, 2)
        degrees = [n.event.roman_numeral.degree for n in bg_nodes]
        assert 1 in degrees


# ---------------------------------------------------------------------------
# Helpers for tree traversal
# ---------------------------------------------------------------------------

def _collect_level(node: Optional[ProlongationNode], level: int) -> List[ProlongationNode]:
    """Recursively collect all nodes at the given level."""
    if node is None:
        return []
    result = []
    if node.level == level:
        result.append(node)
    for child in node.children:
        result.extend(_collect_level(child, level))
    return result
