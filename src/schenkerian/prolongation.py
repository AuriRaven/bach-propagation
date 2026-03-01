"""
Schenkerian prolongation analysis.

Implements a three-level structural reduction of a harmonic event sequence:

  Level 0 — Foreground:   all events (full chord sequence)
  Level 1 — Middleground: after passing, neighbor, and arpeggiation reduction
  Level 2 — Background:   Ursatz skeleton (structural events only)

Sub-stage A: Stateless reduction rules (pure functions).
Sub-stage B: Metric weighting via MeterAnalyzer.build_metric_grid().
Sub-stage C: ProlongationNode tree construction.

All methods are stateless — instantiate ProlongationAnalyzer once and reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from src.core.data_structures import (
    HarmonicEvent,
    HarmonicFunction,
    Score,
)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ProlongationNode:
    """
    One node in the prolongation tree.

    Attributes:
        event:    The HarmonicEvent this node represents.
        level:    Structural level (0=foreground, 1=middleground, 2=background).
        children: Events at the next-lower level that this node prolongs over.
        relation: How this node relates to its parent
                  ('passing' | 'neighbor' | 'arpeggiation' | 'structural').
    """
    event: HarmonicEvent
    level: int
    children: List[ProlongationNode] = field(default_factory=list)
    relation: str = 'structural'


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class ProlongationAnalyzer:
    """
    Build a 3-level prolongation tree from a list of HarmonicEvents.

    Usage::

        from src.schenkerian import ProlongationAnalyzer
        analyzer = ProlongationAnalyzer()
        root = analyzer.analyze(harmonic_events, score)
        # root is a ProlongationNode at level=2 (background)
        # root.children → middleground nodes
        # each middleground node.children → foreground nodes
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        events: List[HarmonicEvent],
        score: Score,
    ) -> Optional[ProlongationNode]:
        """
        Build the 3-level prolongation tree.

        Returns None if events is empty. Returns a single structural node
        if events has only one element.

        Args:
            events: Ordered list of HarmonicEvents (onset-sorted).
            score:  Source score, used to look up time signatures for metric
                    weighting.

        Returns:
            Root ProlongationNode at level=2 (background).
        """
        if not events:
            return None

        # Sub-stage B: assign metric weight to each event
        weighted = self._assign_metric_weights(events, score)

        # Level 0 — foreground: all events
        fg_nodes = self._build_level(events, level=0)

        # Level 1 — middleground: pass + neighbor + arpeggiation reduction
        mg_events = self._reduce(weighted, pass_num=1)
        mg_nodes = self._build_level(mg_events, level=1)

        # Level 2 — background: neighbor + arpeggiation only
        #           (passing already removed; structural events retained)
        mg_ids = {id(ev) for ev in mg_events}
        mg_weighted = [(ev, w) for ev, w in weighted if id(ev) in mg_ids]
        bg_events = self._reduce(mg_weighted, pass_num=2)
        bg_nodes = self._build_level(bg_events, level=2)

        return self._link_levels(fg_nodes, mg_nodes, bg_nodes)

    # ------------------------------------------------------------------
    # Sub-stage A: Reduction rules
    # ------------------------------------------------------------------

    def _is_cadence(
        self,
        events: List[HarmonicEvent],
        i: int,
    ) -> bool:
        """
        Return True if events[i] is the dominant half of an authentic cadence
        (V → I pair). Cadential pairs are never removed by any reduction rule.

        Checks:
        - events[i].roman_numeral.degree == 5 (dominant)
        - events[i+1].roman_numeral.degree == 1 (tonic)
        """
        if i < 0 or i >= len(events) - 1:
            return False
        curr = events[i].roman_numeral
        nxt = events[i + 1].roman_numeral
        return curr.degree == 5 and nxt.degree == 1

    def _apply_passing_reduction(
        self,
        events: List[HarmonicEvent],
    ) -> List[HarmonicEvent]:
        """
        Remove an event X between A and B if:
        1. X shares harmonic function with A or B, AND
        2. The bass moves stepwise A→X→B (abs difference ≤ 2 semitones), AND
        3. Neither the A→X pair nor the X→B pair is a cadential V→I.

        Stepwise bass motion is measured modulo-12 to handle wrap-around.
        """
        if len(events) < 3:
            return list(events)

        keep = [True] * len(events)
        for i in range(1, len(events) - 1):
            if not keep[i]:
                continue
            prev_idx = _last_kept(keep, i)
            next_idx = _next_kept(keep, i, len(events))
            if prev_idx is None or next_idx is None:
                continue

            prev_ev = events[prev_idx]
            curr_ev = events[i]
            next_ev = events[next_idx]

            # Cadential guard
            if self._is_cadence(events, prev_idx) or self._is_cadence(events, i):
                continue

            curr_fn = curr_ev.roman_numeral.function
            prev_fn = prev_ev.roman_numeral.function
            next_fn = next_ev.roman_numeral.function

            shares_function = (
                curr_fn is not None
                and (curr_fn == prev_fn or curr_fn == next_fn)
            )
            if not shares_function:
                continue

            # Stepwise bass check
            a_bass = prev_ev.chord.bass
            x_bass = curr_ev.chord.bass
            b_bass = next_ev.chord.bass
            step_ax = min(abs(x_bass - a_bass), 12 - abs(x_bass - a_bass))
            step_xb = min(abs(b_bass - x_bass), 12 - abs(b_bass - x_bass))
            if step_ax <= 2 and step_xb <= 2:
                keep[i] = False

        return [ev for ev, k in zip(events, keep) if k]

    def _apply_neighbor_reduction(
        self,
        events: List[HarmonicEvent],
    ) -> List[HarmonicEvent]:
        """
        Remove event X between A and B if:
        1. A and B share the same chord root, AND
        2. X has a different chord root than A (it departed and returned), AND
        3. No cadential V→I spans the A→X or X→B pair.
        """
        if len(events) < 3:
            return list(events)

        keep = [True] * len(events)
        for i in range(1, len(events) - 1):
            if not keep[i]:
                continue
            prev_idx = _last_kept(keep, i)
            next_idx = _next_kept(keep, i, len(events))
            if prev_idx is None or next_idx is None:
                continue

            prev_ev = events[prev_idx]
            curr_ev = events[i]
            next_ev = events[next_idx]

            if self._is_cadence(events, prev_idx) or self._is_cadence(events, i):
                continue

            same_frame = prev_ev.chord.root == next_ev.chord.root
            departed = curr_ev.chord.root != prev_ev.chord.root
            if same_frame and departed:
                keep[i] = False

        return [ev for ev, k in zip(events, keep) if k]

    def _apply_arpeggiation_reduction(
        self,
        events: List[HarmonicEvent],
    ) -> List[HarmonicEvent]:
        """
        Remove event X between A and B if:
        1. X shares the same chord root as A (arpeggiation of the same harmony), AND
        2. X has a different inversion than A (it moved within the chord), AND
        3. No cadential V→I spans the A→X pair.
        """
        if len(events) < 2:
            return list(events)

        keep = [True] * len(events)
        for i in range(1, len(events)):
            if not keep[i]:
                continue
            prev_idx = _last_kept(keep, i)
            if prev_idx is None:
                continue

            prev_ev = events[prev_idx]
            curr_ev = events[i]

            if self._is_cadence(events, prev_idx):
                continue

            same_root = curr_ev.chord.root == prev_ev.chord.root
            diff_inv = curr_ev.chord.inversion != prev_ev.chord.inversion
            if same_root and diff_inv:
                keep[i] = False

        return [ev for ev, k in zip(events, keep) if k]

    # ------------------------------------------------------------------
    # Sub-stage B: Metric weighting
    # ------------------------------------------------------------------

    def _assign_metric_weights(
        self,
        events: List[HarmonicEvent],
        score: Score,
    ) -> List[Tuple[HarmonicEvent, float]]:
        """
        Return (event, metric_weight) pairs.

        Weight is derived from MeterAnalyzer.build_metric_grid() using the
        time signature in effect at each event's onset. Beat position is
        computed as onset modulo measure length.

        Fallback weight = 0.1 if no grid entry matches.
        """
        from src.temporal.meter_analyzer import MeterAnalyzer

        analyzer = MeterAnalyzer()
        result: List[Tuple[HarmonicEvent, float]] = []

        for ev in events:
            ts = score.get_time_signature_at(ev.onset)
            grid = analyzer.build_metric_grid(ts)
            measure_len = Fraction(ts.numerator * 4, ts.denominator)
            beat_pos = ev.onset % measure_len
            weight = grid.weight_at(beat_pos)
            result.append((ev, weight))

        return result

    # ------------------------------------------------------------------
    # Sub-stage C: Tree construction
    # ------------------------------------------------------------------

    def _reduce(
        self,
        weighted: List[Tuple[HarmonicEvent, float]],
        pass_num: int,
    ) -> List[HarmonicEvent]:
        """
        Apply reduction rules in order and return the surviving event list.

        pass_num=1 (foreground → middleground): passing + neighbor + arpeggiation
        pass_num=2 (middleground → background): neighbor + arpeggiation only
                   (structural events survive; passing already removed)

        Metrically strong events (weight ≥ 0.8) are never removed.
        """
        events = [ev for ev, _ in weighted]
        weights: Dict[int, float] = {
            id(ev): w for ev, w in weighted
        }

        # Protect metrically strong events from removal by patching keep-logic
        # We do this by cloning events and running rules, then restoring strong events.
        if pass_num == 1:
            events = self._apply_passing_reduction(events)
        events = self._apply_neighbor_reduction(events)
        events = self._apply_arpeggiation_reduction(events)

        # Re-insert any metrically strong events that were removed
        original_events = [ev for ev, _ in weighted]
        strong = {id(ev) for ev, w in weighted if w >= 0.8}
        surviving_ids = {id(ev) for ev in events}
        missing_strong = [
            ev for ev in original_events
            if id(ev) in strong and id(ev) not in surviving_ids
        ]
        if missing_strong:
            events = sorted(
                events + missing_strong,
                key=lambda e: e.onset,
            )

        return events

    def _build_level(
        self,
        events: List[HarmonicEvent],
        level: int,
    ) -> List[ProlongationNode]:
        """Wrap each event in a ProlongationNode at the given level."""
        return [ProlongationNode(event=ev, level=level) for ev in events]

    def _link_levels(
        self,
        fg: List[ProlongationNode],
        mg: List[ProlongationNode],
        bg: List[ProlongationNode],
    ) -> ProlongationNode:
        """
        Build the tree by assigning children across levels.

        Strategy:
        - Each middleground node adopts the foreground nodes whose onset falls
          in its span [mg_start, next_mg_start).
        - The first background node (earliest onset) is the root. It adopts
          ALL remaining background nodes as direct structural children, then
          assigns each background node its own middleground children.
        - This ensures all nodes are reachable from root, even when every
          event survives to background (e.g. all-downbeat sequences).
        """
        # Degenerate cases
        if not bg:
            if not mg:
                return fg[0] if fg else None
            root_mg = mg[0]
            for fg_node in fg:
                root_mg.children.append(fg_node)
            return root_mg

        # Assign foreground → middleground (span-based)
        mg_sorted = sorted(mg, key=lambda n: n.event.onset)
        for idx, mg_node in enumerate(mg_sorted):
            start = mg_node.event.onset
            end = (mg_sorted[idx + 1].event.onset
                   if idx + 1 < len(mg_sorted) else Fraction(10 ** 9))
            for fg_node in fg:
                if start <= fg_node.event.onset < end:
                    if fg_node.event.onset == start:
                        fg_node.relation = 'structural'
                    mg_node.children.append(fg_node)

        # Assign middleground → background (span-based)
        bg_sorted = sorted(bg, key=lambda n: n.event.onset)
        for idx, bg_node in enumerate(bg_sorted):
            start = bg_node.event.onset
            end = (bg_sorted[idx + 1].event.onset
                   if idx + 1 < len(bg_sorted) else Fraction(10 ** 9))
            for mg_node in mg:
                if start <= mg_node.event.onset < end:
                    if mg_node.event.onset == start:
                        mg_node.relation = 'structural'
                    bg_node.children.append(mg_node)

        # Chain background nodes: root adopts all subsequent bg nodes
        # so every node in the tree is reachable from the root.
        root = bg_sorted[0]
        root.relation = 'structural'
        for subsequent_bg in bg_sorted[1:]:
            subsequent_bg.relation = 'structural'
            root.children.append(subsequent_bg)

        return root


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_kept(keep: List[bool], i: int) -> Optional[int]:
    """Return the index of the last True entry before i, or None."""
    for j in range(i - 1, -1, -1):
        if keep[j]:
            return j
    return None


def _next_kept(keep: List[bool], i: int, n: int) -> Optional[int]:
    """Return the index of the next True entry after i, or None."""
    for j in range(i + 1, n):
        if keep[j]:
            return j
    return None
