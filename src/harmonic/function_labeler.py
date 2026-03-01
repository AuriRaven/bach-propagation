"""
Harmonic function labeling with context-sensitive overrides.

Assigns T / PD / D / AMBIGUOUS function labels to RomanNumeral objects.
A two-pass algorithm handles both base lookups and sequential context rules.
"""

import dataclasses
from typing import Dict, List, Tuple

from src.core.data_structures import HarmonicFunction, RomanNumeral

# ---------------------------------------------------------------------------
# Base function by scale degree (mode-independent)
# ---------------------------------------------------------------------------
_BASE_FUNCTION: Dict[int, HarmonicFunction] = {
    1: HarmonicFunction.TONIC,
    2: HarmonicFunction.PREDOMINANT,
    3: HarmonicFunction.AMBIGUOUS,
    4: HarmonicFunction.PREDOMINANT,
    5: HarmonicFunction.DOMINANT,
    6: HarmonicFunction.TONIC,
    7: HarmonicFunction.DOMINANT,
}

# ---------------------------------------------------------------------------
# Context overrides: (prev_degree, curr_degree) → new function for PREV
# "vi preceding ii or IV" → relabel vi as PREDOMINANT
# ---------------------------------------------------------------------------
_CONTEXT_OVERRIDES: Dict[Tuple[int, int], HarmonicFunction] = {
    (6, 2): HarmonicFunction.PREDOMINANT,
    (6, 4): HarmonicFunction.PREDOMINANT,
}


class FunctionLabeler:
    """
    Assign and refine harmonic function labels on RomanNumeral objects.

    All public methods return new RomanNumeral instances — input objects are
    never mutated (frozen dataclass + dataclasses.replace pattern).
    """

    def label(self, rn: RomanNumeral) -> RomanNumeral:
        """
        Assign a harmonic function to a single RomanNumeral without context.

        Rules (in priority order):
          1. If rn.function is already set → return rn unchanged.
          2. If rn.is_secondary is True → function = DOMINANT.
          3. Otherwise → look up _BASE_FUNCTION[rn.degree].

        Returns:
            Same object if already labelled; new RomanNumeral otherwise.
        """
        if rn.function is not None:
            return rn

        if rn.is_secondary:
            func = HarmonicFunction.DOMINANT
        else:
            func = _BASE_FUNCTION.get(rn.degree, HarmonicFunction.AMBIGUOUS)

        return dataclasses.replace(rn, function=func)

    def label_sequence(self, rns: List[RomanNumeral]) -> List[RomanNumeral]:
        """
        Label an entire sequence and apply context-sensitive overrides.

        Pass 1: apply self.label(rn) to each element.
        Pass 2: iterate from index 1; if (prev.degree, curr.degree) is in
                _CONTEXT_OVERRIDES, relabel result[i-1] with the override
                function.

        Context override direction: the *previous* chord (e.g. vi) has its
        function changed when followed by a specific next chord (ii or IV).
        The next chord's function is not altered.

        Does not mutate the input list or any input RomanNumeral.

        Returns:
            New list of RomanNumeral objects.
        """
        if not rns:
            return []

        # Pass 1 — base labeling
        result: List[RomanNumeral] = [self.label(rn) for rn in rns]

        # Pass 2 — context overrides (update result[i-1])
        for i in range(1, len(result)):
            prev = result[i - 1]
            curr = result[i]
            override = _CONTEXT_OVERRIDES.get((prev.degree, curr.degree))
            if override is not None:
                result[i - 1] = dataclasses.replace(prev, function=override)

        return result
