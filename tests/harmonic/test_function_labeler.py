"""
Tests for src/harmonic/function_labeler.py

All tests use synthetic RomanNumeral objects — no file I/O required.
"""

import pytest
from fractions import Fraction

from src.core.data_structures import HarmonicFunction, RomanNumeral
from src.harmonic.function_labeler import FunctionLabeler


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _rn(
    degree: int,
    quality: str = "major",
    inversion: int = 0,
    is_secondary: bool = False,
    function: HarmonicFunction = None,
) -> RomanNumeral:
    return RomanNumeral(
        degree=degree,
        quality=quality,
        inversion=inversion,
        local_key_tonic=0,
        local_key_mode="major",
        is_secondary=is_secondary,
        secondary_target=None,
        function=function,
    )


FL = FunctionLabeler()

T = HarmonicFunction.TONIC
PD = HarmonicFunction.PREDOMINANT
D = HarmonicFunction.DOMINANT
AMB = HarmonicFunction.AMBIGUOUS


# ---------------------------------------------------------------------------
# TestLabel
# ---------------------------------------------------------------------------

class TestLabel:
    def test_degree_1_gets_tonic(self):
        assert FL.label(_rn(1)).function == T

    def test_degree_2_gets_predominant(self):
        assert FL.label(_rn(2)).function == PD

    def test_degree_3_gets_ambiguous(self):
        assert FL.label(_rn(3)).function == AMB

    def test_degree_4_gets_predominant(self):
        assert FL.label(_rn(4)).function == PD

    def test_degree_5_gets_dominant(self):
        assert FL.label(_rn(5)).function == D

    def test_degree_6_gets_tonic(self):
        assert FL.label(_rn(6)).function == T

    def test_degree_7_gets_dominant(self):
        assert FL.label(_rn(7)).function == D

    def test_secondary_dominant_always_gets_dominant(self):
        # is_secondary=True overrides the degree-6 tonic base function
        rn = _rn(6, is_secondary=True)
        assert FL.label(rn).function == D

    def test_already_labelled_returned_unchanged(self):
        rn = _rn(1, function=D)  # manually set to DOMINANT
        result = FL.label(rn)
        assert result.function == D   # not overwritten to TONIC
        assert result is rn           # same object (no copy needed)

    def test_returns_new_object_when_unlabelled(self):
        rn = _rn(1)
        result = FL.label(rn)
        assert result is not rn  # new frozen instance created

    def test_result_is_roman_numeral(self):
        assert isinstance(FL.label(_rn(1)), RomanNumeral)

    def test_non_secondary_degree_6_gets_tonic(self):
        rn = _rn(6, is_secondary=False)
        assert FL.label(rn).function == T


# ---------------------------------------------------------------------------
# TestLabelSequence
# ---------------------------------------------------------------------------

class TestLabelSequence:
    def test_empty_input_returns_empty(self):
        assert FL.label_sequence([]) == []

    def test_single_rn_labelled(self):
        result = FL.label_sequence([_rn(1)])
        assert result[0].function == T

    def test_all_rns_labelled(self):
        rns = [_rn(d) for d in range(1, 8)]
        result = FL.label_sequence(rns)
        assert all(r.function is not None for r in result)

    def test_vi_before_ii_becomes_predominant(self):
        rns = [_rn(6), _rn(2)]
        result = FL.label_sequence(rns)
        # vi's function is relabelled to PREDOMINANT
        assert result[0].function == PD
        # ii's function is still PREDOMINANT (unchanged)
        assert result[1].function == PD

    def test_vi_before_IV_becomes_predominant(self):
        rns = [_rn(6), _rn(4)]
        result = FL.label_sequence(rns)
        assert result[0].function == PD

    def test_vi_before_V_stays_tonic(self):
        # vi → V : no context override → vi stays TONIC
        rns = [_rn(6), _rn(5)]
        result = FL.label_sequence(rns)
        assert result[0].function == T

    def test_vi_before_I_stays_tonic(self):
        rns = [_rn(6), _rn(1)]
        result = FL.label_sequence(rns)
        assert result[0].function == T

    def test_iii_stays_ambiguous_regardless_of_context(self):
        # iii is not in _CONTEXT_OVERRIDES as a trigger
        rns = [_rn(3), _rn(2)]
        result = FL.label_sequence(rns)
        assert result[0].function == AMB

    def test_does_not_mutate_input_list(self):
        rns = [_rn(6), _rn(2)]
        original_ids = [id(r) for r in rns]
        FL.label_sequence(rns)
        assert [id(r) for r in rns] == original_ids

    def test_does_not_mutate_input_rn_objects(self):
        rn = _rn(6)
        original_function = rn.function
        FL.label_sequence([rn, _rn(2)])
        # rn.function must be unchanged (frozen dataclass, but just confirm)
        assert rn.function == original_function

    def test_returns_new_list(self):
        rns = [_rn(1)]
        result = FL.label_sequence(rns)
        assert result is not rns

    def test_long_sequence_all_labelled(self):
        rns = [_rn(d) for d in [1, 4, 5, 1, 6, 2, 5, 1]]
        result = FL.label_sequence(rns)
        assert all(r.function is not None for r in result)
        assert len(result) == 8

    def test_context_override_only_affects_previous(self):
        # vi → ii: result[0] changes; result[1] function is determined by ii alone
        rns = [_rn(6), _rn(2), _rn(5)]
        result = FL.label_sequence(rns)
        assert result[0].function == PD   # vi → PREDOMINANT (override)
        assert result[1].function == PD   # ii → PREDOMINANT (base)
        assert result[2].function == D    # V → DOMINANT (base)
