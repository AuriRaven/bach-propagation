"""
Tests for :mod:`src.corpus.transition_matrix`.

These tests run entirely off synthetic analysis dicts — no Stage-4 cache
and no Supabase connection required. The idea is to pin the *algebra*
of the matrix: counts → smoothed probabilities → row-normalised; mode
separation; JSON round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.corpus.transition_matrix import (
    MATRIX_SCHEMA_VERSION,
    TransitionMatrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analysis(rn_labels: List[str], mode: str = "major", tonic: int = 0) -> Dict[str, object]:
    """Build a minimal analysis dict for the matrix to ingest."""
    return {
        "score_id": f"toy-{mode}",
        "global_key": {"tonic": tonic, "mode": mode, "confidence": 1.0, "label": f"{mode} {tonic}"},
        "chords": [],
        "roman_numerals": [
            {
                "label": lbl,
                "degree": 1,
                "quality": "major",
                "inversion": 0,
                "local_key_tonic": tonic,
                "local_key_mode": mode,
                "is_secondary": False,
                "secondary_target": None,
                "function": "TONIC",
            }
            for lbl in rn_labels
        ],
        "cadences": [],
    }


# ---------------------------------------------------------------------------
# TestBuildFromAnalyses
# ---------------------------------------------------------------------------


class TestBuildFromAnalyses:
    def test_counts_adjacent_pairs(self):
        tm = TransitionMatrix()
        tm.build_from_analyses([_analysis(["I", "V", "I"])])
        # Two pairs: (I, V) and (V, I).
        assert tm.row_count("major", "I") == 1
        assert tm.row_count("major", "V") == 1

    def test_single_chord_piece_contributes_nothing(self):
        tm = TransitionMatrix()
        tm.build_from_analyses([_analysis(["I"])])
        assert tm.labels("major") == []

    def test_major_and_minor_tables_are_independent(self):
        tm = TransitionMatrix()
        tm.build_from_analyses([
            _analysis(["I", "V", "I"], mode="major"),
            _analysis(["i", "iv", "V", "i"], mode="minor"),
        ])
        assert "V" in tm.labels("major")
        assert "iv" not in tm.labels("major")
        assert "iv" in tm.labels("minor")
        assert "V" in tm.labels("minor")


# ---------------------------------------------------------------------------
# TestLaplaceSmoothing
# ---------------------------------------------------------------------------


class TestLaplaceSmoothing:
    def test_unseen_transition_is_nonzero(self):
        tm = TransitionMatrix()
        tm.build_from_analyses([_analysis(["I", "V", "I"])])
        # (V, V) was never observed — but smoothing gives it nonzero mass.
        p = tm.probability("major", "V", "V")
        assert p > 0.0

    def test_unseen_from_rn_returns_uniform(self):
        tm = TransitionMatrix()
        tm.build_from_analyses([_analysis(["I", "V", "I"])])
        # "IV" has never appeared as a source — every successor gets the
        # same smoothed probability.
        labels = tm.labels("major")
        probs = [tm.probability("major", "IV", tgt) for tgt in labels]
        assert len(set(round(p, 9) for p in probs)) == 1

    def test_observed_transitions_are_higher_than_unseen(self):
        tm = TransitionMatrix()
        tm.build_from_analyses([_analysis(["I", "V", "I", "V", "I"])])
        observed = tm.probability("major", "I", "V")
        unseen = tm.probability("major", "I", "I")  # I → I never observed
        assert observed > unseen
        assert unseen > 0.0

    def test_rows_sum_to_one_over_alphabet(self):
        """For every ``from_rn`` the smoothed row must be a valid
        distribution over the mode's label alphabet."""
        tm = TransitionMatrix()
        tm.build_from_analyses([
            _analysis(["I", "IV", "V", "I", "ii", "V", "I"]),
        ])
        labels = tm.labels("major")
        for from_rn in labels:
            total = sum(tm.probability("major", from_rn, to) for to in labels)
            assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestTopK
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top_k_sorts_descending(self):
        tm = TransitionMatrix()
        # Heavily bias V → I by repeating it.
        tm.build_from_analyses([_analysis(["I", "V", "I"] * 10 + ["V", "vi"])])
        top = tm.top_k("major", "V", k=3)
        assert top[0][0] == "I"
        # Probabilities monotonically descending.
        assert all(top[i][1] >= top[i + 1][1] for i in range(len(top) - 1))

    def test_top_k_respects_k(self):
        tm = TransitionMatrix()
        tm.build_from_analyses([_analysis(["I", "V", "I", "vi", "IV", "V", "I"])])
        assert len(tm.top_k("major", "I", k=2)) == 2
        # k larger than alphabet clamps naturally.
        assert len(tm.top_k("major", "I", k=999)) == len(tm.labels("major"))

    def test_top_k_empty_table_returns_empty(self):
        tm = TransitionMatrix()
        assert tm.top_k("major", "I", k=5) == []


# ---------------------------------------------------------------------------
# TestJsonRoundTrip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_roundtrip_preserves_probabilities(self, tmp_path: Path):
        tm = TransitionMatrix()
        tm.build_from_analyses([
            _analysis(["I", "IV", "V", "I", "vi", "ii", "V", "I"]),
            _analysis(["i", "iv", "V", "i"], mode="minor"),
        ])
        path = tmp_path / "matrix.json"
        tm.to_json(path)

        loaded = TransitionMatrix.from_json(path)

        # Probabilities for a sampling of transitions must round-trip.
        for mode in ("major", "minor"):
            for from_rn in tm.labels(mode):
                for to_rn in tm.labels(mode):
                    a = tm.probability(mode, from_rn, to_rn)
                    b = loaded.probability(mode, from_rn, to_rn)
                    assert a == pytest.approx(b, abs=1e-5)

    def test_json_sorted_descending(self, tmp_path: Path):
        tm = TransitionMatrix()
        tm.build_from_analyses([_analysis(["I", "V", "I"] * 5 + ["I", "IV"])])
        path = tmp_path / "matrix.json"
        tm.to_json(path)

        data = json.loads(path.read_text())
        row = data["modes"]["major"]["transitions"]["I"]
        probs = [p for (_lbl, p) in row]
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_json_records_schema_version(self, tmp_path: Path):
        TransitionMatrix().to_json(tmp_path / "m.json")
        data = json.loads((tmp_path / "m.json").read_text())
        assert data["schema_version"] == MATRIX_SCHEMA_VERSION

    def test_from_json_rejects_schema_mismatch(self, tmp_path: Path):
        (tmp_path / "m.json").write_text(json.dumps({
            "schema_version": "9.9.9",
            "modes": {},
        }))
        with pytest.raises(ValueError, match="schema_version"):
            TransitionMatrix.from_json(tmp_path / "m.json")


# ---------------------------------------------------------------------------
# TestBuildFromCache
# ---------------------------------------------------------------------------


class TestBuildFromCache:
    def test_reads_json_files_from_disk(self, tmp_path: Path):
        (tmp_path / "a.json").write_text(json.dumps(_analysis(["I", "V", "I"])))
        (tmp_path / "b.json").write_text(json.dumps(_analysis(["i", "iv", "i"], mode="minor")))

        tm = TransitionMatrix()
        tm.build_from_cache(tmp_path)

        assert "V" in tm.labels("major")
        assert "iv" in tm.labels("minor")

    def test_skips_manifest_json(self, tmp_path: Path):
        # manifest.json would pollute the matrix if not filtered.
        (tmp_path / "manifest.json").write_text(json.dumps({
            "totals": {"ok": 1}, "entries": [],
        }))
        (tmp_path / "a.json").write_text(json.dumps(_analysis(["I", "V", "I"])))

        tm = TransitionMatrix()
        tm.build_from_cache(tmp_path)
        assert tm.labels("major") == ["I", "V"]

    def test_missing_cache_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            TransitionMatrix().build_from_cache(tmp_path / "no-such-dir")

    def test_malformed_json_is_skipped(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("{not valid json")
        (tmp_path / "good.json").write_text(json.dumps(_analysis(["I", "V", "I"])))

        tm = TransitionMatrix()
        tm.build_from_cache(tmp_path)
        assert tm.labels("major") == ["I", "V"]


# ---------------------------------------------------------------------------
# TestModeValidation
# ---------------------------------------------------------------------------


class TestModeValidation:
    def test_probability_rejects_unknown_mode(self):
        tm = TransitionMatrix()
        with pytest.raises(ValueError, match="mode"):
            tm.probability("dorian", "I", "V")

    def test_top_k_rejects_unknown_mode(self):
        tm = TransitionMatrix()
        with pytest.raises(ValueError, match="mode"):
            tm.top_k("dorian", "I", k=3)
