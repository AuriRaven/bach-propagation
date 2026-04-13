"""
Tests for scripts/batch_harmonic_analysis.py

These tests exercise the batch driver against a tiny synthetic
:class:`Score` (no music21 corpus parsing, no disk corpus) so they run
in under a second. The music21-corpus path is smoke-tested separately
by the user via the CLI.

Structure:
    TestAnalyseScore                — pipeline correctness on a toy score
    TestJSONRoundTrip               — per-score JSON validates against schema
    TestRunBatchSkipAndForce        — idempotency + --force semantics
    TestRunBatchFailureRecovery     — one failing task doesn't abort the run
    TestManifest                    — manifest shape and totals
    TestCLI                         — argparse wiring
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import List

import pytest

from src.core.data_structures import MusicalEvent, Score, TimeSignature

# Ensure backend/ is on sys.path so ``scripts.*`` imports work when pytest
# runs from the backend root (which is the normal configuration here).
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from scripts.batch_harmonic_analysis import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    AnalysisResult,
    PipelineConfig,
    _ManifestEntry,
    _Task,
    analyse_score,
    main,
    run_batch,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _c_major_progression() -> Score:
    """Build a tiny synthetic C-major I–IV–V–I chorale.

    Every beat carries a full triad spelled out vertically so the chord
    classifier has obvious unambiguous material — we want to exercise
    the driver, not stress the analyzers.
    """

    def triad(root_pitch: int, third: int, fifth: int, onset: float) -> List[MusicalEvent]:
        return [
            MusicalEvent(
                pitch=p,
                onset=Fraction(onset),
                duration=Fraction(1),
                voice=v,
                beat_strength=1.0,
            )
            for v, p in enumerate([root_pitch, third, fifth])
        ]

    events: List[MusicalEvent] = []
    # I: C E G
    events += triad(48, 52, 55, 0)
    # IV: F A C
    events += triad(53, 57, 60, 1)
    # V: G B D
    events += triad(55, 59, 62, 2)
    # I: C E G
    events += triad(48, 52, 55, 3)

    return Score(
        events=events,
        time_signatures=[(Fraction(0), TimeSignature(4, 4))],
        title="Toy C-major",
        composer="TEST",
    )


@pytest.fixture
def toy_score() -> Score:
    return _c_major_progression()


# ---------------------------------------------------------------------------
# TestAnalyseScore
# ---------------------------------------------------------------------------


class TestAnalyseScore:
    def test_returns_analysis_result(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        assert isinstance(result, AnalysisResult)
        assert result.score_id == "toy"
        assert result.source == "test"
        assert result.pipeline_version == ANALYSIS_SCHEMA_VERSION

    def test_populates_global_key(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        gk = result.global_key
        assert gk["tonic"] == 0          # C
        assert gk["mode"] == "major"
        assert 0.0 <= float(gk["confidence"]) <= 1.0

    def test_produces_some_chords(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        assert result.stats["n_chords"] >= 1
        # RNs and chords line up one-to-one
        assert result.stats["n_roman_numerals"] == result.stats["n_chords"]

    def test_stats_sum_matches_lists(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        assert result.stats["n_chords"] == len(result.chords)
        assert result.stats["n_roman_numerals"] == len(result.roman_numerals)
        assert result.stats["n_cadences"] == len(result.cadences)
        assert result.stats["n_nct_annotations"] == len(result.nct_annotations)

    def test_records_runtime_ms(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        assert result.runtime_ms >= 0


# ---------------------------------------------------------------------------
# TestJSONRoundTrip
# ---------------------------------------------------------------------------


class TestJSONRoundTrip:
    def test_result_dict_is_json_serialisable(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        encoded = json.dumps(result.to_dict())
        decoded = json.loads(encoded)
        assert decoded["score_id"] == "toy"
        assert decoded["pipeline_version"] == ANALYSIS_SCHEMA_VERSION

    def test_fractions_render_as_strings(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        # Onsets and durations must be strings, not floats.
        assert isinstance(result.duration_ql, str)
        for c in result.chords:
            assert isinstance(c["onset"], str)
            assert isinstance(c["duration"], str)

    def test_cadence_indices_point_back_to_harmonic_events(self, toy_score: Score):
        result = analyse_score(
            toy_score,
            score_id="toy",
            source="test",
            input_path=None,
            config=PipelineConfig(),
        )
        for cad in result.cadences:
            # -1 is the sentinel for "not found" — our cadences always come
            # from the detector's own list so indices must be valid.
            assert 0 <= cad["penultimate_index"] < len(result.chords)
            assert 0 <= cad["final_index"] < len(result.chords)


# ---------------------------------------------------------------------------
# Helpers for batch-driver tests
# ---------------------------------------------------------------------------


class _StubLoader:
    """Minimal drop-in for ScoreLoader with just enough surface for the
    run_batch tests below. Maps (kind, payload) -> Score."""

    def __init__(self, mapping, failing: set = frozenset()):
        self._mapping = mapping
        self._failing = failing

    def load_corpus(self, bwv: str) -> Score:
        if bwv in self._failing:
            raise RuntimeError(f"synthetic failure for {bwv}")
        return self._mapping[("corpus", bwv)]

    def load(self, path) -> Score:
        if str(path) in self._failing:
            raise RuntimeError(f"synthetic failure for {path}")
        return self._mapping[("path", str(path))]


@pytest.fixture
def patched_loader(monkeypatch, toy_score):
    """Replace ScoreLoader inside the batch module with a stub that returns
    the toy score for any corpus id."""
    from scripts import batch_harmonic_analysis as mod

    class _AlwaysToy:
        def __init__(self, *_, **__):
            pass

        def load_corpus(self, bwv: str) -> Score:
            return toy_score

        def load(self, path) -> Score:
            return toy_score

    monkeypatch.setattr(mod, "ScoreLoader", _AlwaysToy)
    return _AlwaysToy


# ---------------------------------------------------------------------------
# TestRunBatchSkipAndForce
# ---------------------------------------------------------------------------


class TestRunBatchSkipAndForce:
    def test_first_run_writes_output(self, patched_loader, tmp_path: Path):
        tasks = [_Task(
            score_id="bwv66.6",
            source="corpus",
            loader_args=("corpus", "66.6"),
            relative_output=Path("corpus/bwv66.6.json"),
        )]
        entries = run_batch(
            tasks,
            output_dir=tmp_path,
            config=PipelineConfig(),
            force=False,
            fail_fast=False,
            progress=False,
        )
        assert len(entries) == 1
        assert entries[0].status == "ok"
        assert (tmp_path / "corpus" / "bwv66.6.json").exists()

    def test_second_run_skips_existing(self, patched_loader, tmp_path: Path):
        tasks = [_Task(
            score_id="bwv66.6",
            source="corpus",
            loader_args=("corpus", "66.6"),
            relative_output=Path("corpus/bwv66.6.json"),
        )]
        first = run_batch(
            tasks, output_dir=tmp_path, config=PipelineConfig(),
            force=False, fail_fast=False, progress=False,
        )
        assert first[0].status == "ok"

        second = run_batch(
            tasks, output_dir=tmp_path, config=PipelineConfig(),
            force=False, fail_fast=False, progress=False,
        )
        assert len(second) == 1
        assert second[0].status == "skipped"

    def test_force_re_runs_existing(self, patched_loader, tmp_path: Path):
        tasks = [_Task(
            score_id="bwv66.6",
            source="corpus",
            loader_args=("corpus", "66.6"),
            relative_output=Path("corpus/bwv66.6.json"),
        )]
        run_batch(
            tasks, output_dir=tmp_path, config=PipelineConfig(),
            force=False, fail_fast=False, progress=False,
        )
        forced = run_batch(
            tasks, output_dir=tmp_path, config=PipelineConfig(),
            force=True, fail_fast=False, progress=False,
        )
        assert forced[0].status == "ok"


# ---------------------------------------------------------------------------
# TestRunBatchFailureRecovery
# ---------------------------------------------------------------------------


class TestRunBatchFailureRecovery:
    def test_failing_task_does_not_abort_run(self, monkeypatch, tmp_path: Path, toy_score):
        from scripts import batch_harmonic_analysis as mod

        class _HalfFailing:
            def __init__(self, *_, **__):
                pass

            def load_corpus(self, bwv: str) -> Score:
                if bwv == "BAD":
                    raise RuntimeError("boom")
                return toy_score

            def load(self, path):  # pragma: no cover — not used
                return toy_score

        monkeypatch.setattr(mod, "ScoreLoader", _HalfFailing)

        tasks = [
            _Task("bwvGOOD1", "corpus", ("corpus", "GOOD1"), Path("corpus/g1.json")),
            _Task("bwvBAD", "corpus", ("corpus", "BAD"), Path("corpus/bad.json")),
            _Task("bwvGOOD2", "corpus", ("corpus", "GOOD2"), Path("corpus/g2.json")),
        ]
        entries = run_batch(
            tasks, output_dir=tmp_path, config=PipelineConfig(),
            force=False, fail_fast=False, progress=False,
        )
        statuses = [e.status for e in entries]
        assert statuses == ["ok", "failed", "ok"]
        # Failures log records the bad task
        log_path = tmp_path / "failures.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["score_id"] == "bwvBAD"

    def test_fail_fast_raises_on_first_error(self, monkeypatch, tmp_path: Path, toy_score):
        from scripts import batch_harmonic_analysis as mod

        class _AlwaysFail:
            def __init__(self, *_, **__):
                pass

            def load_corpus(self, bwv: str) -> Score:
                raise RuntimeError("boom")

            def load(self, path):  # pragma: no cover
                return toy_score

        monkeypatch.setattr(mod, "ScoreLoader", _AlwaysFail)

        tasks = [_Task("bwvX", "corpus", ("corpus", "X"), Path("corpus/x.json"))]
        with pytest.raises(RuntimeError, match="boom"):
            run_batch(
                tasks, output_dir=tmp_path, config=PipelineConfig(),
                force=False, fail_fast=True, progress=False,
            )


# ---------------------------------------------------------------------------
# TestManifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_totals_match_entries(self, tmp_path: Path):
        entries = [
            _ManifestEntry("a", "corpus", "ok", output="a.json", runtime_ms=1, n_cadences=0),
            _ManifestEntry("b", "corpus", "ok", output="b.json", runtime_ms=2, n_cadences=1),
            _ManifestEntry("c", "corpus", "failed", error="boom"),
            _ManifestEntry("d", "corpus", "skipped", output="d.json"),
        ]
        path = write_manifest(
            tmp_path, entries,
            source_description="unit-test",
            config=PipelineConfig(),
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["totals"] == {"ok": 2, "failed": 1, "skipped": 1, "total": 4}
        assert data["pipeline_version"] == ANALYSIS_SCHEMA_VERSION
        assert len(data["entries"]) == 4

    def test_manifest_records_pipeline_config(self, tmp_path: Path):
        cfg = PipelineConfig(separate_voices=True, max_voices=4)
        write_manifest(tmp_path, [], source_description="x", config=cfg)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["pipeline_config"]["separate_voices"] is True
        assert data["pipeline_config"]["max_voices"] == 4


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_with_stubbed_loader(self, monkeypatch, tmp_path: Path, toy_score):
        from scripts import batch_harmonic_analysis as mod

        class _AlwaysToy:
            def __init__(self, *_, **__):
                pass

            def load_corpus(self, bwv: str) -> Score:
                return toy_score

            def load(self, path) -> Score:
                return toy_score

        monkeypatch.setattr(mod, "ScoreLoader", _AlwaysToy)

        rc = main([
            "--bwv", "999.9",
            "--output-dir", str(tmp_path),
            "--quiet",
        ])
        assert rc == 0
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "corpus" / "bwv999.9.json").exists()

    def test_main_requires_source(self):
        # argparse treats the mutually-exclusive group as required, so a
        # bare invocation should exit with code 2.
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_main_missing_path_returns_error(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        rc = main([
            "--path", str(missing),
            "--output-dir", str(tmp_path / "out"),
            "--quiet",
        ])
        assert rc == 2
