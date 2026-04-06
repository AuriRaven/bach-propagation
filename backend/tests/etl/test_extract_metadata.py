"""Tests for extract_metadata.py.

All tests are pure unit tests — no Supabase, no music21 I/O, no network.
music21 extraction is tested via mocking.

Run from backend/ project root:
    uv run pytest tests/etl/test_extract_metadata.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.baroque_corpus_etl.load.extract_metadata import (
    infer_form_tag,
    infer_instrument_family,
    parse_bwv_from_filename,
    parse_movement_from_filename,
    extract_metadata,
    _fetch_existing_metadata_ids,
)


# =============================================================================
# parse_bwv_from_filename
# =============================================================================

class TestParseBwvFromFilename:
    def test_standard_uppercase(self):
        assert parse_bwv_from_filename("bach_BWV988.mid") == "988"

    def test_standard_lowercase(self):
        assert parse_bwv_from_filename("bwv1007_prelude.mid") == "1007"

    def test_with_underscore_separator(self):
        assert parse_bwv_from_filename("bwv_525.mid") == "525"

    def test_no_bwv_returns_none(self):
        assert parse_bwv_from_filename("riemenschneider001.krn") is None

    def test_embedded_in_longer_name(self):
        assert parse_bwv_from_filename("my_bach_BWV806_allemande.mid") == "806"


# =============================================================================
# parse_movement_from_filename
# =============================================================================

class TestParseMovementFromFilename:
    def test_prelude(self):
        assert parse_movement_from_filename("bwv1007_prelude.mid") == "Prelude"

    def test_fugue(self):
        assert parse_movement_from_filename("bwv1080_fugue.mid") == "Fugue"

    def test_sarabande(self):
        assert parse_movement_from_filename("bwv_sarabande.mid") == "Sarabande"

    def test_no_match_returns_none(self):
        assert parse_movement_from_filename("riemenschneider001.krn") is None

    def test_case_insensitive(self):
        assert parse_movement_from_filename("BWV806_ALLEMANDE.mid") == "Allemande"

    def test_hyphen_separator(self):
        assert parse_movement_from_filename("bwv-1004-gigue.mid") == "Gigue"


# =============================================================================
# infer_form_tag
# =============================================================================

class TestInferFormTag:
    def test_chorales_always_chorale(self):
        assert infer_form_tag("chorales", None, "r001.krn") == "chorale"

    def test_inventions_invention(self):
        assert infer_form_tag("inventions", None, "bwv772.krn") == "invention"

    def test_inventions_sinfonia(self):
        assert infer_form_tag("inventions", None, "bwv787_sinfonia.krn") == "sinfonia"

    def test_organ_fixed(self):
        assert infer_form_tag("organ", None, "bwv525.mid") == "organ_work"

    def test_fugue_movement(self):
        assert infer_form_tag("keyboard", "Fugue", "bwv542.mid") == "fugue"

    def test_suite_movement(self):
        assert infer_form_tag("solo_instruments", "Allemande", "bwv1007.mid") == "suite_movement"

    def test_orchestral_fugue_from_filename(self):
        assert infer_form_tag("orchestral", None, "art_of_fugue.mid") == "fugue"

    def test_orchestral_fallback(self):
        assert infer_form_tag("orchestral", None, "concerto.mid") == "orchestral_work"

    def test_keyboard_fallback(self):
        assert infer_form_tag("keyboard", None, "bwv772.mid") == "keyboard_work"


# =============================================================================
# infer_instrument_family
# =============================================================================

class TestInferInstrumentFamily:
    def test_chorales(self):
        assert infer_instrument_family("chorales", None) == "choir"

    def test_inventions(self):
        assert infer_instrument_family("inventions", None) == "keyboard"

    def test_keyboard(self):
        assert infer_instrument_family("keyboard", "goldberg") == "keyboard"

    def test_organ(self):
        assert infer_instrument_family("organ", None) == "organ"

    def test_sacred(self):
        assert infer_instrument_family("sacred", None) == "mixed"

    def test_solo_cello(self):
        assert infer_instrument_family("solo_instruments", "cello") == "strings"

    def test_solo_violin(self):
        assert infer_instrument_family("solo_instruments", "violin") == "strings"

    def test_solo_flute(self):
        assert infer_instrument_family("solo_instruments", "flute") == "woodwinds"

    def test_solo_lute(self):
        assert infer_instrument_family("solo_instruments", "lute") == "plucked"

    def test_orchestral_art_of_fugue(self):
        assert infer_instrument_family("orchestral", "art_of_fugue") == "keyboard"

    def test_orchestral_concertos(self):
        assert infer_instrument_family("orchestral", "concertos") == "mixed"


# =============================================================================
# extract_metadata — mocked music21 success path
# =============================================================================

class TestExtractMetadataMusic21Success:
    def test_returns_expected_values(self, tmp_path):
        f = tmp_path / "bwv1007_prelude.mid"
        f.write_bytes(b"fake midi")

        mock_m21_result = {
            "key_signature": "G major",
            "key_mode": "major",
            "time_signature": "3/4",
            "num_measures": 44,
            "num_voices": 1,
            "duration_seconds": 132.0,
            "raw_metadata": {"title": "Prelude"},
            "extraction_method": "music21",
        }

        with patch(
            "scripts.baroque_corpus_etl.load.extract_metadata._extract_with_music21",
            return_value=mock_m21_result,
        ):
            result = extract_metadata(
                abs_path=f,
                silver_path="bach/solo_instruments/cello/cello_suites/bwv1007_prelude.mid",
                collection="solo_instruments",
                subcollection="cello",
            )

        assert result["key_signature"] == "G major"
        assert result["key_mode"] == "major"
        assert result["time_signature"] == "3/4"
        assert result["num_measures"] == 44
        assert result["extraction_method"] == "music21"
        assert result["instrument_family"] == "strings"
        assert result["form_tag"] == "prelude"

    def test_all_required_keys_present(self, tmp_path):
        f = tmp_path / "r001.krn"
        f.write_bytes(b"fake krn")

        with patch(
            "scripts.baroque_corpus_etl.load.extract_metadata._extract_with_music21",
            return_value={
                "key_signature": None, "key_mode": None, "time_signature": "4/4",
                "num_measures": 10, "num_voices": 4, "duration_seconds": 60.0,
                "raw_metadata": None, "extraction_method": "music21",
            },
        ):
            result = extract_metadata(f, "bach/chorales/r001.krn", "chorales", None)

        required = {
            "key_signature", "key_mode", "time_signature", "num_measures",
            "num_voices", "duration_seconds", "instrument_family",
            "form_tag", "extraction_method", "raw_metadata",
        }
        assert required.issubset(result.keys())


# =============================================================================
# extract_metadata — music21 failure → filename fallback
# =============================================================================

class TestExtractMetadataFallback:
    def test_falls_back_on_music21_exception(self, tmp_path):
        f = tmp_path / "bwv1041_allegro.mid"
        f.write_bytes(b"fake midi")

        with patch(
            "scripts.baroque_corpus_etl.load.extract_metadata._extract_with_music21",
            side_effect=Exception("parse error"),
        ):
            result = extract_metadata(
                abs_path=f,
                silver_path="bach/orchestral/concertos/bwv1041_allegro.mid",
                collection="orchestral",
                subcollection="concertos",
            )

        assert result["extraction_method"] == "filename_parsing"
        assert result["key_signature"] is None
        assert result["key_mode"] is None
        assert result["instrument_family"] == "mixed"
        assert result["form_tag"] == "orchestral_work"

    def test_movement_name_still_parsed_on_fallback(self, tmp_path):
        f = tmp_path / "bwv1007_prelude.mid"
        f.write_bytes(b"fake")

        with patch(
            "scripts.baroque_corpus_etl.load.extract_metadata._extract_with_music21",
            side_effect=Exception("timeout"),
        ):
            result = extract_metadata(
                f, "bach/solo_instruments/cello/cello_suites/bwv1007_prelude.mid",
                "solo_instruments", "cello",
            )

        assert result["form_tag"] == "prelude"


# =============================================================================
# _fetch_existing_metadata_ids
# =============================================================================

class TestFetchExistingMetadataIds:
    def test_returns_set_of_ids(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value.data = [
            {"id": "uuid-1"},
            {"id": "uuid-2"},
        ]
        result = _fetch_existing_metadata_ids(mock_client)
        assert result == {"uuid-1", "uuid-2"}

    def test_empty_table_returns_empty_set(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value.data = []
        result = _fetch_existing_metadata_ids(mock_client)
        assert result == set()