# tests/etl/test_source_detector.py
from pathlib import Path
import pytest
from scripts.baroque_corpus_etl.transform.source_detector import detect_source

class TestSourceDetector:
    def test_kernscores_inventions(self):
        p = Path("data/raw/bach/inventions/inven01.krn")
        s = detect_source(p)
        assert s.source_id == "kernscores_inventions"
        assert s.composer == "bach"

    def test_kunstderfuge_js(self):
        p = Path("data/raw/bach_js/bach/bwv772_mid/bach_bwv772_mid_bwv772_full.mid")
        s = detect_source(p)
        assert s.source_id == "kunstderfuge_js"

    def test_bulk_370_flagged_duplicate(self):
        p = Path("data/raw/bulk_xml/bach-chorales-370/kern/chor001.krn")
        s = detect_source(p)
        assert s.source_id == "bulk_chorales_370"
        policy = __import__(
            "scripts.baroque_corpus_etl.transform.catalog_registry",
            fromlist=["get_policy"]
        ).get_policy(s.source_id)
        assert policy.dedup_action == "DUPLICATE_DROP"

    def test_bulk_371_kept(self):
        p = Path("data/raw/bulk_xml/bach-chorales-371/kern/chor001.krn")
        s = detect_source(p)
        assert s.source_id == "bulk_chorales_371"

    def test_handel_kunstderfuge(self):
        p = Path("data/raw/handel/kunstderfuge/handel_kunstderfuge_unknown_hwv-426_1_prelude.mid")
        s = detect_source(p)
        assert s.source_id == "kunstderfuge_handel"
        assert s.composer == "handel"

    def test_scarlatti(self):
        p = Path("data/raw/scarlatti/kunstderfuge/scarlatti_k001.mid")
        s = detect_source(p)
        assert s.source_id == "kunstderfuge_scarlatti"


# tests/etl/test_filename_decoder.py
import pytest
from scripts.baroque_corpus_etl.transform.filename_decoder import (
    decode_filename, decode_kernscores, decode_kunstderfuge_js,
    decode_kunstderfuge_contrast, is_corrupted_filename,
)

class TestCorruptionDetection:
    def test_midi_asp_flagged(self):
        assert is_corrupted_filename("midi.asp?file=vivaldi_4_stagioni") is True

    def test_normal_filename_not_flagged(self):
        assert is_corrupted_filename("bach_bwv772_mid_bwv772_full") is False


class TestKernscoresDecoder:
    def test_invention(self):
        r = decode_kernscores("inven01")
        assert r.catalog_number_resolved == "BWV772"

    def test_sinfonia(self):
        r = decode_kernscores("sinfo03")
        assert r.catalog_number_resolved == "BWV789"

    def test_cello_suite_with_movement(self):
        r = decode_kernscores("cs1_1pre")
        assert r.catalog_number_resolved == "BWV1007"
        assert r.movement_name_decoded == "Prelude"
        assert r.movement_num == "01"

    def test_violin_sonata(self):
        r = decode_kernscores("vs2_3and")
        assert r.catalog_number_resolved == "BWV1003"
        assert r.movement_name_decoded == "Andante"

    def test_wtc_ii_prelude(self):
        r = decode_kernscores("wtcii01a")
        assert r.catalog_number_resolved == "BWV870"
        assert r.movement_name_decoded == "Prelude"

    def test_wtc_ii_fugue(self):
        r = decode_kernscores("wtcii01b")
        assert r.catalog_number_resolved == "BWV870"
        assert r.movement_name_decoded == "Fugue"

    def test_chorale(self):
        r = decode_kernscores("chor001")
        assert r.movement_name_decoded == "Chorale"
        assert r.movement_num == "001"


class TestKunstderfugeJSDecoder:
    def test_explicit_bwv(self):
        r = decode_kunstderfuge_js("bach_bwv772_mid_bwv772_full")
        assert r.catalog_number_resolved == "BWV772"

    def test_goldberg_aria(self):
        r = decode_kunstderfuge_js("bach_988_aria_mid_bwv988_988_full")
        assert r.catalog_number_resolved == "BWV988"
        assert r.movement_name_decoded == "Aria"

    def test_goldberg_variation(self):
        r = decode_kunstderfuge_js("bach_988_v01_mid_bwv988_988_full")
        assert r.catalog_number_resolved == "BWV988"
        assert r.movement_num == "01"

    def test_b_minor_mass(self):
        r = decode_kunstderfuge_js("bach_bjsbmm01_mid_bwv232_full")
        assert r.catalog_number_resolved == "BWV232"

    def test_musical_offering(self):
        r = decode_kunstderfuge_js("bach_1079_02_mid_unknown_full")
        assert r.catalog_number_resolved == "BWV1079"
        assert r.movement_num == "02"

    def test_art_of_fugue(self):
        r = decode_kunstderfuge_js("bach_1080_c01_mid_unknown_full")
        assert r.catalog_number_resolved == "BWV1080"


class TestKunstderfugeContrastDecoder:
    def test_handel_hwv(self):
        r = decode_kunstderfuge_contrast(
            "handel_kunstderfuge_unknown_hwv-426_1_prelude_(c)yamada", "handel"
        )
        assert r.catalog_number_resolved == "HWV426"
        assert r.movement_num == "01"

    def test_scarlatti_kirkpatrick(self):
        r = decode_kunstderfuge_contrast(
            "scarlatti_kunstderfuge_unknown_sonatas_k-001_(c)sankey", "scarlatti"
        )
        assert r.catalog_number_resolved == "K001"

    def test_quarantine_passthrough(self):
        r = decode_filename(
            "vivaldi_kunstderfuge_unknown_midi.asp?file=vivaldi_4_stagioni",
            "kunstderfuge_vivaldi",
            "vivaldi",
        )
        assert r.is_quarantine is True
        assert r.silver_slug_stem == "QUARANTINE"