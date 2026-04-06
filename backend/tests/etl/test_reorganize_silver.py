# tests/etl/test_reorganize_silver.py
"""Tests for Silver reorganizer — routing and unknown resolution."""

import shutil
from pathlib import Path

import pytest

from scripts.baroque_corpus_etl.load.reorganize_silver import (
    ResolvedFile,
    resolve_named_file,
    resolve_unknown_file,
    route_bwv,
)


class TestRouteBWV:
    def test_cello_suite(self):
        assert route_bwv(1007) == "solo_instruments/cello/cello_suites"
        assert route_bwv(1012) == "solo_instruments/cello/cello_suites"

    def test_violin_sonata(self):
        assert route_bwv(1001) == "solo_instruments/violin/sonatas"
        assert route_bwv(1003) == "solo_instruments/violin/sonatas"

    def test_violin_partita(self):
        assert route_bwv(1002) == "solo_instruments/violin/partitas"
        assert route_bwv(1006) == "solo_instruments/violin/partitas"

    def test_flute(self):
        assert route_bwv(1013) == "solo_instruments/flute"

    def test_goldberg(self):
        assert route_bwv(988) == "keyboard/goldberg"

    def test_english_suites(self):
        assert route_bwv(806) == "keyboard/english_suites"
        assert route_bwv(811) == "keyboard/english_suites"

    def test_organ(self):
        assert route_bwv(525) == "organ"
        assert route_bwv(651) == "organ"

    def test_concertos(self):
        assert route_bwv(1041) == "orchestral/concertos"
        assert route_bwv(1065) == "orchestral/concertos"

    def test_musical_offering(self):
        assert route_bwv(1079) == "orchestral/musical_offering"

    def test_art_of_fugue(self):
        assert route_bwv(1080) == "orchestral/art_of_fugue"

    def test_sacred(self):
        assert route_bwv(232) == "sacred"
        assert route_bwv(248) == "sacred"

    def test_keyboard_misc(self):
        assert route_bwv(772) == "keyboard/misc"
        assert route_bwv(906) == "keyboard/misc"


class TestResolveNamedFile:
    def _make(self, tmp_path: Path, name: str) -> Path:
        p = tmp_path / name
        p.write_bytes(b"MThd" + b"\x00" * 10)
        return p

    def test_goldberg_aria(self, tmp_path):
        p = self._make(tmp_path, "bach_BWV988_00_aria.mid")
        r = resolve_named_file(p)
        assert r.target_subdir == "keyboard/goldberg"
        assert r.bwv_num == 988

    def test_cello_suite_routed(self, tmp_path):
        p = self._make(tmp_path, "bach_BWV1007_01_prelude.mid")
        r = resolve_named_file(p)
        assert r.target_subdir == "solo_instruments/cello/cello_suites"

    def test_concerto_routed(self, tmp_path):
        p = self._make(tmp_path, "bach_BWV1041a.mid")
        r = resolve_named_file(p)
        assert r.target_subdir == "orchestral/concertos"

    def test_no_bwv_goes_unresolved(self, tmp_path):
        p = self._make(tmp_path, "bach_something_weird.mid")
        r = resolve_named_file(p)
        assert r.target_subdir == "midi_unresolved"
        assert not r.resolved


class TestResolveUnknownFile:
    def _make(self, tmp_path: Path, name: str) -> Path:
        p = tmp_path / name
        p.write_bytes(b"MThd" + b"\x00" * 10)
        return p

    def test_cello_suite_resolved(self, tmp_path):
        p = self._make(tmp_path, "bach_unknown_bach_cs1_1pre_mid_unknown_full.mid")
        r = resolve_unknown_file(p)
        assert r.bwv_num == 1007
        assert r.target_subdir == "solo_instruments/cello/cello_suites"
        assert "prelude" in r.target_slug

    def test_violin_sonata_resolved(self, tmp_path):
        p = self._make(tmp_path, "bach_unknown_bach_vs1_1ada_mid_unknown_full.mid")
        r = resolve_unknown_file(p)
        assert r.bwv_num == 1001
        assert r.target_subdir == "solo_instruments/violin/sonatas"

    def test_violin_partita_resolved(self, tmp_path):
        p = self._make(tmp_path, "bach_unknown_bach_vp2_1all_mid_unknown_full.mid")
        r = resolve_unknown_file(p)
        assert r.bwv_num == 1004
        assert r.target_subdir == "solo_instruments/violin/partitas"

    def test_flute_partita_resolved(self, tmp_path):
        p = self._make(tmp_path, "bach_unknown_bach_fp_1all_mid_unknown_full.mid")
        r = resolve_unknown_file(p)
        assert r.bwv_num == 1013
        assert r.target_subdir == "solo_instruments/flute"

    def test_pfa_resolved(self, tmp_path):
        p = self._make(tmp_path, "bach_unknown_bach_pfa_1pre_mid_unknown_full.mid")
        r = resolve_unknown_file(p)
        assert r.bwv_num == 998
        assert r.target_subdir == "solo_instruments/lute"

    def test_bjs_resolved(self, tmp_path):
        p = self._make(tmp_path, "bach_unknown_bach_bjs1031a_mid_unknown_full.mid")
        r = resolve_unknown_file(p)
        assert r.bwv_num == 1031

    def test_unresolvable_goes_to_unresolved(self, tmp_path):
        p = self._make(tmp_path, "bach_unknown_bach_jsb212sf_mid_unknown_full.mid")
        r = resolve_unknown_file(p)
        assert r.target_subdir == "midi_unresolved"