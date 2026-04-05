# tests/etl/test_rename_plan.py
"""Tests for rename_plan.py — slug normalization and collision resolution."""

import pytest
from scripts.baroque_corpus_etl.transform.rename_plan import (
    normalize_catalog,
    normalize_slug,
    resolve_collisions,
    silver_subdir,
)


class TestNormalizeCatalog:
    def test_strips_leading_zeros(self):
        assert normalize_catalog("BWV0541") == "BWV541"

    def test_uppercase_prefix(self):
        assert normalize_catalog("bwv772") == "BWV772"

    def test_letter_suffix_lowercase(self):
        assert normalize_catalog("BWV810E") == "BWV810e"

    def test_combined(self):
        assert normalize_catalog("BWV0541P") == "BWV541p"

    def test_kirkpatrick(self):
        assert normalize_catalog("K001") == "K1"

    def test_empty(self):
        assert normalize_catalog("") == ""

    def test_nan(self):
        assert normalize_catalog("nan") == ""


class TestNormalizeSlug:
    def test_unsafe_chars_replaced(self):
        # spaces and parens become underscores, then collapsed
        assert normalize_slug("bach (BWV772).krn") == "bach_BWV772.krn"

    def test_collapses_underscores(self):
        result = normalize_slug("bach__BWV772.krn")
        assert "__" not in result

    def test_preserves_extension(self):
        assert normalize_slug("bach_BWV772.mid").endswith(".mid")


class TestSilverSubdir:
    def test_chorales(self):
        assert silver_subdir("bulk_chorales_371") == "bach/chorales"

    def test_kunstderfuge_js(self):
        assert silver_subdir("kunstderfuge_js") == "bach/midi"

    def test_vivaldi(self):
        assert silver_subdir("kunstderfuge_vivaldi") == "contrast/vivaldi"

    def test_unknown_fallback(self):
        assert silver_subdir("mystery_source") == "misc/mystery_source"


class TestResolveCollisions:
    def _make_rows(self, slugs: list[str], sources: list[str]) -> list[dict]:
        return [
            {
                "source_path": f"/raw/{src}.mid",
                "silver_path": f"/silver/{slug}",
                "silver_slug": slug,
                "action": "RENAME",
                "note": "",
            }
            for slug, src in zip(slugs, sources)
        ]

    def test_no_collision_unchanged(self):
        rows = self._make_rows(
            ["bach_BWV772.krn", "bach_BWV773.krn"],
            ["inven01", "inven02"],
        )
        result = resolve_collisions(rows)
        assert result[0]["action"] == "RENAME"
        assert result[1]["action"] == "RENAME"

    def test_collision_resolved(self):
        rows = self._make_rows(
            ["bach_BWV1080_02.mid", "bach_BWV1080_02.mid"],
            ["1080_c02", "1080c02b"],
        )
        result = resolve_collisions(rows)
        slugs = [r["silver_slug"] for r in result]
        assert len(set(slugs)) == 2
        assert all(r["action"] == "COLLISION_RESOLVED" for r in result)

    def test_art_of_fugue_b_variant(self):
        rows = self._make_rows(
            ["bach_BWV1080_02.mid", "bach_BWV1080_02.mid"],
            ["1080_c02_mid", "1080c02b_mid"],
        )
        result = resolve_collisions(rows)
        b_variant = next(r for r in result if "b" in r["silver_slug"][-6:])
        assert b_variant["silver_slug"].endswith("_b.mid")