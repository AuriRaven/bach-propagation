"""
Tests for :class:`src.models.music_transformer.MusicTransformer`.

Uses tiny vocab sizes (chord=10, rn=15, cadence=5) and short sequences
so every test runs in well under a second on CPU. No real ``vocab.json``
is touched — the ``from_vocab`` test writes a synthetic one to ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.music_transformer import (
    CHECKPOINT_SCHEMA_VERSION,
    MultiStreamEmbedding,
    MusicTransformer,
    MusicTransformerBlock,
)


# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

CHORD_V = 10
RN_V = 15
CAD_V = 5


def _tiny_model(**overrides) -> MusicTransformer:
    """Build a tiny but valid MusicTransformer for fast unit tests."""
    kwargs = dict(
        chord_vocab_size=CHORD_V,
        rn_vocab_size=RN_V,
        cadence_vocab_size=CAD_V,
        d_model=24,        # divisible by both 3 (per-stream split) and n_heads
        n_heads=4,
        n_layers=2,
        d_ff=48,
        dropout=0.0,       # deterministic outputs for round-trip checks
        max_seq_len=32,
    )
    kwargs.update(overrides)
    return MusicTransformer(**kwargs)


def _random_batch(B: int, T: int, vocab_sizes=(CHORD_V, RN_V, CAD_V)):
    """Return three (B, T) LongTensors with ids in valid ranges."""
    g = torch.Generator().manual_seed(0)
    chord = torch.randint(0, vocab_sizes[0], (B, T), generator=g)
    rn = torch.randint(0, vocab_sizes[1], (B, T), generator=g)
    cad = torch.randint(0, vocab_sizes[2], (B, T), generator=g)
    return chord, rn, cad


# ---------------------------------------------------------------------------
# TestForwardShape
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_forward_returns_correct_shape(self):
        model = _tiny_model()
        model.eval()
        B, T = 2, 8
        chord, rn, cad = _random_batch(B, T)
        with torch.no_grad():
            out = model(chord, rn, cad)
        assert out.shape == (B, T, CHORD_V)

    def test_forward_handles_attention_mask(self):
        model = _tiny_model()
        model.eval()
        B, T = 3, 6
        chord, rn, cad = _random_batch(B, T)
        # Mark last two positions of each batch row as padding.
        attn = torch.ones(B, T, dtype=torch.long)
        attn[:, -2:] = 0
        with torch.no_grad():
            out = model(chord, rn, cad, attention_mask=attn)
        assert out.shape == (B, T, CHORD_V)
        assert torch.isfinite(out).all()

    def test_forward_dtype_is_float(self):
        model = _tiny_model()
        model.eval()
        chord, rn, cad = _random_batch(1, 4)
        with torch.no_grad():
            out = model(chord, rn, cad)
        assert out.dtype == torch.float32

    def test_rejects_sequence_longer_than_max_seq_len(self):
        model = _tiny_model(max_seq_len=8)
        chord, rn, cad = _random_batch(1, 16)
        with pytest.raises(ValueError, match="max_seq_len"):
            model(chord, rn, cad)


# ---------------------------------------------------------------------------
# TestCausalMask
# ---------------------------------------------------------------------------


class TestCausalMask:
    def test_causal_mask_function(self):
        from src.models.music_transformer import _causal_mask
        mask = _causal_mask(4, device=torch.device("cpu"), dtype=torch.float32)
        # Lower triangle (incl. diagonal) is 0; upper triangle is -inf.
        for i in range(4):
            for j in range(4):
                if j <= i:
                    assert mask[i, j].item() == 0.0
                else:
                    assert mask[i, j].item() == float("-inf")

    def test_future_token_does_not_affect_past_logits(self):
        """Hallmark of a causal model: changing input at position k must
        not change the output at any position t < k."""
        model = _tiny_model()
        model.eval()
        B, T = 1, 8
        chord, rn, cad = _random_batch(B, T)

        with torch.no_grad():
            base = model(chord, rn, cad)

        # Mutate the *last* timestep across all three streams.
        chord_mut = chord.clone()
        rn_mut = rn.clone()
        cad_mut = cad.clone()
        chord_mut[0, -1] = (chord[0, -1] + 1) % CHORD_V
        rn_mut[0, -1] = (rn[0, -1] + 1) % RN_V
        cad_mut[0, -1] = (cad[0, -1] + 1) % CAD_V

        with torch.no_grad():
            mut = model(chord_mut, rn_mut, cad_mut)

        # All positions strictly before the mutated one must be identical.
        torch.testing.assert_close(base[:, :-1, :], mut[:, :-1, :])

    def test_mid_sequence_perturbation_only_affects_later_positions(self):
        model = _tiny_model()
        model.eval()
        B, T = 1, 10
        chord, rn, cad = _random_batch(B, T)
        with torch.no_grad():
            base = model(chord, rn, cad)

        k = 5
        chord_mut = chord.clone()
        chord_mut[0, k] = (chord[0, k] + 3) % CHORD_V
        with torch.no_grad():
            mut = model(chord_mut, rn, cad)

        # Positions < k unchanged.
        torch.testing.assert_close(base[:, :k, :], mut[:, :k, :])
        # At least one position >= k changed (sanity — perturbation matters).
        assert not torch.allclose(base[:, k:, :], mut[:, k:, :])


# ---------------------------------------------------------------------------
# TestParameterCount
# ---------------------------------------------------------------------------


class TestParameterCount:
    def test_default_config_in_target_range(self):
        """Default config (the one we will train) sits in 5M–20M params."""
        model = MusicTransformer(
            chord_vocab_size=111,
            rn_vocab_size=378,
            cadence_vocab_size=11,
        )
        n = model.count_parameters()
        assert 5_000_000 <= n <= 20_000_000, (
            f"default config has {n:,} params; expected 5M–20M"
        )

    def test_larger_d_model_means_more_params(self):
        small = _tiny_model(d_model=24, n_heads=4)
        big = _tiny_model(d_model=48, n_heads=4)
        assert big.count_parameters() > small.count_parameters()

    def test_more_layers_means_more_params(self):
        shallow = _tiny_model(n_layers=2)
        deep = _tiny_model(n_layers=4)
        assert deep.count_parameters() > shallow.count_parameters()

    def test_count_parameters_is_positive(self):
        assert _tiny_model().count_parameters() > 0


# ---------------------------------------------------------------------------
# TestCheckpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_and_load_round_trip_produces_identical_output(self, tmp_path):
        model = _tiny_model()
        model.eval()
        chord, rn, cad = _random_batch(2, 6)
        with torch.no_grad():
            expected = model(chord, rn, cad)

        ckpt_path = tmp_path / "tiny.pt"
        model.save_checkpoint(ckpt_path, extra={"epoch": 7, "val_loss": 1.234})

        loaded, extra = MusicTransformer.load_checkpoint(ckpt_path)
        loaded.eval()
        with torch.no_grad():
            got = loaded(chord, rn, cad)

        torch.testing.assert_close(got, expected)
        assert extra == {"epoch": 7, "val_loss": 1.234}

    def test_loaded_model_has_matching_hyperparameters(self, tmp_path):
        model = _tiny_model(d_model=36, n_heads=6, n_layers=3)
        ckpt_path = tmp_path / "hp.pt"
        model.save_checkpoint(ckpt_path)
        loaded, _ = MusicTransformer.load_checkpoint(ckpt_path)
        assert loaded.hyperparameters == model.hyperparameters
        assert loaded.vocab_config == model.vocab_config

    def test_checkpoint_carries_schema_version(self, tmp_path):
        model = _tiny_model()
        ckpt_path = tmp_path / "v.pt"
        model.save_checkpoint(ckpt_path)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert raw["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert "model_state_dict" in raw
        assert "hyperparameters" in raw
        assert "vocab_config" in raw

    def test_load_rejects_mismatched_schema_version(self, tmp_path):
        model = _tiny_model()
        ckpt_path = tmp_path / "bad.pt"
        model.save_checkpoint(ckpt_path)
        # Corrupt the version pin.
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw["schema_version"] = "0.0.0-bogus"
        torch.save(raw, ckpt_path)
        with pytest.raises(ValueError, match="schema_version"):
            MusicTransformer.load_checkpoint(ckpt_path)

    def test_extra_defaults_to_empty_dict(self, tmp_path):
        model = _tiny_model()
        ckpt_path = tmp_path / "noex.pt"
        model.save_checkpoint(ckpt_path)
        _, extra = MusicTransformer.load_checkpoint(ckpt_path)
        assert extra == {}


# ---------------------------------------------------------------------------
# TestFromVocab
# ---------------------------------------------------------------------------


class TestFromVocab:
    def _write_vocab(self, path: Path) -> None:
        path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "specials": {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3, "CAD_NONE": 4},
            "chord_to_id": {f"c{i}": i for i in range(20)},
            "rn_to_id":    {f"r{i}": i for i in range(30)},
            "cadence_to_id": {f"k{i}": i for i in range(7)},
        }))

    def test_from_vocab_sizes_pulled_from_file(self, tmp_path):
        vocab_path = tmp_path / "vocab.json"
        self._write_vocab(vocab_path)
        model = MusicTransformer.from_vocab(
            vocab_path,
            d_model=24, n_heads=4, n_layers=2, d_ff=48,
            max_seq_len=32, dropout=0.0,
        )
        assert model.vocab_config == {
            "chord_vocab_size": 20,
            "rn_vocab_size": 30,
            "cadence_vocab_size": 7,
        }
        # And the embedding tables match those sizes.
        assert model.embedding.chord_embed.num_embeddings == 20
        assert model.embedding.rn_embed.num_embeddings == 30
        assert model.embedding.cadence_embed.num_embeddings == 7

    def test_from_vocab_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MusicTransformer.from_vocab(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# TestEmbeddingInternals
# ---------------------------------------------------------------------------


class TestEmbeddingInternals:
    def test_padding_rows_are_zero_after_init(self):
        emb = MultiStreamEmbedding(
            chord_vocab_size=10, rn_vocab_size=15, cadence_vocab_size=5,
            d_model=24, max_seq_len=16, dropout=0.0,
        )
        # padding_idx=0 in each table — row 0 should be all-zeros after init.
        assert torch.all(emb.chord_embed.weight[0] == 0)
        assert torch.all(emb.rn_embed.weight[0] == 0)
        assert torch.all(emb.cadence_embed.weight[0] == 0)

    def test_embedding_output_shape(self):
        emb = MultiStreamEmbedding(
            chord_vocab_size=10, rn_vocab_size=15, cadence_vocab_size=5,
            d_model=24, max_seq_len=16, dropout=0.0,
        )
        chord, rn, cad = _random_batch(2, 6)
        out = emb(chord, rn, cad)
        assert out.shape == (2, 6, 24)


# ---------------------------------------------------------------------------
# TestBlockValidation
# ---------------------------------------------------------------------------


class TestBlockValidation:
    def test_block_rejects_indivisible_d_model(self):
        with pytest.raises(ValueError, match="divisible"):
            MusicTransformerBlock(d_model=10, n_heads=4)

    def test_block_forward_preserves_shape(self):
        from src.models.music_transformer import _causal_mask
        block = MusicTransformerBlock(d_model=24, n_heads=4, d_ff=48, dropout=0.0)
        block.eval()
        x = torch.randn(2, 5, 24)
        cm = _causal_mask(5, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            y = block(x, causal_mask=cm)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# TestGradientFlow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_loss_backward_produces_finite_grads(self):
        """Smoke-test: training-style step yields finite gradients on every
        parameter — catches dimension-mismatch / NaN-init regressions."""
        model = _tiny_model()
        model.train()
        chord, rn, cad = _random_batch(2, 8)
        targets = chord.clone()
        logits = model(chord, rn, cad)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, CHORD_V), targets.reshape(-1)
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no grad on {name}"
                assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
