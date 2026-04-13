"""
Smoke tests for :mod:`scripts.train_transformer` and
:mod:`scripts.verify_checkpoint`.

We bypass the real ~200-timestep chorale corpus entirely: the fixture
writes a handful of synthetic sequences and a minimal ``vocab.json`` to
``tmp_path``. A two-epoch run against a tiny transformer completes in a
couple of seconds on CPU.

Structure
---------
TestTrainTransformerCLI     — argparse defaults, end-to-end training
TestVerifyCheckpointCLI     — loads & dry-runs a saved checkpoint,
                              rejects schema mismatches
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure scripts.* imports resolve under pytest.
_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts import train_transformer, verify_checkpoint
from src.data.harmonic_encoder import BOS, CAD_NONE, EOS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seq(score_id: str, length: int) -> dict:
    body_chord = list(range(5, 5 + max(length - 2, 0)))
    body_rn = list(range(10, 10 + max(length - 2, 0)))
    body_cad = [CAD_NONE] * max(length - 2, 0)
    return {
        "score_id": score_id,
        "chord_tokens":   [BOS, *body_chord, EOS],
        "rn_tokens":      [BOS, *body_rn, EOS],
        "cadence_tokens": [BOS, *body_cad, EOS],
        "onsets":         [""] + [str(i) for i in range(len(body_chord))] + [""],
    }


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal (jsonl, vocab.json) pair to tmp_path."""
    jsonl = tmp_path / "encoded_sequences.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        # 30 distinct score_ids to populate train / val / test buckets.
        for i in range(30):
            f.write(json.dumps(_seq(f"bwv{i}.6", 24)) + "\n")

    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({
        "schema_version": "1.0.0",
        "specials": {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3, "CAD_NONE": 4},
        "chord_to_id":  {f"c{i}": i for i in range(40)},
        "rn_to_id":     {f"r{i}": i for i in range(50)},
        "cadence_to_id": {f"k{i}": i for i in range(8)},
    }))
    return jsonl, vocab


# ---------------------------------------------------------------------------
# TestTrainTransformerCLI
# ---------------------------------------------------------------------------


class TestTrainTransformerCLI:
    def test_parser_has_expected_defaults(self):
        p = train_transformer.build_parser()
        args = p.parse_args([])
        assert args.epochs == 60
        assert args.batch_size == 32
        assert args.lr == 1e-4
        assert args.d_model == 384
        assert args.n_layers == 6
        assert args.warmup_steps == 500
        assert args.device is None

    def test_rejects_missing_jsonl(self, tmp_path, capsys):
        rc = train_transformer.main([
            "--jsonl", str(tmp_path / "nope.jsonl"),
            "--vocab", str(tmp_path / "nope.json"),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
            "--device", "cpu",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "jsonl" in err or "not found" in err

    def test_end_to_end_tiny_run(self, synthetic_corpus, tmp_path):
        jsonl, vocab = synthetic_corpus
        out = tmp_path / "run"
        rc = train_transformer.main([
            "--jsonl", str(jsonl),
            "--vocab", str(vocab),
            "--output-dir", str(out),
            "--epochs", "1",
            "--batch-size", "4",
            "--window-size", "8",
            "--stride", "4",
            "--d-model", "24",
            "--n-heads", "4",
            "--n-layers", "2",
            "--d-ff", "48",
            "--dropout", "0.0",
            "--max-seq-len", "32",
            "--warmup-steps", "1",
            "--patience", "10",
            "--device", "cpu",
        ])
        assert rc == 0
        assert (out / "best.pt").exists()
        assert (out / "last.pt").exists()
        assert (out / "training_log.jsonl").exists()


# ---------------------------------------------------------------------------
# TestVerifyCheckpointCLI
# ---------------------------------------------------------------------------


class TestVerifyCheckpointCLI:
    def _produce_checkpoint(self, synthetic_corpus, tmp_path) -> Path:
        jsonl, vocab = synthetic_corpus
        out = tmp_path / "run"
        train_transformer.main([
            "--jsonl", str(jsonl),
            "--vocab", str(vocab),
            "--output-dir", str(out),
            "--epochs", "1",
            "--batch-size", "4",
            "--window-size", "8",
            "--stride", "4",
            "--d-model", "24", "--n-heads", "4",
            "--n-layers", "2", "--d-ff", "48",
            "--dropout", "0.0", "--max-seq-len", "32",
            "--warmup-steps", "1", "--patience", "10",
            "--device", "cpu",
        ])
        return out / "best.pt"

    def test_verify_reports_and_dry_runs(
        self, synthetic_corpus, tmp_path, capsys
    ):
        ckpt = self._produce_checkpoint(synthetic_corpus, tmp_path)
        rc = verify_checkpoint.main([str(ckpt), "--device", "cpu"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "schema_version" in out
        assert "parameters" in out
        assert "forward pass:" in out

    def test_verify_missing_file_exits_2(self, tmp_path):
        rc = verify_checkpoint.main([str(tmp_path / "missing.pt")])
        assert rc == 2

    def test_verify_rejects_schema_mismatch(
        self, synthetic_corpus, tmp_path
    ):
        ckpt = self._produce_checkpoint(synthetic_corpus, tmp_path)
        import torch
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        raw["schema_version"] = "0.0.0-bogus"
        torch.save(raw, ckpt)
        rc = verify_checkpoint.main([str(ckpt), "--device", "cpu"])
        assert rc == 3
