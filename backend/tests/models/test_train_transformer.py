"""
Tests for :mod:`src.models.train_transformer`.

Uses tiny vocab sizes (chord=10, rn=15, cadence=5), a 2-layer × 24-wide
model, and an in-memory synthetic dataset so the whole file runs in
well under ten seconds on CPU. No real corpus data is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.music_transformer import MusicTransformer
from src.models.train_transformer import (
    TrainerConfig,
    TransformerTrainer,
    _chord_accuracy,
    _compute_ce_loss,
    _compute_specials_penalty,
    make_warmup_cosine_scheduler,
    select_device,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


CHORD_V = 10
RN_V = 15
CAD_V = 5


class _ToyDataset(Dataset):
    """Synthetic next-chord dataset.

    Every item is a length-``T`` window whose chord-target equals a
    deterministic shift of chord-input — a pattern a tiny transformer
    should pick up in 3–5 epochs, which is what the "loss decreases"
    test checks.
    """

    def __init__(self, n_items: int, T: int, seed: int = 0) -> None:
        self._items: List[Dict[str, torch.Tensor]] = []
        g = torch.Generator().manual_seed(seed)
        for _ in range(n_items):
            # Start ids at 5 for chord/rn to avoid emitting specials as
            # targets (which CE would ignore anyway, but clearer this way).
            chord = torch.randint(5, CHORD_V, (T + 1,), generator=g)
            rn = torch.randint(5, RN_V, (T + 1,), generator=g)
            # Cadence vocab only has 5 slots total (ids 0..4), so we
            # sample from the full range — ``4`` (CAD_NONE) is the common
            # case in real data.
            cad = torch.randint(0, CAD_V, (T + 1,), generator=g)
            self._items.append({
                "chord_input":    chord[:-1].long(),
                "rn_input":       rn[:-1].long(),
                "cadence_input":  cad[:-1].long(),
                "chord_target":   chord[1:].long(),
                "attention_mask": torch.ones(T, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._items[idx]


def _tiny_model() -> MusicTransformer:
    return MusicTransformer(
        chord_vocab_size=CHORD_V,
        rn_vocab_size=RN_V,
        cadence_vocab_size=CAD_V,
        d_model=24, n_heads=4, n_layers=2, d_ff=48,
        dropout=0.0, max_seq_len=32,
    )


def _loaders(n_train: int = 8, n_val: int = 4, T: int = 8, batch_size: int = 4):
    return (
        DataLoader(_ToyDataset(n_train, T, seed=0), batch_size=batch_size, shuffle=True),
        DataLoader(_ToyDataset(n_val, T, seed=1),   batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# TestDeviceSelection
# ---------------------------------------------------------------------------


class TestDeviceSelection:
    def test_cpu_override_respected(self):
        assert select_device("cpu") == torch.device("cpu")

    def test_auto_select_returns_torch_device(self):
        dev = select_device()
        # Any of the three backends is acceptable depending on host.
        assert isinstance(dev, torch.device)
        assert dev.type in {"cpu", "cuda", "mps"}


# ---------------------------------------------------------------------------
# TestScheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_warmup_then_cosine_decay(self):
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = make_warmup_cosine_scheduler(opt, warmup_steps=10, total_steps=100)

        lrs: List[float] = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        # Step 0 → ~0; step 10 → ~1.0; step 100 → ~0.
        assert lrs[0] < 0.2
        assert 0.8 < lrs[10] <= 1.0
        assert lrs[-1] < 0.2
        # Post-warmup peak ≥ any post-peak LR.
        peak = max(lrs)
        peak_idx = lrs.index(peak)
        assert peak_idx >= 9
        assert all(lr <= peak + 1e-9 for lr in lrs[peak_idx:])

    def test_scheduler_zero_warmup_starts_at_full_lr(self):
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = make_warmup_cosine_scheduler(opt, warmup_steps=0, total_steps=50)
        # At step 0 with no warmup, cosine(0) = 1.0.
        assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestLossHelpers
# ---------------------------------------------------------------------------


class TestLossHelpers:
    def test_ce_ignores_padding_positions(self):
        # Two positions, one PAD — loss must equal the CE of the single
        # non-pad position in isolation.
        logits = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 0.0, 10.0]]])
        targets = torch.tensor([[1, 0]])  # second is PAD
        loss_full = _compute_ce_loss(logits, targets, pad_id=0).item()
        loss_solo = torch.nn.functional.cross_entropy(
            logits[:, 0, :], targets[:, 0]
        ).item()
        assert abs(loss_full - loss_solo) < 1e-6

    def test_specials_penalty_rewards_low_special_mass(self):
        # Logits that place almost all mass on id=2 (a real chord) at
        # both positions — penalty should be near zero.
        good = torch.full((1, 2, 4), -10.0)
        good[0, :, 2] = 10.0
        targets = torch.tensor([[2, 2]])
        pen_good = _compute_specials_penalty(good, targets).item()

        # Logits that place all mass on PAD=0 — penalty should be ≈1.
        bad = torch.full((1, 2, 4), -10.0)
        bad[0, :, 0] = 10.0
        pen_bad = _compute_specials_penalty(bad, targets).item()

        assert pen_good < 0.01
        assert pen_bad > 0.9

    def test_chord_accuracy_ignores_pad(self):
        # Correct on pos 0, wrong on pos 1, PAD on pos 2 → 1 / 2 = 0.5.
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 1] = 10.0    # argmax 1
        logits[0, 1, 0] = 10.0    # argmax 0
        logits[0, 2, 3] = 10.0    # argmax 3 (but pos is pad)
        targets = torch.tensor([[1, 2, 0]])  # last is PAD
        correct, total = _chord_accuracy(logits, targets)
        assert correct == 1
        assert total == 2


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_runs_and_drops_training_loss(self, tmp_path):
        model = _tiny_model()
        train_loader, val_loader = _loaders(n_train=8, n_val=4, T=8, batch_size=4)
        config = TrainerConfig(
            epochs=5, lr=5e-3, warmup_steps=2, patience=10, log_every=0, seed=0,
        )
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "run",
            config=config,
            device="cpu",
        )
        results = trainer.fit()
        history = results["history"]
        assert len(history) == 5
        assert history[-1]["train_loss"] < history[0]["train_loss"]

    def test_best_checkpoint_is_self_describing(self, tmp_path):
        model = _tiny_model()
        train_loader, val_loader = _loaders(n_train=6, n_val=4, T=6, batch_size=3)
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "run",
            config=TrainerConfig(epochs=2, warmup_steps=1, log_every=0, patience=10),
            device="cpu",
        )
        trainer.fit()

        best = tmp_path / "run" / "best.pt"
        last = tmp_path / "run" / "last.pt"
        assert best.exists()
        assert last.exists()

        reloaded, extra = MusicTransformer.load_checkpoint(best)
        assert reloaded.hyperparameters == model.hyperparameters
        assert reloaded.vocab_config == model.vocab_config
        assert "epoch" in extra and "val_loss" in extra

    def test_training_log_has_one_row_per_epoch(self, tmp_path):
        model = _tiny_model()
        train_loader, val_loader = _loaders(n_train=6, n_val=4, T=6, batch_size=3)
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "run",
            config=TrainerConfig(epochs=3, warmup_steps=1, log_every=0, patience=10),
            device="cpu",
        )
        trainer.fit()
        log = (tmp_path / "run" / "training_log.jsonl").read_text().splitlines()
        assert len(log) == 3
        for line in log:
            row = json.loads(line)
            assert {"epoch", "train_loss", "val_loss", "val_ppl",
                    "val_chord_accuracy", "lr"}.issubset(row.keys())

    def test_early_stopping_triggers(self, tmp_path):
        """With patience=1 and a tiny corpus, the run is very likely to
        plateau (or oscillate) within a few epochs — assert the trainer
        honours the patience budget."""
        model = _tiny_model()
        train_loader, val_loader = _loaders(n_train=4, n_val=4, T=6, batch_size=2)
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "run",
            config=TrainerConfig(
                epochs=30, warmup_steps=1, patience=1, log_every=0,
                min_delta=0.05,  # require a healthy drop each epoch
            ),
            device="cpu",
        )
        results = trainer.fit()
        assert results["stopped_early"] is True
        assert len(results["history"]) < 30


# ---------------------------------------------------------------------------
# TestGradientClipping
# ---------------------------------------------------------------------------


class TestGradientClipping:
    def test_clip_is_applied_during_step(self, tmp_path, monkeypatch):
        """Patch clip_grad_norm_ to observe that the trainer calls it
        with the configured max_norm on every training batch."""
        calls: List[float] = []

        import torch.nn.utils
        original = torch.nn.utils.clip_grad_norm_

        def spy(parameters, max_norm, *args, **kwargs):
            calls.append(float(max_norm))
            return original(parameters, max_norm, *args, **kwargs)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

        model = _tiny_model()
        train_loader, val_loader = _loaders(n_train=4, n_val=2, T=6, batch_size=2)
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "run",
            config=TrainerConfig(
                epochs=1, warmup_steps=1, patience=10, log_every=0,
                grad_clip_norm=0.75,
            ),
            device="cpu",
        )
        trainer.fit()
        # One call per training batch in epoch 1.
        assert len(calls) == len(train_loader)
        assert all(abs(c - 0.75) < 1e-9 for c in calls)


# ---------------------------------------------------------------------------
# TestGrammarWeight
# ---------------------------------------------------------------------------


class TestGrammarWeight:
    def test_zero_weight_path_is_pure_ce(self, tmp_path):
        """With grammar_weight=0 the aux term is inert — training should
        just be vanilla CE. Sanity-check by confirming the loss values
        in the log are finite and match the CE scale."""
        model = _tiny_model()
        train_loader, val_loader = _loaders(n_train=4, n_val=2, T=6, batch_size=2)
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "run",
            config=TrainerConfig(
                epochs=1, warmup_steps=1, patience=10, log_every=0,
                grammar_weight=0.0,
            ),
            device="cpu",
        )
        res = trainer.fit()
        assert res["history"][0]["train_loss"] > 0
        assert res["history"][0]["train_loss"] < 20

    def test_positive_weight_path_runs(self, tmp_path):
        model = _tiny_model()
        train_loader, val_loader = _loaders(n_train=4, n_val=2, T=6, batch_size=2)
        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "run",
            config=TrainerConfig(
                epochs=1, warmup_steps=1, patience=10, log_every=0,
                grammar_weight=0.1,
            ),
            device="cpu",
        )
        res = trainer.fit()
        assert len(res["history"]) == 1


# ---------------------------------------------------------------------------
# TestConfigSerialisation
# ---------------------------------------------------------------------------


class TestConfigSerialisation:
    def test_config_to_dict_is_json_serialisable(self):
        cfg = TrainerConfig(epochs=3, lr=1e-4, grammar_weight=0.05)
        d = cfg.to_dict()
        # Will raise if any value is non-serialisable.
        json.dumps(d)
        assert d["epochs"] == 3
        assert d["lr"] == 1e-4
        assert d["grammar_weight"] == 0.05
