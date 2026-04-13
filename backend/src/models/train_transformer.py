"""
Training loop for :class:`src.models.music_transformer.MusicTransformer`.

Public API
----------

:class:`TrainerConfig`
    Dataclass of every knob the trainer exposes. Defaults are tuned for
    the Bach-chorale corpus on an Apple-Silicon M4 Pro.

:class:`TransformerTrainer`
    Owns the model + loaders + optimiser + scheduler + early stopping.
    Call :meth:`fit` to run the full loop; returns a history dict.

Design
------

* **Optimiser:** ``AdamW`` (decoupled weight decay).
* **Schedule:** linear warmup over ``warmup_steps`` training batches,
  then cosine decay to zero by the end of training.
* **Loss:** token-level cross-entropy on the chord stream only, with
  ``ignore_index=PAD`` so padded positions don't contribute.
* **Grammar auxiliary loss (optional).** A small differentiable penalty
  that pushes the model away from emitting specials (``PAD`` / ``UNK``)
  at real target positions. Enabled by setting ``grammar_weight > 0``.
  It is **not** a full tonal-grammar critic — that work lives in
  :mod:`src.grammar.validator` and runs at generation time. This hook
  exists so future aux-loss experiments have a plumbed entry point.
* **Checkpoints.** Best (lowest val loss) and last (every epoch) are
  written via :meth:`MusicTransformer.save_checkpoint`, so the files
  are self-describing (hyperparameters + vocab_config embedded).
* **Device selection.** MPS → CUDA → CPU, or an explicit override.
* **Gradient clipping.** Global L2 norm via
  :func:`torch.nn.utils.clip_grad_norm_`.
* **Early stopping.** Patience-based on val loss; the loop aborts if
  the best val loss hasn't improved for ``patience`` epochs.

The training loop writes one JSON line per epoch to
``{output_dir}/training_log.jsonl`` for offline analysis.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.harmonic_encoder import PAD, UNK
from src.models.music_transformer import MusicTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """All user-tunable training hyperparameters.

    Defaults target a 384-wide × 6-layer transformer on an M4 Pro.
    """

    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    patience: int = 5
    #: Weight on the aux (specials) penalty; ``0`` disables it.
    grammar_weight: float = 0.0
    #: Print a progress line every ``log_every`` training batches.
    log_every: int = 50
    seed: int = 42
    #: Minimum relative improvement in val loss to count as "better".
    min_delta: float = 1e-4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Return a torch device following the MPS→CUDA→CPU fallback chain.

    Args:
        preferred: ``"mps"``, ``"cuda"``, or ``"cpu"`` to pin the choice.
            ``None`` (the default) lets the function pick the best
            available backend.
    """
    if preferred is not None:
        return torch.device(preferred)
    # MPS first — matches the user's M4 Pro workflow.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup for ``warmup_steps`` then cosine decay to ``min_lr_ratio``.

    Step 0 → LR 0; step ``warmup_steps`` → full LR; step ``total_steps``
    → ``min_lr_ratio * full_lr``. Intermediate values interpolate on a
    cosine curve.
    """
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _compute_ce_loss(
    logits: Tensor, targets: Tensor, pad_id: int = PAD
) -> Tensor:
    """Token-level cross-entropy, ignoring ``pad_id`` positions.

    ``logits``: ``(B, T, V)``. ``targets``: ``(B, T)``.
    """
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=pad_id,
    )


def _compute_specials_penalty(
    logits: Tensor,
    targets: Tensor,
    pad_id: int = PAD,
    unk_id: int = UNK,
) -> Tensor:
    """Average probability mass placed on ``PAD`` / ``UNK`` at non-pad targets.

    The model predicts a full chord distribution at every position. At
    real target positions it has no reason to assign any mass to the
    special-token ids — doing so is a cheap shortcut toward a lower
    softmax entropy. This penalty nudges the distribution away from the
    specials without putting any hard prior on which real chord to pick.
    """
    probs = torch.softmax(logits, dim=-1)          # (B, T, V)
    # Sum over the special ids — guard against vocab_size <= max(id).
    V = probs.size(-1)
    ids = [i for i in (pad_id, unk_id) if 0 <= i < V]
    if not ids:
        return logits.new_tensor(0.0)
    specials_mass = probs[..., ids].sum(dim=-1)    # (B, T)
    valid = (targets != pad_id).float()            # (B, T)
    denom = valid.sum().clamp_min(1.0)
    return (specials_mass * valid).sum() / denom


def _chord_accuracy(
    logits: Tensor, targets: Tensor, pad_id: int = PAD
) -> Tuple[int, int]:
    """Return ``(correct, total)`` counts for top-1 chord prediction,
    ignoring positions where ``target == pad_id``."""
    preds = logits.argmax(dim=-1)
    mask = targets != pad_id
    correct = int((preds[mask] == targets[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


# ---------------------------------------------------------------------------
# TransformerTrainer
# ---------------------------------------------------------------------------


class TransformerTrainer:
    """Owns a :class:`MusicTransformer` plus the machinery to train it.

    Args:
        model: A constructed :class:`MusicTransformer`.
        train_loader: DataLoader yielding dicts with keys ``chord_input``,
            ``rn_input``, ``cadence_input``, ``chord_target``,
            ``attention_mask``.
        val_loader: DataLoader over the validation split, same schema.
        output_dir: Directory where ``best.pt`` / ``last.pt`` and
            ``training_log.jsonl`` will be written. Created if missing.
        config: :class:`TrainerConfig`; defaults applied when ``None``.
        device: Pinned device, or ``None`` to auto-select.
    """

    def __init__(
        self,
        model: MusicTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path,
        config: Optional[TrainerConfig] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = (
            torch.device(device)
            if isinstance(device, (str, torch.device))
            else select_device()
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self._seed_everything(self.config.seed)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        total_steps = max(1, len(train_loader)) * max(1, self.config.epochs)
        self.scheduler = make_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
        )

        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._epochs_without_improvement = 0
        self._global_step = 0

    # ------------------------------------------------------------------
    # Public loop
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, Any]:
        """Run the training loop until convergence or early stopping.

        Returns:
            ``{"history": [...], "best_epoch": int, "best_val_loss": float,
               "best_checkpoint": str, "stopped_early": bool}``.
        """
        log_path = self.output_dir / "training_log.jsonl"
        history: List[Dict[str, Any]] = []
        stopped_early = False

        print(
            f"[trainer] device={self.device} "
            f"params={self.model.count_parameters():,} "
            f"train_batches={len(self.train_loader)} "
            f"val_batches={len(self.val_loader)}"
        )

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.monotonic()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._eval_epoch()
            dt = time.monotonic() - t0

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_ppl": val_metrics["ppl"],
                "val_chord_accuracy": val_metrics["chord_accuracy"],
                "lr": self._current_lr(),
                "epoch_seconds": round(dt, 2),
            }
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            improved = val_metrics["loss"] < self._best_val_loss - self.config.min_delta
            marker = " *" if improved else ""
            print(
                f"[epoch {epoch:>3}/{self.config.epochs}] "
                f"train={train_metrics['loss']:.4f} "
                f"val={val_metrics['loss']:.4f} "
                f"ppl={val_metrics['ppl']:.2f} "
                f"acc={val_metrics['chord_accuracy']:.3f} "
                f"lr={self._current_lr():.2e} "
                f"time={dt:.1f}s{marker}"
            )

            self._save_last(epoch=epoch, extra=row)
            if improved:
                self._best_val_loss = val_metrics["loss"]
                self._best_epoch = epoch
                self._epochs_without_improvement = 0
                self._save_best(epoch=epoch, extra=row)
            else:
                self._epochs_without_improvement += 1

            if self._epochs_without_improvement >= self.config.patience:
                print(
                    f"[trainer] early stop at epoch {epoch} "
                    f"(no improvement for {self.config.patience} epochs)"
                )
                stopped_early = True
                break

        return {
            "history": history,
            "best_epoch": self._best_epoch,
            "best_val_loss": self._best_val_loss,
            "best_checkpoint": str(self.output_dir / "best.pt"),
            "last_checkpoint": str(self.output_dir / "last.pt"),
            "stopped_early": stopped_early,
        }

    # ------------------------------------------------------------------
    # Inner loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(self.train_loader, start=1):
            loss = self._step(batch, backward=True)
            total_loss += loss
            n_batches += 1
            self._global_step += 1
            if self.config.log_every and (i % self.config.log_every == 0):
                print(
                    f"  [train] epoch={epoch} batch={i}/{len(self.train_loader)} "
                    f"loss={loss:.4f} lr={self._current_lr():.2e}"
                )
        return {"loss": total_loss / max(1, n_batches)}

    def _eval_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                logits, targets = self._forward(batch)
                loss = _compute_ce_loss(logits, targets)
                total_loss += float(loss.item())
                n_batches += 1
                c, t = _chord_accuracy(logits, targets)
                correct += c
                total_tokens += t
        avg_loss = total_loss / max(1, n_batches)
        ppl = math.exp(min(20.0, avg_loss))   # clamp to avoid overflow on early epochs
        acc = correct / total_tokens if total_tokens > 0 else 0.0
        return {"loss": avg_loss, "ppl": ppl, "chord_accuracy": acc}

    def _step(self, batch: Dict[str, Tensor], backward: bool) -> float:
        logits, targets = self._forward(batch)
        ce = _compute_ce_loss(logits, targets)
        loss = ce
        if self.config.grammar_weight > 0.0:
            aux = _compute_specials_penalty(logits, targets)
            loss = loss + self.config.grammar_weight * aux
        if backward:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
            self.optimizer.step()
            self.scheduler.step()
        return float(loss.item())

    def _forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        chord = batch["chord_input"].to(self.device, non_blocking=True)
        rn = batch["rn_input"].to(self.device, non_blocking=True)
        cad = batch["cadence_input"].to(self.device, non_blocking=True)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)
        targets = batch["chord_target"].to(self.device, non_blocking=True)
        logits = self.model(chord, rn, cad, attention_mask=mask)
        return logits, targets

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_best(self, epoch: int, extra: Dict[str, Any]) -> None:
        self.model.save_checkpoint(self.output_dir / "best.pt", extra=dict(extra))

    def _save_last(self, epoch: int, extra: Dict[str, Any]) -> None:
        self.model.save_checkpoint(self.output_dir / "last.pt", extra=dict(extra))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


__all__ = [
    "TrainerConfig",
    "TransformerTrainer",
    "make_warmup_cosine_scheduler",
    "select_device",
]
