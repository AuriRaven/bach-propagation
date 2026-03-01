"""
Training loop for BaroqueHarmonyLSTM.

Public API
----------
train_epoch    — single training epoch, returns average loss
validate_epoch — single validation epoch, returns average loss
train          — full training loop with early stopping

Loss function: nn.CrossEntropyLoss(ignore_index=0) — ignores PAD tokens.
Optimiser: Adam, lr=1e-3.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import HarmonyDataset
from src.models.baseline_lstm import BaroqueHarmonyLSTM


def train_epoch(
    model: BaroqueHarmonyLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Run one full training epoch.

    Returns:
        Average cross-entropy loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        logits, _ = model(inp)                             # (B, T, vocab_size)
        # CrossEntropyLoss expects (B, C, T) or flattened
        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), tgt.view(B * T))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def validate_epoch(
    model: BaroqueHarmonyLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Run one full validation epoch (no gradient computation).

    Returns:
        Average cross-entropy loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inp, tgt in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)

            logits, _ = model(inp)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), tgt.view(B * T))
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def train(
    model: BaroqueHarmonyLSTM,
    train_dataset: HarmonyDataset,
    val_dataset: HarmonyDataset,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = 'cpu',
    checkpoint_path: Optional[str] = 'model_best.pt',
    verbose: bool = True,
) -> dict:
    """
    Full training loop with early stopping.

    Saves the best model (by validation loss) to *checkpoint_path* if provided.

    Args:
        model:            BaroqueHarmonyLSTM instance.
        train_dataset:    Training HarmonyDataset.
        val_dataset:      Validation HarmonyDataset.
        epochs:           Maximum number of training epochs.
        batch_size:       DataLoader batch size.
        lr:               Adam learning rate.
        patience:         Early stopping patience (epochs without improvement).
        device:           'cpu' or 'cuda'.
        checkpoint_path:  File path for the best model checkpoint; None to skip saving.
        verbose:          Print epoch summaries.

    Returns:
        {
          'train_losses': list[float],
          'val_losses':   list[float],
          'best_epoch':   int,
          'best_val_loss': float,
        }
    """
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f}"
                  + (" *" if epochs_no_improve == 0 else ""))

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }
