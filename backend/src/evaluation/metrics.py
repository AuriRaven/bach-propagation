"""
Evaluation metrics for the BaroqueHarmonyLSTM.

next_chord_accuracy      — token-level accuracy (non-PAD positions)
function_accuracy        — harmonic function accuracy via majority-vote map
perplexity               — exp(avg cross-entropy loss)
build_transition_matrix  — empirical chord→chord transition probabilities
transition_validity_rate — % of predicted transitions seen in training corpus
random_baseline          — random-predictor reference values
evaluate_all             — run all metrics and return a results dict
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.encoder import Encoder, TOTAL_VOCAB, PAD
from src.models.baseline_lstm import BaroqueHarmonyLSTM


def next_chord_accuracy(
    model: BaroqueHarmonyLSTM,
    data_loader: DataLoader,
    pad_idx: int = PAD,
    device: str = 'cpu',
) -> float:
    """
    Proportion of non-PAD target positions where argmax(logit) == target.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inp, tgt in data_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            logits, _ = model(inp)                  # (B, T, vocab_size)
            preds = logits.argmax(dim=-1)           # (B, T)
            mask = tgt != pad_idx
            correct += (preds[mask] == tgt[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0


def function_accuracy(
    model: BaroqueHarmonyLSTM,
    data_loader: DataLoader,
    encoder: Encoder,
    pad_idx: int = PAD,
    device: str = 'cpu',
) -> float:
    """
    Proportion of non-PAD positions where function(predicted_token) == function(target_token).

    Uses ``encoder.token_function()`` majority-vote map. Call
    ``encoder.build_chord_function_map()`` before using this metric.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inp, tgt in data_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            logits, _ = model(inp)
            preds = logits.argmax(dim=-1)
            mask = tgt != pad_idx

            pred_flat = preds[mask].cpu().tolist()
            tgt_flat = tgt[mask].cpu().tolist()
            for p, t in zip(pred_flat, tgt_flat):
                if encoder.token_function(p) == encoder.token_function(t):
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def perplexity(
    model: BaroqueHarmonyLSTM,
    data_loader: DataLoader,
    pad_idx: int = PAD,
    device: str = 'cpu',
) -> float:
    """
    Compute perplexity = exp(average cross-entropy loss over non-PAD tokens).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inp, tgt in data_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            logits, _ = model(inp)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), tgt.view(B * T))
            total_loss += loss.item()
            total_tokens += (tgt != pad_idx).sum().item()
    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def build_transition_matrix(
    chord_sequences: List[List[int]],
    vocab_size: int = TOTAL_VOCAB,
) -> Dict[Tuple[int, int], float]:
    """
    Count chord→chord bigram transitions and normalise rows to probabilities.

    Args:
        chord_sequences: Lists of integer token sequences (e.g. output of
                         Encoder.encode_sequence, including START/END).
        vocab_size:      Size of the vocabulary (for validation; unused in dict).

    Returns:
        ``{(from_token, to_token): probability}`` where each from_token row sums to 1.
        Returns an empty dict if chord_sequences is empty.
    """
    counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for seq in chord_sequences:
        for i in range(len(seq) - 1):
            counts[seq[i]][seq[i + 1]] += 1

    matrix: Dict[Tuple[int, int], float] = {}
    for from_tok, to_counts in counts.items():
        row_total = sum(to_counts.values())
        for to_tok, cnt in to_counts.items():
            matrix[(from_tok, to_tok)] = cnt / row_total

    return matrix


def transition_validity_rate(
    model: BaroqueHarmonyLSTM,
    data_loader: DataLoader,
    transition_matrix: Dict[Tuple[int, int], float],
    pad_idx: int = PAD,
    device: str = 'cpu',
    threshold: float = 0.01,
) -> float:
    """
    Proportion of consecutive (pred_t, pred_{t+1}) pairs that appear in
    ``transition_matrix`` with probability > *threshold*.
    """
    model.eval()
    valid = 0
    total = 0
    with torch.no_grad():
        for inp, tgt in data_loader:
            inp = inp.to(device)
            logits, _ = model(inp)
            preds = logits.argmax(dim=-1)  # (B, T)
            tgt = tgt.to(device)

            for b in range(preds.shape[0]):
                seq = preds[b].cpu().tolist()
                tgt_seq = tgt[b].cpu().tolist()
                for t in range(len(seq) - 1):
                    if tgt_seq[t] == pad_idx or tgt_seq[t + 1] == pad_idx:
                        continue
                    prob = transition_matrix.get((seq[t], seq[t + 1]), 0.0)
                    if prob > threshold:
                        valid += 1
                    total += 1

    return valid / total if total > 0 else 0.0


def random_baseline(
    vocab_size: int,
    chord_sequences: List[List[int]],
    pad_idx: int = PAD,
) -> dict:
    """
    Compute reference metrics for a uniformly-random predictor.

    Returns:
        {
            'next_chord_accuracy':    float,
            'function_accuracy':      float,
            'transition_validity_rate': float,
            'perplexity':             float,
        }
    """
    # next-chord accuracy: 1/vocab_size
    nca = 1.0 / vocab_size

    # function accuracy: any of 3 function labels equally likely → 1/3 correct
    # (assuming roughly equal distribution of 3 functions in the reference)
    fn_acc = 1.0 / 3

    # transition validity rate: transitions are distributed uniformly at random
    # so a random model has the same validity rate as the empirical coverage
    matrix = build_transition_matrix(chord_sequences, vocab_size)
    valid_pairs = sum(1 for p in matrix.values() if p > 0.01)
    total_pairs = vocab_size * vocab_size
    tvr = valid_pairs / total_pairs if total_pairs > 0 else 0.0

    # perplexity of uniform distribution: vocab_size
    ppl = float(vocab_size)

    return {
        'next_chord_accuracy': nca,
        'function_accuracy': fn_acc,
        'transition_validity_rate': tvr,
        'perplexity': ppl,
    }


def evaluate_all(
    model: BaroqueHarmonyLSTM,
    data_loader: DataLoader,
    encoder: Encoder,
    transition_matrix: Dict[Tuple[int, int], float],
    pad_idx: int = PAD,
    device: str = 'cpu',
) -> dict:
    """
    Run all metrics and return a combined results dict.

    Returns:
        {
            'next_chord_accuracy':     float,
            'function_accuracy':       float,
            'perplexity':              float,
            'transition_validity_rate': float,
        }
    """
    return {
        'next_chord_accuracy': next_chord_accuracy(model, data_loader, pad_idx, device),
        'function_accuracy': function_accuracy(model, data_loader, encoder, pad_idx, device),
        'perplexity': perplexity(model, data_loader, pad_idx, device),
        'transition_validity_rate': transition_validity_rate(
            model, data_loader, transition_matrix, pad_idx, device
        ),
    }
