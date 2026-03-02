"""
Tests for src/analysis/embedding_analysis.py

TestEmbeddingAnalysis
---------------------
  test_embeddings_shape              extracted embeddings are (60, 64)
  test_function_labels_coverage      all 60 tokens have a function/quality label
  test_pca_reduces_correctly         output is (60, 2) with explained variance > 0
  test_cluster_count                 k-means with k=3 produces exactly 3 unique labels
  test_purity_bounds                 purity is between 0 and 1
  test_purity_random_lower_than_model  k-means purity > random on well-separated data
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.embedding_analysis import (
    cluster_embeddings,
    compute_cluster_purity,
    extract_chord_embeddings,
    reduce_dimensions,
)
from src.data.encoder import CHORD_VOCAB_SIZE, Encoder
from src.models.baseline_lstm import BaroqueHarmonyLSTM


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    """Save a freshly-initialised BaroqueHarmonyLSTM to a temp file."""
    path = tmp_path_factory.mktemp('models') / 'test_model.pt'
    model = BaroqueHarmonyLSTM()
    torch.save(model.state_dict(), path)
    return str(path)


@pytest.fixture(scope='module')
def encoder():
    """Encoder with a chord→function map that covers all 60 chord tokens."""
    enc = Encoder()
    qualities = ['major', 'minor', 'diminished', 'augmented', 'dominant7']
    sequences = []
    for root in range(12):
        for q_idx, quality in enumerate(qualities):
            fn = (root * 5 + q_idx) % 3  # deterministic T/PD/D spread
            sequences.append([{
                'chord_root': root,
                'chord_quality': quality,
                'harmonic_function': fn,
            }])
    enc.build_chord_function_map(sequences)
    return enc


@pytest.fixture(scope='module')
def extracted(checkpoint, encoder):
    """Run extract_chord_embeddings once and share the result."""
    return extract_chord_embeddings(checkpoint, encoder)


@pytest.fixture(scope='module')
def embeddings(extracted):
    return extracted[0]


@pytest.fixture(scope='module')
def function_ids(extracted):
    return extracted[2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbeddingAnalysis:

    def test_embeddings_shape(self, embeddings):
        assert embeddings.shape == (60, 64)

    def test_function_labels_coverage(self, extracted):
        _, token_labels, function_ids, quality_labels = extracted
        assert len(token_labels) == CHORD_VOCAB_SIZE
        assert len(function_ids) == CHORD_VOCAB_SIZE
        assert len(quality_labels) == CHORD_VOCAB_SIZE
        # Every function_id must be a valid int in {0, 1, 2}
        assert all(fid in {0, 1, 2} for fid in function_ids)

    def test_pca_reduces_correctly(self, embeddings):
        reduced, pca = reduce_dimensions(embeddings, method='pca', n_components=2)
        assert reduced.shape == (60, 2)
        assert sum(pca.explained_variance_ratio_) > 0

    def test_cluster_count(self, embeddings):
        clusters = cluster_embeddings(embeddings, k_values=[3])
        assert len(np.unique(clusters[3])) == 3

    def test_purity_bounds(self, embeddings, function_ids):
        clusters = cluster_embeddings(embeddings, k_values=[3])
        true_fn = np.array(function_ids, dtype=int)
        purity, breakdown = compute_cluster_purity(clusters[3], true_fn)
        assert 0.0 <= purity <= 1.0
        # ARI is stored separately inside breakdown before being popped in script;
        # here breakdown still has _ari because we call compute_cluster_purity directly
        assert '_ari' in breakdown
        ari = breakdown['_ari']
        assert -1.0 <= ari <= 1.0

    def test_purity_random_lower_than_model(self):
        """
        On clearly separated data k-means purity must exceed random assignment.

        We synthesise 60 points arranged in 3 tight Gaussian clusters (20 each)
        so k=3 k-means should recover near-perfect purity regardless of whether
        a real trained checkpoint is available.
        """
        rng = np.random.default_rng(42)
        synthetic_emb = np.vstack([
            rng.normal([10.0,  0.0], 0.2, (20, 2)),   # class 0 — far right
            rng.normal([-10.0, 0.0], 0.2, (20, 2)),   # class 1 — far left
            rng.normal([0.0,  10.0], 0.2, (20, 2)),   # class 2 — far up
        ])
        true_fn = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=int)

        clusters = cluster_embeddings(synthetic_emb, k_values=[3])
        model_purity, _ = compute_cluster_purity(clusters[3], true_fn)

        random_labels = rng.integers(0, 3, size=60)
        random_purity, _ = compute_cluster_purity(random_labels, true_fn)

        assert random_purity <= model_purity
