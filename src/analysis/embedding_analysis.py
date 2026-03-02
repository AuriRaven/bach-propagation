"""
Unsupervised embedding-space analysis for the trained BaroqueHarmonyLSTM.

Functions
---------
extract_chord_embeddings          Pull chord embedding vectors from a checkpoint.
reduce_dimensions                 PCA / UMAP 2-D projection.
cluster_embeddings                K-Means for multiple k values.
compute_cluster_purity            Purity + ARI against ground-truth function labels.
compute_tonal_distance_correlation  Circle-of-fifths vs. cosine distance Mantel test.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.data.encoder import CHORD_OFFSET, CHORD_VOCAB_SIZE, Encoder
from src.models.baseline_lstm import BaroqueHarmonyLSTM

# Chromatic root index → note name (enharmonic spellings chosen to match encoder labels)
_ROOT_NAMES: List[str] = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'
]


# ---------------------------------------------------------------------------
# 1. Extract embeddings
# ---------------------------------------------------------------------------

def extract_chord_embeddings(
    checkpoint_path: str,
    encoder: Encoder,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    Load a checkpoint and return the chord embedding vectors with metadata.

    Parameters
    ----------
    checkpoint_path : str
        Path to a saved ``model.state_dict()`` file.
    encoder : Encoder
        An Encoder whose ``build_chord_function_map`` has already been called
        so that ``token_function`` returns majority-vote labels.

    Returns
    -------
    embeddings : np.ndarray, shape (60, 64)
        One row per chord token (tokens CHORD_OFFSET … CHORD_OFFSET+59).
    token_labels : list of str
        Human-readable chord names, e.g. ``"C:major"``, ``"D:minor"``.
    function_ids : list of int
        Majority-vote harmonic function (0 = T, 1 = PD, 2 = D) for each token.
    quality_labels : list of str
        Canonical quality string for each token.
    """
    model = BaroqueHarmonyLSTM()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Full embedding matrix: (vocab_size, embed_dim)
    weight: np.ndarray = model.embedding.weight.detach().cpu().numpy()

    # Chord tokens occupy CHORD_OFFSET … CHORD_OFFSET + CHORD_VOCAB_SIZE − 1
    chord_rows = weight[CHORD_OFFSET: CHORD_OFFSET + CHORD_VOCAB_SIZE]  # (60, 64)

    token_labels: List[str] = []
    function_ids: List[int] = []
    quality_labels: List[str] = []

    for offset in range(CHORD_VOCAB_SIZE):
        tok = CHORD_OFFSET + offset
        root, quality = encoder.token_to_chord(tok)
        token_labels.append(f"{_ROOT_NAMES[root]}:{quality}")
        function_ids.append(encoder.token_function(tok))
        quality_labels.append(quality)

    return chord_rows, token_labels, function_ids, quality_labels


# ---------------------------------------------------------------------------
# 2. Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = 'pca',
    n_components: int = 2,
) -> Tuple[np.ndarray, object]:
    """
    Project embeddings to *n_components* dimensions.

    Always standardises with ``StandardScaler`` before reducing.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
    method : {'pca', 'umap'}
        ``'umap'`` falls back to PCA with a warning if *umap-learn* is not installed.
    n_components : int
        Target dimensionality (default 2).

    Returns
    -------
    reduced : np.ndarray, shape (n, n_components)
    reducer : fitted reducer object (PCA or UMAP instance)
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    if method == 'umap':
        try:
            from umap import UMAP  # type: ignore
            reducer: object = UMAP(n_components=n_components, random_state=42)
        except ImportError:
            warnings.warn(
                "umap-learn is not installed; falling back to PCA.",
                ImportWarning,
                stacklevel=2,
            )
            method = 'pca'

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)

    reduced: np.ndarray = reducer.fit_transform(scaled)  # type: ignore[union-attr]
    return reduced, reducer


# ---------------------------------------------------------------------------
# 3. Clustering
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    k_values: List[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Run K-Means for each k in *k_values* and return label arrays.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
    k_values : list of int
        Number-of-clusters values to try. Defaults to ``[3, 5, 7]``.

    Returns
    -------
    dict mapping k → cluster label array of shape (n,)
    """
    if k_values is None:
        k_values = [3, 5, 7]

    results: Dict[int, np.ndarray] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        results[k] = km.fit_predict(embeddings)
    return results


# ---------------------------------------------------------------------------
# 4. Cluster purity
# ---------------------------------------------------------------------------

def compute_cluster_purity(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> Tuple[float, dict]:
    """
    Compute cluster purity and adjusted Rand index.

    Purity = (1/N) × Σ_k  max_c |C_k ∩ T_c|

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (n,)
        K-Means (or any) cluster assignments.
    true_labels : np.ndarray, shape (n,)
        Ground-truth class labels (ints).

    Returns
    -------
    overall_purity : float in [0, 1]
    per_cluster_breakdown : dict
        ``{cluster_id: {'dominant_label': int, 'purity': float, 'size': int}}``
        plus ``'_ari'`` key containing the adjusted Rand index.
    """
    n = len(cluster_labels)
    total_correct = 0
    breakdown: dict = {}

    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        members = true_labels[mask]
        n_classes = int(true_labels.max()) + 1
        counts = np.bincount(members, minlength=n_classes)
        dominant = int(np.argmax(counts))
        correct = int(counts[dominant])
        total_correct += correct
        breakdown[int(cid)] = {
            'dominant_label': dominant,
            'purity': correct / len(members),
            'size': int(len(members)),
        }

    overall_purity = total_correct / n
    ari = float(adjusted_rand_score(true_labels, cluster_labels))
    breakdown['_ari'] = ari
    return overall_purity, breakdown


# ---------------------------------------------------------------------------
# 5. Tonal distance correlation (Mantel test approximation)
# ---------------------------------------------------------------------------

def compute_tonal_distance_correlation(
    embeddings: np.ndarray,
    token_labels: List[str],
) -> Tuple[float, float]:
    """
    Correlate pairwise cosine distances in embedding space with pairwise
    circle-of-fifths distances between chord roots (Mantel test approximation).

    If the model independently recovered tonal structure the correlation
    should be positive and significant.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
    token_labels : list of str
        Labels in ``"RootName:quality"`` format, as returned by
        :func:`extract_chord_embeddings`.

    Returns
    -------
    r : float  — Pearson correlation coefficient
    p : float  — two-tailed p-value
    """
    # Parse chromatic root index from label (e.g. "C#:major" → 1, "Eb:minor" → 3)
    roots = np.array(
        [_ROOT_NAMES.index(lbl.split(':')[0]) for lbl in token_labels],
        dtype=int,
    )

    # Map chromatic root → circle-of-fifths position (0–11)
    # C=0, G=1, D=2, A=3, E=4, B=5, F#=6, Db=7, Ab=8, Eb=9, Bb=10, F=11
    fifths_pos = (roots * 7) % 12

    # Pairwise circle-of-fifths distance: circular distance on a 12-element ring
    n = len(token_labels)
    cof_dists: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(int(fifths_pos[i]) - int(fifths_pos[j]))
            cof_dists.append(float(min(d, 12 - d)))

    cof_arr = np.array(cof_dists)

    # Pairwise cosine distance in embedding space
    emb_dists = pdist(embeddings, metric='cosine')

    r, p = pearsonr(emb_dists, cof_arr)
    return float(r), float(p)
