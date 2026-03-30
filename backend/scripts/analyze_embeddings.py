"""
Analyze the chord embedding space of a trained BaroqueHarmonyLSTM.

Usage
-----
    uv run python scripts/analyze_embeddings.py \\
        --checkpoint model_best.pt \\
        --sequences data/sequences.json \\
        --output-dir results/embedding_analysis

Outputs (all under --output-dir)
---------------------------------
    purity_results.json
    embeddings_2d.csv
    figures/function_ground_truth.png
    figures/clusters_k3.png
    figures/clusters_k5.png
    figures/clusters_k7.png
    figures/combined_analysis.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe in headless environments
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import matplotlib.patches as mpatches
import numpy as np

from src.data.encoder import Encoder
from src.analysis.embedding_analysis import (
    cluster_embeddings,
    compute_cluster_purity,
    compute_tonal_distance_correlation,
    extract_chord_embeddings,
    reduce_dimensions,
)

_FUNCTION_NAMES = ['Tonic', 'Predominant', 'Dominant']
_FUNCTION_COLORS = ['#2196F3', '#FF9800', '#F44336']  # blue, orange, red
_K_VALUES = [3, 5, 7]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter(
    ax: plt.Axes,
    embeddings_2d: np.ndarray,
    colors: list,
    labels: list[str],
    title: str,
    xlabel: str = 'PC 1',
    ylabel: str = 'PC 2',
) -> None:
    """Draw 60 chord points with chord-name annotations."""
    ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=colors, s=60, alpha=0.85, edgecolors='none',
    )
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=5.5, alpha=0.75,
            xytext=(3, 3), textcoords='offset points',
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def _cluster_colors(cluster_ids: np.ndarray, k: int) -> list:
    cmap = plt.get_cmap('tab10')
    return [cmap(int(cid) / max(k - 1, 1)) for cid in cluster_ids]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chord embedding space")
    parser.add_argument('--checkpoint', default='model_best.pt', type=str,
                        help="Path to trained model checkpoint")
    parser.add_argument('--sequences', default='data/sequences.json', type=Path,
                        help="Sequences JSON from extract_all.py")
    parser.add_argument('--output-dir', default='results/embedding_analysis', type=Path,
                        help="Directory to write all outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build encoder with majority-vote chord→function map
    # ------------------------------------------------------------------
    print(f"Loading sequences from {args.sequences} ...")
    with open(args.sequences) as f:
        raw_sequences = json.load(f)
    print(f"  {len(raw_sequences)} sequences loaded.")

    enc = Encoder()
    enc.build_chord_function_map(raw_sequences)

    # ------------------------------------------------------------------
    # 2. Extract chord embeddings
    # ------------------------------------------------------------------
    print(f"Extracting chord embeddings from {args.checkpoint} ...")
    embeddings, token_labels, function_ids, quality_labels = extract_chord_embeddings(
        args.checkpoint, enc
    )
    print(f"  Embeddings shape: {embeddings.shape}")  # (60, 64)

    # ------------------------------------------------------------------
    # 3. PCA → 2D
    # ------------------------------------------------------------------
    print("Reducing to 2D with PCA ...")
    embeddings_2d, pca = reduce_dimensions(embeddings, method='pca', n_components=2)
    ev = pca.explained_variance_ratio_
    print(f"  PC1: {ev[0]:.1%}  PC2: {ev[1]:.1%}  Total: {sum(ev):.1%}")

    # ------------------------------------------------------------------
    # 4. Save 2D CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / 'embeddings_2d.csv'
    with open(csv_path, 'w') as fh:
        fh.write('label,function_id,quality,pc1,pc2\n')
        for i in range(len(token_labels)):
            fh.write(
                f"{token_labels[i]},{function_ids[i]},{quality_labels[i]},"
                f"{embeddings_2d[i, 0]:.6f},{embeddings_2d[i, 1]:.6f}\n"
            )
    print(f"  Saved {csv_path}")

    # ------------------------------------------------------------------
    # 5. K-Means clustering
    # ------------------------------------------------------------------
    print(f"Running K-Means for k = {_K_VALUES} ...")
    clusters = cluster_embeddings(embeddings, k_values=_K_VALUES)

    # ------------------------------------------------------------------
    # 6. Purity + ARI
    # ------------------------------------------------------------------
    true_fn = np.array(function_ids, dtype=int)
    purity_results: dict = {
        'pca_explained_variance': {
            'pc1': float(ev[0]),
            'pc2': float(ev[1]),
            'total': float(sum(ev)),
        }
    }

    for k in _K_VALUES:
        purity, breakdown = compute_cluster_purity(clusters[k], true_fn)
        ari = breakdown.pop('_ari')
        purity_results[f'k{k}'] = {
            'purity': purity,
            'ari': ari,
            'per_cluster': breakdown,
        }
        print(f"  k={k}: purity={purity:.3f}  ARI={ari:.3f}")

    # ------------------------------------------------------------------
    # 7. Tonal distance correlation
    # ------------------------------------------------------------------
    print("Computing tonal distance correlation ...")
    r, p = compute_tonal_distance_correlation(embeddings, token_labels)
    purity_results['tonal_distance_correlation'] = {
        'pearson_r': r,
        'p_value': p,
    }
    significance = "significant" if p < 0.05 else "not significant"
    print(f"  Circle-of-fifths correlation: r={r:.3f}, p={p:.4f} ({significance})")

    # Save JSON
    json_path = out_dir / 'purity_results.json'
    with open(json_path, 'w') as fh:
        json.dump(purity_results, fh, indent=2)
    print(f"  Saved {json_path}")

    # ------------------------------------------------------------------
    # 8. Plot A — Ground truth harmonic functions
    # ------------------------------------------------------------------
    fn_colors = [_FUNCTION_COLORS[fid] for fid in function_ids]
    legend_patches = [
        mpatches.Patch(facecolor=_FUNCTION_COLORS[i], label=_FUNCTION_NAMES[i])
        for i in range(3)
    ]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    _scatter(ax, embeddings_2d, fn_colors, token_labels,
             "Chord Embedding Space — Harmonic Function Labels")
    ax.legend(handles=legend_patches, loc='best', fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / 'function_ground_truth.png')
    plt.close(fig)
    print("  Saved figures/function_ground_truth.png")

    # ------------------------------------------------------------------
    # 9. Plot B — K-Means clusters
    # ------------------------------------------------------------------
    for k in _K_VALUES:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        colors = _cluster_colors(clusters[k], k)
        _scatter(ax, embeddings_2d, colors, token_labels,
                 f"Chord Embedding Space — K-Means Clustering (k={k})")
        plt.tight_layout()
        fig.savefig(fig_dir / f'clusters_k{k}.png')
        plt.close(fig)
        print(f"  Saved figures/clusters_k{k}.png")

    # ------------------------------------------------------------------
    # 10. Plot C — Combined 2×2 thesis figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)

    # Top-left: ground truth functions
    _scatter(axes[0, 0], embeddings_2d, fn_colors, token_labels,
             "Ground Truth Harmonic Functions")
    axes[0, 0].legend(handles=legend_patches, loc='best', fontsize=7)

    # Top-right: k=3 clusters
    _scatter(axes[0, 1], embeddings_2d, _cluster_colors(clusters[3], 3), token_labels,
             "K-Means Clustering (k=3)")

    # Bottom-left: k=5 clusters
    _scatter(axes[1, 0], embeddings_2d, _cluster_colors(clusters[5], 5), token_labels,
             "K-Means Clustering (k=5)")

    # Bottom-right: purity bar chart
    ax_bar = axes[1, 1]
    bar_colors = ['#2196F3', '#4CAF50', '#FF9800']
    purities = [purity_results[f'k{k}']['purity'] for k in _K_VALUES]
    bars = ax_bar.bar([str(k) for k in _K_VALUES], purities, color=bar_colors, alpha=0.85)
    ax_bar.axhline(1 / 3, linestyle='--', color='red', linewidth=1.5,
                   label='Random baseline (33.3%)')
    ax_bar.set_xlabel('k', fontsize=10)
    ax_bar.set_ylabel('Purity', fontsize=10)
    ax_bar.set_title('Cluster Purity by k', fontsize=11)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(fontsize=9)
    for bar, val in zip(bars, purities):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10,
        )

    fig.suptitle("Chord Embedding Space Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_dir / 'combined_analysis.png')
    plt.close(fig)
    print("  Saved figures/combined_analysis.png")

    print("\nDone.")


if __name__ == '__main__':
    main()
