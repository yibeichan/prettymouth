#!/usr/bin/env python
"""
Create focused visualization for clusters 1-4 only (for reviewer response).
Shows within-cluster similarity and across-cluster dispersion.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib.patches as mpatches

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Colors for clusters 1-4
CLUSTER_COLORS = {
    1: '#e41a1c',  # Red
    2: '#377eb8',  # Blue
    3: '#4daf4a',  # Green
    4: '#984ea3',  # Purple
}

# Markers for groups
GROUP_MARKERS = {
    'affair': 'o',
    'paranoia': 's',
    'combined': '^',
    'balanced': 'D',
}


def extract_patterns_for_clusters(base_dir, cluster_info, target_clusters=[1, 2, 3, 4]):
    """Extract patterns only for specified clusters."""
    patterns = []
    labels = []
    metadata = []

    for cluster_id in target_clusters:
        cluster_id_str = str(cluster_id)
        if cluster_id_str not in cluster_info:
            continue

        members = cluster_info[cluster_id_str]['members']

        for member in members:
            group = member['group']
            model = member['model']
            original_state_idx = member['original_state_idx']

            # Parse model name
            n_states = model.split()[-1].replace('states', '')

            # Construct path
            metrics_path = Path(base_dir) / f"04_{group}_hmm_{n_states}states_ntw_native_trimmed" / "statistics" / f"{group}_metrics.pkl"

            if not metrics_path.exists():
                continue

            try:
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)

                state_properties = metrics.get('state_properties', {})

                if original_state_idx in state_properties:
                    state_props = state_properties[original_state_idx]
                elif str(original_state_idx) in state_properties:
                    state_props = state_properties[str(original_state_idx)]
                else:
                    continue

                if 'mean_pattern' in state_props:
                    pattern = np.array(state_props['mean_pattern'])
                    patterns.append(pattern)
                    labels.append(cluster_id)
                    metadata.append({
                        'cluster': cluster_id,
                        'group': group,
                        'model': model,
                        'state_idx': original_state_idx
                    })

            except Exception as e:
                continue

    logger.info(f"Extracted {len(patterns)} patterns from clusters {target_clusters}")
    return patterns, labels, metadata


def create_focused_visualization(patterns, labels, metadata, output_dir):
    """Create visualization focused on clusters 1-4."""

    X = np.array(patterns)
    labels_array = np.array(labels)

    # Calculate correlation distance matrix
    n = len(patterns)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            corr, _ = spearmanr(patterns[i], patterns[j])
            dist = 1 - corr
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Create figure with only one row
    fig = plt.figure(figsize=(15, 5))

    # 1. PCA
    ax1 = plt.subplot(1, 3, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    for cluster_id in sorted(set(labels)):
        mask = labels_array == cluster_id
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[CLUSTER_COLORS[cluster_id]],
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'Cluster {cluster_id} (n={sum(mask)})')

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.set_title('PCA Projection', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. MDS
    ax2 = plt.subplot(1, 3, 2)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    X_mds = mds.fit_transform(dist_matrix)

    for cluster_id in sorted(set(labels)):
        mask = labels_array == cluster_id
        ax2.scatter(X_mds[mask, 0], X_mds[mask, 1],
                   c=[CLUSTER_COLORS[cluster_id]],
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'Cluster {cluster_id}')

    ax2.set_xlabel('MDS Dimension 1')
    ax2.set_ylabel('MDS Dimension 2')
    ax2.set_title('MDS (Correlation Distance)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. t-SNE
    ax3 = plt.subplot(1, 3, 3)
    perplexity = min(15, len(patterns) // 4)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    for cluster_id in sorted(set(labels)):
        mask = labels_array == cluster_id
        ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[CLUSTER_COLORS[cluster_id]],
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'Cluster {cluster_id}')

    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_title(f't-SNE (perplexity={perplexity})', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Calculate statistics for return values
    within_dists = []
    between_dists = []

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                within_dists.append(dist_matrix[i, j])
            else:
                between_dists.append(dist_matrix[i, j])

    mean_within = np.mean(within_dists)
    mean_between = np.mean(between_dists)

    # Calculate within-cluster correlations
    within_corrs = []
    for cluster_id in [1, 2, 3, 4]:
        mask = labels_array == cluster_id
        cluster_patterns = [p for p, m in zip(patterns, mask) if m]
        if len(cluster_patterns) > 1:
            for i in range(len(cluster_patterns)):
                for j in range(i+1, len(cluster_patterns)):
                    corr, _ = spearmanr(cluster_patterns[i], cluster_patterns[j])
                    within_corrs.append(corr)

    plt.suptitle('Brain State Clustering: Within-cluster Similarity & Across-cluster Dispersion\n(Clusters 1-4, Threshold = 0.80)',
                fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save
    output_path = output_dir / 'cluster_viz_focused.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {output_path}")

    return {
        'mean_within_correlation': np.mean(within_corrs),
        'mean_within_distance': mean_within,
        'mean_between_distance': mean_between,
        'separation_ratio': mean_between/mean_within
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()

    # Setup paths
    if args.base_dir:
        base_dir = args.base_dir
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            scratch_dir = os.getenv("SCRATCH_DIR")
            base_dir = os.path.join(scratch_dir, "output_RR") if scratch_dir else "output_RR"
        except:
            base_dir = "output_RR"

    th_str = f"th_{args.threshold:.2f}".replace('.', '')
    output_dir = Path(base_dir) / "07_RR_cluster_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cluster info
    cluster_info_path = Path(base_dir) / f"06_state_pattern_cluster/{th_str}/cluster_info.json"

    logger.info("Creating focused visualization for clusters 1-4...")

    with open(cluster_info_path, 'r') as f:
        cluster_info = json.load(f)

    # Extract patterns for clusters 1-4 only
    patterns, labels, metadata = extract_patterns_for_clusters(base_dir, cluster_info, [1, 2, 3, 4])

    if not patterns:
        logger.error("No patterns extracted")
        return

    # Create visualization
    metrics = create_focused_visualization(patterns, labels, metadata, output_dir)

    print("\n" + "="*60)
    print("SUPPLEMENTARY FIGURE CREATED")
    print("="*60)
    print(f"Mean within-cluster correlation: {metrics['mean_within_correlation']:.3f}")
    print(f"Separation ratio: {metrics['separation_ratio']:.2f}x")
    print(f"Output: {output_dir}/cluster_viz_focused.png")
    print("="*60)


if __name__ == "__main__":
    main()