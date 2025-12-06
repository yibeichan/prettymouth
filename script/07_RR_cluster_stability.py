#!/usr/bin/env python3
"""
Simple analysis: How similar are top clusters across consecutive thresholds?

Answers ONE question:
"What are the top N clusters at each threshold, and how similar are they to the previous threshold's top N?"
"""

import os
import json
import numpy as np
import pandas as pd

# Setup paths
try:
    from dotenv import load_dotenv
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
except:
    scratch_dir = None

base_dir = os.path.join(scratch_dir, "output_RR") if scratch_dir else "output_RR"
input_dir = os.path.join(base_dir, "06_state_pattern_cluster")
output_dir = os.path.join(base_dir, "07_RR_cluster_stability")
os.makedirs(output_dir, exist_ok=True)

thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

NETWORK_NAMES = ['Aud', 'Ctr-A', 'Ctr-B', 'Ctr-C', 'DMN-A', 'DMN-B',
                        'DMN-C', 'DA-A', 'DA-B', 'Lang', 'SVA-A',
                        'SVA-B', 'SM-A', 'SM-B', 'Vis-A', 'Vis-B',
                        'Vis-C']


def jaccard_similarity(vec1, vec2):
    """Calculate Jaccard similarity between two binary vectors."""
    intersection = np.sum(np.logical_and(vec1, vec2))
    union = np.sum(np.logical_or(vec1, vec2))
    return intersection / union if union > 0 else 0.0


def load_top_clusters(threshold, top_n=5):
    """Load top N clusters by size for a given threshold."""
    th_str = f"th_{threshold:.2f}".replace('.', '')
    cluster_path = os.path.join(input_dir, th_str, "cluster_info.json")

    if not os.path.exists(cluster_path):
        return None

    with open(cluster_path, 'r') as f:
        cluster_info = json.load(f)

    # Sort by size and take top N
    clusters = []
    for cluster_id, info in cluster_info.items():
        clusters.append({
            'id': cluster_id,
            'size': info['size'],
            'pattern': np.array(info['consensus_pattern']),
            'occupancy': info['total_fractional_occupancy']
        })

    clusters_sorted = sorted(clusters, key=lambda x: x['size'], reverse=True)
    return clusters_sorted[:top_n]


def compare_threshold_clusters(clusters_curr, clusters_prev):
    """Compare top clusters between two consecutive thresholds."""
    results = []

    for i, curr_cluster in enumerate(clusters_curr, 1):
        # Find best match in previous threshold
        best_sim = 0
        best_match_rank = None

        for j, prev_cluster in enumerate(clusters_prev, 1):
            sim = jaccard_similarity(curr_cluster['pattern'], prev_cluster['pattern'])
            if sim > best_sim:
                best_sim = sim
                best_match_rank = j

        # Get network signature
        active_networks = [NETWORK_NAMES[k] for k, val in enumerate(curr_cluster['pattern']) if val == 1]

        results.append({
            'rank': i,
            'size': curr_cluster['size'],
            'n_networks': len(active_networks),
            'networks': ', '.join(active_networks),
            'best_match_similarity': best_sim,
            'prev_rank': best_match_rank
        })

    return results


def main():
    print("=" * 80)
    print("SIMPLE CLUSTER STABILITY ANALYSIS")
    print("Question: How similar are top 5 clusters across consecutive thresholds?")
    print("=" * 80)
    print()

    # Load top 5 clusters for each threshold
    all_clusters = {}
    for th in thresholds:
        clusters = load_top_clusters(th, top_n=5)
        if clusters and len(clusters) >= 3:
            all_clusters[th] = clusters
            print(f"Threshold {th:.2f}: {len(clusters)} clusters loaded")

    print()

    # Compare consecutive thresholds
    comparison_results = []

    sorted_ths = sorted(all_clusters.keys())
    for i in range(len(sorted_ths) - 1):
        th_curr = sorted_ths[i + 1]
        th_prev = sorted_ths[i]

        clusters_curr = all_clusters[th_curr]
        clusters_prev = all_clusters[th_prev]

        results = compare_threshold_clusters(clusters_curr, clusters_prev)

        # Add to overall results
        for result in results:
            comparison_results.append({
                'threshold_from': th_prev,
                'threshold_to': th_curr,
                **result
            })

    # Create DataFrame
    df = pd.DataFrame(comparison_results)
    df.to_csv(os.path.join(output_dir, 'cluster_stability.csv'), index=False)

    # Summary statistics
    print("RESULTS")
    print("-" * 80)
    print()

    for i in range(len(sorted_ths) - 1):
        th_prev = sorted_ths[i]
        th_curr = sorted_ths[i + 1]

        subset = df[(df['threshold_from'] == th_prev) & (df['threshold_to'] == th_curr)]
        avg_sim = subset['best_match_similarity'].mean()
        n_high_sim = len(subset[subset['best_match_similarity'] >= 0.7])

        print(f"{th_prev:.2f} → {th_curr:.2f}:")
        print(f"  Average similarity: {avg_sim:.3f}")
        print(f"  High similarity (≥0.7): {n_high_sim}/{len(subset)} clusters")
        print()

    # Overall statistics
    overall_avg = df['best_match_similarity'].mean()
    n_high_overall = len(df[df['best_match_similarity'] >= 0.7])
    n_total = len(df)

    print("OVERALL:")
    print(f"  Average similarity across all transitions: {overall_avg:.3f}")
    print(f"  High similarity matches (≥0.7): {n_high_overall}/{n_total} ({n_high_overall/n_total*100:.1f}%)")

    # Verdict
    print()
    if overall_avg >= 0.7:
        verdict = "HIGH STABILITY"
    elif overall_avg >= 0.5:
        verdict = "MODERATE STABILITY"
    else:
        verdict = "LOW STABILITY"

    print(f"VERDICT: {verdict}")
    print()
    print(f"Results saved to: {output_dir}/cluster_stability.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
