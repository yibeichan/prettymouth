import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging
from collections import defaultdict
import re
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import List
from natsort import natsorted


plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': [8.0, 6.0],
    'figure.dpi': 300,
    'font.size': 10,
    'svg.fonttype': 'none',
    'figure.titlesize': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'ytick.labelsize': 6,
    'xtick.labelsize': 6,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

def analyze_cluster_stability(
    base_dir: str,
    groups: List[str],
    output_dir: str,
    thresholds: List[float] = None
):
    """
    Analyze how clusters change with different similarity thresholds.

    Args:
        base_dir: Base directory containing the StatePatternAnalyzer results
        groups: List of groups used in the analysis
        output_dir: Directory to save results
        thresholds: List of similarity thresholds to test
    """
    # Setup logging
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    # Store results
    threshold_dirs = {}
    cluster_data = {}
    pattern_cluster_assignments = {}

    # Initialize pattern lifecycle tracking
    pattern_lifecycles = {}  # Track each unique pattern across thresholds
    
    # Run analysis for each threshold
    from utils.pattern_cluster import StatePatternAnalyzer
    
    logger.info(f"Analyzing cluster stability across thresholds: {thresholds}")
    
    for threshold in thresholds:
        # Define output directory for this threshold
        threshold_str = f"{threshold:.2f}".replace('.', '')
        thresh_dir = os.path.join(output_dir, f"th_{threshold_str}")
        os.makedirs(thresh_dir, exist_ok=True)
        threshold_dirs[threshold] = thresh_dir
        
        # Create and run analyzer with this threshold
        config = {
            'min_activation': 0.1,
            'max_ci_width': 0.3,
            'min_pattern_stability': 0.7,
            'cluster_similarity_threshold': threshold
        }
        
        analyzer = StatePatternAnalyzer(
            base_dir=base_dir,
            groups=groups,
            output_dir=thresh_dir,
            config=config
        )
        
        # Run analysis
        analyzer.run_analysis()
        
        # Store cluster info for this threshold
        with open(os.path.join(thresh_dir, 'cluster_info.json'), 'r') as f:
            cluster_info = json.load(f)
            cluster_data[threshold] = cluster_info
        
        # Track pattern to cluster assignments for each threshold
        pattern_to_cluster = {}

        # For each cluster
        for cluster_id, data in cluster_info.items():
            # For each pattern in the cluster
            for member in data['members']:
                group = member['group']
                pattern_idx = member['pattern_idx']

                # Create a unique pattern ID
                pattern_id = f"{group}_{pattern_idx}"
                pattern_to_cluster[pattern_id] = int(cluster_id)

                # Update lifecycle tracking
                if pattern_id not in pattern_lifecycles:
                    pattern_lifecycles[pattern_id] = {
                        'group': group,
                        'pattern_idx': pattern_idx,
                        'first_threshold': threshold,
                        'last_threshold': threshold,
                        'thresholds_present': [threshold],
                        'cluster_assignments': {threshold: int(cluster_id)},
                        'sizes_at_threshold': {threshold: data['size']},
                        'stability_info': {}
                    }
                else:
                    pattern_lifecycles[pattern_id]['last_threshold'] = threshold
                    pattern_lifecycles[pattern_id]['thresholds_present'].append(threshold)
                    pattern_lifecycles[pattern_id]['cluster_assignments'][threshold] = int(cluster_id)
                    pattern_lifecycles[pattern_id]['sizes_at_threshold'][threshold] = data['size']

        pattern_cluster_assignments[threshold] = pattern_to_cluster
    
    # Analyze pattern lifecycles
    analyze_pattern_lifecycles(pattern_lifecycles, thresholds, output_dir)

    # Collect threshold statistics for visualization
    threshold_stats = []
    for threshold in thresholds:
        clusters = cluster_data[threshold]
        cluster_sizes = [int(c['size']) for c in clusters.values()]

        threshold_stats.append({
            'threshold': threshold,
            'unique_clusters': len(clusters),
            'total_patterns': sum(cluster_sizes),
            'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'fragmentation_ratio': len(clusters) / sum(cluster_sizes) if sum(cluster_sizes) > 0 else 0
        })

    # Create visualization of threshold statistics
    create_threshold_statistics_plot(threshold_stats, output_dir)

    # Now analyze the stability across thresholds
    analyze_stability_results(thresholds, pattern_cluster_assignments, cluster_data, output_dir)

    return threshold_dirs, cluster_data, pattern_lifecycles

def create_threshold_statistics_plot(threshold_stats, output_dir):
    """
    Create a comprehensive visualization of threshold statistics.

    Args:
        threshold_stats: List of dictionaries containing statistics for each threshold
        output_dir: Directory to save the plot
    """
    logger = logging.getLogger(__name__)

    # Extract data from threshold_stats
    thresholds = [stat['threshold'] for stat in threshold_stats]
    unique_clusters = [stat['unique_clusters'] for stat in threshold_stats]
    total_patterns = [stat['total_patterns'] for stat in threshold_stats]
    largest_cluster = [stat['largest_cluster'] for stat in threshold_stats]
    avg_cluster_size = [stat['avg_cluster_size'] for stat in threshold_stats]
    fragmentation_ratio = [stat['fragmentation_ratio'] for stat in threshold_stats]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.15)

    # Color palette
    colors_main = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A']

    # 1. Number of unique clusters
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(thresholds, unique_clusters, 'o-', linewidth=2.5, markersize=10,
             color=colors_main[0], markeredgecolor='white', markeredgewidth=1.5)
    ax1.fill_between(thresholds, unique_clusters, alpha=0.2, color=colors_main[0])
    ax1.set_xlabel('Similarity Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Unique Clusters', fontsize=12, fontweight='bold')
    ax1.set_title('Cluster Count vs. Threshold', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(thresholds, unique_clusters)):
        ax1.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9, color=colors_main[0], fontweight='bold')

    # 2. Size of largest cluster
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(thresholds, largest_cluster, 's-', linewidth=2.5, markersize=10,
             color=colors_main[1], markeredgecolor='white', markeredgewidth=1.5)
    ax2.fill_between(thresholds, largest_cluster, alpha=0.2, color=colors_main[1])
    ax2.set_xlabel('Similarity Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Size of Largest Cluster', fontsize=12, fontweight='bold')
    ax2.set_title('Largest Cluster Size vs. Threshold', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(thresholds, largest_cluster)):
        ax2.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9, color=colors_main[1], fontweight='bold')

    # 3. Average cluster size
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(thresholds, avg_cluster_size, '^-', linewidth=2.5, markersize=10,
             color=colors_main[2], markeredgecolor='white', markeredgewidth=1.5)
    ax3.fill_between(thresholds, avg_cluster_size, alpha=0.2, color=colors_main[2])
    ax3.set_xlabel('Similarity Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Cluster Size', fontsize=12, fontweight='bold')
    ax3.set_title('Average Cluster Size vs. Threshold', fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(thresholds, avg_cluster_size)):
        ax3.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9, color=colors_main[2], fontweight='bold')

    # 4. Fragmentation ratio
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(thresholds, fragmentation_ratio, 'd-', linewidth=2.5, markersize=10,
             color=colors_main[3], markeredgecolor='white', markeredgewidth=1.5)
    ax4.fill_between(thresholds, fragmentation_ratio, alpha=0.2, color=colors_main[3])
    ax4.set_xlabel('Similarity Threshold', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Fragmentation Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Clustering Fragmentation vs. Threshold', fontsize=14, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(thresholds, fragmentation_ratio)):
        ax4.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9, color=colors_main[3], fontweight='bold')

    # 5. Combined normalized plot
    ax5 = fig.add_subplot(gs[2, :])
    # Normalize values to 0-1 for comparison
    unique_clusters_norm = np.array(unique_clusters) / max(unique_clusters)
    largest_cluster_norm = np.array(largest_cluster) / max(largest_cluster)
    avg_cluster_norm = np.array(avg_cluster_size) / max(avg_cluster_size)

    ax5.plot(thresholds, unique_clusters_norm, 'o-', linewidth=2.5, markersize=10,
             label='Unique Clusters (normalized)', color=colors_main[0], alpha=0.8)
    ax5.plot(thresholds, largest_cluster_norm, 's-', linewidth=2.5, markersize=10,
             label='Largest Cluster Size (normalized)', color=colors_main[1], alpha=0.8)
    ax5.plot(thresholds, avg_cluster_norm, '^-', linewidth=2.5, markersize=10,
             label='Average Cluster Size (normalized)', color=colors_main[2], alpha=0.8)

    ax5.set_xlabel('Similarity Threshold', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
    ax5.set_title('Normalized Clustering Metrics Comparison', fontsize=14, fontweight='bold', pad=10)
    ax5.legend(loc='best', fontsize=11, framealpha=0.95)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_ylim(-0.05, 1.05)

    # Add vertical line at potential balance point
    if len(thresholds) > 4:
        balance_idx = len(thresholds) // 2
        balance_threshold = thresholds[balance_idx]
        ax5.axvline(x=balance_threshold, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax5.text(balance_threshold, 0.5, f'θ={balance_threshold:.2f}', rotation=90,
                 verticalalignment='center', fontsize=10, color='gray', alpha=0.7)

    # Main title
    fig.suptitle(f'Clustering Analysis: Threshold Impact on Pattern Organization\n(Total Patterns: {total_patterns[0]})',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add text box with key insights
    fragmentation_change = fragmentation_ratio[-1] / fragmentation_ratio[0] if fragmentation_ratio[0] > 0 else 0
    stabilized_threshold = None
    for i in range(1, len(largest_cluster)):
        if largest_cluster[i] == largest_cluster[i-1]:
            stabilized_threshold = thresholds[i-1]
            break

    # insights = [
    #     'Key Insights:',
    #     '• Lower thresholds → fewer, larger clusters',
    #     '• Higher thresholds → more, smaller clusters',
    # ]
    # if stabilized_threshold:
    #     insights.append(f'• Largest cluster stabilizes at size {largest_cluster[-1]} for θ ≥ {stabilized_threshold:.2f}')
    # if fragmentation_change > 1:
    #     insights.append(f'• Fragmentation increases ~{fragmentation_change:.1f}x from θ={thresholds[0]:.2f} to θ={thresholds[-1]:.2f}')

    # textstr = '\n'.join(insights)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # fig.text(0.8, 0.01, textstr, fontsize=10, ha='right', va='bottom',
    #          bbox=props, transform=fig.transFigure)

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'threshold_statistics_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'threshold_statistics_analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()

    logger.info(f"Threshold statistics plot saved to {output_dir}")

def analyze_pattern_lifecycles(pattern_lifecycles, thresholds, output_dir):
    """
    Analyze and visualize pattern lifecycles across thresholds.

    Args:
        pattern_lifecycles: Dictionary tracking each pattern's presence across thresholds
        thresholds: List of threshold values
        output_dir: Directory to save results
    """
    logger = logging.getLogger(__name__)

    # Calculate lifecycle statistics
    lifecycle_stats = []

    for pattern_id, lifecycle in pattern_lifecycles.items():
        # Check if pattern persists across all thresholds
        persistence = len(lifecycle['thresholds_present']) / len(thresholds)

        # Check for gaps (pattern disappears then reappears)
        thresholds_present = sorted(lifecycle['thresholds_present'])
        has_gaps = False
        if len(thresholds_present) > 1:
            for i in range(len(thresholds_present) - 1):
                idx_current = thresholds.index(thresholds_present[i])
                idx_next = thresholds.index(thresholds_present[i + 1])
                if idx_next - idx_current > 1:
                    has_gaps = True
                    break

        # Check cluster stability (does it stay in same cluster?)
        cluster_changes = len(set(lifecycle['cluster_assignments'].values())) - 1

        # Track size evolution
        size_trend = 'stable'
        if len(lifecycle['sizes_at_threshold']) > 1:
            sizes = list(lifecycle['sizes_at_threshold'].values())
            if sizes[-1] > sizes[0] * 1.2:
                size_trend = 'growing'
            elif sizes[-1] < sizes[0] * 0.8:
                size_trend = 'shrinking'

        lifecycle_stats.append({
            'pattern_id': pattern_id,
            'group': lifecycle['group'],
            'persistence': persistence,
            'n_thresholds': len(lifecycle['thresholds_present']),
            'first_seen': lifecycle['first_threshold'],
            'last_seen': lifecycle['last_threshold'],
            'has_gaps': has_gaps,
            'cluster_changes': cluster_changes,
            'size_trend': size_trend,
            'threshold_range': f"{lifecycle['first_threshold']:.2f}-{lifecycle['last_threshold']:.2f}"
        })

    # Convert to DataFrame
    df_lifecycles = pd.DataFrame(lifecycle_stats)

    # Save lifecycle analysis
    df_lifecycles.to_csv(os.path.join(output_dir, 'pattern_lifecycles.csv'), index=False)

    # Create lifecycle visualization
    create_lifecycle_visualization(df_lifecycles, thresholds, output_dir)

    # Log summary statistics
    logger.info("\nPattern Lifecycle Analysis:")
    logger.info(f"Total unique patterns tracked: {len(pattern_lifecycles)}")
    logger.info(f"Patterns persisting across all thresholds: {len(df_lifecycles[df_lifecycles['persistence'] == 1.0])}")
    logger.info(f"Patterns with gaps: {len(df_lifecycles[df_lifecycles['has_gaps'] == True])}")
    logger.info(f"Average persistence: {df_lifecycles['persistence'].mean():.2f}")

    # Group-specific statistics
    for group in df_lifecycles['group'].unique():
        group_df = df_lifecycles[df_lifecycles['group'] == group]
        logger.info(f"\n{group} patterns:")
        logger.info(f"  Total: {len(group_df)}")
        logger.info(f"  Avg persistence: {group_df['persistence'].mean():.2f}")
        logger.info(f"  Stable across all: {len(group_df[group_df['persistence'] == 1.0])}")

    return df_lifecycles

def calculate_similarity_scores(pattern_assignments, thresholds):
    """
    Calculate similarity scores between consecutive thresholds with improved handling of missing patterns.

    Args:
        pattern_assignments: Dict of threshold -> pattern assignments
        thresholds: List of threshold values

    Returns:
        List of similarity score dictionaries
    """
    similarity_scores = []

    # Get all pattern IDs and their frequencies
    pattern_counts = {}
    for threshold in thresholds:
        for pattern_id in pattern_assignments[threshold].keys():
            pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1

    # Sort patterns by frequency for consistency
    all_patterns = sorted(pattern_counts.keys())

    # Create threshold labels with logging
    threshold_labels = {}
    threshold_labels_full = {}  # Include missing patterns as distinct cluster
    pattern_counts_per_threshold = {}

    # Find max cluster ID across all thresholds for missing pattern assignment
    max_cluster_id = 0
    for threshold in thresholds:
        for cluster_id in pattern_assignments[threshold].values():
            max_cluster_id = max(max_cluster_id, cluster_id)

    missing_cluster_id = max_cluster_id + 1  # Special cluster for missing patterns

    for threshold in thresholds:
        assignments = pattern_assignments[threshold]
        labels = np.zeros(len(all_patterns), dtype=int)
        labels_full = np.zeros(len(all_patterns), dtype=int)
        n_patterns = 0

        for i, pattern_id in enumerate(all_patterns):
            if pattern_id in assignments:
                labels[i] = assignments[pattern_id]
                labels_full[i] = assignments[pattern_id]
                n_patterns += 1
            else:
                labels[i] = -1  # Mark as missing for standard calculation
                labels_full[i] = missing_cluster_id  # Assign to "missing" cluster

        threshold_labels[threshold] = labels
        threshold_labels_full[threshold] = labels_full
        pattern_counts_per_threshold[threshold] = n_patterns

        print(f"Threshold {threshold:.2f}: {n_patterns} patterns, "
              f"{len(set(labels[labels != -1]))} unique clusters")

    # Compare consecutive thresholds with both methods
    for i in range(len(thresholds) - 1):
        t1 = thresholds[i]
        t2 = thresholds[i + 1]

        labels1 = threshold_labels[t1]
        labels2 = threshold_labels[t2]
        labels1_full = threshold_labels_full[t1]
        labels2_full = threshold_labels_full[t2]

        # Method 1: Only common patterns (original)
        mask = (labels1 != -1) & (labels2 != -1)
        common_patterns = np.sum(mask)

        if common_patterns > 0:
            ari_common = adjusted_rand_score(labels1[mask], labels2[mask])
            nmi_common = normalized_mutual_info_score(labels1[mask], labels2[mask])
        else:
            ari_common = 0.0
            nmi_common = 0.0

        # Method 2: All patterns with missing as distinct cluster
        ari_full = adjusted_rand_score(labels1_full, labels2_full)
        nmi_full = normalized_mutual_info_score(labels1_full, labels2_full)

        # Calculate pattern dynamics
        patterns_lost = np.sum((labels1 != -1) & (labels2 == -1))
        patterns_gained = np.sum((labels1 == -1) & (labels2 != -1))
        patterns_stable = common_patterns

        # Calculate additional statistics
        unique_clusters1 = len(set(labels1[labels1 != -1]))
        unique_clusters2 = len(set(labels2[labels2 != -1]))

        similarity_scores.append({
            'Threshold1': t1,
            'Threshold2': t2,
            'ARI': ari_common,  # Keep original for compatibility
            'NMI': nmi_common,  # Keep original for compatibility
            'ARI_full': ari_full,  # New: includes missing patterns
            'NMI_full': nmi_full,  # New: includes missing patterns
            'Common_Patterns': common_patterns,
            'Total_Patterns1': pattern_counts_per_threshold[t1],
            'Total_Patterns2': pattern_counts_per_threshold[t2],
            'Patterns_Lost': patterns_lost,
            'Patterns_Gained': patterns_gained,
            'Patterns_Stable': patterns_stable,
            'Unique_Clusters1': unique_clusters1,
            'Unique_Clusters2': unique_clusters2
        })

        print(f"\nComparing thresholds {t1:.2f} and {t2:.2f}:")
        print(f"Pattern dynamics: {patterns_stable} stable, {patterns_lost} lost, {patterns_gained} gained")
        print(f"Clusters: {unique_clusters1} -> {unique_clusters2}")
        print(f"ARI (common only): {ari_common:.3f}, ARI (all patterns): {ari_full:.3f}")
        print(f"NMI (common only): {nmi_common:.3f}, NMI (all patterns): {nmi_full:.3f}")

    return similarity_scores

# Update the plotting function to show more information
def plot_similarity_scores(similarity_scores, output_dir):
    df_similarity = pd.DataFrame(similarity_scores)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot ARI comparison
    ax = axes[0, 0]
    ax.plot(df_similarity['Threshold1'], df_similarity['ARI'], 'o-', label='Common patterns only', linewidth=2)
    if 'ARI_full' in df_similarity.columns:
        ax.plot(df_similarity['Threshold1'], df_similarity['ARI_full'], 's--', label='All patterns', linewidth=2)
    ax.set_title('Adjusted Rand Index between Consecutive Thresholds')
    ax.set_ylabel('ARI Score')
    ax.set_xlabel('Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot NMI comparison
    ax = axes[0, 1]
    ax.plot(df_similarity['Threshold1'], df_similarity['NMI'], 'o-', label='Common patterns only', linewidth=2)
    if 'NMI_full' in df_similarity.columns:
        ax.plot(df_similarity['Threshold1'], df_similarity['NMI_full'], 's--', label='All patterns', linewidth=2)
    ax.set_title('Normalized Mutual Information between Consecutive Thresholds')
    ax.set_ylabel('NMI Score')
    ax.set_xlabel('Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot pattern dynamics
    ax = axes[1, 0]
    if 'Patterns_Stable' in df_similarity.columns:
        ax.plot(df_similarity['Threshold1'], df_similarity['Patterns_Stable'], 'o-', label='Stable', color='green')
        ax.plot(df_similarity['Threshold1'], df_similarity['Patterns_Lost'], 's-', label='Lost', color='red')
        ax.plot(df_similarity['Threshold1'], df_similarity['Patterns_Gained'], '^-', label='Gained', color='blue')
    else:
        ax.plot(df_similarity['Threshold1'], df_similarity['Common_Patterns'], 'o-', label='Common', color='green')
    ax.set_title('Pattern Dynamics Across Thresholds')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Number of Patterns')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot total patterns and clusters
    ax = axes[1, 1]
    ax.plot(df_similarity['Threshold1'], df_similarity['Total_Patterns1'], 'o-', label='Total Patterns', color='blue')
    ax2 = ax.twinx()
    ax2.plot(df_similarity['Threshold1'], df_similarity['Unique_Clusters1'], 's-', label='Unique Clusters', color='orange')
    ax.set_title('Pattern and Cluster Counts')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Number of Patterns', color='blue')
    ax2.set_ylabel('Number of Clusters', color='orange')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_similarity_scores.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'threshold_similarity_scores.svg'), dpi=300)
    plt.close()
    
def analyze_stability_results(thresholds, pattern_assignments, cluster_data, output_dir):
    """
    Analyze the stability of clustering results across different thresholds.
    
    Args:
        thresholds: List of thresholds used
        pattern_assignments: Dictionary mapping thresholds to pattern assignments
        cluster_data: Dictionary mapping thresholds to cluster information
        output_dir: Directory to save results
    """
    # 1. Calculate similarity scores using our new function
    similarity_scores = calculate_similarity_scores(pattern_assignments, thresholds)
    
    # 2. Track top clusters across thresholds
    top_clusters_data = []
    
    for threshold in thresholds:
        clusters = cluster_data[threshold]
        
        # Get top 5 clusters by size
        top_clusters = sorted(clusters.items(), key=lambda x: int(x[1]['size']), reverse=True)[:5]
        
        for rank, (cluster_id, data) in enumerate(top_clusters, 1):
            # Get cluster signature (e.g., pattern of active features)
            if 'consensus_pattern' in data:
                # Convert to tuple for hashability if it's a list
                signature = tuple(data['consensus_pattern'])
            else:
                signature = "no_pattern"
            
            # Count patterns by group
            group_counts = data.get('group_counts', {})
            
            top_clusters_data.append({
                'Threshold': threshold,
                'Rank': rank,
                'ClusterID': cluster_id,
                'Size': data['size'],
                'Signature': signature,
                'GroupCounts': group_counts,
                'Unique_Clusters': len(clusters),  # Add number of unique clusters
                'Total_Patterns': sum(int(c['size']) for c in clusters.values())  # Add total pattern count
            })
    
    # Create a summary DataFrame
    df_top_clusters = pd.DataFrame(top_clusters_data)
    
    # Calculate Jaccard similarity for tracking clusters across thresholds
    cluster_tracking = track_clusters_across_thresholds(df_top_clusters, output_dir)
    
    # 3. Generate summary figures using our new plotting function
    plot_similarity_scores(similarity_scores, output_dir)
    
    # Create additional stability visualizations
    create_stability_visualizations(similarity_scores, df_top_clusters, cluster_tracking, output_dir)
    
    # 4. Save detailed tabular data
    df_similarity = pd.DataFrame(similarity_scores)
    df_similarity.to_csv(os.path.join(output_dir, 'threshold_similarity_scores.csv'), index=False)
    df_top_clusters.to_csv(os.path.join(output_dir, 'top_clusters_by_threshold.csv'), index=False)
    
    # Log comprehensive summary statistics
    logger.info("\nStability Analysis Summary:")
    logger.info(f"Number of thresholds analyzed: {len(thresholds)}")
    logger.info(f"Average ARI between consecutive thresholds: {df_similarity['ARI'].mean():.3f}")
    logger.info(f"Average NMI between consecutive thresholds: {df_similarity['NMI'].mean():.3f}")
    
    # Check if our new metrics are in the dataframe before logging them
    if 'Common_Patterns' in df_similarity.columns:
        logger.info(f"Average number of common patterns: {df_similarity['Common_Patterns'].mean():.1f}")
    
    # Log threshold-specific statistics
    for threshold in thresholds:
        threshold_data = df_top_clusters[df_top_clusters['Threshold'] == threshold]
        if not threshold_data.empty:
            logger.info(f"\nThreshold {threshold:.2f}:")
            if 'Unique_Clusters' in threshold_data.columns:
                logger.info(f"  Total unique clusters: {threshold_data['Unique_Clusters'].iloc[0]}")
            if 'Total_Patterns' in threshold_data.columns:
                logger.info(f"  Total patterns: {threshold_data['Total_Patterns'].iloc[0]}")
            logger.info(f"  Size of largest cluster: {threshold_data['Size'].iloc[0]}")
    
    return df_similarity, df_top_clusters, cluster_tracking


def track_clusters_across_thresholds(df_top_clusters, output_dir):
    """Track how clusters evolve across thresholds using pattern signatures."""
    # First collect all signatures and count their occurrences
    signature_counts = {}
    for _, row in df_top_clusters.iterrows():
        signature = row['Signature']
        if signature == "no_pattern":
            continue
            
        # Convert signature to hashable form if needed
        if not isinstance(signature, (str, tuple)):
            signature = str(signature)
            
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    
    # Sort signatures by count (persistence) in descending order
    sorted_signatures = sorted(signature_counts.items(), 
                             key=lambda x: (-x[1], x[0]))  # Sort by count desc, then signature
    
    # Create mapping with sorted group IDs
    signature_to_group = {
        signature: idx + 1 
        for idx, (signature, _) in enumerate(sorted_signatures)
    }
    
    # Now collect tracking data with sorted group IDs
    tracking_data = []
    for _, row in df_top_clusters.iterrows():
        signature = row['Signature']
        
        if signature == "no_pattern":
            continue
        
        # Convert signature to hashable form if needed
        if not isinstance(signature, (str, tuple)):
            signature = str(signature)
        
        # Get group_id from mapping, with fallback
        try:
            group_id = signature_to_group.get(signature)
            if group_id is None:
                print(f"Warning: No group ID found for signature: {signature}")
                continue
        except Exception as e:
            print(f"Error processing signature: {signature}")
            print(f"Error details: {str(e)}")
            continue
        
        tracking_data.append({
            'Threshold': row['Threshold'],
            'ClusterID': row['ClusterID'],
            'Rank': row['Rank'],
            'Size': row['Size'],
            'ClusterGroup': f"G{group_id}",
            'Signature': signature,
            'Persistence': signature_counts[signature]  # Add persistence count
        })
    
    # Convert to DataFrame
    df_tracking = pd.DataFrame(tracking_data)
    
    if df_tracking.empty:
        raise ValueError("No valid tracking data generated")
    
    # Calculate additional statistics
    stats = {
        'total_patterns': len(signature_to_group),
        'patterns_by_threshold': df_tracking.groupby('Threshold')['ClusterGroup'].nunique().to_dict(),
        'most_persistent_patterns': df_tracking[df_tracking['Persistence'] == df_tracking['Persistence'].max()]['ClusterGroup'].unique().tolist(),
        'persistence_by_group': df_tracking.groupby('ClusterGroup')['Persistence'].first().to_dict()
    }
    
    # Save main tracking DataFrame
    tracking_file = os.path.join(output_dir, 'cluster_tracking.csv')
    df_tracking.to_csv(tracking_file, index=False)
    
    # Save statistics as JSON
    stats_file = os.path.join(output_dir, 'tracking_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Saved cluster tracking results to: {output_dir}")
    print(f"Found {stats['total_patterns']} unique patterns across all thresholds")
    print("Persistence by group:")
    for group, count in sorted(stats['persistence_by_group'].items()):
        print(f"  {group}: appears in {count} thresholds")
    print(f"Most persistent patterns: {', '.join(stats['most_persistent_patterns'])}")
    
    return df_tracking

def plot_cluster_tracking(cluster_tracking, output_dir):
    """
    Create enhanced visualizations for cluster tracking across thresholds.
    """
    # Get total number of groups at the start
    total_groups = len(cluster_tracking['ClusterGroup'].unique())
    
    # Create main subplot for tracking
    plt.figure(figsize=(5, 3))
    gs = plt.GridSpec(1, 5)
    ax_main = plt.subplot(gs[0, :4])
    ax_persist = plt.subplot(gs[0, 4])
    
    # Get unique persistence values for each group
    persistence_by_group = cluster_tracking.groupby('ClusterGroup')['Persistence'].first()
    
    # Create group order and color mapping
    group_order = [f"G{i}" for i in range(1, total_groups + 1)]
    group_order.reverse()  # G1 at top
    
    # Create color mapping using Set3
    color_map = {
        group: plt.cm.tab10(i % 10)  # divide by 10 since tab10 expects values between 0 and 1
        for i, group in enumerate(group_order)
    }
    
    # Plot main tracking visualization
    for group_name, group_data in cluster_tracking.groupby('ClusterGroup'):
        group_data = group_data.sort_values('Threshold')
        
        # Use color from color_map
        color = color_map[group_name]
        persistence = group_data['Persistence'].iloc[0]
        
        # Plot points with size proportional to cluster size
        ax_main.scatter(group_data['Threshold'], group_data['Rank'], 
                       s=group_data['Size'], 
                       color=color,
                       label=f"{group_name} (n={persistence})",
                       alpha=1)        
        # Connect points with lines
        ax_main.plot(group_data['Threshold'], group_data['Rank'], 
                    color=color, linestyle='-', alpha=1)
        
        # Add cluster ID and size labels
        # for _, row in group_data.iterrows():
        #     ax_main.text(row['Threshold'], row['Rank'], 
        #                 f"(n={int(row['Size'])})", 
        #                 fontsize=6, ha='center', va='center')
    
    # set ytickslabels to be C1, C2, C3, etc.
    ax_main.set_yticks(range(1, len(persistence_by_group) + 1))  # Start from 1
    ax_main.set_yticklabels([f"C{i}" for i in range(1, len(persistence_by_group) + 1)])
    ax_main.invert_yaxis()
    ax_main.set_title('Pattern Evolution Across Thresholds')
    ax_main.set_xlabel('')
    ax_main.set_ylabel('')
    # ax_main.set_xlabel('Similarity Threshold')
    # ax_main.set_ylabel('Pattern Rank')
    ax_main.grid(True, alpha=0.2)
    ax_main.set_ylim(5.5, 0.5)
    
    # Persistence bar plot with same colors
    persistence_by_group = persistence_by_group[group_order]
    
    bars = ax_persist.barh(range(len(persistence_by_group)), 
                          persistence_by_group.values,
                          color=[color_map[group] for group in group_order])
    
    # remove top and right spines
    # ax_persist.spines['top'].set_visible(False)
    # ax_persist.spines['right'].set_visible(False)
    ax_persist.set_xlim(0, 7)
    ax_persist.set_yticks(range(len(persistence_by_group)))
    ax_persist.set_yticklabels(group_order)  # Use reversed group order
    # ax_persist.set_xlabel('# Thresholds')
    ax_persist.set_title('Pattern Persistence')
    
    # # Add threshold range annotations (using reversed order)
    # for idx, group in enumerate(group_order):
    #     group_data = cluster_tracking[cluster_tracking['ClusterGroup'] == group]
    #     thresh_range = f"{group_data['Threshold'].min():.2f}-{group_data['Threshold'].max():.2f}"
    #     ax_persist.text(persistence_by_group[group], idx, 
    #                    f" Th: [{thresh_range}]", 
    #                    va='center', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_tracking_main.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cluster_tracking_main.svg'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pattern size evolution plot
    plt.figure(figsize=(4, 2))
    
    for group_name, group_data in cluster_tracking.groupby('ClusterGroup'):
        group_data = group_data.sort_values('Threshold')
        group_num = int(group_name.replace('G', ''))
        color = plt.cm.Set3(1 - (group_num - 1) / total_groups)  # Consistent color scheme
        
        persistence = group_data['Persistence'].iloc[0]
        plt.plot(group_data['Threshold'], group_data['Size'], 
                'o-', color=color, 
                label=f"{group_name} (n={persistence})",
                alpha=0.7)
    
    plt.title('Pattern Size Evolution Across Thresholds')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Pattern Size (# States)')
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_tracking_sizes.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cluster_tracking_sizes.svg'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Pattern stability heatmap
    thresholds = sorted(cluster_tracking['Threshold'].unique())
    groups = sorted(cluster_tracking['ClusterGroup'].unique(), 
                   key=lambda x: int(x.replace('G', '')))  # Sort by group number
    
    stability_matrix = np.zeros((len(groups), len(thresholds)))
    for i, group in enumerate(groups):
        for j, thresh in enumerate(thresholds):
            mask = (cluster_tracking['ClusterGroup'] == group) & \
                  (cluster_tracking['Threshold'] == thresh)
            if mask.any():
                stability_matrix[i, j] = cluster_tracking[mask]['Size'].iloc[0]
    
    plt.figure(figsize=(4, 2))
    sns.heatmap(stability_matrix, 
                xticklabels=[f"{t:.2f}" for t in thresholds],
                yticklabels=groups,
                cmap='YlOrRd',
                annot=True,
                fmt='.0f',
                annot_kws={'fontsize': 6},
                cbar = False)
    
    
    # plt.title('Pattern Stability and Size Across Thresholds')
    # plt.xlabel('Similarity Threshold')
    # plt.ylabel('Pattern Group')
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_tracking_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cluster_tracking_heatmap.svg'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_lifecycle_visualization(df_lifecycles, thresholds, output_dir):
    """
    Create visualizations for pattern lifecycles.

    Args:
        df_lifecycles: DataFrame with lifecycle statistics
        thresholds: List of threshold values
        output_dir: Directory to save visualizations
    """
    # 1. Persistence distribution by group
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Persistence histogram
    ax = axes[0, 0]
    for group in df_lifecycles['group'].unique():
        group_df = df_lifecycles[df_lifecycles['group'] == group]
        ax.hist(group_df['persistence'], bins=10, alpha=0.5, label=group)
    ax.set_xlabel('Persistence (fraction of thresholds)')
    ax.set_ylabel('Number of Patterns')
    ax.set_title('Pattern Persistence Distribution')
    ax.legend()

    # Size trend by group
    ax = axes[0, 1]
    size_trend_counts = df_lifecycles.groupby(['group', 'size_trend']).size().unstack(fill_value=0)
    size_trend_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Group')
    ax.set_ylabel('Number of Patterns')
    ax.set_title('Pattern Size Evolution')
    ax.legend(title='Size Trend')

    # Cluster changes distribution
    ax = axes[1, 0]
    df_lifecycles.groupby('group')['cluster_changes'].hist(ax=ax, alpha=0.5, bins=range(0,
                                                            df_lifecycles['cluster_changes'].max() + 2))
    ax.set_xlabel('Number of Cluster Changes')
    ax.set_ylabel('Number of Patterns')
    ax.set_title('Pattern Cluster Stability')

    # Patterns with gaps
    ax = axes[1, 1]
    gap_counts = df_lifecycles.groupby(['group', 'has_gaps']).size().unstack(fill_value=0)
    gap_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Group')
    ax.set_ylabel('Number of Patterns')
    ax.set_title('Patterns with Discontinuous Presence')
    ax.legend(title='Has Gaps', labels=['Continuous', 'Discontinuous'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pattern_lifecycles.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'pattern_lifecycles.svg'), dpi=300)
    plt.close()

    # 2. Lifecycle timeline visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort patterns by group and persistence for better visualization
    df_sorted = df_lifecycles.sort_values(['group', 'persistence'], ascending=[True, False])

    # Create a timeline for top persistent patterns (limit for visibility)
    max_patterns_to_show = 50
    patterns_to_show = df_sorted.head(max_patterns_to_show)

    y_positions = {}
    colors_map = {'affair': '#E1BE6A', 'paranoia': '#40B0A6', 'combined': '#6B15A7', 'balanced': '#3096DF'}

    for i, (_, pattern) in enumerate(patterns_to_show.iterrows()):
        y_positions[pattern['pattern_id']] = i

        # Draw line for threshold range
        ax.plot([pattern['first_seen'], pattern['last_seen']], [i, i],
               linewidth=3, color=colors_map.get(pattern['group'], 'gray'),
               alpha=0.7, solid_capstyle='round')

        # Mark if pattern has gaps
        if pattern['has_gaps']:
            ax.scatter(pattern['first_seen'] + (pattern['last_seen'] - pattern['first_seen'])/2,
                      i, marker='x', s=30, color='red', zorder=5)

    ax.set_yticks(range(len(patterns_to_show)))
    ax.set_yticklabels([f"{p['group']}_{p['pattern_id'].split('_')[-1]}"
                        for _, p in patterns_to_show.iterrows()], fontsize=8)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Pattern ID')
    ax.set_title(f'Pattern Lifecycle Timeline (Top {len(patterns_to_show)} Most Persistent)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, lw=3, label=g)
                      for g, c in colors_map.items() if g in patterns_to_show['group'].values]
    legend_elements.append(Line2D([0], [0], marker='x', color='red', lw=0, label='Has Gaps'))
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pattern_lifecycle_timeline.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pattern_lifecycle_timeline.svg'), dpi=300, bbox_inches='tight')
    plt.close()

def create_stability_visualizations(similarity_scores, df_top_clusters, cluster_tracking, output_dir):
    """
    Create visualizations for stability analysis results.
    
    Args:
        similarity_scores: Matrix of similarity scores between patterns
        df_top_clusters: DataFrame containing top clusters information
        cluster_tracking: DataFrame with cluster tracking information
        thresholds: List of threshold values used
        output_dir: Directory to save visualization outputs
    """
    # 2. Plot top cluster sizes by threshold
    plt.figure(figsize=(12, 8))
    
    # Transform data for better plotting
    df_plot = df_top_clusters.pivot(index='Threshold', columns='Rank', values='Size')
    
    # Plot cluster sizes
    sns.heatmap(df_plot, annot=True, fmt='.0f', cmap='YlGnBu', 
                linewidths=0.5, cbar_kws={'label': 'Cluster Size'})
    
    plt.title('Top 5 Cluster Sizes by Threshold')
    plt.ylabel('Similarity Threshold')
    plt.xlabel('Cluster Rank')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_cluster_sizes.png'), dpi=300)
    plt.close()
    
    # 3. Visualize cluster tracking
    if not cluster_tracking.empty:
        plot_cluster_tracking(cluster_tracking, output_dir)
    
    # 4. Group composition analysis
    if 'GroupCounts' in df_top_clusters.columns:
        # Unpack the group counts from each row
        group_data = []
        
        for _, row in df_top_clusters.iterrows():
            threshold = row['Threshold']
            cluster_id = row['ClusterID']
            rank = row['Rank']
            
            # Extract group counts
            group_counts = row['GroupCounts']
            if isinstance(group_counts, dict):
                for group, count in group_counts.items():
                    if count > 0:
                        group_data.append({
                            'Threshold': threshold,
                            'ClusterID': cluster_id,
                            'Rank': rank,
                            'Group': group,
                            'Count': count
                        })
        
if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    load_dotenv()
        
    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = os.path.join(scratch_dir, "output_RR")
    output_dir = os.path.join(base_dir, "06_state_pattern_cluster")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    groups = ["affair", "paranoia", "combined", "balanced"]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(output_dir, '06_state_pattern_cluster.log'))]
    )
    logger = logging.getLogger(__name__)

    # Run analysis
    analyze_cluster_stability(
        base_dir=base_dir,
        groups=groups,
        output_dir=output_dir,
        thresholds=thresholds
    )