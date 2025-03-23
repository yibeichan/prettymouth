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
        
        pattern_cluster_assignments[threshold] = pattern_to_cluster
    
    # Now analyze the stability across thresholds
    analyze_stability_results(thresholds, pattern_cluster_assignments, cluster_data, output_dir)
    
    return threshold_dirs, cluster_data

def calculate_similarity_scores(pattern_assignments, thresholds):
    """
    Calculate similarity scores between consecutive thresholds with additional checks.
    
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
    pattern_counts_per_threshold = {}
    
    for threshold in thresholds:
        assignments = pattern_assignments[threshold]
        labels = np.zeros(len(all_patterns), dtype=int)
        n_patterns = 0
        
        for i, pattern_id in enumerate(all_patterns):
            if pattern_id in assignments:
                labels[i] = assignments[pattern_id]
                n_patterns += 1
            else:
                labels[i] = -1
        
        threshold_labels[threshold] = labels
        pattern_counts_per_threshold[threshold] = n_patterns
        
        print(f"Threshold {threshold:.2f}: {n_patterns} patterns, "
              f"{len(set(labels[labels != -1]))} unique clusters")
    
    # Compare consecutive thresholds
    for i in range(len(thresholds) - 1):
        t1 = thresholds[i]
        t2 = thresholds[i + 1]
        
        labels1 = threshold_labels[t1]
        labels2 = threshold_labels[t2]
        
        # Get patterns present in both thresholds
        mask = (labels1 != -1) & (labels2 != -1)
        common_patterns = np.sum(mask)
        
        if common_patterns > 0:
            # Calculate metrics only on common patterns
            ari = adjusted_rand_score(labels1[mask], labels2[mask])
            nmi = normalized_mutual_info_score(labels1[mask], labels2[mask])
            
            # Calculate additional statistics
            unique_clusters1 = len(set(labels1[mask]))
            unique_clusters2 = len(set(labels2[mask]))
            
            similarity_scores.append({
                'Threshold1': t1,
                'Threshold2': t2,
                'ARI': ari,
                'NMI': nmi,
                'Common_Patterns': common_patterns,
                'Total_Patterns1': pattern_counts_per_threshold[t1],
                'Total_Patterns2': pattern_counts_per_threshold[t2],
                'Unique_Clusters1': unique_clusters1,
                'Unique_Clusters2': unique_clusters2
            })
            
            print(f"\nComparing thresholds {t1:.2f} and {t2:.2f}:")
            print(f"Common patterns: {common_patterns}")
            print(f"Clusters: {unique_clusters1} -> {unique_clusters2}")
            print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
    
    return similarity_scores

# Update the plotting function to show more information
def plot_similarity_scores(similarity_scores, output_dir):
    df_similarity = pd.DataFrame(similarity_scores)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot ARI
    sns.lineplot(ax=ax1, x='Threshold1', y='ARI', data=df_similarity, 
                marker='o', linewidth=2)
    ax1.set_title('Adjusted Rand Index between Consecutive Thresholds')
    ax1.set_ylabel('ARI Score')
    ax1.grid(True, alpha=0.3)
    
    # Plot NMI
    sns.lineplot(ax=ax2, x='Threshold1', y='NMI', data=df_similarity, 
                marker='o', linewidth=2, color='orange')
    ax2.set_title('Normalized Mutual Information between Consecutive Thresholds')
    ax2.set_ylabel('NMI Score')
    ax2.grid(True, alpha=0.3)
    
    # Plot pattern counts
    sns.lineplot(ax=ax3, x='Threshold1', y='Common_Patterns', data=df_similarity,
                marker='o', label='Common Patterns', color='green')
    sns.lineplot(ax=ax3, x='Threshold1', y='Total_Patterns1', data=df_similarity,
                marker='s', label='Total Patterns', color='blue')
    ax3.set_title('Pattern Counts across Thresholds')
    ax3.set_xlabel('Similarity Threshold')
    ax3.set_ylabel('Number of Patterns')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
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
    base_dir = os.path.join(scratch_dir, "output")
    output_dir = os.path.join(base_dir, "06_state_pattern_cluster")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    groups = ["affair", "paranoia", "combined"]

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