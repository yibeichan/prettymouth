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


def analyze_stability_results(thresholds, pattern_assignments, cluster_data, output_dir):
    """
    Analyze the stability of clustering results across different thresholds.
    
    Args:
        thresholds: List of thresholds used
        pattern_assignments: Dictionary mapping thresholds to pattern assignments
        cluster_data: Dictionary mapping thresholds to cluster information
        output_dir: Directory to save results
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing stability of clustering across thresholds")
    
    # 1. Compare consecutive thresholds using ARI and NMI
    similarity_scores = []
    
    # Create lists of cluster assignments for each threshold
    threshold_labels = {}
    all_patterns = set()
    
    # Get all pattern IDs across all thresholds
    for threshold in thresholds:
        all_patterns.update(pattern_assignments[threshold].keys())
    
    # Create sorted list of pattern IDs
    all_patterns = sorted(list(all_patterns))
    
    # For each threshold, create array of cluster assignments
    for threshold in thresholds:
        assignments = pattern_assignments[threshold]
        labels = np.zeros(len(all_patterns), dtype=int)
        
        for i, pattern_id in enumerate(all_patterns):
            labels[i] = assignments.get(pattern_id, -1)  # -1 for patterns not in this threshold
        
        threshold_labels[threshold] = labels
    
    # Compare consecutive thresholds
    for i in range(len(thresholds) - 1):
        t1 = thresholds[i]
        t2 = thresholds[i + 1]
        
        labels1 = threshold_labels[t1]
        labels2 = threshold_labels[t2]
        
        # Only compare patterns present in both thresholds
        mask = (labels1 != -1) & (labels2 != -1)
        
        if np.sum(mask) > 0:
            ari = adjusted_rand_score(labels1[mask], labels2[mask])
            nmi = normalized_mutual_info_score(labels1[mask], labels2[mask])
            
            similarity_scores.append({
                'Threshold1': t1,
                'Threshold2': t2,
                'ARI': ari,
                'NMI': nmi,
                'Patterns': np.sum(mask)
            })
    
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
                'GroupCounts': group_counts
            })
    
    # Create a summary DataFrame
    df_top_clusters = pd.DataFrame(top_clusters_data)
    
    # Calculate Jaccard similarity for tracking clusters across thresholds
    cluster_tracking = track_clusters_across_thresholds(df_top_clusters, thresholds)
    
    # 3. Generate summary figures
    create_stability_visualizations(similarity_scores, df_top_clusters, cluster_tracking, thresholds, output_dir)
    
    # 4. Save tabular data
    df_similarity = pd.DataFrame(similarity_scores)
    df_similarity.to_csv(os.path.join(output_dir, 'threshold_similarity_scores.csv'), index=False)
    df_top_clusters.to_csv(os.path.join(output_dir, 'top_clusters_by_threshold.csv'), index=False)
    
    # Log summary statistics
    logger.info(f"Average ARI between consecutive thresholds: {df_similarity['ARI'].mean():.3f}")
    logger.info(f"Average NMI between consecutive thresholds: {df_similarity['NMI'].mean():.3f}")
    
    return df_similarity, df_top_clusters, cluster_tracking


def track_clusters_across_thresholds(df_top_clusters, thresholds):
    """
    Track how clusters evolve across thresholds using pattern signatures.
    
    Args:
        df_top_clusters: DataFrame with top clusters for each threshold
        thresholds: List of thresholds
    
    Returns:
        DataFrame with cluster tracking information
    """
    # Group by signature to find similar clusters across thresholds
    tracking_data = []
    
    # Create a dictionary to track cluster groups
    signature_to_group = {}
    next_group_id = 1
    
    # For each threshold and cluster
    for _, row in df_top_clusters.iterrows():
        signature = row['Signature']
        
        # Skip if no signature (should not happen for consensus patterns)
        if signature == "no_pattern":
            continue
        
        # Convert to a hashable form if needed (e.g., string representation)
        if not isinstance(signature, (str, tuple)):
            signature = str(signature)
        
        # Assign to group or create new group
        if signature in signature_to_group:
            group_id = signature_to_group[signature]
        else:
            group_id = next_group_id
            signature_to_group[signature] = group_id
            next_group_id += 1
        
        # Store tracking data
        tracking_data.append({
            'Threshold': row['Threshold'],
            'ClusterID': row['ClusterID'],
            'Rank': row['Rank'],
            'Size': row['Size'],
            'ClusterGroup': f"Group_{group_id}"
        })
    
    return pd.DataFrame(tracking_data)


def create_stability_visualizations(similarity_scores, df_top_clusters, cluster_tracking, thresholds, output_dir):
    """
    Create visualizations for cluster stability analysis.
    
    Args:
        similarity_scores: List of dicts with ARI and NMI scores
        df_top_clusters: DataFrame with top clusters for each threshold
        cluster_tracking: DataFrame with cluster tracking across thresholds
        thresholds: List of thresholds
        output_dir: Directory to save visualizations
    """
    # 1. Plot ARI and NMI scores
    df_similarity = pd.DataFrame(similarity_scores)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    sns.lineplot(x='Threshold1', y='ARI', data=df_similarity, marker='o', linewidth=2)
    plt.title('Adjusted Rand Index between Consecutive Thresholds')
    plt.ylabel('ARI Score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    sns.lineplot(x='Threshold1', y='NMI', data=df_similarity, marker='o', linewidth=2, color='orange')
    plt.title('Normalized Mutual Information between Consecutive Thresholds')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('NMI Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_similarity_scores.png'), dpi=300)
    plt.close()
    
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
        plt.figure(figsize=(14, 8))
        
        # Create a scatter plot with connected lines for each cluster group
        for group_name, group_data in cluster_tracking.groupby('ClusterGroup'):
            # Sort by threshold
            group_data = group_data.sort_values('Threshold')
            
            # Plot points with size proportional to cluster size
            plt.scatter(group_data['Threshold'], group_data['Rank'], 
                       s=group_data['Size']/10, label=group_name, alpha=0.7)
            
            # Connect points with lines
            plt.plot(group_data['Threshold'], group_data['Rank'], 'k-', alpha=0.3)
            
            # Add cluster ID labels
            for _, row in group_data.iterrows():
                plt.text(row['Threshold'], row['Rank'], f"C{row['ClusterID']}", 
                        fontsize=8, ha='center', va='center')
        
        plt.gca().invert_yaxis()  # Invert y-axis so rank 1 is at the top
        plt.title('Tracking Top Clusters Across Similarity Thresholds')
        plt.xlabel('Similarity Threshold')
        plt.ylabel('Cluster Rank')
        plt.grid(True, alpha=0.2)
        plt.ylim(5.5, 0.5)  # Set fixed y-axis limits
        
        # Legend with smaller font and outside plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_tracking.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
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
        
        if group_data:
            df_groups = pd.DataFrame(group_data)
            
            # Plot group composition
            plt.figure(figsize=(14, 10))
            
            # Plot stacked bars for each threshold and rank
            pivot_data = df_groups.pivot_table(
                index=['Threshold', 'Rank'], 
                columns='Group', 
                values='Count',
                aggfunc='sum',
                fill_value=0
            )
            
            # Reset index for plotting
            plot_data = pivot_data.reset_index()
            
            # Create a categorical column for x-axis
            plot_data['ThresholdRank'] = plot_data['Threshold'].astype(str) + '-' + plot_data['Rank'].astype(str)
            
            # Prepare data for plotting
            melted_data = pd.melt(
                plot_data, 
                id_vars=['ThresholdRank', 'Threshold', 'Rank'], 
                var_name='Group', 
                value_name='Count'
            )
            
            # Order by threshold then rank
            melted_data = melted_data.sort_values(['Threshold', 'Rank'])
            
            # Create a categorical x-axis variable
            threshold_rank_order = melted_data['ThresholdRank'].unique()
            melted_data['ThresholdRank'] = pd.Categorical(
                melted_data['ThresholdRank'], 
                categories=threshold_rank_order, 
                ordered=True
            )
            
            # Plot
            sns.barplot(x='ThresholdRank', y='Count', hue='Group', data=melted_data)
            
            plt.title('Group Composition of Top Clusters Across Thresholds')
            plt.xlabel('Threshold-Rank')
            plt.ylabel('Pattern Count')
            plt.xticks(rotation=90)
            
            # Add a grid for readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'group_composition.png'), dpi=300)
            plt.close()


if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    load_dotenv()
        
    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = os.path.join(scratch_dir, "output")
    output_dir = os.path.join(base_dir, "06_state_pattern_cluster")
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    groups = ["affair", "paranoia", "combined"]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Run analysis
    analyze_cluster_stability(
        base_dir=base_dir,
        groups=groups,
        output_dir=output_dir,
        thresholds=thresholds
    )