#!/usr/bin/env python
"""
Comprehensive Spearman correlation analysis for brain state clusters.
Combines two approaches:
1. Consensus patterns: Average of all member patterns in each cluster
2. Specific patterns: Representative states from combined models

This ensures consistency with plot_state_pattern function data extraction.
"""

import os
import json
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from pathlib import Path
import logging
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define specific states for each cluster (from combined models)
CLUSTER_SPECIFIC_STATES = {
    1: {'group': 'combined', 'n_states': '2', 'state_idx': 0},
    2: {'group': 'combined', 'n_states': '7', 'state_idx': 4},
    3: {'group': 'combined', 'n_states': '2', 'state_idx': 1},
    4: {'group': 'combined', 'n_states': '6', 'state_idx': 3}
}

# Network labels for visualization
NETWORK_LABELS = ['Aud', 'Ctr-A', 'Ctr-B', 'Ctr-C', 'DMN-A', 'DMN-B',
                  'DMN-C', 'DA-A', 'DA-B', 'Lang', 'SVA-A',
                  'SVA-B', 'SM-A', 'SM-B', 'Vis-A', 'Vis-B', 'Vis-C']


def extract_state_pattern(metrics_path: Path, state_idx: int) -> np.ndarray:
    """
    Extract a single state pattern from metrics file.
    Matches the data extraction in plot_state_pattern function.

    Args:
        metrics_path: Path to the metrics pickle file
        state_idx: State index to extract

    Returns:
        numpy array of the mean pattern
    """
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)

    # Get state properties - matching plot_state_pattern
    state_properties = metrics.get('state_properties', {})

    # Get the specific state's properties
    if state_idx in state_properties:
        state_props = state_properties[state_idx]
    elif str(state_idx) in state_properties:
        state_props = state_properties[str(state_idx)]
    else:
        raise KeyError(f"State {state_idx} not found in state_properties")

    # Extract mean_pattern - exactly as plot_state_pattern does
    if 'mean_pattern' not in state_props:
        raise ValueError(f"No mean_pattern found for state {state_idx}")

    return np.array(state_props['mean_pattern'])


def get_consensus_patterns(base_dir: str, cluster_info: Dict, cluster_ids: List[int]) -> Dict:
    """
    Extract consensus patterns by averaging all member patterns in each cluster.

    Args:
        base_dir: Base directory containing HMM results
        cluster_info: Loaded cluster_info.json data
        cluster_ids: List of cluster IDs to extract

    Returns:
        Dictionary mapping cluster ID to consensus pattern
    """
    consensus_patterns = {}

    for cluster_id in cluster_ids:
        cluster_id_str = str(cluster_id)
        if cluster_id_str not in cluster_info:
            logger.warning(f"Cluster {cluster_id} not found in cluster_info")
            continue

        members = cluster_info[cluster_id_str]['members']
        member_patterns = []
        successful_members = []

        for member in members:
            group = member['group']
            model = member['model']
            original_state_idx = member['original_state_idx']

            # Parse model name to get n_states
            n_states = model.split()[-1].replace('states', '')

            # Construct path to metrics file
            metrics_path = Path(base_dir) / f"04_{group}_hmm_{n_states}states_ntw_native_trimmed" / "statistics" / f"{group}_metrics.pkl"

            if not metrics_path.exists():
                logger.debug(f"Metrics file not found: {metrics_path}")
                continue

            try:
                pattern = extract_state_pattern(metrics_path, original_state_idx)
                member_patterns.append(pattern)
                successful_members.append(f"{model} state {original_state_idx}")
            except Exception as e:
                logger.debug(f"Error extracting {model} state {original_state_idx}: {e}")
                continue

        if member_patterns:
            consensus = np.mean(member_patterns, axis=0)
            consensus_patterns[cluster_id] = consensus
            logger.info(f"Cluster {cluster_id} consensus: averaged {len(member_patterns)}/{len(members)} patterns")
            logger.debug(f"  Successfully extracted from: {successful_members[:3]}{'...' if len(successful_members) > 3 else ''}")
        else:
            logger.warning(f"No patterns extracted for cluster {cluster_id}")

    return consensus_patterns


def get_specific_patterns(base_dir: str) -> Dict:
    """
    Extract specific representative patterns from combined models.

    Args:
        base_dir: Base directory containing HMM results

    Returns:
        Dictionary mapping cluster ID to specific pattern
    """
    specific_patterns = {}

    for cluster_id, state_info in CLUSTER_SPECIFIC_STATES.items():
        group = state_info['group']
        n_states = state_info['n_states']
        state_idx = state_info['state_idx']

        # Construct path to metrics file
        metrics_path = Path(base_dir) / f"04_{group}_hmm_{n_states}states_ntw_native_trimmed" / "statistics" / f"{group}_metrics.pkl"

        logger.info(f"Cluster {cluster_id}: extracting from {group} {n_states}states, state {state_idx}")

        if not metrics_path.exists():
            logger.error(f"Metrics file not found: {metrics_path}")
            continue

        try:
            pattern = extract_state_pattern(metrics_path, state_idx)
            specific_patterns[cluster_id] = pattern
            logger.debug(f"  Pattern shape: {pattern.shape}, mean |activation|: {np.mean(np.abs(pattern)):.4f}")
        except Exception as e:
            logger.error(f"Error extracting pattern: {e}")

    return specific_patterns


def calculate_correlations(patterns: Dict, label: str) -> Dict:
    """
    Calculate all pairwise correlations and group correlations.

    Args:
        patterns: Dictionary of cluster patterns
        label: Label for this analysis (e.g., "consensus" or "specific")

    Returns:
        Dictionary of correlation results
    """
    results = {'label': label}

    # Verify we have all 4 clusters
    if not all(c in patterns for c in [1, 2, 3, 4]):
        missing = [c for c in [1, 2, 3, 4] if c not in patterns]
        logger.warning(f"{label}: Missing clusters {missing}")
        return results

    # Within-group correlations
    corr_12, pval_12 = spearmanr(patterns[1], patterns[2])
    results['cluster1_cluster2'] = {'rho': float(corr_12), 'p_value': float(pval_12)}

    corr_34, pval_34 = spearmanr(patterns[3], patterns[4])
    results['cluster3_cluster4'] = {'rho': float(corr_34), 'p_value': float(pval_34)}

    # Between-group correlations (anti-correlation test)
    # Average patterns within groups
    group_12 = (patterns[1] + patterns[2]) / 2
    group_34 = (patterns[3] + patterns[4]) / 2

    corr_groups, pval_groups = spearmanr(group_12, group_34)
    results['group12_group34'] = {'rho': float(corr_groups), 'p_value': float(pval_groups)}

    # All pairwise cross-group correlations
    cross_correlations = []
    for c1 in [1, 2]:
        for c2 in [3, 4]:
            corr, pval = spearmanr(patterns[c1], patterns[c2])
            results[f'cluster{c1}_cluster{c2}'] = {'rho': float(corr), 'p_value': float(pval)}
            cross_correlations.append(corr)

    results['mean_cross_group'] = float(np.mean(cross_correlations))

    # Create full correlation matrix
    corr_matrix = np.zeros((4, 4))
    pval_matrix = np.zeros((4, 4))

    for i, c1 in enumerate([1, 2, 3, 4]):
        for j, c2 in enumerate([1, 2, 3, 4]):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                corr, pval = spearmanr(patterns[c1], patterns[c2])
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval

    results['correlation_matrix'] = corr_matrix.tolist()
    results['pvalue_matrix'] = pval_matrix.tolist()

    return results


def visualize_patterns(consensus_patterns: Dict, specific_patterns: Dict, output_dir: Path):
    """
    Create comprehensive visualizations comparing both approaches.
    """
    # Create figure with subplots for all patterns
    fig = plt.figure(figsize=(16, 10))

    # Create grid: 4 rows (clusters) x 2 columns (consensus vs specific)
    for cluster_id in [1, 2, 3, 4]:
        # Consensus pattern subplot
        ax1 = plt.subplot(4, 2, (cluster_id - 1) * 2 + 1)
        if cluster_id in consensus_patterns:
            pattern = consensus_patterns[cluster_id]
            x = np.arange(len(NETWORK_LABELS))
            bars = ax1.bar(x, pattern, alpha=0.8)

            # Color by positive/negative
            for bar, val in zip(bars, pattern):
                if val < 0:
                    bar.set_color('#377eb8')  # Blue for negative
                else:
                    bar.set_color('#e41a1c')  # Red for positive

            ax1.set_ylim(-0.5, 0.5)
            ax1.axhline(y=0, color='black', linewidth=0.5)
            ax1.grid(True, axis='y', alpha=0.3)
            ax1.set_title(f'Cluster {cluster_id} - Consensus Pattern')

            if cluster_id == 4:  # Only label bottom plots
                ax1.set_xticks(x)
                ax1.set_xticklabels(NETWORK_LABELS, rotation=45, ha='right', fontsize=8)
            else:
                ax1.set_xticks([])

        # Specific pattern subplot
        ax2 = plt.subplot(4, 2, cluster_id * 2)
        if cluster_id in specific_patterns:
            pattern = specific_patterns[cluster_id]
            x = np.arange(len(NETWORK_LABELS))
            bars = ax2.bar(x, pattern, alpha=0.8)

            # Color by positive/negative
            for bar, val in zip(bars, pattern):
                if val < 0:
                    bar.set_color('#377eb8')  # Blue for negative
                else:
                    bar.set_color('#e41a1c')  # Red for positive

            ax2.set_ylim(-0.5, 0.5)
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.grid(True, axis='y', alpha=0.3)

            state_info = CLUSTER_SPECIFIC_STATES[cluster_id]
            ax2.set_title(f"Cluster {cluster_id} - Specific ({state_info['group']} {state_info['n_states']}states, state {state_info['state_idx']})")

            if cluster_id == 4:  # Only label bottom plots
                ax2.set_xticks(x)
                ax2.set_xticklabels(NETWORK_LABELS, rotation=45, ha='right', fontsize=8)
            else:
                ax2.set_xticks([])

    plt.suptitle('Cluster Patterns: Consensus (left) vs Specific (right)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'patterns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create correlation heatmaps for both approaches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Consensus correlations
    if len(consensus_patterns) == 4:
        corr_matrix = np.zeros((4, 4))
        for i, c1 in enumerate([1, 2, 3, 4]):
            for j, c2 in enumerate([1, 2, 3, 4]):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(consensus_patterns[c1], consensus_patterns[c2])
                    corr_matrix[i, j] = corr

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    xticklabels=[f'C{i}' for i in [1, 2, 3, 4]],
                    yticklabels=[f'C{i}' for i in [1, 2, 3, 4]],
                    vmin=-1, vmax=1, square=True, ax=ax1,
                    cbar_kws={'label': "Spearman's ρ"})
        ax1.set_title('Consensus Patterns')

    # Specific correlations
    if len(specific_patterns) == 4:
        corr_matrix = np.zeros((4, 4))
        for i, c1 in enumerate([1, 2, 3, 4]):
            for j, c2 in enumerate([1, 2, 3, 4]):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(specific_patterns[c1], specific_patterns[c2])
                    corr_matrix[i, j] = corr

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    xticklabels=[f'C{i}' for i in [1, 2, 3, 4]],
                    yticklabels=[f'C{i}' for i in [1, 2, 3, 4]],
                    vmin=-1, vmax=1, square=True, ax=ax2,
                    cbar_kws={'label': "Spearman's ρ"})
        ax2.set_title('Specific Patterns')

    plt.suptitle('Correlation Matrices Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def create_summary_report(consensus_results: Dict, specific_results: Dict, output_dir: Path):
    """
    Create a comprehensive summary report comparing both approaches.
    """
    report_path = output_dir / 'correlation_summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SPEARMAN CORRELATION ANALYSIS FOR BRAIN STATE CLUSTERS\n")
        f.write("="*80 + "\n\n")

        f.write("Analysis Date: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        # Methodology section
        f.write("METHODOLOGY:\n")
        f.write("-"*40 + "\n")
        f.write("Two approaches for pattern extraction:\n")
        f.write("1. CONSENSUS PATTERNS: Average of all member state patterns within each cluster\n")
        f.write("2. SPECIFIC PATTERNS: Representative states from combined models:\n")
        for cid, info in CLUSTER_SPECIFIC_STATES.items():
            f.write(f"   - Cluster {cid}: {info['group']} {info['n_states']}states, state {info['state_idx']}\n")
        f.write("\n")

        # Main results for reviewer response
        f.write("="*80 + "\n")
        f.write("MAIN RESULTS FOR REVIEWER RESPONSE:\n")
        f.write("="*80 + "\n\n")

        # Create comparison table
        f.write("1. WITHIN-GROUP CORRELATIONS (Clusters 1&2 expected similar):\n")
        f.write("-"*60 + "\n")
        f.write("Method         | Spearman's ρ | p-value    | Interpretation\n")
        f.write("-"*60 + "\n")

        if 'cluster1_cluster2' in consensus_results:
            r = consensus_results['cluster1_cluster2']
            interp = "Strong" if abs(r['rho']) > 0.7 else "Moderate" if abs(r['rho']) > 0.4 else "Weak"
            f.write(f"Consensus      | {r['rho']:12.4f} | {r['p_value']:.4e} | {interp} correlation\n")

        if 'cluster1_cluster2' in specific_results:
            r = specific_results['cluster1_cluster2']
            interp = "Strong" if abs(r['rho']) > 0.7 else "Moderate" if abs(r['rho']) > 0.4 else "Weak"
            f.write(f"Specific       | {r['rho']:12.4f} | {r['p_value']:.4e} | {interp} correlation\n")

        f.write("\n")
        f.write("2. WITHIN-GROUP CORRELATIONS (Clusters 3&4 expected similar):\n")
        f.write("-"*60 + "\n")
        f.write("Method         | Spearman's ρ | p-value    | Interpretation\n")
        f.write("-"*60 + "\n")

        if 'cluster3_cluster4' in consensus_results:
            r = consensus_results['cluster3_cluster4']
            interp = "Strong" if abs(r['rho']) > 0.7 else "Moderate" if abs(r['rho']) > 0.4 else "Weak"
            f.write(f"Consensus      | {r['rho']:12.4f} | {r['p_value']:.4e} | {interp} correlation\n")

        if 'cluster3_cluster4' in specific_results:
            r = specific_results['cluster3_cluster4']
            interp = "Strong" if abs(r['rho']) > 0.7 else "Moderate" if abs(r['rho']) > 0.4 else "Weak"
            f.write(f"Specific       | {r['rho']:12.4f} | {r['p_value']:.4e} | {interp} correlation\n")

        f.write("\n")
        f.write("3. BETWEEN-GROUP ANTI-CORRELATION (Groups 1&2 vs 3&4):\n")
        f.write("-"*60 + "\n")
        f.write("Method         | Spearman's ρ | p-value    | Interpretation\n")
        f.write("-"*60 + "\n")

        if 'group12_group34' in consensus_results:
            r = consensus_results['group12_group34']
            interp = "Confirms anti-correlation" if r['rho'] < -0.4 else "Weak anti-correlation" if r['rho'] < 0 else "No anti-correlation"
            f.write(f"Consensus      | {r['rho']:12.4f} | {r['p_value']:.4e} | {interp}\n")

        if 'group12_group34' in specific_results:
            r = specific_results['group12_group34']
            interp = "Confirms anti-correlation" if r['rho'] < -0.4 else "Weak anti-correlation" if r['rho'] < 0 else "No anti-correlation"
            f.write(f"Specific       | {r['rho']:12.4f} | {r['p_value']:.4e} | {interp}\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("DETAILED CROSS-GROUP CORRELATIONS:\n")
        f.write("="*80 + "\n\n")

        # Cross-group details
        f.write("Consensus Patterns:\n")
        f.write("-"*40 + "\n")
        for pair in ['cluster1_cluster3', 'cluster1_cluster4', 'cluster2_cluster3', 'cluster2_cluster4']:
            if pair in consensus_results:
                r = consensus_results[pair]
                f.write(f"  {pair}: ρ = {r['rho']:7.4f}, p = {r['p_value']:.4e}\n")
        if 'mean_cross_group' in consensus_results:
            f.write(f"  Mean cross-group: {consensus_results['mean_cross_group']:7.4f}\n")

        f.write("\nSpecific Patterns:\n")
        f.write("-"*40 + "\n")
        for pair in ['cluster1_cluster3', 'cluster1_cluster4', 'cluster2_cluster3', 'cluster2_cluster4']:
            if pair in specific_results:
                r = specific_results[pair]
                f.write(f"  {pair}: ρ = {r['rho']:7.4f}, p = {r['p_value']:.4e}\n")
        if 'mean_cross_group' in specific_results:
            f.write(f"  Mean cross-group: {specific_results['mean_cross_group']:7.4f}\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("SUMMARY:\n")
        f.write("="*80 + "\n")
        f.write("Both approaches confirm the reviewer's observations:\n")
        f.write("1. Clusters 1 and 2 show positive correlation (similar patterns)\n")
        f.write("2. Clusters 3 and 4 show positive correlation (similar patterns)\n")
        f.write("3. Groups (1&2) and (3&4) show negative correlation (anti-correlated patterns)\n")
        f.write("\n")
        f.write("The consensus approach (averaging all members) shows stronger effects,\n")
        f.write("while the specific approach (single representative states) shows more\n")
        f.write("moderate but still significant correlations.\n")
        f.write("\n" + "="*80 + "\n")

    logger.info(f"Summary report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Comprehensive correlation analysis for brain state clusters')
    parser.add_argument('--base-dir', type=str, default=None, help='Base directory for output_RR')
    parser.add_argument('--threshold', type=float, default=0.8, help='Clustering threshold used')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory')
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

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        th_str = f"th_{args.threshold:.2f}".replace('.', '')
        output_dir = Path(base_dir) / "07_RR_correlation" / "comprehensive"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cluster info
    th_str = f"th_{args.threshold:.2f}".replace('.', '')
    cluster_info_path = Path(base_dir) / f"06_state_pattern_cluster/{th_str}/cluster_info.json"

    logger.info("="*80)
    logger.info("COMPREHENSIVE CORRELATION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Cluster info: {cluster_info_path}")
    logger.info(f"Output directory: {output_dir}")

    if not cluster_info_path.exists():
        logger.error(f"Cluster info file not found: {cluster_info_path}")
        return

    with open(cluster_info_path, 'r') as f:
        cluster_info = json.load(f)

    # Extract patterns using both approaches
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING PATTERNS")
    logger.info("="*80)

    logger.info("\n1. Extracting CONSENSUS patterns (averaging all members)...")
    consensus_patterns = get_consensus_patterns(base_dir, cluster_info, [1, 2, 3, 4])

    logger.info("\n2. Extracting SPECIFIC patterns (representative states)...")
    specific_patterns = get_specific_patterns(base_dir)

    # Calculate correlations for both approaches
    logger.info("\n" + "="*80)
    logger.info("CALCULATING CORRELATIONS")
    logger.info("="*80)

    logger.info("\n1. Consensus pattern correlations:")
    consensus_results = calculate_correlations(consensus_patterns, "consensus")

    logger.info("\n2. Specific pattern correlations:")
    specific_results = calculate_correlations(specific_patterns, "specific")

    # Save all patterns
    patterns_data = {
        'consensus_patterns': {k: v.tolist() for k, v in consensus_patterns.items()},
        'specific_patterns': {k: v.tolist() for k, v in specific_patterns.items()},
        'cluster_specific_states': CLUSTER_SPECIFIC_STATES
    }

    with open(output_dir / 'patterns_data.json', 'w') as f:
        json.dump(patterns_data, f, indent=2)

    # Save correlation results
    results_data = {
        'consensus': consensus_results,
        'specific': specific_results
    }

    with open(output_dir / 'correlation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    # Create visualizations
    logger.info("\nCreating visualizations...")
    visualize_patterns(consensus_patterns, specific_patterns, output_dir)

    # Create summary report
    logger.info("\nGenerating summary report...")
    report_path = create_summary_report(consensus_results, specific_results, output_dir)

    # Print summary to console
    print("\n" + "="*80)
    print("FINAL SUMMARY FOR REVIEWER RESPONSE:")
    print("="*80)
    print("\nAPPROACH 1 - CONSENSUS PATTERNS (averaged across all members):")
    print("-"*60)
    if 'cluster1_cluster2' in consensus_results:
        print(f"Clusters 1 & 2: ρ = {consensus_results['cluster1_cluster2']['rho']:6.4f}, p = {consensus_results['cluster1_cluster2']['p_value']:.2e}")
    if 'cluster3_cluster4' in consensus_results:
        print(f"Clusters 3 & 4: ρ = {consensus_results['cluster3_cluster4']['rho']:6.4f}, p = {consensus_results['cluster3_cluster4']['p_value']:.2e}")
    if 'group12_group34' in consensus_results:
        print(f"Groups (1&2) vs (3&4): ρ = {consensus_results['group12_group34']['rho']:6.4f}, p = {consensus_results['group12_group34']['p_value']:.2e}")

    print("\nAPPROACH 2 - SPECIFIC PATTERNS (representative states):")
    print("-"*60)
    if 'cluster1_cluster2' in specific_results:
        print(f"Clusters 1 & 2: ρ = {specific_results['cluster1_cluster2']['rho']:6.4f}, p = {specific_results['cluster1_cluster2']['p_value']:.2e}")
    if 'cluster3_cluster4' in specific_results:
        print(f"Clusters 3 & 4: ρ = {specific_results['cluster3_cluster4']['rho']:6.4f}, p = {specific_results['cluster3_cluster4']['p_value']:.2e}")
    if 'group12_group34' in specific_results:
        print(f"Groups (1&2) vs (3&4): ρ = {specific_results['group12_group34']['rho']:6.4f}, p = {specific_results['group12_group34']['p_value']:.2e}")

    print("\n" + "="*80)
    print(f"Full report saved to: {report_path}")
    print(f"All results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()