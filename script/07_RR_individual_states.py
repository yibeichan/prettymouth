"""
Script to plot all individual states within a specific cluster to understand
how the consensus pattern is formed and identify any discrepancies.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup plotting
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

# Network labels
NETWORK_LABELS = ['Aud', 'Ctr-A', 'Ctr-B', 'Ctr-C', 'DMN-A', 'DMN-B',
                  'DMN-C', 'DA-A', 'DA-B', 'Lang', 'SVA-A',
                  'SVA-B', 'SM-A', 'SM-B', 'Vis-A', 'Vis-B', 'Vis-C']

def load_state_pattern(base_dir, group, n_states, state_idx):
    """
    Load the actual state pattern from the metrics file.

    Args:
        base_dir: Base directory for outputs
        group: Group name (e.g., 'combined', 'affair')
        n_states: Number of states in the model
        state_idx: Original state index

    Returns:
        Dictionary with mean_pattern, CI, and metadata
    """
    metrics_path = Path(base_dir) / f"04_{group}_hmm_{n_states}states_ntw_native_trimmed" / "statistics" / f"{group}_metrics.pkl"

    if not metrics_path.exists():
        print(f"Warning: Metrics file not found: {metrics_path}")
        return None

    try:
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

        if state_idx not in metrics['state_properties']:
            print(f"Warning: State {state_idx} not found in {group} {n_states}states")
            return None

        state_props = metrics['state_properties'][state_idx]

        return {
            'mean_pattern': np.array(state_props['mean_pattern']),
            'mean_pattern_ci': state_props['mean_pattern_ci'],
            'pattern_stability': state_props.get('pattern_stability', None),
            'group': group,
            'n_states': n_states,
            'state_idx': state_idx
        }
    except Exception as e:
        print(f"Error loading state pattern: {e}")
        return None

def extract_significant_features(mean_pattern, ci, min_activation=0.1):
    """
    Extract significant features using the same logic as StatePatternAnalyzer.

    Args:
        mean_pattern: Array of network activations
        ci: Confidence interval dict or array
        min_activation: Minimum activation threshold

    Returns:
        Boolean array indicating significant features
    """
    # Base criterion: activation above threshold
    reliable_activation = mean_pattern > min_activation

    # Statistical reliability
    if isinstance(ci, dict):
        ci_lower = np.array(ci['lower'])
    else:
        ci_lower = ci[:, 0]

    # CI threshold - using same logic as in pattern_cluster.py
    # For now, use a simple threshold
    ci_threshold = 0.05  # Conservative threshold
    reliable_direction = ci_lower > ci_threshold

    return reliable_activation & reliable_direction

def plot_cluster_states(cluster_id, cluster_info_path, base_dir, output_dir):
    """
    Plot all states in a specific cluster with their patterns.

    Args:
        cluster_id: Cluster ID to analyze
        cluster_info_path: Path to cluster_info.json
        base_dir: Base directory for outputs
        output_dir: Directory to save plots
    """
    # Load cluster info
    with open(cluster_info_path, 'r') as f:
        cluster_info = json.load(f)

    if str(cluster_id) not in cluster_info:
        print(f"Cluster {cluster_id} not found in cluster_info.json")
        return

    cluster_data = cluster_info[str(cluster_id)]
    members = cluster_data['members']
    consensus_pattern = np.array(cluster_data['consensus_pattern'])

    print(f"\nAnalyzing Cluster {cluster_id}")
    print(f"Number of members: {len(members)}")
    print(f"Consensus pattern active networks: {[NETWORK_LABELS[i] for i, v in enumerate(consensus_pattern) if v == 1]}")

    # Collect all state patterns
    all_patterns = []
    all_significant_features = []
    pattern_labels = []

    for idx, member in enumerate(members):
        group = member['group']
        model = member['model']
        original_state_idx = member['original_state_idx']

        # Extract n_states from model string
        import re
        match = re.search(r'(\d+)states', model)
        if not match:
            print(f"Warning: Could not extract n_states from model: {model}")
            continue
        n_states = int(match.group(1))

        # Load the actual state pattern
        state_data = load_state_pattern(base_dir, group, n_states, original_state_idx)

        if state_data is None:
            continue

        mean_pattern = state_data['mean_pattern']
        ci = state_data['mean_pattern_ci']

        # Extract significant features using the same criteria
        significant_features = extract_significant_features(mean_pattern, ci)

        all_patterns.append(mean_pattern)
        all_significant_features.append(significant_features.astype(int))
        pattern_labels.append(f"{group[:3]}-{n_states}s-S{original_state_idx}")

        # Print comparison
        sig_networks = [NETWORK_LABELS[i] for i, v in enumerate(significant_features) if v]
        print(f"\n{idx+1}. {model}, State {original_state_idx}:")
        print(f"   Significant networks: {sig_networks}")

    if len(all_patterns) == 0:
        print("No patterns could be loaded!")
        return

    # Convert to arrays
    all_patterns = np.array(all_patterns)
    all_significant_features = np.array(all_significant_features)

    # Calculate consensus from significant features
    calculated_consensus = (np.mean(all_significant_features, axis=0) >= 0.5).astype(int)

    print(f"\nCalculated consensus from significant features:")
    print(f"Active networks: {[NETWORK_LABELS[i] for i, v in enumerate(calculated_consensus) if v == 1]}")
    print(f"\nStored consensus pattern:")
    print(f"Active networks: {[NETWORK_LABELS[i] for i, v in enumerate(consensus_pattern) if v == 1]}")
    print(f"\nDo they match? {np.array_equal(calculated_consensus, consensus_pattern)}")

    # Create visualization with more space between subplots
    fig = plt.figure(figsize=(12, max(8, len(members) * 0.3)))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(members), 2], hspace=0.35)

    # Top panel: Heatmap of significant features
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(all_significant_features, aspect='auto', cmap='RdYlGn',
                     interpolation='nearest', vmin=0, vmax=1)
    ax1.set_yticks(range(len(pattern_labels)))
    ax1.set_yticklabels(pattern_labels, fontsize=7)
    ax1.set_xticks(range(len(NETWORK_LABELS)))
    ax1.set_xticklabels(NETWORK_LABELS, rotation=90)
    ax1.set_title(f'Cluster {cluster_id}: Significant Features for All Member States',
                  fontweight='bold', pad=8)

    # Add consensus pattern as vertical lines
    for i, val in enumerate(consensus_pattern):
        if val == 1:
            ax1.axvline(x=i, color='blue', linewidth=2, alpha=0.3)

    # Add legend for top panel in the space between plots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Network active (significant)'),
        Patch(facecolor='red', alpha=0.7, label='Network inactive'),
        plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.3, label='In stored consensus')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.085),
               ncol=3, frameon=True, fontsize=8)

    # Bottom panel: Network frequency across states
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(NETWORK_LABELS))

    # Calculate frequency of each network across all states
    freq_per_network = np.mean(all_significant_features, axis=0)

    # Create bars with orange color
    bars = ax2.bar(x, freq_per_network * 100, width=0.7, color='orange', alpha=0.8)

    # Add 50% threshold line
    ax2.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Add percentage value labels on top of bars
    for i, (freq, in_consensus) in enumerate(zip(freq_per_network, consensus_pattern)):
        height = freq * 100
        label_text = f'{height:.0f}%'
        # Add warning emoji for errors (in consensus but <50%)
        if in_consensus == 1 and freq < 0.5:
            label_text = f'{height:.0f}%⚠️'
        ax2.text(i, height + 1, label_text, ha='center', va='bottom', fontsize=7)

    ax2.set_ylabel('Frequency in States (%)', fontweight='bold')
    ax2.set_xlabel('Network', fontweight='bold')
    ax2.set_title('Network Activation Frequency Across All Member States', fontweight='bold', pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(NETWORK_LABELS, rotation=90)
    ax2.set_ylim(0, 140)  # Space for labels
    ax2.grid(True, axis='y', alpha=0.3)

    # Save figure
    output_path = Path(output_dir) / f'cluster{cluster_id}_individual_states_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    plt.show()
    plt.close()

    # Save detailed comparison to CSV
    df = pd.DataFrame({
        'Network': NETWORK_LABELS,
        'Calculated_Consensus': calculated_consensus,
        'Stored_Consensus': consensus_pattern,
        'Frequency_in_States': freq_per_network,
        'Match': calculated_consensus == consensus_pattern
    })

    csv_path = Path(output_dir) / f'cluster{cluster_id}_consensus_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison to: {csv_path}")

    return df, all_patterns, all_significant_features

if __name__ == "__main__":
    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = Path(scratch_dir) / "output_RR"

    # Cluster analysis directory - using threshold 0.80
    threshold_str = '080'
    cluster_dir = base_dir / "06_state_pattern_cluster" / f"th_{threshold_str}"
    cluster_info_path = cluster_dir / "cluster_info.json"

    # Output directory for this analysis
    output_dir = base_dir / "07_RR_individual_states" / f"th_{threshold_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cluster info to get all cluster IDs
    with open(cluster_info_path, 'r') as f:
        cluster_info = json.load(f)

    all_cluster_ids = sorted([int(k) for k in cluster_info.keys()])

    # For testing, only process specific clusters
    # Uncomment the next line to test with specific clusters only
    # all_cluster_ids = [1, 2, 6, 31, 66]  # Test with some correct and some with errors

    print(f"Found {len(all_cluster_ids)} clusters to analyze: {all_cluster_ids}")
    print("="*80)

    # Analyze all clusters
    for cluster_id in all_cluster_ids:
        print(f"\n{'='*80}")
        print(f"Processing Cluster {cluster_id}")
        print(f"{'='*80}")

        try:
            df, patterns, sig_features = plot_cluster_states(
                cluster_id=cluster_id,
                cluster_info_path=cluster_info_path,
                base_dir=base_dir,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")
            continue

    print("\n" + "="*80)
    print("Analysis complete for all clusters!")
    print("="*80)
