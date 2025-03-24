import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time
from natsort import natsorted
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import pandas as pd
import argparse
import re
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import gridspec

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

COLORS = {
    'affair': '#e41a1c',      # Red
    'affair_light': '#ff6666', # Light red
    'paranoia': '#4daf4a',    # Green
    'paranoia_light': '#90ee90', # Light green
    'combined': '#984ea3',    # Purple
    'combined_light': '#d8b2d8', # Light purple
    'constructed': '#377eb8',   # Blue
    'constructed_light': '#99c2ff' # Light blue
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class StatePatternAnalyzer:
    def __init__(self, base_dir, groups, output_dir, config=None):
        """
        Initialize the state pattern analyzer with configurable parameters.
        
        Args:
            base_dir: Base directory containing HMM results
            groups: List of group identifiers to analyze
            output_dir: Directory for output files
            config: Dictionary of analysis parameters (optional)
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.groups = groups
        print("Extracting patterns for groups:", self.groups)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        default_config = {
            'min_activation': 0.1,
            'max_ci_width': 0.3,
            'min_pattern_stability': 0.5,
            'cluster_similarity_threshold': 0.7
        }
        
        # Use provided config or defaults
        self.config = config if config else default_config
        
        # Initialize data storage
        self.patterns = {group: None for group in groups}  # Store patterns by group
        self.cluster_info = {}  # Store cluster information
        self.state_mappings = {}  # Store state mappings from original to sorted indices
        
        self.logger.info(f"Initialized StatePatternAnalyzer for groups: {', '.join(groups)}")
    
    def find_metrics_files(self, group):
        """Find all metrics files for a specific group."""
        # Try multiple glob patterns to find files
        patterns = [
            f"*_{group}_hmm_*states_*/statistics/{group}_metrics.pkl",
            f"*/*_{group}_*/statistics/{group}_metrics.pkl",
            f"**/{group}_metrics.pkl"  # Recursive search as last resort
        ]
        
        all_files = []
        for pattern in patterns:
            files = natsorted(self.base_dir.glob(pattern))
            all_files.extend(files)
            if files:
                self.logger.info(f"Found {len(files)} metrics files using pattern: {pattern}")
        
        # Remove duplicates while maintaining order
        unique_files = []
        seen = set()
        for file in all_files:
            if str(file) not in seen:
                unique_files.append(file)
                seen.add(str(file))
        
        if not unique_files:
            self.logger.warning(f"No metrics files found for group {group}")
        else:
            self.logger.info(f"Found {len(unique_files)} unique metrics files for group {group}")
        
        return unique_files
    
    def _normalize_model_name(self, model_path):
        """Extract standardized model name from file path."""
        path_str = str(model_path)
        
        # Extract group name
        group_match = None
        for group in self.groups:
            if group in path_str:
                group_match = group
                break
        
        if not group_match:
            return "unknown"
        
        # Extract number of states
        state_match = re.search(r'(\d+)_?states', path_str)
        if state_match:
            n_states = state_match.group(1)
            return f"{group_match} {n_states}states"
        else:
            return f"{group_match} unknown"
    
    def _sort_states_by_occupancy(self, metrics, model_name):
        """
        Sort states by fractional occupancy and create mapping.
        Returns a dict with mapping from original state idx to sorted state idx
        and fractional occupancy information.
        """
        try:
            if 'state_metrics' not in metrics or 'group_level' not in metrics['state_metrics']:
                self.logger.warning(f"Missing state metrics structure in {model_name}")
                return {}
            
            group_metrics = metrics['state_metrics']['group_level']
            if 'fractional_occupancy' not in group_metrics:
                self.logger.warning(f"Missing fractional_occupancy in {model_name}")
                return {}
            
            # Get fractional occupancy for each state
            occupancy = group_metrics['fractional_occupancy']
            
            # Dictionary to store both mapping and occupancy information
            result = {
                'mapping': {},
                'occupancy': {}
            }
            
            # Handle different data structures - numpy array vs dict
            if isinstance(occupancy, np.ndarray):
                # If occupancy is an array, state indices are just 0...n-1
                state_indices = list(range(len(occupancy)))
                # Create pairs of (state_idx, occupancy) for sorting
                state_occupancy_pairs = [(idx, float(occupancy[idx])) for idx in state_indices]
                # Sort by occupancy (higher occupancy first)
                state_occupancy_pairs.sort(key=lambda x: x[1], reverse=True)
                # Create mapping and store occupancy
                for new_idx, (original_idx, occ) in enumerate(state_occupancy_pairs):
                    result['mapping'][original_idx] = new_idx
                    result['occupancy'][original_idx] = occ
                    
            elif isinstance(occupancy, dict):
                # If occupancy is a dict, keys are state indices
                state_indices = list(occupancy.keys())
                # Create pairs of (state_idx, occupancy) for sorting
                state_occupancy_pairs = [(idx, float(occupancy[idx])) for idx in state_indices]
                # Sort by occupancy (higher occupancy first)
                state_occupancy_pairs.sort(key=lambda x: x[1], reverse=True)
                # Create mapping and store occupancy
                for new_idx, (original_idx, occ) in enumerate(state_occupancy_pairs):
                    result['mapping'][original_idx] = new_idx
                    result['occupancy'][original_idx] = occ
            else:
                self.logger.warning(f"Unexpected occupancy data type: {type(occupancy)}")
                return {}
            
            return result
        except Exception as e:
            self.logger.warning(f"Error sorting states for {model_name}: {str(e)}")
            return {}
        
    def extract_patterns(self, group):
        """Extract reliable patterns for a group."""
        metrics_files = self.find_metrics_files(group)
        all_patterns = []
        pattern_metadata = []
        group_state_mappings = {}
        
        # Process each metrics file
        for file_path in metrics_files:
            try:
                with open(file_path, 'rb') as f:
                    metrics = pickle.load(f)
                
                # Extract state properties
                state_properties = metrics.get('state_properties', {})
                if not state_properties:
                    self.logger.warning(f"No state_properties found in {file_path}")
                    continue
                
                # Remove separability as it's a global metric
                if 'separability' in state_properties:
                    state_properties.pop('separability')
                
                # Get normalized model name
                model_name = self._normalize_model_name(file_path)
                
                # Sort states by fractional occupancy and create mapping
                state_mapping_data = self._sort_states_by_occupancy(metrics, model_name)
                
                # Store the mapping and occupancy
                if state_mapping_data:
                    # Get the mapping part
                    state_mapping = state_mapping_data.get('mapping', {})
                    state_occupancy = state_mapping_data.get('occupancy', {})
                    
                    # Store in the group_state_mappings with both mapping and occupancy
                    group_state_mappings[model_name] = {
                        'mapping': state_mapping,
                        'occupancy': state_occupancy
                    }
                else:
                    state_mapping = {}
                    state_occupancy = {}
                
                # Process each state
                for state_idx, state_data in state_properties.items():
                    if not isinstance(state_data, dict):
                        continue
                    
                    # Validate required metrics exist
                    required_metrics = ['mean_pattern', 'mean_pattern_ci', 'pattern_stability']
                    missing_metrics = [m for m in required_metrics if m not in state_data]
                    if missing_metrics:
                        continue
                    
                    # Extract pattern data
                    mean_pattern = np.array(state_data['mean_pattern'])
                    
                    # Handle CI in different formats
                    mean_pattern_ci = state_data['mean_pattern_ci']
                    if isinstance(mean_pattern_ci, dict):
                        ci = np.array([mean_pattern_ci['lower'], mean_pattern_ci['upper']]).T
                    else:
                        ci = np.array(mean_pattern_ci)
                    
                    pattern_stability = state_data['pattern_stability']
                    
                    # Apply filtering criteria
                    if self._is_reliable_pattern(mean_pattern, ci, pattern_stability):
                        # Select significant features
                        significant_features = self._select_significant_features(
                            mean_pattern, ci, pattern_stability
                        )
                        
                        if np.any(significant_features):
                            all_patterns.append(significant_features)
                            
                            # Get the sorted state index - handle state_idx that might be a string
                            try:
                                # Try to convert to int first if it's a string
                                if isinstance(state_idx, str):
                                    state_idx_key = int(state_idx)
                                else:
                                    state_idx_key = state_idx
                                
                                sorted_state_idx = state_mapping.get(state_idx_key, state_idx)
                                
                                # Get the occupancy for this state if available
                                # Use proper dictionary lookup with a default
                                state_occ_value = state_occupancy.get(state_idx_key, None)
                            except (ValueError, TypeError):
                                # If conversion fails, use as is
                                sorted_state_idx = state_idx
                                state_occ_value = None
                            
                            # Add to pattern metadata with occupancy information
                            pattern_metadata.append({
                                'file': str(file_path),
                                'model': model_name,
                                'original_state_idx': state_idx,
                                'sorted_state_idx': sorted_state_idx,
                                'occupancy': state_occ_value,  # Add occupancy as a value, not a dict
                                'active_features': np.where(significant_features)[0].tolist(),
                                'activation_values': mean_pattern[significant_features].tolist(),
                                'pattern_stability': pattern_stability
                            })
            
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {str(e)}")
        
        # Store extracted patterns and state mappings
        self.patterns[group] = {
            'patterns': np.array(all_patterns) if all_patterns else np.array([]),
            'metadata': pattern_metadata
        }
        
        self.state_mappings[group] = group_state_mappings
        
        self.logger.info(f"Group {group}: Extracted {len(all_patterns)} patterns")
        return len(all_patterns)
    
    def _is_reliable_pattern(self, mean_pattern, ci, pattern_stability):
        """Check if a pattern meets basic reliability criteria."""
        max_abs_activation = np.max(np.abs(mean_pattern))
        
        # Ensure ci has correct dimensions
        if ci.ndim != 2:
            return False
            
        ci_width = ci[:, 1] - ci[:, 0] if ci.shape[1] >= 2 else np.zeros_like(mean_pattern)
        max_ci_width = np.max(ci_width)
        
        return (max_abs_activation >= self.config['min_activation'] and
                max_ci_width <= self.config['max_ci_width'] and
                pattern_stability >= self.config['min_pattern_stability'])
    
    def _select_significant_features(self, mean_pattern, ci, pattern_stability):
        """
        Select significant features using statistical reliability criteria.
        
        For HMM state patterns, we want features that are:
        1. Consistently activated above threshold
        2. Statistically reliable (lower CI bound > 0)
        
        We ignore feature importance since co-activation is our primary concern.
        """
        # Activation threshold - features must exceed minimum activation
        reliable_activation = mean_pattern > self.config['min_activation']
        
        # Statistical reliability - lower confidence bound must be positive
        if ci.ndim != 2 or ci.shape[1] < 2:
            self.logger.warning(f"Invalid CI shape: {ci.shape if hasattr(ci, 'shape') else 'unknown'}")
            reliable_direction = np.zeros_like(mean_pattern, dtype=bool)
        else:
            reliable_direction = ci[:, 0] > 0
        
        # For very stable patterns, we could be less strict on CI bounds
        if pattern_stability > 0.8:  # High stability patterns
            # For highly stable patterns, activation above threshold is sufficient
            return reliable_activation
        else:
            # For less stable patterns, require statistical reliability
            return reliable_activation & reliable_direction
    
    def cluster_patterns_across_groups(self):
        """Cluster patterns across all groups using Jaccard distance."""
        # Collect all patterns
        all_patterns = []
        pattern_sources = []
        
        for group in self.groups:
            if group not in self.patterns or self.patterns[group] is None:
                continue
                
            group_patterns = self.patterns[group]['patterns']
            if len(group_patterns) == 0:
                continue
                
            # Add each pattern with its source info
            for pattern_idx, pattern in enumerate(group_patterns):
                all_patterns.append(pattern)
                pattern_sources.append({
                    'group': group,
                    'pattern_idx': pattern_idx
                })
        
        all_patterns_array = np.array(all_patterns)
        
        if len(all_patterns_array) == 0:
            self.logger.warning("No patterns to cluster")
            return
        
        self.logger.info(f"Clustering {len(all_patterns_array)} patterns across groups")
        
        # Perform clustering with Jaccard distance
        distance_matrix = squareform(pdist(all_patterns_array, metric='jaccard'))
        linkage_matrix = linkage(squareform(distance_matrix), method='average')
        
        # Apply clustering with the configured threshold
        distance_cutoff = 1.0 - self.config['cluster_similarity_threshold']
        cluster_labels = fcluster(linkage_matrix, distance_cutoff, criterion='distance')
        
        # Store cluster assignments
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(cluster_labels):
            clusters[int(cluster_id)].append({
                'pattern': all_patterns[i],
                **pattern_sources[i]
            })
        
        # Sort clusters by size and reassign cluster IDs
        # Changed: Sort by total fractional occupancy instead of just size
        sorted_clusters = sorted(clusters.items(), 
                               key=lambda x: sum(
                                   self.patterns[item['group']]['metadata'][item['pattern_idx']].get('occupancy', 0) 
                                   for item in x[1] if item['group'] in self.patterns
                               ), 
                               reverse=True)
        
        # Create new clusters dictionary with reassigned IDs
        new_clusters = {}
        for new_id, (_, cluster_members) in enumerate(sorted_clusters, 1):  # Start IDs from 1
            new_clusters[new_id] = cluster_members
            
        # Replace original clusters with sorted version
        clusters = new_clusters
        
        # Compute cluster information
        self._compute_cluster_info(clusters)
        self.logger.info(f"Created {len(self.cluster_info)} pattern clusters, sorted by total fractional occupancy")
    
    def _compute_cluster_info(self, clusters):
        """Compute detailed information about each cluster with fractional occupancy."""
        self.cluster_info = {}
        
        for cluster_id, members in clusters.items():
            patterns = np.array([m['pattern'] for m in members])
            
            # Compute consensus pattern
            consensus = np.mean(patterns, axis=0) >= 0.5
            
            # Get unique groups as a list
            groups_in_cluster = sorted(set(m['group'] for m in members))
            
            # Collect member information with source details
            member_info = []
            total_fractional_occupancy = 0.0  # Track total fractional occupancy
            
            for member in members:
                group = member['group']
                pattern_idx = member['pattern_idx']
                
                # Find original metadata for this pattern
                metadata = {}
                if group in self.patterns and 'metadata' in self.patterns[group]:
                    if pattern_idx < len(self.patterns[group]['metadata']):
                        metadata = self.patterns[group]['metadata'][pattern_idx]
                
                # Extract model and state info
                model_key = metadata.get('model', 'unknown')
                original_state_idx = metadata.get('original_state_idx', 'unknown')
                sorted_state_idx = metadata.get('sorted_state_idx', 'unknown')
                
                # Extract fractional occupancy - this is key for our new sorting
                fractional_occupancy = metadata.get('occupancy', 0.0)
                if fractional_occupancy is not None:
                    total_fractional_occupancy += float(fractional_occupancy)
                
                member_info.append({
                    'group': group,
                    'pattern_idx': pattern_idx,
                    'model': model_key,
                    'original_state_idx': original_state_idx, 
                    'sorted_state_idx': sorted_state_idx,
                    'fractional_occupancy': fractional_occupancy  # Include in member info
                })
            
            # Group membership counts
            group_counts = {}
            for group in self.groups:
                count = sum(1 for m in members if m['group'] == group)
                group_counts[group] = count
            
            # Store cluster information
            self.cluster_info[cluster_id] = {
                'size': len(members),
                'total_fractional_occupancy': total_fractional_occupancy,  # New field
                'consensus_pattern': consensus.astype(int),
                'groups': groups_in_cluster,
                'members': member_info,
                'group_counts': group_counts
            }
    
    def visualize_results(self):
        """Create visualizations for clustering results."""
        if not self.cluster_info:
            self.logger.warning("No clustering results to visualize")
            return
        
        # Sort clusters by total fractional occupancy instead of size
        sorted_clusters = sorted(
            self.cluster_info.items(), 
            key=lambda x: x[1]['total_fractional_occupancy'], 
            reverse=True
        )
        
        # Create visualizations
        self._create_visualization(sorted_clusters, COLORS)
        self._create_model_summary()
    
    def _create_visualization(self, sorted_clusters, group_colors):
        """Create visualizations of clusters as separate plots."""
        # Get network labels for better axis labeling
        network_labels = ['Aud', 'Ctr-A', 'Ctr-B', 'Ctr-C', 'DMN-A', 'DMN-B',
                        'DMN-C', 'DA-A', 'DA-B', 'Lang', 'SVA-A',
                        'SVA-B', 'SM-A', 'SM-B', 'Vis-A', 'Vis-B',
                        'Vis-C']
        
        self._create_consensus_pattern_plot(sorted_clusters, network_labels)        

    def _create_consensus_pattern_plot(self, sorted_clusters, network_labels):
        """Create a plot of consensus patterns with a stacked bar plot of group counts."""
        num_clusters_to_show = min(25, len(sorted_clusters))
        
        if num_clusters_to_show == 0:
            self.logger.warning("No clusters to display in consensus pattern plot")
            return
        
        # Extract cluster info and consensus patterns
        cluster_sizes = np.array([data['size'] for _, data in sorted_clusters[:num_clusters_to_show]])
        total_occupancies = np.array([data['total_fractional_occupancy'] for _, data in sorted_clusters[:num_clusters_to_show]])
        consensus_patterns = np.array([data['consensus_pattern'] for _, data in sorted_clusters[:num_clusters_to_show]])
        
        # Get unique group names from all clusters
        group_names = set()
        for _, data in sorted_clusters[:num_clusters_to_show]:
            group_names.update(data['group_counts'].keys())
        group_names = sorted(list(group_names))
        
        # Create figure with gridspec for 3 panels with adjusted ratios for a more square heatmap
        fig = plt.figure(figsize=(7, 6))  # Adjust overall figure proportions to be closer to square
        gs = gridspec.GridSpec(2, 2, 
                              height_ratios=[2, 3],      # Keep the top histogram height ratio
                              width_ratios=[2.5, 2.5],     # Make the heatmap proportionally narrower 
                              hspace=0,               # Add minimal spacing for clarity
                              wspace=0)               # Add minimal spacing for clarity

        ax_hist = plt.subplot(gs[0, 0])   # Top histogram
        ax_heat = plt.subplot(gs[1, 0])   # Main heatmap
        ax_bars = plt.subplot(gs[1, 1])   # Right stacked bars

        # Create cluster labels with occupancy information
        cluster_labels = []
        
        for i, (cluster_id, data) in enumerate(sorted_clusters[:num_clusters_to_show]):
            # Format label with size and occupancy
            occ_formatted = f"{data['total_fractional_occupancy']:.2f}" if data['total_fractional_occupancy'] else "0.00"
            label = f"C{cluster_id} (n={data['size']}, occ={occ_formatted})"
            cluster_labels.append(label)
        
        # Use total_occupancies for weighting the visualization
        total_occupancies_column = total_occupancies.reshape(-1, 1)
        
        # Multiply consensus patterns by occupancy for weighted visualization
        weighted_patterns = consensus_patterns[:num_clusters_to_show] * total_occupancies_column

        # Create heatmap with weighted values
        im = ax_heat.imshow(weighted_patterns, 
                        aspect='auto', 
                        cmap='BuPu',
                        interpolation='nearest')
        
        # Add hatching for non-zero patterns
        rows, cols = weighted_patterns.shape
        for i in range(rows):
            for j in range(cols):
                if weighted_patterns[i, j] > 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                      fill=False,
                                      hatch='///',
                                      color='blueviolet',
                                      alpha=0.5, 
                                      zorder=2)
                    ax_heat.add_patch(rect)
        
        # Set y-axis labels (clusters)
        ax_heat.set_yticks(range(len(cluster_labels)))
        ax_heat.set_yticklabels(cluster_labels)
        
        # Set x-axis labels (features)
        feature_indices = np.arange(0, consensus_patterns.shape[1], 1)
        ax_heat.set_xticks(feature_indices)
        ax_heat.set_xticklabels([network_labels[i] for i in range(len(feature_indices))], rotation=90)
        
        # Count how many clusters show activity for each network
        column_counts = np.sum(consensus_patterns != 0, axis=0)
        bars = ax_hist.bar(feature_indices, column_counts, align='center', alpha=0.7, color='slateblue', width=0.9)
        
        # Add count labels on the bars instead of on top
        for bar in bars:
            height = bar.get_height()
            y_position = height / 2  # Place text in the middle of the bar
            
            # Adjust text color based on bar height for visibility
            text_color = 'white' if height > (max(column_counts) / 3) else 'darkblue'
            
            ax_hist.text(bar.get_x() + bar.get_width()/2., y_position,
                        f'{int(height)}', color=text_color, fontsize=5,
                        ha='center', va='center', rotation=0, fontweight='bold')
        
        # Remove spines for histogram
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.spines['left'].set_visible(False)
        ax_hist.spines['bottom'].set_visible(False)
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        
        # Share x-axis between histogram and heatmap
        ax_hist.set_xlim(ax_heat.get_xlim())
        
        # Create stacked bar plot for group counts
        # Define colors for groups
        color_map = {group: COLORS[group] for group in group_names}
        
        # Create matrix for stacked bars
        group_data = np.zeros((num_clusters_to_show, len(group_names)))
        for i, (_, data) in enumerate(sorted_clusters[:num_clusters_to_show]):
            for j, group in enumerate(group_names):
                group_data[i, j] = data['group_counts'].get(group, 0)
        
        # Plot stacked bars
        bottoms = np.zeros(num_clusters_to_show)
        for j, group in enumerate(group_names):
            values = group_data[:, j]
            bars = ax_bars.barh(range(num_clusters_to_show), values, left=bottoms, 
                             label=group, color=color_map[group], height=0.9)
            
            # Add count labels on bars if there's enough space
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width > 0:
                    # Only show text if the segment is wide enough
                    if width > max(values) * 0.05:  # Minimum width threshold for showing text
                        ax_bars.text(bottoms[i] + width/2, i, 
                                    f'{int(width)}', color='white', fontsize=5,
                                    ha='center', va='center', fontweight='bold')
            
            # Update bottoms INSIDE the loop after each group's bars are drawn
            bottoms += values
        
        ax_bars.set_xticks([])
        ax_bars.set_yticks([])
        ax_bars.set_yticklabels([])  # No labels since they're already on the heatmap

        ax_bars.set_ylim(ax_heat.get_ylim())
        # remove spines
        ax_bars.spines['bottom'].set_visible(False)
        
        # Set bar plot title and legend
        # ax_bars.set_title('Group Counts', fontsize=10)
        
        # Adjust legend - place it underneath the bar plot
        # First, get current position of the bar plot
        pos = ax_bars.get_position()
        
        # Make space for legend by reducing height of the bar plot
        ax_bars.set_position([pos.x0, pos.y0, pos.width, pos.height])
        

        legend = ax_bars.legend(title="Groups", 
                            loc='lower right',          # Position at lower right
                            fontsize='x-small',         # Smaller font size (smaller than 'small')
                            title_fontsize='small',     # Smaller title font
                            ncol=1,  # More columns to make it more compact
                            frameon=True,
                            framealpha=0.8,
                            markerscale=0.7,            # Make the markers smaller
                            handlelength=1.5,           # Shorter handles
                            handletextpad=0.5)          # Less padding between handle and text
        
        # Remove unnecessary spines
        ax_bars.spines['top'].set_visible(False)
        ax_bars.spines['right'].set_visible(False)
        
        # Adjust overall layout
        # fig.suptitle('Consensus Patterns for Top Clusters\n(Color Intensity ∝ Fractional Occupancy)')
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        
        # Save figure
        plt.savefig(self.output_dir / 'consensus_patterns_with_groups.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'consensus_patterns_with_groups.svg', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_model_summary(self):
        """Create clear and informative model-to-cluster mappings."""
        # Collect data for all patterns by group
        group_tracking_data = {group: [] for group in self.groups}
        
        for cluster_id, cluster_data in self.cluster_info.items():
            for member in cluster_data['members']:
                group = member['group']
                model = member.get('model', 'unknown')
                original_state_idx = member.get('original_state_idx', 'unknown')
                sorted_state_idx = member.get('sorted_state_idx', 'unknown')
                
                # Extract model number for better sorting
                model_num = None
                if isinstance(model, str):
                    match = re.search(r'(\d+)states', model)
                    if match:
                        model_num = int(match.group(1))
                
                # Only add to tracking data if the model belongs to the current group
                if group in model:
                    group_tracking_data[group].append({
                        'Group': group,
                        'Model': model,
                        'Model_Num': model_num,
                        'Original_State_Index': original_state_idx,
                        'Sorted_State_Index': sorted_state_idx,
                        'Cluster_ID': cluster_id
                    })
        
        # Save full tracking data for reference
        all_tracking_data = []
        for group_data in group_tracking_data.values():
            all_tracking_data.extend(group_data)
        
        if not all_tracking_data:
            self.logger.warning("No tracking data available")
            return
            
        # Convert to DataFrame and save
        full_df = pd.DataFrame(all_tracking_data)
        full_df.to_csv(self.output_dir / 'pattern_tracking.csv', index=False)
        
        # Process each group separately
        for group, tracking_data in group_tracking_data.items():
            print("Processing group for cluster plot:", group)
            if not tracking_data:
                self.logger.info(f"No data available for group {group}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(tracking_data)
            
            # Sort by model using natsort
            if 'Model' in df.columns:
                model_order = natsorted(df['Model'].unique())
                df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
                df = df.sort_values(by=['Model', 'Sorted_State_Index'])
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['Model', 'Sorted_State_Index'])
            
            # Create state-cluster heatmap for easier visual pattern identification
            try:
                # Get numeric state indices where possible
                df['State_Num'] = pd.to_numeric(df['Sorted_State_Index'], errors='coerce')
                
                # Create pivot table
                pivot_df = df.pivot(index='Model', columns='Sorted_State_Index', values='Cluster_ID')
                
                # Sort rows using natural sort
                pivot_df = pivot_df.loc[natsorted(pivot_df.index)]
                
                # Create heatmap
                plt.figure(figsize=(5, max(1, len(pivot_df)*0.2)))
                
                # Use a colormap with good distinguishability for discrete values
                cmap = plt.get_cmap('tab20', len(self.cluster_info)+1)
                
                ax = sns.heatmap(pivot_df, 
                                 annot=True, 
                                 fmt='g', 
                                 cmap="rainbow", 
                                 cbar=False,
                                 linewidths=0.5, 
                                 linecolor='white',
                                 annot_kws={'size': 6})
                
                # Set title and labels
                # plt.title(f'{group}: Model State to Cluster Mapping (Sorted States)')
                
                # Set x-axis ticks for all states
                plt.xticks(np.arange(20) + 0.5, range(1, 21))
                plt.xlabel('')
                plt.ylabel('')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{group}_model_clusters.png', dpi=300)
                plt.savefig(self.output_dir / f'{group}_model_clusters.svg', dpi=300)
                print(f"Saved {group}_model_clusters.png and {group}_model_clusters.svg")
                plt.close()
                
            except Exception as e:
                print(f"Error creating heatmap for {group}: {str(e)}")
                print("DataFrame contents:")
                print(df)
        
        # Create cross-group cluster visualization 
        self._create_cluster_overlap_diagram()


    def _create_cluster_overlap_diagram(self):
        """Create diagram showing how clusters overlap between groups using a stacked bar plot."""
        # Extract data
        group_overlaps = {}
        
        # Get count of patterns per group for each cluster
        for cluster_id, data in self.cluster_info.items():
            if len(data['groups']) > 1:  # Cluster has patterns from multiple groups
                group_overlaps[cluster_id] = data['group_counts']
        
        if not group_overlaps:
            return
        
        # Create overlap visualization
        plt.figure(figsize=(5, 2))
        
        # Convert to DataFrame for easier plotting
        overlap_data = []
        for cluster_id, counts in group_overlaps.items():
            for group, count in counts.items():
                if count > 0:
                    overlap_data.append({
                        'Cluster': f"C{cluster_id}",
                        'Group': group,
                        'Count': count
                    })
        
        df = pd.DataFrame(overlap_data)
        
        # Pivot the DataFrame to get it in the right format for stacking
        df_pivot = df.pivot(index='Cluster', columns='Group', values='Count').fillna(0)
        df_pivot = df_pivot.loc[natsorted(df_pivot.index)]

        # Create stacked bar chart
        ax = df_pivot.plot(
            kind='bar',
            stacked=True,
            color=[COLORS[group] for group in df_pivot.columns],
            figsize=(5, 2),
            width=1
        )
        
        # # Add total count on top of each bar
        # totals = df_pivot.sum(axis=1)
        # for i, total in enumerate(totals):
        #     ax.text(i, total, f'{int(total)}', 
        #             ha='center', va='bottom')
        
        # plt.title('Pattern Distribution Across Clusters')
        # plt.xlabel('Cluster ID')
        # plt.ylabel('Pattern Count')
        # plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=90)
        plt.xlabel('')
        plt.ylabel('')
        # remove legend
        ax.legend().remove()
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_overlap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'group_overlap.svg', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """Save all results to output directory."""
        # Save cluster information
        with open(self.output_dir / 'cluster_info.json', 'w') as f:
            json.dump(self.cluster_info, f, indent=2, cls=NumpyEncoder)
        
        # Save state mappings
        with open(self.output_dir / 'state_mappings.json', 'w') as f:
            json.dump(self.state_mappings, f, indent=2, cls=NumpyEncoder)
        
        # Save configuration
        with open(self.output_dir / 'analysis_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def run_analysis(self):
        """Run the complete pattern extraction and clustering pipeline."""
        start_time = time.time()
        
        # Step 1: Extract patterns for each group
        self.logger.info("Extracting patterns for each group...")
        for group in self.groups:
            pattern_count = self.extract_patterns(group)
            if pattern_count == 0:
                self.logger.warning(f"No patterns extracted for group {group}")
        
        # Step 2: Cluster patterns across groups
        self.logger.info("Clustering patterns across groups...")
        self.cluster_patterns_across_groups()
        
        # Step 3: Visualize results
        self.logger.info("Creating visualizations...")
        self.visualize_results()
        
        # Step 4: Save results
        self.logger.info("Saving results...")
        self.save_results()
        
        execution_time = time.time() - start_time
        self.logger.info(f"Analysis complete. Execution time: {execution_time:.2f} seconds")
        
        return self.cluster_info

# def main():
#     from dotenv import load_dotenv
#     load_dotenv()
        
#     # Setup paths
#     scratch_dir = os.getenv("SCRATCH_DIR")
#     base_dir = os.path.join(scratch_dir, "output")
#     output_dir = os.path.join(base_dir, "06_state_pattern_clusters")

#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Extract and cluster state patterns from HMM results')
    
#     parser.add_argument('--groups', type=str, nargs='+', default=["affair", "paranoia", "combined"], 
#                         help='List of groups to analyze')
#     parser.add_argument('--min_activation', type=float, default=0.1, help='Minimum activation threshold')
#     parser.add_argument('--max_ci_width', type=float, default=0.3, help='Maximum CI width threshold')
#     parser.add_argument('--min_pattern_stability', type=float, default=0.7, help='Minimum pattern stability')
#     parser.add_argument('--cluster_similarity', type=float, default=0.7, help='Cluster similarity threshold')
#     parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
#     args = parser.parse_args()
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Setup logging
#     log_level = logging.DEBUG if args.verbose else logging.INFO
#     logging.basicConfig(
#         level=log_level,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(os.path.join(output_dir, "pattern_analysis.log")),
#             logging.StreamHandler()
#         ]
#     )
    
#     # Create config from args
#     config = {
#         'min_activation': args.min_activation,
#         'max_ci_width': args.max_ci_width,
#         'min_pattern_stability': args.min_pattern_stability,
#         'cluster_similarity_threshold': args.cluster_similarity
#     }
    
#     try:
#         # Create and run analyzer
#         analyzer = StatePatternAnalyzer(
#             base_dir=base_dir,
#             groups=args.groups,
#             output_dir=output_dir,
#             config=config
#         )
        
#         analyzer.run_analysis()
        
#     except Exception as e:
#         logging.error(f"Error during analysis: {e}", exc_info=True)
#         raise

# if __name__ == '__main__':
#     main()