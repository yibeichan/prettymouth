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

COLORS = {
    'affair': '#e41a1c',      # Red
    'affair_light': '#ff6666', # Light red
    'paranoia': '#4daf4a',    # Green
    'paranoia_light': '#90ee90', # Light green
    'combined': '#984ea3',    # Purple
    'combined_light': '#d8b2d8' # Light purple
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
        # Get clusters sorted by size (largest first)
        sorted_clusters = sorted(clusters.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True)
        
        # Create new clusters dictionary with reassigned IDs
        new_clusters = {}
        for new_id, (_, cluster_members) in enumerate(sorted_clusters, 1):  # Start IDs from 1
            new_clusters[new_id] = cluster_members
            
        # Replace original clusters with sorted version
        clusters = new_clusters
        
        # Compute cluster information
        self._compute_cluster_info(clusters)
        self.logger.info(f"Created {len(self.cluster_info)} pattern clusters")
    
    def _compute_cluster_info(self, clusters):
        """Compute detailed information about each cluster."""
        self.cluster_info = {}
        
        for cluster_id, members in clusters.items():
            patterns = np.array([m['pattern'] for m in members])
            
            # Compute consensus pattern
            consensus = np.mean(patterns, axis=0) >= 0.5
            
            # Get unique groups as a list
            groups_in_cluster = sorted(set(m['group'] for m in members))
            
            # Collect member information with source details
            member_info = []
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
                
                member_info.append({
                    'group': group,
                    'pattern_idx': pattern_idx,
                    'model': model_key,
                    'original_state_idx': original_state_idx, 
                    'sorted_state_idx': sorted_state_idx
                })
            
            # Group membership counts
            group_counts = {}
            for group in self.groups:
                count = sum(1 for m in members if m['group'] == group)
                group_counts[group] = count
            
            # Store cluster information
            self.cluster_info[cluster_id] = {
                'size': len(members),
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
        
        # Sort clusters by size
        sorted_clusters = sorted(
            self.cluster_info.items(), 
            key=lambda x: x[1]['size'], 
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
        
        # 1. Plot consensus patterns for top clusters with better visualization
        self._create_consensus_pattern_plot(sorted_clusters, network_labels)
        
        # 2. Plot pattern distributions across clusters as a separate visualization
        self._create_cluster_distribution_plot(sorted_clusters, group_colors)

    def _create_consensus_pattern_plot(self, sorted_clusters, network_labels):
        """Create a plot of consensus patterns for the top clusters with size-based coloring."""
        num_clusters_to_show = min(30, len(sorted_clusters))
        
        if num_clusters_to_show == 0:
            self.logger.warning("No clusters to display in consensus pattern plot")
            return
        
        # Extract cluster sizes and consensus patterns
        cluster_sizes = np.array([data['size'] for _, data in sorted_clusters[:num_clusters_to_show]])
        consensus_patterns = np.array([data['consensus_pattern'] for _, data in sorted_clusters[:num_clusters_to_show]])
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0)  # Set hspace=0 to remove gap
        ax_hist = plt.subplot(gs[0])
        ax_heat = plt.subplot(gs[1])

        # Create cluster labels and calculate group counts for coloring
        cluster_labels = []
        group_counts_array = []  # Store group counts for coloring
        
        for i, (cluster_id, data) in enumerate(sorted_clusters[:num_clusters_to_show]):
            # Format label with size and group counts
            label = f"C{cluster_id} (n={data['size']}): "  # Add 1 to cluster_id
            group_counts = []
            total_group_count = sum(data['group_counts'].values())  # Calculate total count
            
            for group, count in data['group_counts'].items():
                if count > 0:
                    group_counts.append(f"{group}({count})")
            label += " ".join(group_counts)
            cluster_labels.append(label)
            
            # Store normalized group counts for coloring
            group_counts_array.append(total_group_count)
        
        # Convert group counts to a column vector for broadcasting
        group_counts_array = np.array(group_counts_array).reshape(-1, 1)
        
        # Multiply consensus patterns by group counts for weighted visualization
        weighted_patterns = consensus_patterns[:num_clusters_to_show] * group_counts_array

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
                                      fill=False,  # Don't fill the rectangle
                                      hatch='///',  # Add hatching
                                      color='blueviolet',  # No edge color
                                      alpha=0.5, 
                                      zorder=2)  # Place above heatmap
                    ax_heat.add_patch(rect)
        
        # Set y-axis labels (clusters)
        ax_heat.set_yticks(range(len(cluster_labels)))
        ax_heat.set_yticklabels(cluster_labels)
        
        # Set x-axis labels (features)
        feature_indices = np.arange(0, consensus_patterns.shape[1], 1)
        ax_heat.set_xticks(feature_indices)
        ax_heat.set_xticklabels([network_labels[i] for i in range(len(feature_indices))], rotation=90)
        
        # Add colorbar
        # cbar = fig.colorbar(im, ax=[ax_hist, ax_heat], shrink=0.6, pad=0.01)
        # cbar.set_label('Pattern Strength × Group Count')
        
        # Count how many clusters show activity for each network
        column_counts = np.sum(consensus_patterns != 0, axis=0)  # Count non-zero entries in each column
        bars = ax_hist.bar(feature_indices, column_counts, align='center', alpha=0.7, color='slateblue')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax_hist.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',  color='darkblue', weight='bold',
                        ha='center', va='bottom', rotation=0)
        
        # remove top right bottom spines
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.spines['left'].set_visible(False)
        ax_hist.spines['bottom'].set_visible(False)
        ax_hist.set_xticks([])  # Remove x-ticks for histogram
        ax_hist.set_yticks([])  # Remove y-ticks for histogram
        
        # Share x-axis between histogram and heatmap
        ax_hist.set_xlim(ax_heat.get_xlim())
        
        # Adjust layout (modify the rect parameters if needed)
        fig.suptitle('Consensus Patterns for Top Clusters\n(Color Intensity ∝ Group Count)', 
                 fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])  # Adjusted bottom margin
        
        # Save figure
        plt.savefig(self.output_dir / 'consensus_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_cluster_distribution_plot(self, sorted_clusters, group_colors):
        """Create a distribution plot showing patterns across clusters by group."""
        # Collect data for plot
        model_cluster_data = []
        for cluster_id, data in sorted_clusters[:30]:  # Show top 20 clusters
            for group, count in data['group_counts'].items():
                if count > 0:
                    model_cluster_data.append({
                        'Cluster': f"C{cluster_id}",
                        'Group': group, 
                        'Count': count
                    })
        
        if not model_cluster_data:
            self.logger.warning("No data available for cluster distribution plot")
            return
        
        df = pd.DataFrame(model_cluster_data)
        
        # Create figure with appropriate size
        plt.figure(figsize=(14, 10))
        
        # Create pivot table with improved readability
        try:
            cluster_counts = df.pivot(index='Cluster', columns='Group', values='Count').fillna(0)
            
            # Use a better colormap for the heatmap
            cmap = sns.color_palette("Blues", as_cmap=True)
            
            # Create the heatmap
            ax = sns.heatmap(cluster_counts, annot=True, fmt='.0f', cmap=cmap, 
                        linewidths=0.5, cbar=True)
            
            # Add total counts
            for i, row_label in enumerate(ax.get_yticklabels()):
                cluster_id = row_label.get_text()
                if cluster_id in cluster_counts.index:
                    total = cluster_counts.loc[cluster_id].sum()
                    ax.text(len(cluster_counts.columns) + 0.2, i, f"Total: {int(total)}", 
                        va='center', fontweight='bold')
            
            plt.title('Pattern Distribution Across Clusters by Group', fontsize=16)
            
        except Exception as e:
            self.logger.warning(f"Error creating heatmap: {e}")
            # Alternative: use a grouped bar chart with group colors
            palette = {group: group_colors.get(group, '#333333') for group in df['Group'].unique()}
            
            ax = sns.barplot(x='Cluster', y='Count', hue='Group', data=df, 
                        palette=palette)
            
            plt.title('Pattern Distribution Across Clusters by Group', fontsize=16)
            plt.xlabel('Cluster ID', fontsize=14)
            plt.ylabel('Pattern Count', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add value labels on top of the bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.0f')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
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
                plt.figure(figsize=(10, max(5, len(pivot_df)*0.5)))
                
                # Use a colormap with good distinguishability for discrete values
                cmap = plt.get_cmap('tab20', len(self.cluster_info)+1)
                
                ax = sns.heatmap(pivot_df, 
                                 annot=True, 
                                 fmt='g', 
                                 cmap="rainbow", 
                                 cbar=False,
                                 linewidths=0.5, 
                                 linecolor='white')
                
                # Set title and labels
                plt.title(f'{group}: Model State to Cluster Mapping (Sorted States)', fontsize=14)
                
                # Set x-axis ticks for all states
                plt.xticks(np.arange(20) + 0.5, range(1, 21))
                plt.xlabel('Sorted State Index', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{group}_model_clusters.png', dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Error creating heatmap for {group}: {str(e)}")
                print("DataFrame contents:")
                print(df)
        
        # Create cross-group cluster visualization 
        self._create_cluster_overlap_diagram()


    def _create_cluster_overlap_diagram(self):
        """Create diagram showing how clusters overlap between groups."""
        # Extract data
        group_overlaps = {}
        
        # Get count of patterns per group for each cluster
        for cluster_id, data in self.cluster_info.items():
            if len(data['groups']) > 1:  # Cluster has patterns from multiple groups
                group_overlaps[cluster_id] = data['group_counts']
        
        if not group_overlaps:
            return
        
        # Create overlap visualization
        plt.figure(figsize=(12, 6))
        
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
        
        palette = {group: COLORS[group] for group in df['Group'].unique()}
        # Create grouped bar chart
        sns.barplot(x='Cluster', y='Count', hue='Group', data=df, palette=palette)
        
        plt.title('Shared Clusters Across Groups', fontsize=14)
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Pattern Count', fontsize=12)
        plt.legend(title='Group')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_overlap.png', dpi=300)
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

def main():
    from dotenv import load_dotenv
    load_dotenv()
        
    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = os.path.join(scratch_dir, "output")
    output_dir = os.path.join(base_dir, "06_state_pattern_clusters")

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract and cluster state patterns from HMM results')
    
    parser.add_argument('--groups', type=str, nargs='+', default=["affair", "paranoia", "combined"], 
                        help='List of groups to analyze')
    parser.add_argument('--min_activation', type=float, default=0.1, help='Minimum activation threshold')
    parser.add_argument('--max_ci_width', type=float, default=0.3, help='Maximum CI width threshold')
    parser.add_argument('--min_pattern_stability', type=float, default=0.7, help='Minimum pattern stability')
    parser.add_argument('--cluster_similarity', type=float, default=0.7, help='Cluster similarity threshold')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "pattern_analysis.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create config from args
    config = {
        'min_activation': args.min_activation,
        'max_ci_width': args.max_ci_width,
        'min_pattern_stability': args.min_pattern_stability,
        'cluster_similarity_threshold': args.cluster_similarity
    }
    
    try:
        # Create and run analyzer
        analyzer = StatePatternAnalyzer(
            base_dir=base_dir,
            groups=args.groups,
            output_dir=output_dir,
            config=config
        )
        
        analyzer.run_analysis()
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()