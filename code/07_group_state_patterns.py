import os
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

class StatePatternAnalyzer:
    def __init__(self, base_path: str, group: str, output_dir: str, 
             min_activation: float = 0.1,
             max_ci_width: float = 0.3,
             min_pattern_stability: float = 0.5,
             importance_percentile: float = 75,
             similarity_threshold: float = 0.8):
        """
        Initialize the StatePatternAnalyzer.
        
        Args:
            base_path (str): Base path to the data directory
            group (str): Group identifier
            output_dir (str): Directory for output files
            min_activation (float): Minimum activation threshold for reliable states
            max_ci_width (float): Maximum confidence interval width for reliable states
            min_pattern_stability (float): Minimum pattern stability score
            importance_percentile (float): Percentile threshold for feature importance
            similarity_threshold (float): Similarity threshold for pattern merging
        """
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic attributes
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.group = group
        self.state_properties = []
        self.unique_patterns = None
        self.pattern_frequencies = None
        
        # Store analysis parameters
        self.analysis_params = {
            'min_activation': min_activation,
            'max_ci_width': max_ci_width,
            'min_pattern_stability': min_pattern_stability,
            'importance_percentile': importance_percentile,
            'similarity_threshold': similarity_threshold
        }
        
        # Log initialization parameters
        self.logger.info(f"""
            Initializing StatePatternAnalyzer:
            - Group: {group}
            - Base path: {base_path}
            - Output directory: {output_dir}
            Analysis Parameters:
            - Minimum activation: {min_activation}
            - Maximum CI width: {max_ci_width}
            - Minimum pattern stability: {min_pattern_stability}
            - Importance percentile: {importance_percentile}
            - Similarity threshold: {similarity_threshold}
        """)
        
        # Get metrics files
        self.metrics_files = self._get_metrics_files()
        
    def _get_metrics_files(self) -> List[Path]:
        """Get all metrics files for the group."""
        files = sorted(self.base_path.glob(f"{self.group}_hmm_*states_ntw_native_trimmed/statistics/{self.group}_metrics.pkl"))
        self.logger.info(f"Found {len(files)} metrics files")
        return files
        
    def load_state_properties(self):
        """
        Load state properties with enhanced metrics for feature selection.
        """
        self.state_properties = []
        processed_files = 0
        processed_states = 0
        
        for metrics_file in self.metrics_files:
            try:
                with open(metrics_file, 'rb') as f:
                    metrics = pickle.load(f)
                
                # Extract state properties
                state_properties = metrics.get('state_properties', {})
                if not state_properties or 'separability' not in state_properties:
                    continue
                    
                # Remove separability as it's a global metric
                state_properties.pop('separability')
                
                # Process each state
                for state_idx, state_data in state_properties.items():
                    if not isinstance(state_data, dict):
                        continue
                        
                    try:
                        # Validate required metrics exist
                        required_metrics = [
                            'mean_pattern',
                            'mean_pattern_ci',
                            'std_pattern',
                            'feature_importance',
                            'pattern_stability'
                        ]
                        
                        if not all(metric in state_data for metric in required_metrics):
                            continue
                            
                        # Convert CI from dict to array format if needed
                        mean_pattern_ci = state_data['mean_pattern_ci']
                        if isinstance(mean_pattern_ci, dict):
                            ci = np.array([
                                mean_pattern_ci['lower'],
                                mean_pattern_ci['upper']
                            ]).T
                        else:
                            ci = mean_pattern_ci
                            
                        # Store all relevant metrics
                        self.state_properties.append({
                            'file': metrics_file,
                            'state_idx': state_idx,
                            'features': {
                                'mean_pattern': np.array(state_data['mean_pattern']),
                                'mean_pattern_ci': ci,
                                'std_pattern': np.array(state_data['std_pattern']),
                                'feature_importance': np.array(state_data['feature_importance']),
                                'pattern_stability': state_data['pattern_stability'],
                                'cv_pattern': np.array(state_data['cv_pattern'])  # Add coefficient of variation
                            }
                        })
                        
                        processed_states += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing state {state_idx} in {metrics_file}: {str(e)}")
                        continue
                
                processed_files += 1
                
            except Exception as e:
                self.logger.error(f"Error processing file {metrics_file}: {str(e)}")
                continue
        
        self.logger.info(f"Processed {processed_files}/{len(self.metrics_files)} files")
        self.logger.info(f"Loaded {processed_states} states")
        
        if not self.state_properties:
            raise ValueError("No valid state properties were loaded")
            
        return len(self.state_properties)

    def filter_reliable_states(self, 
                          min_activation: float = 0.1,  # Lower threshold to catch more meaningful states
                          max_ci_width: float = 0.5,     # Relaxed to account for natural variability
                          min_pattern_stability: float = 0.3):  # Key metric for reliability
        """
        Filter states based on core reliability criteria, focusing on activation strength,
        confidence interval stability, and pattern stability.
        
        Args:
            min_activation: Minimum absolute activation threshold
            max_ci_width: Maximum allowed width of confidence interval
            min_pattern_stability: Minimum pattern stability score
        """
        reliable_states = []
        
        # Store filtering statistics
        filter_stats = {
            'total_states': len(self.state_properties),
            'failed_activation': 0,
            'failed_ci_width': 0,
            'failed_stability': 0,
            'state_metrics': []
        }
        
        for state in self.state_properties:
            features = state['features']
            mean_pattern = features['mean_pattern']
            ci = features['mean_pattern_ci']
            pattern_stability = features['pattern_stability']
            
            # Calculate core metrics
            max_abs_activation = np.max(np.abs(mean_pattern))
            ci_width = ci[:, 1] - ci[:, 0]
            max_ci_width_observed = np.max(ci_width)
            
            # Store metrics for diagnostics
            state_metrics = {
                'state_idx': state['state_idx'],
                'max_abs_activation': max_abs_activation,
                'max_ci_width': max_ci_width_observed,
                'pattern_stability': pattern_stability
            }
            filter_stats['state_metrics'].append(state_metrics)
            
            # Apply core criteria
            passes_activation = max_abs_activation >= min_activation
            passes_ci_width = max_ci_width_observed <= max_ci_width
            passes_stability = pattern_stability >= min_pattern_stability
            
            # Track failures for diagnostics
            if not passes_activation:
                filter_stats['failed_activation'] += 1
            if not passes_ci_width:
                filter_stats['failed_ci_width'] += 1
            if not passes_stability:
                filter_stats['failed_stability'] += 1
                
            # Apply all criteria
            if (passes_activation and passes_ci_width and passes_stability):
                reliable_states.append(state)
        
        # Calculate summary statistics
        activation_values = [m['max_abs_activation'] for m in filter_stats['state_metrics']]
        ci_width_values = [m['max_ci_width'] for m in filter_stats['state_metrics']]
        stability_values = [m['pattern_stability'] for m in filter_stats['state_metrics']]
        
        self.logger.info(f"""
            State filtering results:
            Total states: {filter_stats['total_states']}
            States passing all criteria: {len(reliable_states)}
            
            Failure breakdown:
            - Failed activation threshold ({min_activation}): {filter_stats['failed_activation']}
            - Failed CI width threshold ({max_ci_width}): {filter_stats['failed_ci_width']}
            - Failed stability threshold ({min_pattern_stability}): {filter_stats['failed_stability']}
            
            Data characteristics:
            Activation: range [{min(activation_values):.3f}, {max(activation_values):.3f}], median {np.median(activation_values):.3f}
            CI width: range [{min(ci_width_values):.3f}, {max(ci_width_values):.3f}], median {np.median(ci_width_values):.3f}
            Stability: range [{min(stability_values):.3f}, {max(stability_values):.3f}], median {np.median(stability_values):.3f}
        """)
        
        # Store filter statistics
        self.filter_stats = filter_stats
        
        return reliable_states

    def select_significant_features(self, state: Dict,
                          min_activation: float = 0.2,  # Direct threshold
                          min_stability: float = 0.5) -> np.ndarray:
        """Select significant features using a direct activation threshold."""
        mean_pattern = state['features']['mean_pattern']
        ci = state['features']['mean_pattern_ci']
        importance = state['features']['feature_importance']
        pattern_stability = state['features']['pattern_stability']
        
        # Simple activation threshold
        reliable_activation = mean_pattern > min_activation
        
        # Other reliability checks
        reliable_direction = ci[:, 0] > 0  # Lower CI bound positive
        important_features = importance > np.percentile(importance, 75)
        stable_pattern = pattern_stability > min_stability
        
        significant_features = (reliable_activation & 
                            reliable_direction & 
                            important_features & 
                            stable_pattern)
        
        return significant_features

    def merge_similar_patterns(self, patterns: np.ndarray, similarity_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge patterns that are very similar to each other.
        
        Args:
            patterns: Array of binary patterns (n_patterns x n_features)
            similarity_threshold: Jaccard similarity threshold for merging
            
        Returns:
            Tuple of (merged patterns array, mapping from old to new pattern indices)
        """
        n_patterns = len(patterns)
        # Initialize mapping where each pattern maps to itself
        pattern_mapping = np.arange(n_patterns)
        
        # Calculate Jaccard similarities between all pairs
        for i in range(n_patterns):
            if i % 50 == 0:
                self.logger.info(f"Processing pattern {i}/{n_patterns}")
                
            for j in range(i + 1, n_patterns):
                # Skip if j has already been merged
                if pattern_mapping[j] != j:
                    continue
                    
                # Calculate Jaccard similarity
                intersection = np.sum(patterns[i] & patterns[j])
                union = np.sum(patterns[i] | patterns[j])
                similarity = intersection / union if union > 0 else 0
                
                # If similar enough, merge j into i
                if similarity >= similarity_threshold:
                    pattern_mapping[j] = pattern_mapping[i]
        
        # Create new pattern set
        unique_pattern_ids = np.unique(pattern_mapping)
        merged_patterns = np.zeros((len(unique_pattern_ids), patterns.shape[1]), dtype=int)
        
        # For each merged group, take the most common feature combinations
        for new_idx, original_idx in enumerate(unique_pattern_ids):
            # Get all patterns that map to this group
            group_patterns = patterns[pattern_mapping == original_idx]
            # Take features that appear in majority of patterns
            merged_patterns[new_idx] = (np.mean(group_patterns, axis=0) > 0.5).astype(int)
        
        return merged_patterns, pattern_mapping

    def extract_unique_patterns(self):
        """Extract unique co-activation patterns with improved merging."""
        # Get reliable states first with updated criteria
        reliable_states = self.filter_reliable_states(
            min_activation=self.analysis_params["min_activation"],
            max_ci_width=self.analysis_params["max_ci_width"],
            min_pattern_stability=self.analysis_params["min_pattern_stability"]
        )
        
        if not reliable_states:
            raise ValueError("No reliable states found after filtering")
        
        # Extract significant features for each reliable state
        all_patterns = []
        pattern_info = []
        
        for state in reliable_states:
            significant_features = self.select_significant_features(
                state,
                min_stability=self.analysis_params["min_pattern_stability"]
            )
            
            if np.any(significant_features):
                # Store more detailed information about the pattern
                mean_pattern = state['features']['mean_pattern']
                active_indices = np.where(significant_features)[0]
                
                all_patterns.append(significant_features)
                pattern_info.append({
                    'file': str(state['file']),
                    'state_idx': state['state_idx'],
                    'n_active': np.sum(significant_features),
                    'active_features': active_indices.tolist(),
                    'activation_values': mean_pattern[active_indices].tolist(),
                    'pattern_stability': state['features']['pattern_stability']
                })

        if not all_patterns:
            raise ValueError("No valid patterns found after feature selection")

        # Convert to numpy array for processing
        all_patterns_array = np.array(all_patterns)
        
        # First get unique patterns
        unique_patterns, indices, initial_counts = np.unique(
            all_patterns_array, 
            axis=0, 
            return_index=True, 
            return_counts=True
        )
        
        # Log pattern characteristics before merging
        self.logger.info("Pattern characteristics before merging:")
        self._log_pattern_statistics(unique_patterns, initial_counts, pattern_info)
        
        # Then merge similar patterns
        merged_patterns, pattern_mapping = self.merge_similar_patterns(
            unique_patterns,
            similarity_threshold=self.analysis_params["similarity_threshold"]
        )
        
        # Calculate new frequencies after merging
        final_counts = np.zeros(len(merged_patterns), dtype=int)
        merged_pattern_info = []  # Track merged pattern information
        
        # Create mapping from old patterns to merged patterns
        unique_pattern_ids = np.unique(pattern_mapping)
        for new_idx, pattern_id in enumerate(unique_pattern_ids):
            # Find all patterns that were merged into this one
            old_indices = np.where(pattern_mapping == pattern_id)[0]
            
            # Sum up counts
            final_counts[new_idx] = np.sum(initial_counts[old_indices])
            
            # Collect info about merged patterns
            merged_info = {
                'original_patterns': [pattern_info[indices[i]] for i in old_indices],
                'merged_count': final_counts[new_idx],
                'constituent_counts': initial_counts[old_indices].tolist()
            }
            merged_pattern_info.append(merged_info)
        
        # Convert patterns to a serializable format
        self.unique_patterns = merged_patterns.astype(bool)  # Store as boolean array
        self.pattern_frequencies = {
            str(i): count  # Use simple integer indices as keys
            for i, (pattern, count) in enumerate(zip(merged_patterns, final_counts))
        }
        
        # Store additional pattern information
        self.pattern_details = {
            'pre_merge': {
                'patterns': unique_patterns.tolist(),
                'counts': initial_counts.tolist(),
                'info': pattern_info
            },
            'post_merge': {
                'patterns': merged_patterns.tolist(),
                'counts': final_counts.tolist(),
                'info': merged_pattern_info
            }
        }
        
        # Log detailed statistics
        self.logger.info("\nFinal pattern statistics:")
        self._log_pattern_statistics(merged_patterns, final_counts, merged_pattern_info)
        
        return self.pattern_details
    
    def _log_pattern_statistics(self, patterns, counts, info):
        """Helper function to log pattern statistics."""
        self.logger.info(f"Number of patterns: {len(patterns)}")
        self.logger.info(f"Pattern frequency distribution: {sorted(counts, reverse=True)}")
        
        # Calculate and log pattern characteristics
        n_features = patterns.shape[1]
        active_features_per_pattern = np.sum(patterns, axis=1)
        
        self.logger.info(f"""
            Pattern characteristics:
            - Total features available: {n_features}
            - Mean active features per pattern: {np.mean(active_features_per_pattern):.2f}
            - Median active features per pattern: {np.median(active_features_per_pattern):.2f}
            - Min active features: {np.min(active_features_per_pattern)}
            - Max active features: {np.max(active_features_per_pattern)}
            - Info: {info}
        """)
        
    def visualize_patterns(self):
        """Create comprehensive visualization of co-activation patterns with improved styling."""
        if self.unique_patterns is None or self.pattern_frequencies is None:
            raise ValueError("Run extract_unique_patterns before visualization")

        COLORS = {
            'heatmap': '#e41a1c',
            'pattern_freq': '#ff7f00',
            'feature_freq': '#377eb8'
        }
        
        network_labels = [
            'Auditory', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC',
            'DorsAttnA', 'DorsAttnB', 'Language', 'SalVenAttnA', 'SalVenAttnB',
            'SomMotA', 'SomMotB', 'VisualA', 'VisualB', 'VisualC', 'Subcortex'
        ]
        
        # Data preparation
        patterns_array = np.array(self.unique_patterns, dtype=int)
        frequencies = np.array(list(self.pattern_frequencies.values()))
        
        if len(patterns_array) == 0:
            self.logger.warning("No patterns to visualize")
            return

        # Sort patterns by frequency
        sort_idx = np.argsort(frequencies)[::-1]
        patterns_array = patterns_array[sort_idx]
        frequencies = frequencies[sort_idx]
        
        # Create figure with adjusted height ratio for larger cells
        fig = plt.figure(figsize=(20, 18))
        gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[0.8, 4],
                        hspace=0, wspace=0)  # Small spacing for visual separation
        
        # 1. Feature frequency barplot (top) with adjusted bar width
        ax_top = fig.add_subplot(gs[0, 0])
        feature_freq = patterns_array.sum(axis=0)
        
        # Calculate bar width to match heatmap cells (leaving small gaps)
        bar_width = 0.9  # Adjust this value between 0 and 1 to control gap size
        
        bars = ax_top.bar(range(patterns_array.shape[1]), feature_freq,
                        width=bar_width,
                        color=COLORS['feature_freq'], alpha=0.9)
        
        # Add count annotations with larger font
        for bar in bars:
            height = bar.get_height()
            ax_top.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12)  # Increased font size
        
        ax_top.set_xlim(-0.5, patterns_array.shape[1] - 0.5)
        ax_top.set_title("Feature Participation Frequency", pad=10, fontsize=12)
        ax_top.set_xticks([])
        # Remove y-axis completely
        ax_top.set_yticks([])
        ax_top.spines['left'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['top'].set_visible(False)
        
        # 2. Main heatmap (bottom left)
        ax_main = fig.add_subplot(gs[1, 0])
        
        im = ax_main.imshow(patterns_array,
                        aspect='auto',
                        cmap=ListedColormap(['white', COLORS['heatmap']]),
                        interpolation='nearest')
        
        # Add pattern IDs without ticks
        pattern_ids = [f"P{i+1}" for i in range(len(patterns_array))]
        ax_main.set_yticks(range(len(pattern_ids)))
        ax_main.set_yticklabels(pattern_ids)
        ax_main.tick_params(axis='y', length=0)
        
        # Add network labels without ticks
        ax_main.set_xticks(range(len(network_labels)))
        ax_main.set_xticklabels(network_labels, rotation=0, ha='center')
        ax_main.tick_params(axis='x', length=0)
        
        # Grid lines
        for x in range(len(network_labels)+1):
            ax_main.axvline(x-0.5, color='black', linewidth=0.8, alpha=0.3)
        for y in range(len(pattern_ids)+1):
            ax_main.axhline(y-0.5, color='black', linewidth=0.8, alpha=0.3)
        
        # 3. Pattern frequency barplot (right) with larger numbers
        ax_right = fig.add_subplot(gs[1, 1])
        bars = ax_right.barh(range(len(frequencies)), frequencies,
                            color=COLORS['pattern_freq'], alpha=0.9)
        
        # Add count annotations with larger font
        for bar in bars:
            width = bar.get_width()
            text_offset = max(frequencies) * 0.02
            ax_right.text(width + text_offset, 
                        bar.get_y() + bar.get_height()/2,
                        f'{int(width)}',
                        ha='left', va='center', 
                        fontsize=12)  # Increased font size
        
        ax_right.set_ylim(ax_main.get_ylim())
        ax_right.set_title("Pattern\nFrequency", fontsize=12, pad=10)
        ax_right.set_yticks([])
        # Remove x-axis completely
        ax_right.set_xticks([])
        ax_right.spines['bottom'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        
        # Updated main plot styling
        ax_main.set_xticklabels(network_labels, rotation=0, ha='center')
        ax_main.set_ylabel("Co-Activation Patterns", fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{self.group}_patterns.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        self.logger.info(f"Saved visualization to {output_path}")
        plt.close()

    def create_coactivation_plot(self):
        """Create a network co-activation rate heatmap."""
        if self.unique_patterns is None:
            raise ValueError("Run extract_unique_patterns before creating co-activation plot")
        
        # Network labels
        network_labels = [
            'Auditory', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC',
            'DorsAttnA', 'DorsAttnB', 'Language', 'SalVenAttnA', 'SalVenAttnB',
            'SomMotA', 'SomMotB', 'VisualA', 'VisualB', 'VisualC', 'Subcortex'
        ]
        
        n_networks = len(network_labels)
        patterns_array = np.array(self.unique_patterns, dtype=bool)
        pattern_frequencies = np.array([freq for freq in self.pattern_frequencies.values()])
        total_occurrences = np.sum(pattern_frequencies)

        # Initialize co-activation matrix
        coactivation_matrix = np.zeros((n_networks, n_networks))

        # Calculate weighted co-activation rates
        for pattern, freq in zip(patterns_array, pattern_frequencies):
            # Get indices where networks are active
            active_networks = np.where(pattern)[0]
            # For each pair of active networks
            for i in active_networks:
                for j in active_networks:
                    if i != j:  # Skip diagonal
                        coactivation_matrix[i, j] += freq

        # Normalize by total pattern occurrences
        coactivation_matrix = coactivation_matrix / total_occurrences
        
        # Create figure with large size for readability
        plt.figure(figsize=(15, 12))
        
        # Create mask for upper triangle and diagonal
        mask = np.triu(np.ones_like(coactivation_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(coactivation_matrix,
                    mask=mask,
                    square=True,
                    cmap='YlOrRd',
                    xticklabels=network_labels,
                    yticklabels=network_labels,
                    cbar_kws={'label': 'Co-activation Rate'})
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Add title
        plt.title(f'Network Co-activation Rate Matrix ({self.group})', pad=20)
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{self.group}_coactivation_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        self.logger.info(f"Saved co-activation matrix visualization to {output_path}")
        plt.close()

        # Calculate and return co-activation statistics
        tril_indices = np.tril_indices(n_networks, k=-1)
        tril_values = coactivation_matrix[tril_indices]
        
        # Find top co-activating pairs
        top_k = 5  # Number of top pairs to track
        top_indices = np.argsort(tril_values)[-top_k:]
        top_pairs = []
        for idx in top_indices:
            i, j = tril_indices[0][idx], tril_indices[1][idx]
            top_pairs.append({
                'networks': [network_labels[i], network_labels[j]],
                'rate': float(tril_values[idx])
            })
        
        stats = {
            'mean_rate': float(np.mean(tril_values)),
            'max_rate': float(np.max(tril_values)),
            'top_pairs': top_pairs
        }
        
        return stats

    def _create_pattern_summary_plots(self):
        """Create additional summary visualizations."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Pattern similarity matrix
        plt.figure(figsize=(10, 8))
        patterns_array = np.array(self.unique_patterns, dtype=bool)  # Convert to boolean array
        similarity_matrix = np.zeros((len(patterns_array), len(patterns_array)))
        
        for i in range(len(patterns_array)):
            for j in range(len(patterns_array)):
                # Calculate Jaccard similarity using boolean operations
                intersection = np.sum(patterns_array[i] & patterns_array[j])
                union = np.sum(patterns_array[i] | patterns_array[j])
                similarity_matrix[i, j] = intersection / union if union > 0 else 0
        
        sns.heatmap(similarity_matrix, 
                    cmap='viridis', 
                    xticklabels=False, 
                    yticklabels=False)
        plt.title(f"Pattern Similarity Matrix ({self.group})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.group}_pattern_similarity.png"))
        plt.close()
        
        # 2. Pattern complexity distribution
        plt.figure(figsize=(8, 6))
        pattern_sizes = np.sum(patterns_array, axis=1)
        plt.hist(pattern_sizes, bins='auto', alpha=0.7)
        plt.axvline(np.mean(pattern_sizes), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(pattern_sizes):.2f}')
        plt.xlabel("Number of Active Features")
        plt.ylabel("Count")
        plt.title(f"Pattern Complexity Distribution ({self.group})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.group}_pattern_complexity.png"))
        plt.close()

    def save_results(self, output_dir: Path):
        """Save analysis results to file."""
        if self.unique_patterns is None or self.pattern_frequencies is None:
            raise ValueError("Run analysis before saving results.")
            
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save patterns as JSON
        results = {
            'unique_patterns': self.unique_patterns.tolist(),
            'pattern_frequencies': {
                str(k): int(v) for k, v in self.pattern_frequencies.items()  # Convert numpy types to native Python types
            }
        }
        
        with open(output_dir / f"{self.group}_patterns.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {output_dir}")

    def _convert_to_serializable(self, obj):
        """Convert numpy types to regular Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        analysis_start_time = time.time()
        analysis_summary = {
            'success': False,
            'steps_completed': [],
            'errors': [],
            'warnings': [],
            'statistics': {},
            'results': None
        }
        
        try:
            # Step 1: Load state properties
            self.logger.info("Step 1/4: Loading state properties...")
            n_states = self.load_state_properties()
            analysis_summary['steps_completed'].append('load_state_properties')
            analysis_summary['statistics']['total_states'] = n_states
            
            # Step 2: Extract unique patterns
            self.logger.info("Step 2/4: Extracting unique patterns...")
            pattern_details = self.extract_unique_patterns()
            analysis_summary['steps_completed'].append('extract_patterns')
            
            # Collect pattern statistics
            pattern_stats = {
                'n_initial_patterns': len(pattern_details['pre_merge']['patterns']),
                'n_final_patterns': len(pattern_details['post_merge']['patterns']),
                'pattern_frequencies': [int(x) for x in pattern_details['post_merge']['counts']],  # Convert to regular integers
                'avg_features_per_pattern': float(np.mean([np.sum(p) for p in pattern_details['post_merge']['patterns']]))
            }
            analysis_summary['statistics'].update(pattern_stats)
            
            # Step 3: Create visualizations
            self.logger.info("Step 3/4: Creating visualizations...")
            try:
                # Create main pattern visualization
                self.visualize_patterns()
                
                # Create co-activation plot and collect statistics
                coactivation_stats = self.create_coactivation_plot()
                analysis_summary['statistics']['coactivation'] = {
                    'mean_rate': float(coactivation_stats['mean_rate']),
                    'max_rate': float(coactivation_stats['max_rate']),
                    'top_pairs': coactivation_stats['top_pairs']
                }
                
                analysis_summary['steps_completed'].append('create_visualizations')
                
            except Exception as vis_error:
                error_msg = f"Error in visualization step: {str(vis_error)}"
                self.logger.warning(error_msg)
                analysis_summary['warnings'].append({
                    'step': 'visualizations',
                    'warning': error_msg
                })
            
            analysis_summary['success'] = True
            analysis_summary['results'] = self._convert_to_serializable(pattern_details)
            
        except Exception as e:
            error_msg = f"Error in analysis pipeline: {str(e)}"
            self.logger.error(error_msg)
            analysis_summary['errors'].append({
                'step': analysis_summary['steps_completed'][-1] if analysis_summary['steps_completed'] else 'unknown',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
            
        finally:
            # Calculate execution time
            execution_time = time.time() - analysis_start_time
            analysis_summary['execution_time'] = execution_time
            
            # Log final summary
            self._log_analysis_summary(analysis_summary)
            
            # Convert analysis_summary to JSON-serializable format
            serializable_summary = self._convert_to_serializable(analysis_summary)
            
            # Save analysis summary
            summary_path = os.path.join(self.output_dir, f"{self.group}_analysis_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(serializable_summary, f, indent=2)
        
        return analysis_summary

    def _validate_patterns(self):
        """
        Perform validation checks on extracted patterns without stability calculation.
        """
        validation_results = {
            'feature_consistency': [],
            'pattern_distinctiveness': []
        }
        
        if self.unique_patterns is not None:
            # Calculate feature consistency
            feature_freq = np.sum(self.unique_patterns, axis=0)
            validation_results['feature_consistency'] = feature_freq.tolist()
            
            # Calculate pattern distinctiveness (using Jaccard distances)
            n_patterns = len(self.unique_patterns)
            distinctiveness_matrix = np.zeros((n_patterns, n_patterns))
            
            for i in range(n_patterns):
                for j in range(i+1, n_patterns):
                    intersection = np.sum(self.unique_patterns[i] & self.unique_patterns[j])
                    union = np.sum(self.unique_patterns[i] | self.unique_patterns[j])
                    jaccard_dist = 1 - (intersection / union if union > 0 else 0)
                    distinctiveness_matrix[i,j] = jaccard_dist
                    distinctiveness_matrix[j,i] = jaccard_dist
            
            validation_results['pattern_distinctiveness'] = distinctiveness_matrix.tolist()
        
        return validation_results

    def _save_detailed_results(self, results: Dict):
        """
        Save detailed analysis results to files.
        """
        output_files = {
            'patterns': f"{self.group}_patterns.json",
            'validation': f"{self.group}_validation.json",
            'statistics': f"{self.group}_statistics.json"
        }
        
        for key, filename in output_files.items():
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(results.get(key, {}), f, indent=2)

    def _log_analysis_summary(self, summary: Dict):
        """
        Log comprehensive analysis summary.
        """
        self.logger.info("\n=== Analysis Summary ===")
        self.logger.info(f"Success: {summary['success']}")
        self.logger.info(f"Steps completed: {', '.join(summary['steps_completed'])}")
        self.logger.info(f"Execution time: {summary['execution_time']:.2f} seconds")
        
        if summary['statistics']:
            self.logger.info("\nKey Statistics:")
            for key, value in summary['statistics'].items():
                self.logger.info(f"- {key}: {value}")
        
        if summary['warnings']:
            self.logger.warning("\nWarnings:")
            for warning in summary['warnings']:
                self.logger.warning(f"- {warning}")
        
        if summary['errors']:
            self.logger.error("\nErrors:")
            for error in summary['errors']:
                self.logger.error(f"- Step: {error['step']}")
                self.logger.error(f"  Error: {error['error']}")
    
def main():
    import argparse
    from dotenv import load_dotenv
    import json
    import yaml
    from datetime import datetime
    
    # Set up logging with file output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def setup_logging(output_dir: Path, group: str):
        log_dir = output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'{group}_analysis_{timestamp}.log'
        
        # Set up file and console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    # Load environment variables
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    
    if not scratch_dir:
        raise ValueError("SCRATCH_DIR environment variable not set")
    
    # Set up argument parser with more options
    parser = argparse.ArgumentParser(description='Analyze recurring state patterns')
    
    # Required arguments
    parser.add_argument('--group', type=str, required=True,
                       help='Group identifier')
    
    # Optional arguments with defaults
    parser.add_argument('--res', type=str, default="native",
                       help='Resolution of the atlas')
    parser.add_argument('--min-activation', type=float, default=0.1,
                       help='Minimum activation threshold')
    parser.add_argument('--max-ci-width', type=float, default=0.3,
                       help='Maximum CI width threshold')
    parser.add_argument('--min-pattern-stability', type=float, default=0.5,
                       help='Minimum pattern stability threshold')
    parser.add_argument('--importance-percentile', type=float, default=75,
                       help='Percentile threshold for feature importance')
    parser.add_argument('--similarity-threshold', type=float, default=0.8,
                       help='Similarity threshold for pattern merging')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML)')
    
    args = parser.parse_args()
    
    # Set up paths
    base_path = Path(scratch_dir) / "output"
    output_dir = base_path / f"{args.group}_state_patterns_{args.res}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, args.group)
    
    # Log system information
    logger.info("=== Analysis Configuration ===")
    logger.info(f"Start time: {timestamp}")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration file if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {e}")
    
    # Combine command line arguments and configuration file
    analysis_params = {
        'min_activation': args.min_activation,
        'max_ci_width': args.max_ci_width,
        'min_pattern_stability': args.min_pattern_stability,
        'importance_percentile': args.importance_percentile,
        'similarity_threshold': args.similarity_threshold
    }
    
    # Config file overrides command line defaults but not explicit arguments
    if config:
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                analysis_params[key] = value
    
    # Save final configuration
    config_file = output_dir / f'analysis_config_{timestamp}.json'
    with open(config_file, 'w') as f:
        json.dump({
            'arguments': vars(args),
            'analysis_parameters': analysis_params,
            'timestamp': timestamp
        }, f, indent=2)
    
    try:
        # Initialize analyzer with parameters
        analyzer = StatePatternAnalyzer(
            base_path=base_path,
            group=args.group,
            output_dir=output_dir,
            **analysis_params
        )
        
        # Run analysis
        results = analyzer.run_analysis()
        
        # Save results
        analyzer.save_results(output_dir)
        
        logger.info("=== Analysis Complete ===")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Configuration saved to: {config_file}")
        
    except Exception as e:
        logger.error("=== Analysis Failed ===")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Save final status
        status_file = output_dir / f'analysis_status_{timestamp}.json'
        with open(status_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'success': 'results' in locals(),
                'error': str(e) if 'e' in locals() else None
            }, f, indent=2)

if __name__ == '__main__':
    main()
