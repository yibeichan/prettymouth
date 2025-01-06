import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import itertools
from dotenv import load_dotenv

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class GroupPatternComparison:
    def __init__(self, base_dir: str, groups: List[str] = ["affair", "paranoia", "combined"]):
        """
        Initialize group comparison analysis.
        
        Args:
            base_dir: Base directory containing group results
            groups: List of group names to compare
        """
        self.base_dir = Path(base_dir)
        self.groups = groups
        self.patterns = {}
        self.coactivation_matrices = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Load data for all groups
        self._load_group_data()
        
    def _load_group_data(self):
        """Load pattern and coactivation data for all groups."""
        for group in self.groups:
            group_dir = self.base_dir / f"{group}_state_patterns_native"
            
            # Load patterns
            pattern_file = group_dir / f"{group}_patterns.json"
            try:
                with open(pattern_file, 'r') as f:
                    self.patterns[group] = json.load(f)
                    # Convert patterns to numpy array
                    self.patterns[group]['unique_patterns'] = np.array(self.patterns[group]['unique_patterns'])
                self.logger.info(f"Loaded patterns for group: {group}")
                
                # Calculate coactivation matrix for this group's patterns
                coact_matrix = self._calculate_coactivation_matrix(
                    self.patterns[group]['unique_patterns'],
                    self.patterns[group]['pattern_frequencies']
                )
                self.coactivation_matrices[group] = coact_matrix
                
            except Exception as e:
                self.logger.error(f"Error loading data for group {group}: {e}")
                raise

    def create_pattern_comparison_plot(self, output_dir: Path):
        network_labels = [
            'Auditory', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC',
            'DorsAttnA', 'DorsAttnB', 'Language', 'SalVenAttnA', 'SalVenAttnB',
            'SomMotA', 'SomMotB', 'VisualA', 'VisualB', 'VisualC', 'Subcortex'
        ]

        # Define group ordering and colors
        group_order = [
            'affair+paranoia+combined',
            'affair+paranoia only',
            'affair+combined only',
            'paranoia+combined only',
            'affair only',
            'paranoia only',
            'combined only'
        ]
        
        colors = {
            'affair+paranoia+combined': '#4285F4',  # Blue
            'affair+paranoia only': '#34A853',           # Green
            'affair+combined only': '#8B4513',           # Yellow
            'paranoia+combined only': '#EA4335',         # Red
            'affair only': '#FF9900',                    # Orange
            'paranoia only': '#9334E6',                  # Purple
            'combined only': '#00CED1'                   # Pink
        }

        # Collect and filter patterns
        unique_patterns = {}
        for group in self.groups:
            patterns = self.patterns[group]['unique_patterns']
            freqs = self.patterns[group]['pattern_frequencies']
            
            for pat_idx, pattern in enumerate(patterns):
                if freqs[str(pat_idx)] >= 3:  # Frequency filter
                    pat_key = tuple(pattern.astype(int))
                    if pat_key not in unique_patterns:
                        unique_patterns[pat_key] = {
                            'pattern': pattern,
                            'groups': set([group]),
                            'total_freq': freqs[str(pat_idx)]
                        }
                    else:
                        unique_patterns[pat_key]['groups'].add(group)
                        unique_patterns[pat_key]['total_freq'] += freqs[str(pat_idx)]

        # Categorize patterns
        categorized_patterns = {category: [] for category in group_order}
        
        for pat_key, info in unique_patterns.items():
            groups = info['groups']
            if groups == {'affair', 'paranoia', 'combined'}:
                category = 'affair+paranoia+combined'
            elif groups == {'affair', 'paranoia'}:
                category = 'affair+paranoia only'
            elif groups == {'affair', 'combined'}:
                category = 'affair+combined only'
            elif groups == {'paranoia', 'combined'}:
                category = 'paranoia+combined only'
            elif groups == {'affair'}:
                category = 'affair only'
            elif groups == {'paranoia'}:
                category = 'paranoia only'
            elif groups == {'combined'}:
                category = 'combined only'
            else:
                continue
                
            categorized_patterns[category].append(info['pattern'])

        # Stack patterns in order
        all_patterns = []
        pattern_colors = []
        for category in group_order:
            patterns = categorized_patterns[category]
            all_patterns.extend(patterns)
            pattern_colors.extend([colors[category]] * len(patterns))

        patterns_array = np.array(all_patterns)
        
        # Create heatmap with colored patterns
        plt.figure(figsize=(25, len(patterns_array) * 0.3))
        
        # Plot binary values
        plt.imshow(patterns_array, aspect='auto', cmap='binary')
        
        # Overlay colors for group membership
        for i, color in enumerate(pattern_colors):
            plt.axhspan(i-0.5, i+0.5, color=color, alpha=0.3)

        # Add vertical grid lines between networks
        for x in range(len(network_labels)-1):
            plt.axvline(x+0.5, color='gray', linewidth=0.5, alpha=0.5)

        # Set x-axis labels horizontal
        plt.xticks(range(len(network_labels)), network_labels, rotation=0)
        
        # Set y-axis labels with pattern numbers
        plt.yticks(range(len(patterns_array)), [f'P{i}' for i in range(len(patterns_array))])

        # Remove ticks
        plt.tick_params(axis='both', which='both', length=0)

        plt.title('Pattern Comparison Across Groups')

        # Add legend
        legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[group], 
                        label=group, alpha=0.3) for group in group_order]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1))

        plt.tight_layout()
        plt.savefig(output_dir / 'pattern_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'total_patterns': len(patterns_array),
            'patterns_per_category': {cat: len(pats) for cat, pats in categorized_patterns.items()}
        }

    def _calculate_coactivation_matrix(self, patterns: np.ndarray, frequencies: Dict[str, int]) -> np.ndarray:
        """Calculate coactivation matrix from patterns and their frequencies."""
        n_networks = patterns.shape[1]
        coactivation_matrix = np.zeros((n_networks, n_networks))
        
        # Convert frequencies dict to array matching pattern order
        freq_array = np.array([frequencies[str(i)] for i in range(len(patterns))])
        total_occurrences = np.sum(freq_array)
        
        # Calculate weighted co-activation rates
        for pattern, freq in zip(patterns, freq_array):
            active_networks = np.where(pattern)[0]
            for i in active_networks:
                for j in active_networks:
                    if i != j:  # Skip diagonal
                        coactivation_matrix[i, j] += freq
        
        # Normalize by total occurrences
        coactivation_matrix = coactivation_matrix / total_occurrences
        return coactivation_matrix
    
    def _calculate_pattern_overlap_stats(self, pattern_group_map: List[List[str]]) -> Dict:
        """
        Calculate comprehensive statistics about pattern overlap between groups.
        
        Args:
            pattern_group_map: List where each element is a list of groups that share a pattern
            
        Returns:
            Dictionary containing various overlap statistics
        """
        stats = {
            'total_patterns': len(pattern_group_map),
            'patterns_per_group': {group: 0 for group in self.groups},
            'group_overlaps': {
                'all_groups': [],
                'affair_paranoia': [],
                'affair_combined': [],
                'paranoia_combined': []
            },
            'unique_patterns': {group: [] for group in self.groups},
            'summary': {
                'overlap_percentages': {},
                'unique_percentages': {}
            }
        }
        
        # Analyze each pattern
        for pattern_idx, groups in enumerate(pattern_group_map):
            # Count patterns per group
            for group in groups:
                stats['patterns_per_group'][group] += 1
            
            # Categorize overlap
            groups_set = set(groups)
            if len(groups) == 1:
                # Unique to one group
                group = groups[0]
                stats['unique_patterns'][group].append(pattern_idx)
            elif groups_set == set(self.groups):
                # Shared by all groups
                stats['group_overlaps']['all_groups'].append(pattern_idx)
            elif groups_set == {'affair', 'paranoia'}:
                stats['group_overlaps']['affair_paranoia'].append(pattern_idx)
            elif groups_set == {'affair', 'combined'}:
                stats['group_overlaps']['affair_combined'].append(pattern_idx)
            elif groups_set == {'paranoia', 'combined'}:
                stats['group_overlaps']['paranoia_combined'].append(pattern_idx)
        
        # Calculate percentages
        total = stats['total_patterns']
        if total > 0:
            # Overlap percentages
            stats['summary']['overlap_percentages'] = {
                'all_groups': len(stats['group_overlaps']['all_groups']) / total * 100,
                'affair_paranoia': len(stats['group_overlaps']['affair_paranoia']) / total * 100,
                'affair_combined': len(stats['group_overlaps']['affair_combined']) / total * 100,
                'paranoia_combined': len(stats['group_overlaps']['paranoia_combined']) / total * 100
            }
            
            # Unique percentages
            stats['summary']['unique_percentages'] = {
                group: len(patterns) / total * 100
                for group, patterns in stats['unique_patterns'].items()
            }
        
        # Add pattern counts to summary
        stats['summary']['counts'] = {
            'total_patterns': total,
            'patterns_per_group': stats['patterns_per_group'],
            'shared_patterns': {
                'all_groups': len(stats['group_overlaps']['all_groups']),
                'affair_paranoia': len(stats['group_overlaps']['affair_paranoia']),
                'affair_combined': len(stats['group_overlaps']['affair_combined']),
                'paranoia_combined': len(stats['group_overlaps']['paranoia_combined'])
            },
            'unique_patterns': {
                group: len(patterns)
                for group, patterns in stats['unique_patterns'].items()
            }
        }
        
        return stats

    def compare_coactivation_matrices(self, output_dir: Path):
        """
        Compare coactivation matrices using multiple distance measures.
        """
        network_labels = [
            'Auditory', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC',
            'DorsAttnA', 'DorsAttnB', 'Language', 'SalVenAttnA', 'SalVenAttnB',
            'SomMotA', 'SomMotB', 'VisualA', 'VisualB', 'VisualC', 'Subcortex'
        ]
        
        # Define group pairs for comparison
        group_pairs = [
            ('affair', 'paranoia'),
            ('affair', 'combined'),
            ('paranoia', 'combined')
        ]
        
        # Initialize matrices for different measures
        diff_matrices = {}  # Absolute differences
        similarity_stats = {}  # Overall statistics
        
        # Calculate all measures for each pair
        for group1, group2 in group_pairs:
            pair_key = f"{group1}_vs_{group2}"
            mat1 = self.coactivation_matrices[group1]
            mat2 = self.coactivation_matrices[group2]
            
            # Get lower triangular indices (excluding diagonal)
            tril_indices = np.tril_indices_from(mat1, k=-1)
            
            # Extract lower triangular values
            tril1 = mat1[tril_indices]
            tril2 = mat2[tril_indices]
            
            # Calculate absolute differences matrix
            diff_matrices[pair_key] = mat1 - mat2
            
            # Calculate overall Pearson correlation between matrices
            corr, p_val = pearsonr(tril1, tril2)
            
            # Calculate cosine similarity between matrices (using lower triangular parts)
            cos_sim = np.dot(tril1, tril2) / (np.linalg.norm(tril1) * np.linalg.norm(tril2))
            cos_dist = 1 - cos_sim
            
            # Calculate MSE (using lower triangular parts)
            mse = np.mean((tril1 - tril2) ** 2)
            rmse = np.sqrt(mse)
            
            # Store similarity statistics
            similarity_stats[pair_key] = {
                'pearson_correlation': float(corr),
                'pearson_pvalue': float(p_val),
                'cosine_distance': float(cos_dist),
                'mse': float(mse),
                'rmse': float(rmse),
                'mean_absolute_difference': float(np.mean(np.abs(diff_matrices[pair_key]))),
                'max_absolute_difference': float(np.max(np.abs(diff_matrices[pair_key])))
            }
        
        # Visualization
        for group1, group2 in group_pairs:
            pair_key = f"{group1}_vs_{group2}"
            
            # Create figure with absolute differences and statistics
            plt.figure(figsize=(8, 8))
            
            # Plot absolute differences
            sns.heatmap(diff_matrices[pair_key],
                    center=0,
                    cmap='RdBu_r',
                    vmin=-0.3,
                    vmax=0.3,
                    xticklabels=network_labels,
                    yticklabels=network_labels)
            
            plt.xticks(rotation=90)
            
            # Add comprehensive statistics in title
            stats = similarity_stats[pair_key]
            title = f'{group1.capitalize()} vs {group2.capitalize()}\n'
            title += f'Pearson ρ = {stats["pearson_correlation"]:.3f} (p = {stats["pearson_pvalue"]:.3e})\n'
            title += f'Cosine Distance = {stats["cosine_distance"]:.3f}\n'
            title += f'RMSE = {stats["rmse"]:.3f}'
            
            plt.title(title)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_dir / f'distance_measures_{pair_key}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Convert numpy arrays to lists for JSON serialization
        results = {
            'similarity_stats': similarity_stats,
            'absolute_differences': {
                key: matrix.tolist() 
                for key, matrix in diff_matrices.items()
            }
        }
        
        return results
        
    def run_analysis(self, output_dir: str):
        """
        Run complete group comparison analysis.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run pattern comparison
        self.logger.info("Running pattern comparison analysis...")
        pattern_stats = self.create_pattern_comparison_plot(output_path)
        
        # Run coactivation comparison
        self.logger.info("Running coactivation matrix comparison...")
        coactivation_stats = self.compare_coactivation_matrices(output_path)
        
        # Save results
        results = {
            'pattern_comparison': pattern_stats,
            'coactivation_comparison': coactivation_stats
        }
        
        with open(output_path / 'group_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Analysis complete. Results saved to {output_path}")
        return results

def main():
    load_dotenv()

    scratch_dir = os.getenv('SCRATCH_DIR')
    base_dir = os.path.join(scratch_dir, 'output')
    output_dir = os.path.join(base_dir, '08_group_coact_pattern_comp')
    
    # Run analysis
    analyzer = GroupPatternComparison(base_dir)
    results = analyzer.run_analysis(output_dir)
    
if __name__ == '__main__':
    main()