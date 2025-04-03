import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging
from collections import defaultdict, Counter
import networkx as nx
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from natsort import natsorted
from scipy.stats import entropy, chi2_contingency, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import scipy

# Define color scheme
COLORS = {
    'affair': '#e41a1c',      # Red
    'affair_light': '#ff6666', # Light red
    'paranoia': '#4daf4a',    # Green
    'paranoia_light': '#90ee90', # Light green
    'combined': '#984ea3',    # Purple
    'combined_light': '#d8b2d8', # Light purple
    'balanced': '#377eb8',   # Blue
    'balanced_light': '#99c2ff' # Light blue
}

class GroupModelComparison:
    def __init__(self, cluster_info_path, state_mappings_path, output_dir, random_seed=42):
        """
        Initialize the Group Model Comparison tool.
        
        Args:
            cluster_info_path: Path to the cluster_info.json file
            state_mappings_path: Path to the state_mappings.json file
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Set output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load cluster and state mapping data
        self.load_data(cluster_info_path, state_mappings_path)
        
        # Extract group information
        self.extract_group_data()
        
    def load_data(self, cluster_info_path, state_mappings_path):
        """Load cluster and state mapping data from JSON files."""
        try:
            with open(cluster_info_path, 'r') as f:
                self.cluster_info = json.load(f)
                self.logger.info(f"Loaded cluster info with {len(self.cluster_info)} clusters")
            
            with open(state_mappings_path, 'r') as f:
                self.state_mappings = json.load(f)
                self.logger.info(f"Loaded state mappings for {len(self.state_mappings)} groups")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def extract_group_data(self):
        """Extract group-specific model and state data."""
        # Get list of groups
        self.groups = list(self.state_mappings.keys())
        self.logger.info(f"Found groups: {', '.join(self.groups)}")
        
        # Create group-specific dataframes
        self.group_data = {}
        
        for group in self.groups:
            # Extract data for this group
            group_models = self.state_mappings.get(group, {})
            
            # Create mapping from model to state to cluster
            model_state_cluster = []
            
            # Track which clusters appear in this group
            group_clusters = set()
            
            # Process each pattern in each cluster
            for cluster_id, cluster_data in self.cluster_info.items():
                # Get members that belong to this group
                for member in cluster_data['members']:
                    if member['group'] == group:
                        model = member.get('model', 'unknown')
                        original_state_idx = member.get('original_state_idx', -1)
                        sorted_state_idx = member.get('sorted_state_idx', -1)
                        
                        model_state_cluster.append({
                            'Group': group,
                            'Model': model,
                            'OriginalStateIdx': original_state_idx,
                            'SortedStateIdx': sorted_state_idx,
                            'ClusterID': int(cluster_id)
                        })
                        
                        group_clusters.add(int(cluster_id))
            
            # Convert to DataFrame
            df = pd.DataFrame(model_state_cluster)
            
            # Add to group data
            self.group_data[group] = {
                'df': df,
                'clusters': sorted(list(group_clusters))
            }
            
            self.logger.info(f"Group {group}: {len(df)} state-cluster mappings, {len(group_clusters)} clusters")
    
    def create_state_cluster_matrices(self):
        """Create state-cluster matrices for each group."""
        self.state_cluster_matrices = {}
        self.model_info = {}
        self.state_frequency_matrices = {}  # New: track state frequencies for statistical testing
        
        for group, data in self.group_data.items():
            df = data['df']
            
            # Get unique models and sorted state indices
            models = df['Model'].unique()
            
            # Store model information
            model_data = []
            
            for model in models:
                # Get states for this model
                model_df = df[df['Model'] == model]
                
                # Get max sorted state index
                max_state = max(model_df['SortedStateIdx'].astype(int)) if not model_df.empty else 0
                
                model_data.append({
                    'Model': model,
                    'StateCount': max_state + 1,
                    'StateRange': list(range(max_state + 1))
                })
            
            self.model_info[group] = pd.DataFrame(model_data)
            
            # Create a matrix representation
            matrix_data = []
            
            # Track cluster frequencies at each position
            max_state_idx = max([max(model_df['SortedStateIdx'].astype(int)) + 1 
                                for model in models 
                                for model_df in [df[df['Model'] == model]]
                                if not model_df.empty], default=0)
            
            # Get max cluster ID across all groups
            max_cluster_id = max([c for group_data in self.group_data.values() 
                                 for c in group_data['clusters']], default=0)
            
            # Initialize frequency matrix
            freq_matrix = np.zeros((max_state_idx, max_cluster_id + 1))
            
            for model in models:
                model_df = df[df['Model'] == model]
                
                # For each state, get its cluster
                for state_idx in range(max(model_df['SortedStateIdx'].astype(int)) + 1):
                    # Find the row with this state
                    state_data = model_df[model_df['SortedStateIdx'].astype(int) == state_idx]
                    
                    if not state_data.empty:
                        cluster_id = state_data.iloc[0]['ClusterID']
                    else:
                        cluster_id = np.nan
                    
                    matrix_data.append({
                        'Model': model,
                        'SortedStateIdx': state_idx,
                        'ClusterID': cluster_id
                    })
                    
                    # Update frequency matrix
                    if not pd.isna(cluster_id):
                        freq_matrix[state_idx, int(cluster_id)] += 1
            
            # Convert to DataFrame
            self.state_cluster_matrices[group] = pd.DataFrame(matrix_data)
            self.state_frequency_matrices[group] = freq_matrix
            
            # Add normalized frequency for statistical comparisons
            if len(models) > 0:
                self.state_frequency_matrices[group] = freq_matrix / len(models)
    
    def calculate_position_weighted_similarity(self, group1_df, group2_df, max_positions=None):
        """
        Calculate similarity between group state-cluster matrices with position awareness.
        
        Args:
            group1_df: DataFrame with columns 'Model', 'SortedStateIdx', 'ClusterID' for group 1
            group2_df: DataFrame with columns 'Model', 'SortedStateIdx', 'ClusterID' for group 2
            max_positions: Maximum number of positions to compare (None = all positions)
        
        Returns:
            float: Similarity score between 0 and 1
            list: Position-specific similarity scores
            dict: Additional metrics for more comprehensive comparison
        """
        # Get all unique models for each group
        models1 = group1_df['Model'].unique()
        models2 = group2_df['Model'].unique()
        
        # Create model-position-cluster matrices for more efficient processing
        max_state_idx1 = int(group1_df['SortedStateIdx'].max()) if not group1_df.empty else 0
        max_state_idx2 = int(group2_df['SortedStateIdx'].max()) if not group2_df.empty else 0
        
        # Determine positions to compare
        if max_positions is None:
            max_positions = min(max_state_idx1, max_state_idx2) + 1
        else:
            max_positions = min(max_positions, max_state_idx1 + 1, max_state_idx2 + 1)
        
        # Calculate position-specific similarity scores
        position_scores = []
        position_ari_scores = []  # Adjusted Rand Index scores by position
        position_ami_scores = []  # Adjusted Mutual Information scores by position
        position_chi2_stats = []  # Chi-square statistics for each position
        position_chi2_pvals = []  # Chi-square p-values for each position
        position_cluster_dists = []  # Cluster distribution differences
        
        for pos in range(max_positions):
            # Get position-specific clusters for each model in each group
            pos_clusters1 = {}
            pos_clusters2 = {}
            
            # Get clusters at this position for each model in group 1
            g1_pos_data = group1_df[group1_df['SortedStateIdx'] == pos]
            for _, row in g1_pos_data.iterrows():
                if not pd.isna(row['ClusterID']):
                    pos_clusters1[row['Model']] = int(row['ClusterID'])
            
            # Get clusters at this position for each model in group 2
            g2_pos_data = group2_df[group2_df['SortedStateIdx'] == pos]
            for _, row in g2_pos_data.iterrows():
                if not pd.isna(row['ClusterID']):
                    pos_clusters2[row['Model']] = int(row['ClusterID'])
            
            # Count cluster overlaps at this position
            overlap_count = 0
            total_comparisons = 0
            
            # Create cluster arrays for ARI and AMI calculations
            clusters1_array = []
            clusters2_array = []
            models_intersection = set(pos_clusters1.keys()) & set(pos_clusters2.keys())
            
            for model in sorted(models_intersection):
                clusters1_array.append(pos_clusters1[model])
                clusters2_array.append(pos_clusters2[model])
            
            # Calculate ARI and AMI if we have enough data
            if len(clusters1_array) >= 2:
                ari_score = adjusted_rand_score(clusters1_array, clusters2_array)
                ami_score = adjusted_mutual_info_score(clusters1_array, clusters2_array)
                position_ari_scores.append(ari_score)
                position_ami_scores.append(ami_score)
            else:
                position_ari_scores.append(0)
                position_ami_scores.append(0)
            
            # Get cluster counts for chi-square test
            all_clusters = sorted(set([c for c in pos_clusters1.values()] + [c for c in pos_clusters2.values()]))
            g1_counts = Counter(pos_clusters1.values())
            g2_counts = Counter(pos_clusters2.values())
            
            # Prepare contingency table
            contingency = np.zeros((2, len(all_clusters)))
            for i, c in enumerate(all_clusters):
                contingency[0, i] = g1_counts.get(c, 0)
                contingency[1, i] = g2_counts.get(c, 0)
            
            # Calculate Jensen-Shannon distance between cluster distributions
            p1 = np.array([g1_counts.get(c, 0) for c in all_clusters])
            p2 = np.array([g2_counts.get(c, 0) for c in all_clusters])
            
            if p1.sum() > 0:
                p1 = p1 / p1.sum()
            if p2.sum() > 0:
                p2 = p2 / p2.sum()
            
            # Calculate the mixture distribution
            m = 0.5 * (p1 + p2)
            
            # Calculate JS divergence carefully
            js_divergence = 0.0
            
            # First component: KL(p1||m)
            mask1 = p1 > 0
            if np.any(mask1):
                js_divergence += 0.5 * entropy(p1[mask1], m[mask1])
            
            # Second component: KL(p2||m)
            mask2 = p2 > 0
            if np.any(mask2):
                js_divergence += 0.5 * entropy(p2[mask2], m[mask2])
            
            position_cluster_dists.append(1 - js_divergence)  # Convert to similarity
            
            # Run chi-square test if we have enough data
            if contingency.shape[1] > 1 and contingency.sum() > 0 and all(contingency.sum(axis=1) > 0):
                try:
                    chi2, p_val, _, _ = chi2_contingency(contingency)
                    position_chi2_stats.append(chi2)
                    position_chi2_pvals.append(p_val)
                except:
                    position_chi2_stats.append(0)
                    position_chi2_pvals.append(1.0)
            else:
                position_chi2_stats.append(0)
                position_chi2_pvals.append(1.0)
            
            # Calculate pairwise similarity for the original metric
            for model1, cluster1 in pos_clusters1.items():
                for model2, cluster2 in pos_clusters2.items():
                    total_comparisons += 1
                    if cluster1 == cluster2:
                        overlap_count += 1
            
            # Calculate position similarity
            if total_comparisons > 0:
                position_scores.append(overlap_count / total_comparisons)
            else:
                position_scores.append(0)
        
        # Apply position weighting (earlier positions could be more important)
        weights = np.linspace(1.0, 0.5, len(position_scores))
        weighted_scores = np.array(position_scores) * weights
        
        # Calculate FDR-corrected p-values
        _, position_chi2_qvals, _, _ = multipletests(
            position_chi2_pvals, 
            alpha=0.05, 
            method='fdr_bh'
        ) if position_chi2_pvals else (None, [], None, None)
        
        # Return weighted average, position scores, and additional metrics
        additional_metrics = {
            'ari_scores': position_ari_scores,
            'ami_scores': position_ami_scores,
            'chi2_stats': position_chi2_stats,
            'chi2_pvals': position_chi2_pvals,
            'chi2_qvals': position_chi2_qvals,  # FDR corrected p-values
            'js_similarities': position_cluster_dists
        }
        
        if np.sum(weights) > 0:
            return np.sum(weighted_scores) / np.sum(weights), position_scores, additional_metrics
        else:
            return 0, position_scores, additional_metrics
    
    def compare_transition_patterns(self, group1_df, group2_df, max_cluster_id, max_positions=None):
        """
        Compare position-specific transition patterns between groups.
        
        Args:
            group1_df: DataFrame with columns 'Model', 'SortedStateIdx', 'ClusterID' for group 1
            group2_df: DataFrame with columns 'Model', 'SortedStateIdx', 'ClusterID' for group 2
            max_cluster_id: Maximum cluster ID
            max_positions: Maximum number of positions to compare (None = all positions)
        
        Returns:
            float: Similarity score between position transition matrices,
            dict: Position-specific transition matrices and similarities
        """
        # Get maximum positions to compare
        max_state_idx1 = int(group1_df['SortedStateIdx'].max()) if not group1_df.empty else 0
        max_state_idx2 = int(group2_df['SortedStateIdx'].max()) if not group2_df.empty else 0
        
        if max_positions is None:
            max_positions = min(max_state_idx1, max_state_idx2)
        else:
            max_positions = min(max_positions, max_state_idx1, max_state_idx2)
        
        if max_positions < 1:  # Need at least two positions for transitions
            return 0, {}
        
        # Create position-specific transition matrices (how often a cluster at position i
        # is followed by a cluster at position i+1)
        def create_position_transition_matrices(group_df, max_pos):
            # Dictionary to store transition matrices for each position
            position_matrices = {}
            
            # Get unique models
            models = group_df['Model'].unique()
            
            # For each position transition (pos → pos+1)
            for pos in range(max_pos):
                # Initialize matrix for this position transition
                matrix = np.zeros((max_cluster_id + 1, max_cluster_id + 1))
                
                for model in models:
                    # Get cluster at position pos
                    from_data = group_df[(group_df['Model'] == model) & 
                                        (group_df['SortedStateIdx'] == pos)]
                    
                    # Get cluster at position pos+1
                    to_data = group_df[(group_df['Model'] == model) & 
                                      (group_df['SortedStateIdx'] == pos+1)]
                    
                    # If both positions have data, record the transition
                    if not from_data.empty and not to_data.empty:
                        from_cluster = from_data.iloc[0]['ClusterID']
                        to_cluster = to_data.iloc[0]['ClusterID']
                        
                        if not pd.isna(from_cluster) and not pd.isna(to_cluster):
                            from_idx = int(from_cluster)
                            to_idx = int(to_cluster)
                            if from_idx < matrix.shape[0] and to_idx < matrix.shape[1]:
                                matrix[from_idx, to_idx] += 1
                
                # Normalize matrix
                row_sums = matrix.sum(axis=1, keepdims=True)
                normalized_matrix = np.divide(matrix, row_sums, 
                                            where=row_sums!=0, 
                                            out=np.zeros_like(matrix))
                
                position_matrices[pos] = {
                    'raw': matrix.copy(),  # Keep raw counts for statistical testing
                    'normalized': normalized_matrix
                }
            
            return position_matrices
        
        # Create position-specific transition matrices for each group
        matrices1 = create_position_transition_matrices(group1_df, max_positions)
        matrices2 = create_position_transition_matrices(group2_df, max_positions)
        
        # Calculate similarities for each position transition
        position_similarities = {}
        position_statistical_tests = {}  # Store statistical test results
        
        for pos in range(max_positions):
            if pos in matrices1 and pos in matrices2:
                mat1 = matrices1[pos]['normalized']
                mat2 = matrices2[pos]['normalized']
                
                # Calculate cosine similarity
                flat1 = mat1.flatten()
                flat2 = mat2.flatten()
                
                dot_product = np.dot(flat1, flat2)
                norm1 = np.linalg.norm(flat1)
                norm2 = np.linalg.norm(flat2)
                
                similarity = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
                position_similarities[pos] = similarity
                
                # Conduct statistical tests for differences in transition probabilities
                raw1 = matrices1[pos]['raw']
                raw2 = matrices2[pos]['raw']
                
                # Find significant transitions
                significant_transitions = []
                transition_p_values = []
                
                # For each potential transition
                for i in range(mat1.shape[0]):
                    for j in range(mat1.shape[1]):
                        # If there are non-zero observations in either group
                        if raw1[i, j] > 0 or raw2[i, j] > 0:
                            # Create contingency table
                            # [from_cluster_i to_cluster_j, from_cluster_i to_other]
                            # [from_other to_cluster_j, from_other to_other]
                            
                            # Group 1
                            a = raw1[i, j]
                            b = raw1[i, :].sum() - a
                            c = raw1[:, j].sum() - a
                            d = raw1.sum() - a - b - c
                            
                            # Group 2
                            e = raw2[i, j]
                            f = raw2[i, :].sum() - e
                            g = raw2[:, j].sum() - e
                            h = raw2.sum() - e - f - g
                            
                            # Fisher's exact test using combined counts
                            table = np.array([[a+e, b+f], [c+g, d+h]])
                            
                            # Perform test if we have enough data
                            if table.min() > 0 and table.sum() > 10:
                                try:
                                    _, p_val = fisher_exact(table)
                                    
                                    # Record significant transitions
                                    if p_val < 0.05:
                                        # Calculate transition probabilities
                                        prob1 = mat1[i, j] if raw1[i, :].sum() > 0 else 0
                                        prob2 = mat2[i, j] if raw2[i, :].sum() > 0 else 0
                                        
                                        significant_transitions.append({
                                            'from_cluster': i,
                                            'to_cluster': j,
                                            'prob_group1': prob1,
                                            'prob_group2': prob2,
                                            'p_value': p_val
                                        })
                                        
                                        transition_p_values.append(p_val)
                                except:
                                    pass
                
                # Apply FDR correction if we have p-values
                if transition_p_values:
                    # Apply FDR correction
                    _, q_values, _, _ = multipletests(
                        transition_p_values, 
                        alpha=0.05, 
                        method='fdr_bh'
                    )
                    
                    # Update significant transitions with q-values
                    for i, trans in enumerate(significant_transitions):
                        trans['q_value'] = q_values[i]
                
                # Store test results
                position_statistical_tests[pos] = {
                    'significant_transitions': significant_transitions
                }
            else:
                position_similarities[pos] = 0
                position_statistical_tests[pos] = {
                    'significant_transitions': []
                }
        
        # Calculate overall similarity as weighted average
        weights = np.linspace(1.0, 0.5, len(position_similarities))
        weighted_sims = [position_similarities[pos] * weights[pos] for pos in range(len(weights))]
        overall_similarity = sum(weighted_sims) / sum(weights) if sum(weights) > 0 else 0
        
        # Return overall similarity and position-specific data
        return overall_similarity, {
            'position_matrices1': {pos: data['normalized'] for pos, data in matrices1.items()},
            'position_matrices2': {pos: data['normalized'] for pos, data in matrices2.items()},
            'position_similarities': position_similarities,
            'statistical_tests': position_statistical_tests
        }
    
    def calculate_advanced_similarities(self, max_positions=8):
        """
        Calculate advanced similarity metrics between groups.
        
        Args:
            max_positions: Maximum number of state positions to compare
        """
        if not hasattr(self, 'state_cluster_matrices'):
            self.create_state_cluster_matrices()
        
        # Get max cluster ID across all groups
        max_cluster_id = 0
        for group_data in self.group_data.values():
            if group_data['clusters']:
                max_cluster_id = max(max_cluster_id, max(group_data['clusters']))
        
        # Calculate different similarity measures
        self.position_weighted_similarity = {}
        self.position_scores = {}
        self.position_advanced_metrics = {}  # New: store advanced metrics
        self.transition_similarity = {}
        self.transition_data = {}
        
        # Create group pairs to compare
        group_pairs = []
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i <= j:  # Include self-comparisons for consistency
                    group_pairs.append((group1, group2))
        
        # Calculate all similarities
        for group1, group2 in group_pairs:
            # Get state-cluster matrices
            matrix1 = self.state_cluster_matrices[group1]
            matrix2 = self.state_cluster_matrices[group2]
            
            # Skip if either matrix is empty
            if matrix1.empty or matrix2.empty:
                pair_key = f"{group1}_{group2}"
                self.position_weighted_similarity[pair_key] = 0.0
                self.position_scores[pair_key] = []
                self.position_advanced_metrics[pair_key] = {}
                self.transition_similarity[pair_key] = 0.0
                self.transition_data[pair_key] = {}
                continue
            
            # Calculate position-weighted similarity
            pos_sim, pos_scores, advanced_metrics = self.calculate_position_weighted_similarity(
                matrix1, matrix2, max_positions=max_positions
            )
            
            # Calculate transition pattern similarity
            trans_sim, trans_data = self.compare_transition_patterns(
                matrix1, matrix2, max_cluster_id, max_positions=max_positions
            )
            
            # Store results
            pair_key = f"{group1}_{group2}"
            self.position_weighted_similarity[pair_key] = pos_sim
            self.position_scores[pair_key] = pos_scores
            self.position_advanced_metrics[pair_key] = advanced_metrics
            self.transition_similarity[pair_key] = trans_sim
            self.transition_data[pair_key] = trans_data
            
            if group1 != group2:  # Don't log self-comparisons
                # Create summary of statistical significance
                significant_positions = sum(1 for p in advanced_metrics['chi2_qvals'] if p < 0.05)
                self.logger.info(
                    f"Groups {group1} vs {group2}: Position-weighted similarity = {pos_sim:.3f}, "
                    f"Transition similarity = {trans_sim:.3f}, "
                    f"Significant position differences: {significant_positions}/{len(advanced_metrics['chi2_qvals'])}"
                )
    
    def cluster_frequency_analysis(self):
        """Analyze and compare cluster frequency distributions between groups."""
        if not hasattr(self, 'state_frequency_matrices'):
            self.create_state_cluster_matrices()
            
        # Calculate frequency differences and statistical significance
        group_comparisons = {}
        
        # Compare each pair of groups
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i >= j:  # Skip self-comparisons and redundant pairs
                    continue
                
                # Get frequency matrices
                freq1 = self.state_frequency_matrices.get(group1)
                freq2 = self.state_frequency_matrices.get(group2)
                
                if freq1 is None or freq2 is None:
                    continue
                
                # Calculate differences
                min_rows = min(freq1.shape[0], freq2.shape[0])
                min_cols = min(freq1.shape[1], freq2.shape[1])
                
                # Extract comparable portions
                freq1_trimmed = freq1[:min_rows, :min_cols]
                freq2_trimmed = freq2[:min_rows, :min_cols]
                
                # Calculate absolute differences
                diff_matrix = freq1_trimmed - freq2_trimmed
                
                # Perform statistical tests for each position
                position_results = []
                
                for pos in range(min_rows):
                    # Get distributions at this position
                    dist1 = freq1_trimmed[pos, :]
                    dist2 = freq2_trimmed[pos, :]
                    
                    # Calculate Jensen-Shannon distance
                    if np.sum(dist1) > 0 and np.sum(dist2) > 0:
                        # Normalize if needed
                        p1 = dist1 / np.sum(dist1)
                        p2 = dist2 / np.sum(dist2)
                        
                        # Calculate the mixture distribution
                        m = 0.5 * (p1 + p2)
                        
                        # Calculate JS divergence without using the where parameter
                        js_divergence = 0.0
                        
                        # First component: KL(p1||m) - only for non-zero p1
                        mask1 = p1 > 0
                        if np.any(mask1):
                            js_divergence += 0.5 * entropy(p1[mask1], m[mask1])
                        
                        # Second component: KL(p2||m) - only for non-zero p2
                        mask2 = p2 > 0
                        if np.any(mask2):
                            js_divergence += 0.5 * entropy(p2[mask2], m[mask2])
                        
                        # Convert to distance (0-1 range)
                        js_dist = np.sqrt(js_divergence) if js_divergence >= 0 else 0
                    else:
                        js_dist = 1.0  # Maximum difference if one distribution is empty
                    
                    # Find top differences
                    top_diff_indices = np.argsort(np.abs(diff_matrix[pos, :]))[-5:]  # Top 5 differences
                    top_diffs = [(idx, diff_matrix[pos, idx]) for idx in top_diff_indices if abs(diff_matrix[pos, idx]) > 0.05]
                    
                    # Perform chi-square test
                    contingency = np.zeros((2, min_cols))
                    for c in range(min_cols):
                        contingency[0, c] = freq1[pos, c] * self.model_info[group1].shape[0]  # Scale back to counts
                        contingency[1, c] = freq2[pos, c] * self.model_info[group2].shape[0]
                    
                    # Only run test if we have sufficient data
                    if contingency.sum() > 0 and all(contingency.sum(axis=1) > 0):
                        valid_columns = contingency.sum(axis=0) > 0
                        if sum(valid_columns) > 1:  # Need at least 2 columns for chi-square
                            try:
                                chi2, p_val, _, _ = chi2_contingency(contingency[:, valid_columns])
                                significant = p_val < 0.05
                            except:
                                chi2, p_val, significant = 0, 1.0, False
                        else:
                            chi2, p_val, significant = 0, 1.0, False
                    else:
                        chi2, p_val, significant = 0, 1.0, False
                    
                    position_results.append({
                        'position': pos,
                        'js_distance': js_dist,
                        'chi2': chi2,
                        'p_value': p_val,
                        'significant': significant,
                        'top_differences': top_diffs
                    })
                
                # Apply FDR correction to p-values
                if position_results:
                    p_values = [r['p_value'] for r in position_results]
                    _, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
                    
                    # Update with q-values
                    for i, r in enumerate(position_results):
                        r['q_value'] = q_values[i]
                        r['significant_fdr'] = q_values[i] < 0.05
                
                group_comparisons[f"{group1}_{group2}"] = {
                    'diff_matrix': diff_matrix,
                    'position_results': position_results
                }
        
        self.cluster_frequency_comparisons = group_comparisons
        
        # Log summary results
        for pair, results in group_comparisons.items():
            group1, group2 = pair.split('_')
            pos_results = results['position_results']
            sig_positions = sum(1 for r in pos_results if r.get('significant_fdr', False))
            
            self.logger.info(
                f"Cluster distribution comparison {group1} vs {group2}: "
                f"{sig_positions}/{len(pos_results)} positions with significant differences"
            )
    
    def plot_position_similarities(self):
        """Visualize position-specific similarities between groups."""
        if not hasattr(self, 'position_scores'):
            self.calculate_advanced_similarities()
        
        # Create a plot showing position-specific similarities for each group pair
        plt.figure(figsize=(12, 8))
        
        # Plot position similarities for each group pair
        for pair, scores in self.position_scores.items():
            if '_' in pair and not pair.startswith('_') and not pair.endswith('_'):
                group1, group2 = pair.split('_')
                if group1 != group2:  # Skip self-comparisons
                    # Get statistical significance markers
                    if pair in self.position_advanced_metrics:
                        q_vals = self.position_advanced_metrics[pair].get('chi2_qvals', [])
                        
                        # Plot with statistical markers
                        x_vals = range(len(scores))
                        plt.plot(x_vals, scores, 
                               marker='o', linewidth=2, markersize=8, 
                               label=f"{group1} vs {group2}")
                        
                        # Mark statistically significant differences
                        for i, (score, q) in enumerate(zip(scores, q_vals)):
                            if q < 0.05:
                                plt.plot(i, score, 'r*', markersize=12)
                    else:
                        plt.plot(range(len(scores)), scores, 
                               marker='o', linewidth=2, markersize=8, 
                               label=f"{group1} vs {group2}")
        
        plt.title('Position-Specific Similarity Scores', fontsize=14)
        plt.xlabel('Position Index', fontsize=12)
        plt.ylabel('Similarity Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add annotation explaining the significance markers
        plt.annotate('* indicates statistically significant differences (q < 0.05)', 
                    xy=(0.5, -0.1), xycoords='axes fraction', 
                    ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'position_similarities.png', dpi=300)
        plt.close()
        
        # Now create more detailed plots with additional metrics
        if hasattr(self, 'position_advanced_metrics'):
            # Plot ARI and AMI scores
            plt.figure(figsize=(12, 8))
            
            for pair, metrics in self.position_advanced_metrics.items():
                if '_' in pair and not pair.startswith('_') and not pair.endswith('_'):
                    group1, group2 = pair.split('_')
                    if group1 != group2:  # Skip self-comparisons
                        x_vals = range(len(metrics.get('ari_scores', [])))
                        
                        # Plot ARI scores
                        plt.plot(x_vals, metrics.get('ari_scores', []), 
                               marker='o', linewidth=2, markersize=6, 
                               label=f"{group1} vs {group2} (ARI)")
                        
                        # Plot AMI scores
                        plt.plot(x_vals, metrics.get('ami_scores', []), 
                               marker='s', linewidth=2, linestyle='--', markersize=6, 
                               label=f"{group1} vs {group2} (AMI)")
            
            plt.title('Position-Specific Adjusted Rand Index and Adjusted Mutual Information', fontsize=14)
            plt.xlabel('Position Index', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'position_ari_ami_scores.png', dpi=300)
            plt.close()
            
            # Plot JS similarity scores
            plt.figure(figsize=(12, 8))
            
            for pair, metrics in self.position_advanced_metrics.items():
                if '_' in pair and not pair.startswith('_') and not pair.endswith('_'):
                    group1, group2 = pair.split('_')
                    if group1 != group2:  # Skip self-comparisons
                        js_sims = metrics.get('js_similarities', [])
                        if js_sims:
                            x_vals = range(len(js_sims))
                            plt.plot(x_vals, js_sims, 
                                   marker='o', linewidth=2, markersize=8, 
                                   label=f"{group1} vs {group2}")
            
            plt.title('Position-Specific Jensen-Shannon Similarity', fontsize=14)
            plt.xlabel('Position Index', fontsize=12)
            plt.ylabel('Similarity Score (1 - JS Distance)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'position_js_similarities.png', dpi=300)
            plt.close()
    
    def plot_cluster_frequency_comparisons(self):
        """Visualize cluster frequency comparisons between groups."""
        if not hasattr(self, 'cluster_frequency_comparisons'):
            self.cluster_frequency_analysis()
        
        # For each group pair, create visualizations
        for pair, data in self.cluster_frequency_comparisons.items():
            group1, group2 = pair.split('_')
            diff_matrix = data['diff_matrix']
            position_results = data['position_results']
            
            # How many positions to show
            max_positions = min(8, diff_matrix.shape[0])
            
            # Create heatmap of differences
            plt.figure(figsize=(max(10, diff_matrix.shape[1] // 3 + 5), max(8, max_positions + 2)))
            
            # Create custom colormap centered at zero
            max_abs_val = max(abs(np.max(diff_matrix[:max_positions, :])), 
                             abs(np.min(diff_matrix[:max_positions, :])))
            norm = plt.Normalize(-max_abs_val, max_abs_val)
            
            # Draw heatmap
            ax = plt.gca()
            im = ax.imshow(diff_matrix[:max_positions, :], cmap='RdBu_r', norm=norm)
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label(f'Frequency Difference ({group1} - {group2})')
            
            # Add position labels
            plt.yticks(range(max_positions), [f"Pos {i}" for i in range(max_positions)])
            
            # Add cluster labels
            plt.xticks(range(diff_matrix.shape[1]), [f"C{i}" for i in range(diff_matrix.shape[1])], 
                     rotation=90, fontsize=8)
            
            # Add significance markers
            for pos in range(max_positions):
                if pos < len(position_results) and position_results[pos].get('significant_fdr', False):
                    plt.text(-1.5, pos, "*", fontsize=20, ha='center', va='center', color='red')
            
            plt.title(f'Cluster Frequency Differences: {group1} vs {group2}', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'cluster_freq_diff_{group1}_{group2}.png', dpi=300)
            plt.close()
            
            # Create detailed position-specific bar charts for significant positions
            significant_positions = [p for p in position_results if p.get('significant_fdr', False)]
            
            # Only create detail plots if we have significant positions
            if significant_positions:
                # How many positions to show (at most 4)
                positions_to_show = min(4, len(significant_positions))
                
                plt.figure(figsize=(15, 4 * positions_to_show))
                
                for i, pos_result in enumerate(significant_positions[:positions_to_show]):
                    pos = pos_result['position']
                    
                    # Get cluster frequencies at this position
                    if pos < diff_matrix.shape[0]:
                        # Find non-zero indices
                        non_zero = np.where(abs(diff_matrix[pos, :]) > 0.05)[0]
                        
                        if len(non_zero) > 0:
                            ax = plt.subplot(positions_to_show, 1, i+1)
                            
                            # Get frequencies for each group
                            g1_freqs = self.state_frequency_matrices[group1][pos, non_zero]
                            g2_freqs = self.state_frequency_matrices[group2][pos, non_zero]
                            
                            # Set up bar positions
                            x = np.arange(len(non_zero))
                            width = 0.35
                            
                            # Create bars
                            ax.bar(x - width/2, g1_freqs, width, label=group1, color=COLORS.get(group1, '#333333'))
                            ax.bar(x + width/2, g2_freqs, width, label=group2, color=COLORS.get(group2, '#333333'))
                            
                            # Add labels and titles
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Position {pos} (p={pos_result["p_value"]:.3f}, q={pos_result.get("q_value", 1.0):.3f})')
                            ax.set_xticks(x)
                            ax.set_xticklabels([f'C{idx}' for idx in non_zero])
                            ax.legend()
                            
                            # Add specific difference values
                            for j, idx in enumerate(non_zero):
                                diff = diff_matrix[pos, idx]
                                if abs(diff) > 0.1:  # Only show notable differences
                                    ax.annotate(f'{diff:.2f}', 
                                              xy=(j, max(g1_freqs[j], g2_freqs[j]) + 0.05),
                                              ha='center', va='center', 
                                              color='black' if diff > 0 else 'red', 
                                              fontweight='bold')
                
                plt.suptitle(f'Significant Cluster Frequency Differences: {group1} vs {group2}', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the overall title
                plt.savefig(self.output_dir / f'cluster_freq_detail_{group1}_{group2}.png', dpi=300)
                plt.close()
    
    def plot_transition_matrices(self):
        """Visualize position-specific transition matrices for each group."""
        if not hasattr(self, 'transition_data'):
            self.calculate_advanced_similarities()
        
        # For each group, create a visualization of meaningful transition patterns
        for group in self.groups:
            # Get self-comparison data
            pair_key = f"{group}_{group}"
            if pair_key not in self.transition_data:
                continue
            
            matrices = self.transition_data[pair_key].get('position_matrices1', {})
            if not matrices:
                continue
            
            # Determine positions to visualize
            positions = sorted(matrices.keys())
            if not positions:
                continue
            
            # Initialize figure for this group's transition patterns
            max_pos_to_show = min(4, len(positions))  # Show at most 4 positions
            
            # Create a summary version showing top transitions for each position
            plt.figure(figsize=(12, 3.5))
            
            # Create subplots in a horizontal layout
            for pos_idx, pos in enumerate(positions[:max_pos_to_show]):
                matrix = matrices[pos]
                
                # Only keep the top transitions
                top_n = 5  # Show top 5 transitions
                flat_idx = np.argsort(matrix.flatten())[-top_n:]
                row_idx, col_idx = np.unravel_index(flat_idx, matrix.shape)
                
                # If there are no significant transitions, continue
                if len(row_idx) == 0 or np.max(matrix) < 0.01:
                    continue
                
                # Create subplot
                ax = plt.subplot(1, max_pos_to_show, pos_idx + 1)
                
                # Create a clean table-like visualization
                cell_text = []
                for i, j in zip(row_idx, col_idx):
                    if matrix[i, j] > 0.01:  # Only show probabilities > 1%
                        cell_text.append([f"C{i} → C{j}", f"{matrix[i, j]:.2f}"])
                
                if not cell_text:
                    ax.text(0.5, 0.5, "No significant transitions", 
                            ha='center', va='center', transform=ax.transAxes)
                else:
                    # Create table
                    table = ax.table(
                        cellText=cell_text,
                        colLabels=['Transition', 'Probability'],
                        loc='center',
                        cellLoc='center'
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 1.5)
                
                ax.set_title(f'Position {pos}→{pos+1}')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{group}_transition_summary.png', dpi=300)
            plt.close()
            
            # Now create a visualization of the transition network
            plt.figure(figsize=(10, 8))
            
            # Combine transitions across positions for an overall view
            G = nx.DiGraph()
            
            # Add transitions from all positions as edges
            all_edges = []
            for pos in positions:
                matrix = matrices[pos]
                
                # Get significant transitions (probability > 0.1)
                significant_mask = matrix > 0.1
                if not np.any(significant_mask):
                    continue
                    
                rows, cols = np.where(significant_mask)
                for i, j in zip(rows, cols):
                    if matrix[i, j] > 0.1:  # Double-check
                        # Add the edge with position info and weight
                        all_edges.append((
                            f"C{i}", f"C{j}", 
                            {"weight": matrix[i, j], "pos": pos}
                        ))
            
            # If no significant transitions, skip network visualization
            if not all_edges:
                plt.close()
                continue
                
            # Add the edges to the graph
            G.add_edges_from(all_edges)
            
            # Get positions using spring layout with weight attribute
            pos = nx.spring_layout(G, seed=self.random_seed)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700, 
                                node_color='skyblue', alpha=0.8)
            
            # Define edge colors based on position
            position_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            # Draw edges for each position with different colors
            for pos_idx in positions:
                # Get edges for this position
                pos_edges = [(u, v) for u, v, d in all_edges if d['pos'] == pos_idx]
                if not pos_edges:
                    continue
                    
                # Get weights for these edges
                edge_weights = [G[u][v]['weight'] * 3 for u, v in pos_edges]
                
                # Draw edges for this position
                nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=edge_weights,
                                    edge_color=position_colors[pos_idx % len(position_colors)],
                                    alpha=0.7, arrowsize=15)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # Create legend for positions
            legend_elements = [
                plt.Line2D([0], [0], color=position_colors[p % len(position_colors)], lw=2,
                        label=f'Position {p}→{p+1}')
                for p in positions if [(u, v) for u, v, d in all_edges if d['pos'] == p]
            ]
            plt.legend(handles=legend_elements)
            
            plt.title(f'Transition Network: {group}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{group}_transition_network.png', dpi=300)
            plt.close()
        
        # Create group comparison visualizations for significant transition differences
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i >= j:  # Skip self-comparisons and redundant pairs
                    continue
                    
                pair_key = f"{group1}_{group2}"
                if pair_key not in self.transition_data:
                    continue
                    
                # Get data including statistical tests
                data = self.transition_data[pair_key]
                statistical_tests = data.get('statistical_tests', {})
                
                # Find positions with significant transitions
                significant_positions = []
                for pos, pos_data in statistical_tests.items():
                    sig_trans = pos_data.get('significant_transitions', [])
                    if sig_trans:
                        # Keep only transitions that are still significant after FDR correction
                        sig_trans = [t for t in sig_trans if t.get('q_value', 1.0) < 0.05]
                        if sig_trans:
                            significant_positions.append((pos, sig_trans))
                
                # Skip if no significant differences
                if not significant_positions:
                    continue
                
                # Only show the first few significant positions
                max_pos_to_show = min(4, len(significant_positions))
                
                # Create figure
                plt.figure(figsize=(max(12, max_pos_to_show * 4), 4))
                
                # Plot each significant position
                for idx, (pos, sig_trans) in enumerate(significant_positions[:max_pos_to_show]):
                    ax = plt.subplot(1, max_pos_to_show, idx + 1)
                    
                    # Create a table of the significant transitions
                    cell_text = []
                    cell_colors = []
                    
                    for trans in sorted(sig_trans, key=lambda t: t['q_value']):
                        from_c = trans['from_cluster']
                        to_c = trans['to_cluster']
                        p1 = trans['prob_group1']
                        p2 = trans['prob_group2']
                        q = trans['q_value']
                        
                        # Add row to the table
                        cell_text.append([
                            f"C{from_c}→C{to_c}", 
                            f"{p1:.2f}", 
                            f"{p2:.2f}", 
                            f"{q:.3f}"
                        ])
                        
                        # Color based on which group has higher probability
                        if p1 > p2:
                            row_color = [COLORS.get(f"{group1}_light", '#f0f0f0')] * 4
                        else:
                            row_color = [COLORS.get(f"{group2}_light", '#f0f0f0')] * 4
                        
                        cell_colors.append(row_color)
                    
                    if cell_text:
                        # Create table
                        table = ax.table(
                            cellText=cell_text,
                            colLabels=['Transition', group1, group2, 'q-val'],
                            loc='center',
                            cellLoc='center',
                            cellColours=cell_colors
                        )
                        table.auto_set_font_size(False)
                        table.set_fontsize(9)
                        table.scale(1, 1.5)
                    else:
                        ax.text(0.5, 0.5, "No significant differences", 
                                ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Position {pos}→{pos+1}')
                    ax.axis('off')
                
                plt.suptitle(f'Significant Transition Differences: {group1} vs {group2}', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
                plt.savefig(self.output_dir / f'sig_transition_diff_{group1}_{group2}.png', dpi=300)
                plt.close()
    
    def plot_similarity_matrices(self):
        """Visualize similarity matrices using different metrics."""
        if not hasattr(self, 'position_weighted_similarity'):
            self.calculate_advanced_similarities()
        
        # Create similarity matrices
        n_groups = len(self.groups)
        pos_similarity_matrix = np.zeros((n_groups, n_groups))
        trans_similarity_matrix = np.zeros((n_groups, n_groups))
        
        # Also create statistical significance matrices
        significant_positions_matrix = np.zeros((n_groups, n_groups))
        
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                # Get the right key
                pair_key = f"{group1}_{group2}"
                if pair_key in self.position_weighted_similarity:
                    pos_similarity_matrix[i, j] = self.position_weighted_similarity[pair_key]
                    trans_similarity_matrix[i, j] = self.transition_similarity[pair_key]
                    
                    # Count significant positions if we have advanced metrics
                    if pair_key in self.position_advanced_metrics:
                        metrics = self.position_advanced_metrics[pair_key]
                        if 'chi2_qvals' in metrics:
                            significant_positions_matrix[i, j] = sum(1 for q in metrics['chi2_qvals'] if q < 0.05)
                else:
                    pair_key = f"{group2}_{group1}"
                    if pair_key in self.position_weighted_similarity:
                        pos_similarity_matrix[i, j] = self.position_weighted_similarity[pair_key]
                        trans_similarity_matrix[i, j] = self.transition_similarity[pair_key]
                        
                        # Count significant positions
                        if pair_key in self.position_advanced_metrics:
                            metrics = self.position_advanced_metrics[pair_key]
                            if 'chi2_qvals' in metrics:
                                significant_positions_matrix[i, j] = sum(1 for q in metrics['chi2_qvals'] if q < 0.05)
        
        # Create combined heatmap with both metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Position-weighted similarity
        sns.heatmap(pos_similarity_matrix, cmap='YlGnBu', annot=True, fmt='.3f',
                   xticklabels=self.groups,
                   yticklabels=self.groups,
                   linewidths=0.5, cbar_kws={'label': 'Similarity'}, ax=ax1)
        
        ax1.set_title('Position-Weighted Similarity', fontsize=14)
        
        # Transition pattern similarity
        sns.heatmap(trans_similarity_matrix, cmap='YlGnBu', annot=True, fmt='.3f',
                   xticklabels=self.groups,
                   yticklabels=self.groups,
                   linewidths=0.5, cbar_kws={'label': 'Similarity'}, ax=ax2)
        
        ax2.set_title('Transition Pattern Similarity', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_similarity_matrices.png', dpi=300)
        plt.close()
        
        # Create a matrix showing number of significantly different positions
        plt.figure(figsize=(10, 8))
        
        # Mask the diagonal for better visualization (setting to NaN for diagonal)
        mask = np.eye(n_groups, dtype=bool)
        masked_matrix = significant_positions_matrix.copy()
        masked_matrix[mask] = np.nan
        
        sns.heatmap(masked_matrix, cmap='Reds', annot=True, fmt='.0f',
                   xticklabels=self.groups,
                   yticklabels=self.groups,
                   linewidths=0.5, 
                   cbar_kws={'label': 'Number of Significantly Different Positions (q < 0.05)'})
        
        plt.title('Statistical Significance of Positional Differences', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_significant_differences.png', dpi=300)
        plt.close()
    
    def plot_mds_projection(self):
        """Visualize groups using MDS based on both similarity metrics."""
        if not hasattr(self, 'transition_similarity') or not hasattr(self, 'position_weighted_similarity'):
            self.calculate_advanced_similarities()
        
        # Create distance matrices (1 - similarity) for both metrics
        n_groups = len(self.groups)
        distance_matrix_trans = np.zeros((n_groups, n_groups))
        distance_matrix_pos = np.zeros((n_groups, n_groups))
        
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i != j:
                    # Get pair key
                    pair_key = f"{group1}_{group2}"
                    rev_pair_key = f"{group2}_{group1}"
                    
                    # Get transition similarity
                    if pair_key in self.transition_similarity:
                        trans_sim = self.transition_similarity[pair_key]
                    elif rev_pair_key in self.transition_similarity:
                        trans_sim = self.transition_similarity[rev_pair_key]
                    else:
                        trans_sim = 0
                    
                    # Get position-weighted similarity
                    if pair_key in self.position_weighted_similarity:
                        pos_sim = self.position_weighted_similarity[pair_key]
                    elif rev_pair_key in self.position_weighted_similarity:
                        pos_sim = self.position_weighted_similarity[rev_pair_key]
                    else:
                        pos_sim = 0
                    
                    # Convert to distances
                    distance_matrix_trans[i, j] = 1 - trans_sim
                    distance_matrix_pos[i, j] = 1 - pos_sim
        
        # Create single combined visualization with both metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Perform MDS for transition similarity
        mds_trans = MDS(n_components=2, dissimilarity='precomputed', random_state=self.random_seed)
        points_trans = mds_trans.fit_transform(distance_matrix_trans)
        
        # Plot transition-based MDS
        for i, group in enumerate(self.groups):
            ax1.scatter(points_trans[i, 0], points_trans[i, 1], 
                      c=[COLORS.get(group, '#333333')], 
                      label=group, s=100)
            
            ax1.annotate(group, (points_trans[i, 0], points_trans[i, 1]), 
                        fontsize=12, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_title('Transition Pattern Similarity')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Perform MDS for position-weighted similarity
        mds_pos = MDS(n_components=2, dissimilarity='precomputed', random_state=self.random_seed)
        points_pos = mds_pos.fit_transform(distance_matrix_pos)
        
        # Plot position-based MDS
        for i, group in enumerate(self.groups):
            ax2.scatter(points_pos[i, 0], points_pos[i, 1], 
                      c=[COLORS.get(group, '#333333')], 
                      label=group, s=100)
            
            ax2.annotate(group, (points_pos[i, 0], points_pos[i, 1]), 
                        fontsize=12, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_title('Position-Weighted Similarity')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_similarity_mds.png', dpi=300)
        plt.close()
        
        # Also create a t-SNE visualization which can sometimes capture more complex relationships
        plt.figure(figsize=(10, 8))
        
        # Combine distance matrices for a single t-SNE projection
        combined_distance = (distance_matrix_pos + distance_matrix_trans) / 2
        
        # Apply t-SNE (with small perplexity due to few samples)
        tsne = TSNE(n_components=2, metric='precomputed', 
                  perplexity=max(3, n_groups/2), random_state=self.random_seed)
        
        try:
            points_tsne = tsne.fit_transform(combined_distance)
            
            # Plot t-SNE projection
            for i, group in enumerate(self.groups):
                plt.scatter(points_tsne[i, 0], points_tsne[i, 1], 
                          c=[COLORS.get(group, '#333333')], 
                          label=group, s=150)
                
                plt.annotate(group, (points_tsne[i, 0], points_tsne[i, 1]), 
                            fontsize=12, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points')
            
            plt.title('t-SNE Projection of Combined Similarity', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'group_similarity_tsne.png', dpi=300)
            plt.close()
        except:
            # t-SNE can fail with too few points, so handle gracefully
            self.logger.warning("t-SNE projection failed, likely due to insufficient data points")
            plt.close()
    
    def save_analysis_results(self):
        """Save analysis results to files."""
        # Save position-specific data
        if hasattr(self, 'position_scores'):
            position_data = {
                'position_scores': self.position_scores,
                'position_weighted_similarity': self.position_weighted_similarity
            }
            
            with open(self.output_dir / 'position_similarity_data.json', 'w') as f:
                json.dump(position_data, f, indent=2)
        
        # Save advanced metrics
        if hasattr(self, 'position_advanced_metrics'):
            # Convert non-serializable numpy types to Python types
            serializable_metrics = {}
            
            for pair, metrics in self.position_advanced_metrics.items():
                serializable_metrics[pair] = {}
                
                for metric_name, values in metrics.items():
                    if isinstance(values, list) or isinstance(values, np.ndarray):
                        serializable_metrics[pair][metric_name] = [float(v) if not np.isnan(v) else None for v in values]
                    else:
                        serializable_metrics[pair][metric_name] = values
            
            with open(self.output_dir / 'advanced_metrics_data.json', 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        
        # Save transition data
        if hasattr(self, 'transition_similarity'):
            # We can't directly serialize NumPy arrays in JSON, so extract the scores
            transition_data = {
                'transition_similarity': self.transition_similarity,
                'position_similarities': {
                    pair: data.get('position_similarities', {}) 
                    for pair, data in self.transition_data.items()
                } if hasattr(self, 'transition_data') else {}
            }
            
            # Extract statistical test results
            if hasattr(self, 'transition_data'):
                test_results = {}
                
                for pair, data in self.transition_data.items():
                    if 'statistical_tests' in data:
                        # Convert tests to serializable format
                        position_tests = {}
                        
                        for pos, pos_data in data['statistical_tests'].items():
                            # Extract significant transitions
                            sig_trans = []
                            
                            for trans in pos_data.get('significant_transitions', []):
                                # Convert to serializable dict
                                sig_trans.append({
                                    'from_cluster': int(trans['from_cluster']),
                                    'to_cluster': int(trans['to_cluster']),
                                    'prob_group1': float(trans['prob_group1']),
                                    'prob_group2': float(trans['prob_group2']),
                                    'p_value': float(trans['p_value']),
                                    'q_value': float(trans.get('q_value', 1.0))
                                })
                            
                            position_tests[str(pos)] = sig_trans
                        
                        test_results[pair] = position_tests
                
                transition_data['statistical_tests'] = test_results
            
            with open(self.output_dir / 'transition_similarity_data.json', 'w') as f:
                json.dump(transition_data, f, indent=2)
        
        # Save cluster frequency analysis results
        if hasattr(self, 'cluster_frequency_comparisons'):
            cluster_freq_data = {}
            
            for pair, data in self.cluster_frequency_comparisons.items():
                # Convert position results to serializable format
                pos_results = []
                
                for result in data['position_results']:
                    pos_result = {
                        'position': int(result['position']),
                        'js_distance': float(result['js_distance']),
                        'chi2': float(result['chi2']),
                        'p_value': float(result['p_value']),
                        'q_value': float(result.get('q_value', 1.0)),
                        'significant': bool(result['significant']),
                        'significant_fdr': bool(result.get('significant_fdr', False)),
                        'top_differences': [
                            [int(idx), float(diff)] for idx, diff in result['top_differences']
                        ]
                    }
                    pos_results.append(pos_result)
                
                cluster_freq_data[pair] = pos_results
            
            with open(self.output_dir / 'cluster_frequency_data.json', 'w') as f:
                json.dump(cluster_freq_data, f, indent=2)
        
        # Save state-cluster matrices (only once, no need to save for each run)
        if hasattr(self, 'state_cluster_matrices') and not (self.output_dir / f'{self.groups[0]}_state_cluster_matrix.csv').exists():
            for group, matrix_df in self.state_cluster_matrices.items():
                matrix_df.to_csv(self.output_dir / f'{group}_state_cluster_matrix.csv', index=False)
        
        # Save model info (only once)
        if hasattr(self, 'model_info') and not (self.output_dir / f'{self.groups[0]}_model_info.csv').exists():
            for group, info_df in self.model_info.items():
                info_df.to_csv(self.output_dir / f'{group}_model_info.csv', index=False)
        
        # Save compact analysis summary
        summary_data = {
            'groups': self.groups,
            'group_data_summary': {
                group: {
                    'pattern_count': len(data['df']),
                    'cluster_count': len(data['clusters']),
                    'model_count': len(data['df']['Model'].unique())
                } for group, data in self.group_data.items()
            },
            'similarity_summary': {
                'position_weighted': {
                    pair: score for pair, score in self.position_weighted_similarity.items()
                    if '_' in pair and pair.split('_')[0] != pair.split('_')[1]  # Skip self-comparisons
                } if hasattr(self, 'position_weighted_similarity') else {},
                'transition': {
                    pair: score for pair, score in self.transition_similarity.items()
                    if '_' in pair and pair.split('_')[0] != pair.split('_')[1]  # Skip self-comparisons
                } if hasattr(self, 'transition_similarity') else {}
            }
        }
        
        # Add statistical significance summary
        if hasattr(self, 'position_advanced_metrics'):
            sig_summary = {}
            
            for pair, metrics in self.position_advanced_metrics.items():
                if '_' in pair and pair.split('_')[0] != pair.split('_')[1]:  # Skip self-comparisons
                    if 'chi2_qvals' in metrics:
                        sig_count = sum(1 for q in metrics['chi2_qvals'] if q < 0.05)
                        sig_summary[pair] = {
                            'significant_positions': sig_count,
                            'total_positions': len(metrics['chi2_qvals']),
                            'significant_percentage': round(sig_count * 100 / len(metrics['chi2_qvals']), 1) if len(metrics['chi2_qvals']) > 0 else 0
                        }
            
            summary_data['statistical_significance'] = sig_summary
        
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def run_analysis(self, max_positions=8):
        """
        Run the complete group comparison analysis.
        
        Args:
            max_positions: Maximum number of state positions to compare
        """
        self.logger.info(f"Starting advanced group comparison analysis (max_positions={max_positions})...")
        
        # Step 1: Create state-cluster matrices
        self.logger.info("Creating state-cluster matrices...")
        self.create_state_cluster_matrices()
        
        # Step 2: Calculate advanced similarities
        self.logger.info("Calculating advanced similarity metrics...")
        self.calculate_advanced_similarities(max_positions=max_positions)
        
        # Step 3: Perform cluster frequency analysis
        self.logger.info("Analyzing cluster frequency distributions...")
        self.cluster_frequency_analysis()
        
        # Step 4: Visualize results
        self.logger.info("Creating visualizations...")
        try:
            self.plot_position_similarities()
            self.logger.info("Position similarities visualized")
        except Exception as e:
            self.logger.error(f"Error visualizing position similarities: {e}")
            
        try:
            self.plot_cluster_frequency_comparisons()
            self.logger.info("Cluster frequency comparisons visualized")
        except Exception as e:
            self.logger.error(f"Error visualizing cluster frequency comparisons: {e}")
            
        try:
            self.plot_transition_matrices()
            self.logger.info("Transition matrices visualized")
        except Exception as e:
            self.logger.error(f"Error visualizing transition matrices: {e}")
            
        try:
            self.plot_similarity_matrices()
            self.logger.info("Similarity matrices visualized")
        except Exception as e:
            self.logger.error(f"Error visualizing similarity matrices: {e}")
            
        try:
            self.plot_mds_projection()
            self.logger.info("MDS projection visualized")
        except Exception as e:
            self.logger.error(f"Error creating MDS projection: {e}")
        
        # Step 5: Save results
        self.logger.info("Saving results...")
        self.save_analysis_results()
        
        self.logger.info("Advanced group comparison analysis complete.")
        
        return {
            'position_weighted_similarity': self.position_weighted_similarity,
            'transition_similarity': self.transition_similarity,
            'position_scores': self.position_scores,
            'significant_positions': self.position_advanced_metrics if hasattr(self, 'position_advanced_metrics') else {}
        }


if __name__ == "__main__":
    import sys
    import argparse
    
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Compare group models with cluster assignments')
    parser.add_argument('--max-positions', type=int, default=8, 
                        help='Maximum number of state positions to compare')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--detailed', action='store_true',
                        help='Enable detailed analysis (more computationally intensive)')
    args = parser.parse_args()

    # Setup paths
    try:
        from dotenv import load_dotenv
        load_dotenv()
        scratch_dir = os.getenv("SCRATCH_DIR")
    except:
        scratch_dir = None
        
    base_dir = os.path.join(scratch_dir, "output") if scratch_dir else "output"
    output_dir = os.path.join(base_dir, "07_group_cluster_comp")
    groups = ["affair", "paranoia", "combined", "balanced"]
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # Setup logging
    # Before creating the FileHandler, make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist

    # Now create the file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "analysis.log"))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting group comparison analysis for thresholds: {thresholds}")
    logger.info(f"Using max_positions: {args.max_positions}, random seed: {args.seed}")
    
    for th in thresholds:
        th_str = f"th_{th:.2f}".replace('.', '')
        logger.info(f"\nProcessing threshold {th}...")
        
        cluster_info_path = os.path.join(base_dir, f"06_state_pattern_cluster/{th_str}/cluster_info.json")
        state_mappings_path = os.path.join(base_dir, f"06_state_pattern_cluster/{th_str}/state_mappings.json")
        th_dir = os.path.join(output_dir, f"{th_str}")
        
        # Check if files exist
        if not os.path.exists(cluster_info_path):
            logger.warning(f"Warning: Cluster info file not found: {cluster_info_path}")
            continue
            
        if not os.path.exists(state_mappings_path):
            logger.warning(f"Warning: State mappings file not found: {state_mappings_path}")
            continue
        
        # Create analyzer and run
        try:
            analyzer = GroupModelComparison(
                cluster_info_path=cluster_info_path,
                state_mappings_path=state_mappings_path,
                output_dir=th_dir,
                random_seed=args.seed
            )
            
            analyzer.run_analysis(max_positions=args.max_positions)
            logger.info(f"Analysis for threshold {th} completed successfully")
        except Exception as e:
            logger.error(f"Error processing threshold {th}: {e}", exc_info=True)
    
    logger.info("\nAll analyses completed")