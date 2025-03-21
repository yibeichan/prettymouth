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
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score
from natsort import natsorted
import pickle

# Import color scheme from the main script
COLORS = {
    'affair': '#e41a1c',      # Red
    'affair_light': '#ff6666', # Light red
    'paranoia': '#4daf4a',    # Green
    'paranoia_light': '#90ee90', # Light green
    'combined': '#984ea3',    # Purple
    'combined_light': '#d8b2d8' # Light purple
}

class GroupModelComparison:
    def __init__(self, cluster_info_path, state_mappings_path, output_dir):
        """
        Initialize the Group Model Comparison tool.
        
        Args:
            cluster_info_path: Path to the cluster_info.json file
            state_mappings_path: Path to the state_mappings.json file
            output_dir: Directory to save results
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Set output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            # Convert to DataFrame
            self.state_cluster_matrices[group] = pd.DataFrame(matrix_data)
    
    def compare_group_cluster_distributions(self):
        """Compare cluster distributions between groups."""
        # Get cluster counts for each group
        cluster_counts = {}
        
        for group, data in self.group_data.items():
            # Count occurrences of each cluster
            counts = Counter(data['df']['ClusterID'])
            
            # Create a vector with counts for all clusters
            all_clusters = set()
            for g, d in self.group_data.items():
                all_clusters.update(d['clusters'])
            
            # Create vector with counts
            count_vector = np.zeros(len(self.cluster_info))
            for cluster_id, count in counts.items():
                # Cluster IDs start from 1 in the JSON, but we need 0-indexed
                idx = int(cluster_id) - 1
                if 0 <= idx < len(count_vector):
                    count_vector[idx] = count
                
            cluster_counts[group] = count_vector
        
        # Calculate pairwise similarity between groups
        self.group_similarity = {}
        self.group_overlap_clusters = {}
        
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i < j:  # Avoid redundant comparisons
                    # Get count vectors
                    vec1 = cluster_counts[group1]
                    vec2 = cluster_counts[group2]
                    
                    # Calculate Jaccard similarity
                    binary_vec1 = (vec1 > 0).astype(int)
                    binary_vec2 = (vec2 > 0).astype(int)
                    
                    # Get shared clusters
                    shared_clusters = []
                    
                    for cluster_id in range(1, len(binary_vec1) + 1):
                        idx = cluster_id - 1  # 0-indexed
                        if binary_vec1[idx] > 0 and binary_vec2[idx] > 0:
                            shared_clusters.append(cluster_id)
                    
                    # Calculate Jaccard similarity
                    if np.any(binary_vec1) and np.any(binary_vec2):
                        jaccard = jaccard_score(binary_vec1, binary_vec2)
                    else:
                        jaccard = 0.0
                    
                    pair_key = f"{group1}_{group2}"
                    self.group_similarity[pair_key] = jaccard
                    self.group_overlap_clusters[pair_key] = shared_clusters
                    
                    self.logger.info(f"Groups {group1} vs {group2}: Jaccard={jaccard:.3f}, {len(shared_clusters)} shared clusters")
    
    def characterize_groups(self):
        """Characterize each group based on its cluster distribution."""
        # For each group, get its top clusters and distribution
        self.group_profiles = {}
        
        for group, data in self.group_data.items():
            df = data['df']
            
            # Count patterns per cluster
            cluster_counts = Counter(df['ClusterID'])
            
            # Get top clusters
            top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create profile
            profile = {
                'top_clusters': top_clusters,
                'cluster_count': len(set(df['ClusterID'])),
                'pattern_count': len(df),
                'models': len(df['Model'].unique()),
                'distribution': {str(k): v for k, v in cluster_counts.items()}
            }
            
            self.group_profiles[group] = profile
            
            # Log summary
            top_clusters_str = ", ".join([f"C{c}({n})" for c, n in top_clusters[:5]])
            self.logger.info(f"Group {group}: {profile['pattern_count']} patterns across {profile['cluster_count']} clusters. Top: {top_clusters_str}")
    
    def visualize_group_model_matrices(self):
        """Visualize state-cluster matrices for each group."""
        if not hasattr(self, 'state_cluster_matrices'):
            self.create_state_cluster_matrices()
        
        for group, matrix_df in self.state_cluster_matrices.items():
            # Convert to wide format for heatmap
            pivot_df = matrix_df.pivot(index='Model', columns='SortedStateIdx', values='ClusterID')
            pivot_df = pivot_df.loc[natsorted(pivot_df.index)]
            # Create colormap for cluster IDs
            n_clusters = len(self.cluster_info)
            cmap = plt.colormaps['tab20'].resampled(n_clusters)
            
            # Create figure
            plt.figure(figsize=(12, max(6, len(pivot_df) * 0.4)))
            
            # Use a discrete colormap for cluster IDs
            ax = sns.heatmap(pivot_df, cmap=cmap, cbar=False, 
                           linewidths=0.5, annot=True, fmt='.0f')
            
            plt.title(f'{group}: Model State to Cluster Mapping')
            plt.ylabel('Model')
            plt.xlabel('Sorted State Index')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{group}_state_cluster_matrix.png', dpi=300)
            plt.close()
            
            # Also save data
            pivot_df.to_csv(self.output_dir / f'{group}_state_cluster_matrix.csv')
    
    def visualize_group_profiles(self):
        """Visualize group profiles based on cluster distributions."""
        if not hasattr(self, 'group_profiles'):
            self.characterize_groups()
        
        # Create distribution comparison
        plt.figure(figsize=(14, 8))
        
        # For each group, plot top clusters
        data = []
        
        for group, profile in self.group_profiles.items():
            # Get top 10 clusters with their counts
            top_clusters = profile['top_clusters'][:10]
            
            for cluster_id, count in top_clusters:
                data.append({
                    'Group': group,
                    'ClusterID': f"C{cluster_id}",
                    'Count': count
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Color mapping
        palette = {group: COLORS.get(group, '#333333') for group in self.groups}
        
        # Create grouped bar chart
        ax = sns.barplot(x='ClusterID', y='Count', hue='Group', data=df, palette=palette)
        
        plt.title('Top Clusters by Group', fontsize=14)
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Pattern Count', fontsize=12)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
        
        plt.legend(title='Group')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_clusters_by_group.png', dpi=300)
        plt.close()
        
        # Create cluster distribution heatmap
        # Prepare data for heatmap
        all_clusters = set()
        for group, profile in self.group_profiles.items():
            all_clusters.update([int(c) for c in profile['distribution'].keys()])
        
        # Create a matrix of cluster counts
        cluster_matrix = np.zeros((len(self.groups), len(all_clusters)))
        
        # Fill matrix
        all_clusters = sorted(list(all_clusters))
        for i, group in enumerate(self.groups):
            distribution = self.group_profiles[group]['distribution']
            for j, cluster_id in enumerate(all_clusters):
                cluster_matrix[i, j] = int(distribution.get(str(cluster_id), 0))
        
        # Create heatmap
        plt.figure(figsize=(max(10, len(all_clusters) * 0.4), 6))
        
        # Create cluster heatmap
        sns.heatmap(cluster_matrix, cmap='YlGnBu', annot=True, fmt='.2f',
                   xticklabels=[f"C{c}" for c in all_clusters],
                   yticklabels=self.groups,
                   linewidths=0.5, cbar_kws={'label': 'Pattern Count'})
        
        plt.title('Cluster Distribution by Group', fontsize=14)
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Group', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_distribution_heatmap.png', dpi=300)
        plt.close()
        
    def visualize_group_similarity(self):
        """Visualize similarity between groups."""
        if not hasattr(self, 'group_similarity'):
            self.compare_group_cluster_distributions()
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(self.groups), len(self.groups)))
        
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i == j:  # Self-similarity is 1.0
                    similarity_matrix[i, j] = 1.0
                else:
                    # Get similarity value
                    pair_key1 = f"{group1}_{group2}"
                    pair_key2 = f"{group2}_{group1}"
                    
                    if pair_key1 in self.group_similarity:
                        similarity_matrix[i, j] = self.group_similarity[pair_key1]
                    elif pair_key2 in self.group_similarity:
                        similarity_matrix[i, j] = self.group_similarity[pair_key2]
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(similarity_matrix, cmap='YlGnBu', annot=True, fmt='.3f',
                   xticklabels=self.groups,
                   yticklabels=self.groups,
                   linewidths=0.5, cbar_kws={'label': 'Jaccard Similarity'})
        
        plt.title('Group Similarity (Jaccard Index)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_similarity_matrix.png', dpi=300)
        plt.close()
        
        # Visualize shared clusters
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i < j:  # Avoid redundant comparisons
                    pair_key = f"{group1}_{group2}"
                    shared_clusters = self.group_overlap_clusters.get(pair_key, [])
                    
                    if shared_clusters:
                        # Create bar chart of shared clusters
                        plt.figure(figsize=(10, 6))
                        
                        # Collect data for shared clusters
                        data = []
                        
                        for cluster_id in shared_clusters:
                            # Get counts for each group
                            count1 = int(self.group_profiles[group1]['distribution'].get(str(cluster_id), 0))
                            count2 = int(self.group_profiles[group2]['distribution'].get(str(cluster_id), 0))
                            
                            data.append({
                                'ClusterID': f"C{cluster_id}",
                                'Group': group1,
                                'Count': count1
                            })
                            
                            data.append({
                                'ClusterID': f"C{cluster_id}",
                                'Group': group2,
                                'Count': count2
                            })
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(data)
                        
                        # Color mapping
                        palette = {
                            group1: COLORS.get(group1, '#333333'),
                            group2: COLORS.get(group2, '#666666')
                        }
                        
                        # Create grouped bar chart
                        ax = sns.barplot(x='ClusterID', y='Count', hue='Group', data=df, palette=palette)
                        
                        plt.title(f'Shared Clusters: {group1} vs {group2}', fontsize=14)
                        plt.xlabel('Cluster ID', fontsize=12)
                        plt.ylabel('Pattern Count', fontsize=12)
                        
                        # Add value labels
                        for container in ax.containers:
                            ax.bar_label(container, fmt='%d')
                        
                        plt.legend(title='Group')
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        plt.tight_layout()
                        plt.savefig(self.output_dir / f'shared_clusters_{group1}_{group2}.png', dpi=300)
                        plt.close()
    
    def create_group_network_visualization(self):
        """Create a network visualization of groups and their shared clusters."""
        if not hasattr(self, 'group_similarity'):
            self.compare_group_cluster_distributions()
        
        # Create network graph
        G = nx.Graph()
        
        # Add group nodes
        for group in self.groups:
            G.add_node(group, type='group', size=len(self.group_data[group]['df']),
                      color=COLORS.get(group, '#333333'))
        
        # Add shared cluster edges
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i < j:  # Avoid redundant comparisons
                    pair_key = f"{group1}_{group2}"
                    similarity = self.group_similarity.get(pair_key, 0)
                    shared_clusters = self.group_overlap_clusters.get(pair_key, [])
                    
                    if shared_clusters:
                        G.add_edge(group1, group2, weight=similarity, 
                                  shared=len(shared_clusters),
                                  clusters=shared_clusters)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Create positions
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_sizes = [G.nodes[n]['size'] / 5 for n in G.nodes()]
        node_colors = [G.nodes[n].get('color', '#333333') for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Draw edges with width proportional to similarity
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add edge labels for shared cluster count
        edge_labels = {(u, v): f"{G[u][v]['shared']} clusters" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title('Group Similarity Network', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_network.png', dpi=300)
        plt.close()
    
    def analyze_group_distances(self):
        """Analyze and visualize distances between groups using t-SNE."""
        if not hasattr(self, 'group_profiles'):
            self.characterize_groups()
        
        # Create a distance matrix based on cluster distributions
        distance_matrix = np.zeros((len(self.groups), len(self.groups)))
        
        # Get all clusters across all groups
        all_clusters = set()
        for group, profile in self.group_profiles.items():
            all_clusters.update([int(c) for c in profile['distribution'].keys()])
        
        # Create normalized distribution vectors
        distributions = {}
        
        for group, profile in self.group_profiles.items():
            vec = np.zeros(max(all_clusters) + 1)
            for cluster_id, count in profile['distribution'].items():
                vec[int(cluster_id)] = count
            
            # Normalize to probabilities
            total = np.sum(vec)
            if total > 0:
                vec = vec / total
            
            distributions[group] = vec
        
        # Calculate distances
        for i, group1 in enumerate(self.groups):
            for j, group2 in enumerate(self.groups):
                if i != j:
                    # Get distribution vectors
                    vec1 = distributions[group1]
                    vec2 = distributions[group2]
                    
                    # Calculate distance (1 - Jaccard similarity)
                    if hasattr(self, 'group_similarity'):
                        pair_key = f"{group1}_{group2}"
                        pair_key_rev = f"{group2}_{group1}"
                        
                        if pair_key in self.group_similarity:
                            similarity = self.group_similarity[pair_key]
                        elif pair_key_rev in self.group_similarity:
                            similarity = self.group_similarity[pair_key_rev]
                        else:
                            similarity = 0
                        
                        distance_matrix[i, j] = 1 - similarity
                    else:
                        # Fallback to euclidean distance
                        min_len = min(len(vec1), len(vec2))
                        distance_matrix[i, j] = np.sqrt(np.sum((vec1[:min_len] - vec2[:min_len])**2))
        
        # Change the t-SNE initialization
        tsne = TSNE(
            n_components=2,
            metric='precomputed',
            random_state=42,
            init='random',  # Change from 'pca' to 'random'
            perplexity=min(30, len(self.groups) - 1)  # Adjust perplexity if needed
        )

        # Perform t-SNE
        points = tsne.fit_transform(distance_matrix)

        # Create visualization
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.groups)))
        
        for i, group in enumerate(self.groups):
            plt.scatter(points[i, 0], points[i, 1], c=[colors[i]], label=group, s=100)
        
        plt.title('t-SNE Visualization of Group Distances')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'group_distances_tsne.png'))
        plt.close()

        self.logger.info("t-SNE visualization saved.")
    
    def save_analysis_results(self):
        """Save analysis results to files."""
        # Save group profiles
        if hasattr(self, 'group_profiles'):
            with open(self.output_dir / 'group_profiles.json', 'w') as f:
                json.dump(self.group_profiles, f, indent=2)
        
        # Save group similarity data
        if hasattr(self, 'group_similarity'):
            similarity_data = {
                'pairwise_similarity': self.group_similarity,
                'overlap_clusters': self.group_overlap_clusters
            }
            
            with open(self.output_dir / 'group_similarity.json', 'w') as f:
                json.dump(similarity_data, f, indent=2)
        
        # Save state-cluster matrices
        if hasattr(self, 'state_cluster_matrices'):
            for group, matrix_df in self.state_cluster_matrices.items():
                matrix_df.to_csv(self.output_dir / f'{group}_state_cluster_matrix.csv', index=False)
        
        # Save model info
        if hasattr(self, 'model_info'):
            for group, info_df in self.model_info.items():
                info_df.to_csv(self.output_dir / f'{group}_model_info.csv', index=False)
        
        # Save full analysis results
        all_results = {
            'groups': self.groups,
            'group_data_summary': {
                group: {
                    'pattern_count': len(data['df']),
                    'cluster_count': len(data['clusters']),
                    'model_count': len(data['df']['Model'].unique())
                } for group, data in self.group_data.items()
            }
        }
        
        if hasattr(self, 'group_similarity'):
            all_results['group_similarity'] = self.group_similarity
        
        if hasattr(self, 'group_profiles'):
            all_results['group_profiles'] = {
                group: {
                    'top_clusters': profile['top_clusters'],
                    'cluster_count': profile['cluster_count'],
                    'pattern_count': profile['pattern_count'],
                    'models': profile['models']
                } for group, profile in self.group_profiles.items()
            }
        
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def run_analysis(self):
        """Run the complete group comparison analysis."""
        self.logger.info("Starting group comparison analysis...")
        
        # Step 1: Create state-cluster matrices
        self.logger.info("Creating state-cluster matrices...")
        self.create_state_cluster_matrices()
        
        # Step 2: Characterize groups
        self.logger.info("Characterizing groups...")
        self.characterize_groups()
        
        # Step 3: Compare group distributions
        self.logger.info("Comparing group distributions...")
        self.compare_group_cluster_distributions()
        
        # Step 4: Visualize results
        self.logger.info("Creating visualizations...")
        self.visualize_group_model_matrices()
        self.visualize_group_profiles()
        self.visualize_group_similarity()
        self.create_group_network_visualization()
        self.analyze_group_distances()
        
        # Step 5: Save results
        self.logger.info("Saving results...")
        self.save_analysis_results()
        
        self.logger.info("Group comparison analysis complete.")
        
        return {
            'group_profiles': self.group_profiles if hasattr(self, 'group_profiles') else None,
            'group_similarity': self.group_similarity if hasattr(self, 'group_similarity') else None
        }


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
        
    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = os.path.join(scratch_dir, "output")
    output_dir = os.path.join(base_dir, "07_group_cluster_comp")
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    groups = ["affair", "paranoia", "combined"]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    for th in thresholds:
        th_str = f"th_{th:.2f}".replace('.', '')  
        cluster_info_path = os.path.join(base_dir, f"06_state_pattern_cluster/{th_str}/cluster_info.json")
        state_mappings_path = os.path.join(base_dir, f"06_state_pattern_cluster/{th_str}/state_mappings.json")
        th_dir = os.path.join(output_dir, f"{th_str}")
        analyzer = GroupModelComparison(
            cluster_info_path=cluster_info_path,
            state_mappings_path=state_mappings_path,
            output_dir=th_dir
        )
    
        analyzer.run_analysis()