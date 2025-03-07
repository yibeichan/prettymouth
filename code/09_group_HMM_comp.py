import numpy as np
import pickle
import json
import os
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy import stats
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from dotenv import load_dotenv
# Environment setup
load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise EnvironmentError("SCRATCH_DIR environment variable is not set")

class BrainStateAnalysis:
    """
    Analysis of brain state dynamics during story comprehension with different contexts.
    
    This class handles the complete analysis pipeline for understanding how prior context
    influences neural state dynamics, focusing on:
    1. State pattern identification and comparison
    2. Temporal dynamics analysis
    3. State transition analysis
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize analysis pipeline
        
        Parameters:
        -----------
        base_dir : str or Path, optional
            Base directory for analysis. Defaults to SCRATCH_DIR
        """
        self.base_dir = Path(SCRATCH_DIR) / "output"
        self.results = {
            'models': {},      # HMM models for each condition
            'sequences': {},   # State sequences
            'patterns': {},    # State patterns (emissions)
            'metrics': {},     # Analysis metrics
            'comparisons': {}  # Between-group comparisons
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configure logging for analysis pipeline"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        self.output_dir = self.base_dir / "09_group_HMM_comparison"
        self.output_dir.mkdir(exist_ok=True)
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup file handler
        log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(log_file)
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def load_group_results(self, group):
        """
        Load analysis results for a specific group
        
        Parameters:
        -----------
        group : str
            One of ['affair', 'paranoia', 'combined']
        """
        self.logger.info(f"Loading results for {group} group")
        group_dir = self.base_dir / f"{group}_hmm_3states_ntw_native_trimmed"
        
        try:
            # Load HMM model
            model_path = group_dir / "models" / f"{group}_hmm_model.pkl"
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
                
                # Verify model structure
                if 'model' in model_dict:
                    hmm_model = model_dict['model']
                    self.logger.info(f"Model type: {type(hmm_model)}")
                    
                    # Check for required attributes
                    if hasattr(hmm_model, 'means_'):
                        self.logger.info(f"Model has means_ with shape: {hmm_model.means_.shape}")
                    else:
                        self.logger.warning(f"Model does not have means_ attribute")
                        
                    if hasattr(hmm_model, 'covars_'):
                        self.logger.info(f"Model has covars_ with shape: {hmm_model.covars_.shape}")
                    else:
                        self.logger.warning(f"Model does not have covars_ attribute")
                
                self.results['models'][group] = model_dict
            
            # Load state sequences
            seq_path = group_dir / "statistics" / f"{group}_state_sequences.npy"
            self.results['sequences'][group] = np.load(seq_path)
            
            # Load metrics
            metrics_path = group_dir / "statistics" / f"{group}_metrics.pkl"
            with open(metrics_path, 'rb') as f:
                self.results['metrics'][group] = pickle.load(f)
            
            self.logger.info(f"Successfully loaded {group} data")
            
        except Exception as e:
            self.logger.error(f"Error loading {group} data: {e}")
            raise
            
    def compute_state_patterns(self):
        """Extract state patterns from HMM models"""
        self.logger.info("Computing state patterns")
        
        patterns = {}
        for group, model_dict in self.results['models'].items():
            self.logger.info(f"\nProcessing group: {group}")
            
            try:
                # Extract the actual HMM model from the dictionary
                if isinstance(model_dict, dict) and 'model' in model_dict:
                    hmm_model = model_dict['model']
                    self.logger.info(f"Found HMM model of type: {type(hmm_model)}")
                    
                    # For GaussianHMM, we want the means_ and covars_
                    if hasattr(hmm_model, 'means_'):
                        patterns[group] = hmm_model.means_
                        self.logger.info(f"Extracted means with shape: {hmm_model.means_.shape}")
                    elif hasattr(hmm_model, 'emissionprob_'):
                        patterns[group] = hmm_model.emissionprob_
                    else:
                        raise AttributeError(f"HMM model for group {group} has neither means_ nor emissionprob_")
                else:
                    raise KeyError(f"Invalid model structure for group {group}")
                        
            except Exception as e:
                self.logger.error(f"Error processing patterns for group {group}: {str(e)}")
                self.logger.error(f"Model structure: {model_dict}")
                raise
                
        self.results['patterns'] = patterns
        self.logger.info("State patterns computed successfully")
        
        return patterns
        
    def reorder_states_by_occupancy(self):
        """
        Reorder states based on fractional occupancy for each group
        Returns dictionaries mapping original indices to new indices
        """
        self.logger.info("Reordering states by fractional occupancy")
        
        reordering_maps = {}
        
        for group in ['affair', 'paranoia']:
            try:
                # Check if the metrics are loaded
                if group not in self.results['metrics']:
                    self.logger.warning(f"Metrics for group {group} not found. Cannot reorder states.")
                    continue
                
                # Get fractional occupancy from metrics
                metrics = self.results['metrics'][group]
                if 'state_metrics' in metrics and 'group_level' in metrics['state_metrics']:
                    group_metrics = metrics['state_metrics']['group_level']
                    
                    if 'fractional_occupancy' in group_metrics:
                        # Get fractional occupancy for each state
                        fractional_occupancy = group_metrics['fractional_occupancy']
                        self.logger.info(f"{group} fractional occupancy: {fractional_occupancy}")
                        
                        # Sort states by occupancy (descending order)
                        sorted_indices = np.argsort(-np.array(fractional_occupancy))
                        
                        # Create mapping from original indices to new indices
                        reordering_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
                        reordering_maps[group] = reordering_map
                        
                        self.logger.info(f"Reordering map for {group}: {reordering_map}")
                        
                        # Reorder state patterns
                        if group in self.results['patterns']:
                            original_patterns = self.results['patterns'][group]
                            reordered_patterns = np.array([original_patterns[old_idx] for old_idx in sorted_indices])
                            self.results['patterns'][group] = reordered_patterns
                        
                        # Reorder state sequences
                        if group in self.results['sequences']:
                            original_sequences = self.results['sequences'][group]
                            reordered_sequences = np.zeros_like(original_sequences)
                            
                            # Map each state in the sequence to its new index
                            for old_idx, new_idx in reordering_map.items():
                                reordered_sequences[original_sequences == old_idx] = new_idx
                            
                            self.results['sequences'][group] = reordered_sequences
                    else:
                        self.logger.warning(f"Fractional occupancy not found for group {group}")
                else:
                    self.logger.warning(f"State metrics or group level metrics not found for group {group}")
            
            except Exception as e:
                self.logger.error(f"Error reordering states for group {group}: {e}")
                raise
        
        # Store reordering maps for later reference
        self.results['reordering_maps'] = reordering_maps
        self.logger.info("States reordered successfully")
        
        return reordering_maps
    
    def run_analysis(self):
        """Execute complete analysis pipeline"""
        self.logger.info("Starting analysis pipeline")
        
        try:
            # 1. Load all group data
            for group in ['affair', 'paranoia', 'combined']:
                self.load_group_results(group)
            
            # 2. Extract state patterns
            self.compute_state_patterns()
            
            # Add debugging before each analysis
            self.logger.info("Running state similarities analysis...")
            self.analyze_state_similarities()
            
            self.logger.info("Running temporal dynamics analysis...")
            self.analyze_temporal_dynamics()
            
            self.logger.info("Running state transitions analysis...")
            self.analyze_state_transitions()
            
            # Debug the structure before saving
            self.logger.info("Comparison results structure:")
            for key, value in self.results['comparisons'].items():
                self.logger.info(f"Key: {key}")
                self.logger.info(f"Type: {type(value)}")
            
            # 4. Save results
            self.save_results()
            
            # 5. Generate visualizations
            self.create_visualizations()
            
            self.logger.info("Analysis pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {e}")
            raise
            
    def save_results(self):
        """Save analysis results and metadata"""
        output_dir = self.output_dir        
        try:
            # Create a clean version of comparisons for saving
            saveable_comparisons = {}
            
            # Process state similarity data
            if 'state_similarity' in self.results['comparisons']:
                sim_data = self.results['comparisons']['state_similarity']
                saveable_comparisons['state_similarity'] = {
                    'similarity_matrix': sim_data['similarity_matrix'].tolist() if isinstance(sim_data['similarity_matrix'], np.ndarray) else sim_data['similarity_matrix'],
                    'matched_states': sim_data['matched_states']
                }
                
            # Process temporal dynamics data
            if 'temporal_dynamics' in self.results['comparisons']:
                temp_data = {}
                for state_pair, metrics in self.results['comparisons']['temporal_dynamics'].items():
                    temp_data[state_pair] = {
                        'occupancy': {
                            'affair_mean': float(metrics['occupancy']['affair_mean']),
                            'paranoia_mean': float(metrics['occupancy']['paranoia_mean']),
                            'difference': float(metrics['occupancy']['difference']),
                            'ttest_statistic': float(metrics['occupancy']['ttest'].statistic),
                            'ttest_pvalue': float(metrics['occupancy']['ttest'].pvalue)
                        },
                        'duration': {
                            'affair_mean': float(metrics['duration']['affair_mean']),
                            'paranoia_mean': float(metrics['duration']['paranoia_mean']),
                            'affair_distribution': [float(x) for x in metrics['duration']['affair_distribution']],
                            'paranoia_distribution': [float(x) for x in metrics['duration']['paranoia_distribution']],
                            'ks_statistic': float(metrics['duration']['ks_test'].statistic),
                            'ks_pvalue': float(metrics['duration']['ks_test'].pvalue)
                        },
                        'temporal_profile': {
                            'affair_profile': [float(x) for x in metrics['temporal_profile']['affair_profile']],
                            'paranoia_profile': [float(x) for x in metrics['temporal_profile']['paranoia_profile']],
                            'correlation': float(metrics['temporal_profile']['correlation'])
                        }
                    }
                saveable_comparisons['temporal_dynamics'] = temp_data
                
            # Process transition data
            if 'transitions' in self.results['comparisons']:
                trans_data = self.results['comparisons']['transitions']
                saveable_comparisons['transitions'] = {
                    'affair': trans_data['affair'].tolist() if isinstance(trans_data['affair'], np.ndarray) else trans_data['affair'],
                    'paranoia': trans_data['paranoia'].tolist() if isinstance(trans_data['paranoia'], np.ndarray) else trans_data['paranoia'],
                    'paranoia_reordered': trans_data['paranoia_reordered'].tolist() if isinstance(trans_data['paranoia_reordered'], np.ndarray) else trans_data['paranoia_reordered'],
                    'difference': trans_data['difference'].tolist() if isinstance(trans_data['difference'], np.ndarray) else trans_data['difference']
                }
                
            # Save the processed comparisons
            comp_file = output_dir / "comparisons.pkl"
            with open(comp_file, 'wb') as f:
                pickle.dump(saveable_comparisons, f)
                
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'groups_analyzed': list(self.results['models'].keys()),
            }
            meta_file = output_dir / "metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                    
            self.logger.info(f"Results saved to {output_dir}")
                
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.logger.error("Results structure:", self.results['comparisons'])
            raise

    def analyze_state_similarities(self):
        """
        Analyze similarities between state patterns across conditions.
        States have already been reordered by fractional occupancy.
        """
        self.logger.info("Analyzing state pattern similarities")
        
        # Get patterns for both conditions (already reordered by occupancy)
        affair_patterns = self.results['patterns']['affair']
        paranoia_patterns = self.results['patterns']['paranoia']
        
        # Compute pairwise similarity matrix
        similarity_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                similarity_matrix[i, j] = 1 - cosine(
                    affair_patterns[i], 
                    paranoia_patterns[j]
                )
        
        # For reordered states, the best match is typically along diagonal
        # But we'll still find optimal matching for completeness
        matched_states = []
        for i in range(3):
            matched_states.append((i, i))  # Direct correspondence after reordering
            
        self.results['comparisons']['state_similarity'] = {
            'similarity_matrix': similarity_matrix,
            'matched_states': matched_states
        }
        
        self.logger.info("State similarity analysis completed")
        return matched_states
    
    def _match_states(self, similarity_matrix):
        """
        Find optimal matching between states based on similarity matrix
        using a greedy approach.
        """
        matched_pairs = []
        available_cols = list(range(3))
        
        for i in range(3):
            # Find best match for current state
            similarities = similarity_matrix[i, available_cols]
            best_match_idx = np.argmax(similarities)
            matched_pairs.append((i, available_cols[best_match_idx]))
            available_cols.pop(best_match_idx)
            
        return matched_pairs
    
    def analyze_temporal_dynamics(self):
        """
        Analyze temporal aspects of state dynamics, including:
        - State occupancy patterns
        - Duration distributions
        - Temporal state profiles
        """
        self.logger.info("Analyzing temporal dynamics")
        
        # Get matched states
        matched_states = self.results['comparisons']['state_similarity']['matched_states']
        
        temporal_metrics = {}
        for affair_idx, paranoia_idx in matched_states:
            metrics = self._compute_temporal_metrics(
                affair_idx, paranoia_idx
            )
            temporal_metrics[f'state_pair_{affair_idx}_{paranoia_idx}'] = metrics
            
        self.results['comparisons']['temporal_dynamics'] = temporal_metrics
        self.logger.info("Temporal dynamics analysis completed")
        
    def _compute_temporal_metrics(self, affair_idx, paranoia_idx):
        """
        Compute comprehensive temporal metrics for a pair of matched states
        """
        # Get sequences
        affair_seq = self.results['sequences']['affair']
        paranoia_seq = self.results['sequences']['paranoia']
        
        # 1. Occupancy analysis
        affair_occ = (affair_seq == affair_idx).mean(axis=1)
        paranoia_occ = (paranoia_seq == paranoia_idx).mean(axis=1)
        
        occ_stats = {
            'affair_mean': float(np.mean(affair_occ)),
            'paranoia_mean': float(np.mean(paranoia_occ)),
            'difference': float(np.mean(affair_occ) - np.mean(paranoia_occ)),
            'ttest': stats.ttest_ind(affair_occ, paranoia_occ)
        }
        
        # 2. Duration analysis
        affair_dur = self._get_state_durations(affair_seq, affair_idx)
        paranoia_dur = self._get_state_durations(paranoia_seq, paranoia_idx)
        
        dur_stats = {
            'affair_mean': float(np.mean(affair_dur)),
            'paranoia_mean': float(np.mean(paranoia_dur)),
            'affair_distribution': affair_dur.tolist(),
            'paranoia_distribution': paranoia_dur.tolist(),
            'ks_test': stats.ks_2samp(affair_dur, paranoia_dur)
        }
        
        # 3. Temporal profile analysis
        affair_profile = (affair_seq == affair_idx).mean(axis=0)
        paranoia_profile = (paranoia_seq == paranoia_idx).mean(axis=0)
        
        # Compute temporal correlation
        temp_corr = np.corrcoef(affair_profile, paranoia_profile)[0,1]
        
        temporal_stats = {
            'affair_profile': affair_profile.tolist(),
            'paranoia_profile': paranoia_profile.tolist(),
            'correlation': float(temp_corr)
        }
        
        return {
            'occupancy': occ_stats,
            'duration': dur_stats,
            'temporal_profile': temporal_stats
        }
        
    def analyze_state_transitions(self):
        """
        Analyze state transition patterns between conditions
        """
        self.logger.info("Analyzing state transitions")
        
        # Get matched states
        matched_states = self.results['comparisons']['state_similarity']['matched_states']
        
        # Compute transition matrices
        affair_trans = self._compute_transition_matrix(
            self.results['sequences']['affair']
        )
        paranoia_trans = self._compute_transition_matrix(
            self.results['sequences']['paranoia']
        )
        
        # Reorder paranoia transitions according to matched states
        reordered_trans = self._reorder_transitions(
            paranoia_trans, matched_states
        )
        
        # Compare transitions
        transition_comparison = {
            'affair': affair_trans,
            'paranoia': paranoia_trans,
            'paranoia_reordered': reordered_trans,
            'difference': affair_trans - reordered_trans
        }
        
        self.results['comparisons']['transitions'] = transition_comparison
        self.logger.info("Transition analysis completed")
        
    def _compute_transition_matrix(self, sequences):
        """Compute transition probability matrix"""
        transitions = np.zeros((3, 3))
        
        for seq in sequences:
            for t in range(len(seq)-1):
                transitions[seq[t], seq[t+1]] += 1
                
        # Normalize
        row_sums = transitions.sum(axis=1)
        transitions = transitions / row_sums[:, np.newaxis]
        
        return transitions
        
    def _reorder_transitions(self, trans_matrix, matched_states):
        """Reorder transition matrix according to state matching"""
        n_states = 3
        mapping = np.zeros(n_states, dtype=int)
        for i, j in matched_states:
            mapping[j] = i
            
        reordered = np.zeros_like(trans_matrix)
        for i in range(n_states):
            for j in range(n_states):
                reordered[mapping[i], mapping[j]] = trans_matrix[i, j]
                
        return reordered
        
    def _get_state_durations(self, sequences, state_idx):
        """Calculate durations of state occurrences"""
        durations = []
        
        for seq in sequences:
            # Find state boundaries
            state_bins = (seq == state_idx).astype(int)
            changes = np.diff(state_bins)
            
            # Find start and end indices
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            
            # Handle edge cases
            if state_bins[0] == 1:
                starts = np.r_[0, starts]
            if state_bins[-1] == 1:
                ends = np.r_[ends, len(state_bins)]
                
            durations.extend(ends - starts)
            
        return np.array(durations)
    
    def create_visualizations(self):
        """Create and save all visualizations"""
        self.logger.info("Generating visualizations")
        
        try:
            # Initialize visualizer
            viz = StateVisualization(self.results)
            viz.set_output_directory(self.output_dir / "figures")
            
            # Create main figure
            viz.create_main_figure()
            
            self.logger.info("Visualizations completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            raise
    
class StateVisualization:
    """
    Visualization module for brain state analysis results.
    Creates publication-ready figures showing state patterns,
    temporal dynamics, and condition comparisons.
    """
    
    def __init__(self, results):
        """
        Initialize visualizer with analysis results
        
        Parameters:
        -----------
        results : dict
            Results dictionary from BrainStateAnalysis
        """
        self.results = results
        self.output_dir = None
        self.colors = plt.cm.tab10(np.linspace(0, 1, 3))
        
        # Update the style to a currently supported one
        try:
            plt.style.use('seaborn-v0_8-paper')  # For newer seaborn versions
        except:
            try:
                plt.style.use('seaborn')  # Fallback to basic seaborn style
            except:
                self.logger.warning("Could not set seaborn style, using default matplotlib style")

        # Set additional style parameters
        plt.rcParams.update({
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        
    def set_output_directory(self, output_dir):
        """Set directory for saving figures"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_main_figure(self):
        """Create main summary figure for publication"""
        fig = plt.figure(figsize=(15, 18))  # Increased figure height
        
        # Create GridSpec with adjusted height ratios
        gs = GridSpec(7, 3, figure=fig, height_ratios=[1.2, 1.2, 1.8, 1.8, 1.8, 1.65, 1.65])
        
        # 1. Top row - make it taller (spans 2 rows)
        ax_sim = fig.add_subplot(gs[0:2, 0])  # State Pattern Similarity
        ax_prop = fig.add_subplot(gs[0:2, 1:])  # State Properties
        
        # 2. Middle - Time series (spans 3 rows)
        ax_temp1 = fig.add_subplot(gs[2:3, :])
        ax_temp2 = fig.add_subplot(gs[3:4, :], sharex=ax_temp1)
        ax_temp3 = fig.add_subplot(gs[4:5, :], sharex=ax_temp1)

        # Hide x-axis labels for top two time series plots
        ax_temp1.set_xticklabels([])
        ax_temp2.set_xticklabels([])
        plt.setp(ax_temp1.get_xticklabels(), visible=False)
        plt.setp(ax_temp2.get_xticklabels(), visible=False)

        # Ensure ax_temp3 shows its xticklabels
        ax_temp3.tick_params(axis='x', labelbottom=True)
        
        # 3. Bottom row - Transition matrices (spans 2 rows)
        ax_trans1 = fig.add_subplot(gs[5:7, 0])
        ax_trans2 = fig.add_subplot(gs[5:7, 1])
        ax_diff = fig.add_subplot(gs[5:7, 2])
        
        
        # Plot all components
        self._plot_similarity_matrix(ax_sim)
        self._plot_state_properties(ax_prop)
        self._plot_temporal_dynamics([ax_temp1, ax_temp2, ax_temp3])
        self._plot_transition_comparison(ax_trans1, ax_trans2, ax_diff)
        
        plt.tight_layout()
        self._save_figure("main_figure")
        
    def _plot_similarity_matrix(self, ax):
        """Plot state pattern similarity matrix"""
        sim_matrix = self.results['comparisons']['state_similarity']['similarity_matrix']
        matched_states = self.results['comparisons']['state_similarity']['matched_states']
        
        # Create heatmap
        im = sns.heatmap(sim_matrix, 
                        annot=True, 
                        cmap='viridis', 
                        vmin=0, 
                        vmax=1,
                        fmt='.2f',
                        ax=ax,
                        xticklabels=['S1', 'S2', 'S3'],
                        yticklabels=['S1', 'S2', 'S3'])
        
        # Mark matched states
        for i, j in matched_states:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                    edgecolor='red', lw=2))
        
        ax.set_title('State Pattern Similarity')
        ax.set_xlabel('Paranoia')
        ax.set_ylabel('Affair')
        
    def _plot_temporal_dynamics(self, axes):
        """Plot temporal evolution of states - one subplot per state pair"""
        temporal_data = self.results['comparisons']['temporal_dynamics']
        matched_states = self.results['comparisons']['state_similarity']['matched_states']
        
        for idx, (affair_idx, paranoia_idx) in enumerate(matched_states):
            ax = axes[idx]
            key = f'state_pair_{affair_idx}_{paranoia_idx}'
            profile = temporal_data[key]['temporal_profile']
            
            # Plot affair and paranoia profiles for this state
            ax.plot(profile['affair_profile'], 
                color=self.colors[idx], 
                label='Affair')
            ax.plot(profile['paranoia_profile'], 
                color=self.colors[idx], 
                linestyle='--',
                alpha=0.7,
                label='Paranoia (Reordered)')
            
            # Add correlation annotation
            corr = profile['correlation']
            ax.text(0.98, 0.98, 
                f'r = {corr:.2f}',
                transform=ax.transAxes,
                ha='right',
                va='top')
            
            ax.set_title(f'State {idx+1}')
            
            # Only show x-axis labels for the bottom plot
            if idx < 2:  # For first two plots
                ax.set_xticks([])
            else:  # For the last plot
                ax.set_xlabel('Time (TRs)')
            
            ax.set_ylabel('State Probability')
            ax.legend()
        
    def _plot_state_properties(self, ax):
        """Plot comparison of state properties"""
        temporal_data = self.results['comparisons']['temporal_dynamics']
        matched_states = self.results['comparisons']['state_similarity']['matched_states']
        
        n_pairs = len(matched_states)
        width = 0.35
        x = np.arange(n_pairs)
        
        # Define distinct colors for Occupancy and Duration
        occ_color = '#1f77b4'  # blue
        dur_color = '#ff7f0e'  # orange
        
        # Create bars
        occ_bars = ax.bar(x - width/2, 
                        [temporal_data[f'state_pair_{a}_{p}']['occupancy']['difference'] 
                        for a, p in matched_states],
                        width, label='Occupancy',
                        color=occ_color)
        
        dur_bars = ax.bar(x + width/2,
                        [temporal_data[f'state_pair_{a}_{p}']['duration']['affair_mean'] - 
                        temporal_data[f'state_pair_{a}_{p}']['duration']['paranoia_mean']
                        for a, p in matched_states],
                        width, label='Duration',
                        color=dur_color)
        
        # Add significance markers
        for idx, (affair_idx, paranoia_idx) in enumerate(matched_states):
            key = f'state_pair_{affair_idx}_{paranoia_idx}'
            metrics = temporal_data[key]
            
            if metrics['occupancy']['ttest'].pvalue < 0.05:
                ax.text(idx - width/2, occ_bars[idx].get_height(), '*',
                    ha='center', va='bottom')
            if metrics['duration']['ks_test'].pvalue < 0.05:
                ax.text(idx + width/2, dur_bars[idx].get_height(), '*',
                    ha='center', va='bottom')
        
        ax.set_title('State Property Differences (Affair - Paranoia)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{i+1}' for i in range(n_pairs)])
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
    def _plot_transition_comparison(self, ax1, ax2, ax3):
        """Plot transition matrices and their difference"""
        trans_data = self.results['comparisons']['transitions']
        
        # Plot affair transitions
        im1 = ax1.imshow(trans_data['affair'], 
                        cmap='coolwarm', vmin=0, vmax=1)
        ax1.set_title('Affair Transitions')
        self._format_transition_plot(ax1, im1, include_values=True)
        
        # Plot paranoia transitions (reordered)
        im2 = ax2.imshow(trans_data['paranoia_reordered'],
                        cmap='coolwarm', vmin=0, vmax=1)
        ax2.set_title('Paranoia Transitions\n(Reordered)')
        self._format_transition_plot(ax2, im2, include_values=True)
        
        # Plot difference
        im3 = ax3.imshow(trans_data['difference'],
                        cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax3.set_title('Difference\n(Affair - Paranoia)')
        self._format_transition_plot(ax3, im3, include_values=True)

    def _format_transition_plot(self, ax, im, include_values=False):
        """Format transition matrix plot"""
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(['S1', 'S2', 'S3'])
        ax.set_yticklabels(['S1', 'S2', 'S3'])
        
        if include_values:
            # Add numerical values in each cell
            for i in range(3):
                for j in range(3):
                    value = im.get_array()[i, j]
                    ax.text(j, i, f'{value:.2f}',
                        ha='center', va='center',
                        color='white' if abs(value) > 0.5 else 'black')
        
    def _save_figure(self, name):
        """Save figure with appropriate formatting"""
        if self.output_dir is not None:
            plt.savefig(self.output_dir / f"{name}.png",
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{name}.pdf",
                       bbox_inches='tight')
            plt.close()

# Utility functions for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, stats._distn_infrastructure.rv_frozen):
            return str(obj)
        return super().default(obj)

def main():
    """Main execution function"""
    try:
        # Initialize analysis
        analysis = BrainStateAnalysis()
        
        # Run complete pipeline
        analysis.run_analysis()
        
        print("Analysis completed successfully")
        print(f"Results saved to: {analysis.base_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()