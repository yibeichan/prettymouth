# Import necessary libraries
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Union, Optional
import yaml
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from dotenv import load_dotenv

# Set up basic logging configuration at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set up configuration
class Config:
    """Configuration class for analysis parameters"""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Analysis parameters
        self.analysis_params = config['analysis_params']
        self.alpha_level = self.analysis_params.get('alpha_level', 0.05)
        self.correction_method = self.analysis_params.get('correction_method', 'fdr_bh')
        self.n_states = self.analysis_params.get('n_states')
        if not self.n_states:
            raise ValueError("Number of states (n_states) must be specified in config")
            
        # Verify n_states is reasonable
        if not isinstance(self.n_states, int) or self.n_states <= 0 or self.n_states > 10:
            raise ValueError(f"Invalid number of states: {self.n_states}")
        
        # Visualization parameters
        self.viz_params = config['visualization']
        self.colors = self.viz_params['colors']
        self.figsize = self.viz_params['figsize']
        self.dpi = self.viz_params['dpi']
        
        # Paths
        self.data_paths = config['data_paths']
        self.output_dir = Path(config['output']['base_dir'])
        self.fig_dir = self.output_dir / config['output']['figures_dir']
        
    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'brain_state_analysis_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing data"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def load_data(self) -> tuple:
        """
        Load all required data files
        
        Returns:
            tuple: (annotations_df, affair_states, paranoia_states)
        """
        try:
            # Load annotations
            annotations_df = pd.read_csv(self.config.data_paths['annotations'])
            self.logger.info(f"Loaded annotations with shape: {annotations_df.shape}")
            
            # Load brain states for both groups
            affair_states = self._load_group_states('affair')
            paranoia_states = self._load_group_states('paranoia')
            
            # Verify data alignment
            self._verify_alignment(annotations_df, affair_states, paranoia_states)
            
            return annotations_df, affair_states, paranoia_states
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
            
    def _load_group_states(self, group: str) -> dict:
        """
        Load brain states for a specific group
        
        Args:
            group: Either 'affair' or 'paranoia'
            
        Returns:
            Dictionary containing state data and parameters
        """
        state_paths = self.config.data_paths[f'{group}_states']
        group_data = {}
        
        try:
            # Load model parameters
            with open(state_paths['model'], 'rb') as f:
                group_data['parameters'] = pickle.load(f)
                
            # Load state sequences - shape should be (n_subj, n_timepoints)
            state_seq = np.load(state_paths['sequences'])
            self.logger.info(f"Original {group} state sequence shape: {state_seq.shape}")
            self.logger.info(f"Unique states: {np.unique(state_seq)}")
            
            # Calculate probability of each state at each timepoint
            n_states = self.config.n_states  # Should be 3
            n_timepoints = state_seq.shape[1]
            n_subjects = state_seq.shape[0]
            
            self.logger.info(f"Computing probabilities for {n_states} states across {n_timepoints} timepoints from {n_subjects} subjects")
            
            # Initialize probability array
            state_probs = np.zeros((n_states, n_timepoints))
            
            # For each state
            for state in range(n_states):
                # Count how many subjects are in this state at each timepoint
                state_probs[state] = (state_seq == state).sum(axis=0) / n_subjects
                
            group_data['state_probabilities'] = state_probs
            
            # Store raw sequences for reference
            group_data['state_sequence'] = state_seq
            
            # Verify probabilities sum to 1
            prob_sums = state_probs.sum(axis=0)
            self.logger.info(f"Probability sums - mean: {prob_sums.mean():.3f}, std: {prob_sums.std():.3f}")
            
            # Log probability stats for each state
            for state in range(n_states):
                stats = {
                    'mean': state_probs[state].mean(),
                    'std': state_probs[state].std(),
                    'min': state_probs[state].min(),
                    'max': state_probs[state].max()
                }
                self.logger.info(f"State {state} probability stats: {stats}")
            
            return group_data
            
        except Exception as e:
            self.logger.error(f"Error loading {group} states: {e}")
            raise
        
    def _verify_alignment(self, annotations_df: pd.DataFrame, 
                         affair_states: dict, paranoia_states: dict) -> None:
        """Verify alignment of all data"""
        expected_length = len(annotations_df)
        
        # Check affair states
        affair_length = affair_states['state_sequence'].shape[1]
        if affair_length != expected_length:
            self.logger.warning(
                f"Length mismatch: affair states ({affair_length}) "
                f"vs annotations ({expected_length})"
            )
            
        # Check paranoia states
        paranoia_length = paranoia_states['state_sequence'].shape[1]
        if paranoia_length != expected_length:
            self.logger.warning(
                f"Length mismatch: paranoia states ({paranoia_length}) "
                f"vs annotations ({expected_length})"
            )

class TRMatrix:
    """Class for creating and managing the TR-by-TR analysis matrix"""
    def __init__(self, annotations_df: pd.DataFrame, 
                 affair_states: dict, paranoia_states: dict, 
                 config: Config, logger: logging.Logger):
        self.annotations_df = annotations_df
        self.affair_states = affair_states
        self.paranoia_states = paranoia_states
        self.config = config  # Add config parameter
        self.logger = logger
        self.tr_matrix = None
        
    def create_matrix(self) -> pd.DataFrame:
        """
        Create combined TR-by-TR matrix with all features
        
        Returns:
            DataFrame with all features and state probabilities
        """
        try:
            # Start with annotations
            self.tr_matrix = self.annotations_df.copy()
            
            # Add state probabilities for both groups
            for group in ['affair', 'paranoia']:
                states_data = getattr(self, f'{group}_states')
                if 'state_probabilities' in states_data:
                    state_probs = states_data['state_probabilities']
                    
                    # Add each state's probability as a column
                    for state in range(self.config.n_states):
                        self.tr_matrix[f'{group}_state_{state}'] = state_probs[state]
                else:
                    self.logger.warning(f"No state probabilities found for {group}")
            
            # Process linguistic features
            self._process_linguistic_features()
            
            # Add binary indicators
            self._add_binary_indicators()
            
            # Create onset markers
            self._create_onset_markers()
            
            self.logger.info(f"Created TR matrix with shape: {self.tr_matrix.shape}")
            
            # Verify state probability columns
            for group in ['affair', 'paranoia']:
                state_cols = [col for col in self.tr_matrix.columns 
                            if col.startswith(f'{group}_state_')]
                self.logger.info(f"{group} state columns: {state_cols}")
                
                if state_cols:
                    probs = self.tr_matrix[state_cols].values
                    self.logger.info(f"{group} probability stats:")
                    self.logger.info(f"  Range: [{probs.min():.3f}, {probs.max():.3f}]")
                    self.logger.info(f"  Sum range: [{probs.sum(axis=1).min():.3f}, "
                                   f"{probs.sum(axis=1).max():.3f}]")
            
            return self.tr_matrix
            
        except Exception as e:
            self.logger.error(f"Error creating TR matrix: {e}")
            raise
            
    def _add_state_probabilities(self, group: str) -> None:
        """Add state probabilities for a group"""
        states = getattr(self, f'{group}_states')
        state_seq = states['state_sequence']
        
        n_states = state_seq.shape[0]  # Number of states
        
        for state in range(n_states):
            # Calculate occupancy probability across subjects
            state_prob = (state_seq == state).mean(axis=0)
            self.tr_matrix[f'{group}_state_{state}'] = state_prob
            
    def _process_linguistic_features(self) -> None:
        """Process linguistic features"""
        # Convert string columns to numeric
        for feat in ['n_verb', 'n_noun', 'n_adj', 'n_adv']:
            if feat in self.tr_matrix:
                self.tr_matrix[feat] = pd.to_numeric(
                    self.tr_matrix[feat], 
                    errors='coerce'
                ).fillna(0)
                
        # Create binary indicators
        for feat in ['verb', 'noun', 'adj', 'adv']:
            count_col = f'n_{feat}'
            binary_col = f'has_{feat}'
            if count_col in self.tr_matrix:
                self.tr_matrix[binary_col] = self.tr_matrix[count_col] > 0
                
    def _add_binary_indicators(self) -> None:
        """Add binary indicators for various conditions"""
        # Speaker indicators
        speaker_cols = ['arthur_speaking', 'lee_speaking', 'girl_speaking']
        for col in speaker_cols:
            if col in self.tr_matrix:
                self.tr_matrix[col] = self.tr_matrix[col].astype(bool)
                
        # Add lee_girl_together if needed
        if 'lee_girl_together' not in self.tr_matrix and 'lee_speaking' in self.tr_matrix:
            self.tr_matrix['lee_girl_together'] = (
                self.tr_matrix['lee_speaking'] & 
                self.tr_matrix['girl_speaking']
            )
            
    def _create_onset_markers(self) -> None:
        """Create onset markers for various features"""
        features = [
            'lee_girl_together', 
            'arthur_speaking',
            'has_verb', 
            'has_noun', 
            'has_adj', 
            'has_adv'
        ]
        
        for feature in features:
            if feature in self.tr_matrix:
                current_feature = self.tr_matrix[feature].astype(bool)
                shifted_feature = current_feature.shift(1)
                shifted_feature = shifted_feature.fillna(False)
                
                onset_col = f'{feature}_onset'
                self.tr_matrix[onset_col] = current_feature & ~shifted_feature

class BrainStateAnalysis:
    """Main analysis class"""
    def __init__(self, tr_matrix: pd.DataFrame, config: Config, logger: logging.Logger):
        self.tr_matrix = tr_matrix
        self.config = config
        self.logger = logger
        self.results = {}
        
        # Define feature sets
        self.feature_sets = {
            'set1': {
                'features': ['lee_girl_together', 'has_verb'],
                'name': 'Lee-Girl & Verb Interaction'
            },
            'set2': {
                'features': ['arthur_speaking', 'has_adj', 'has_adv'],
                'name': 'Arthur & Modifier Interaction'
            }
        }
        
    def run_analysis(self) -> Dict:
        """Run complete analysis pipeline"""
        try:
            # 1. General feature exploration
            self.logger.info("Running general feature exploration")
            self.results['general'] = self._analyze_general_features()
            
            # 2. Feature set specific analyses
            for set_name, set_info in self.feature_sets.items():
                self.logger.info(f"Analyzing {set_info['name']}")
                self.results[set_name] = self._analyze_feature_set(
                    set_info['features'],
                    set_name
                )
                
            # 3. Apply multiple comparison correction
            self.logger.info("Applying multiple comparison correction")
            self._correct_multiple_comparisons()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {e}")
            raise
            
    def _analyze_general_features(self) -> Dict:
        """Analyze general content features"""
        general_results = {}
        
        # Define feature groups
        feature_groups = {
            'dialog': ['is_dialog'],
            'speakers': ['arthur_speaking', 'lee_speaking', 'girl_speaking'],
            'linguistic': {
                'verb': ['has_verb', 'n_verb'],
                'noun': ['has_noun', 'n_noun'],
                'adj': ['has_adj', 'n_adj'],
                'adv': ['has_adv', 'n_adv']
            }
        }
        
        # Analyze each feature group
        for group_name, features in feature_groups.items():
            if isinstance(features, dict):
                group_results = {}
                for subgroup, subfeatures in features.items():
                    group_results[subgroup] = self._analyze_feature_group(subfeatures)
                general_results[group_name] = group_results
            else:
                general_results[group_name] = self._analyze_feature_group(features)
                
        return general_results
        
    def _analyze_feature_group(self, features: List[str]) -> Dict:
        """Analyze a group of related features"""
        results = {}
        
        for feature in features:
            if feature not in self.tr_matrix.columns:
                self.logger.warning(f"Feature {feature} not found in TR matrix")
                continue
                
            # Get feature type
            if self.tr_matrix[feature].dtype == bool:
                results[feature] = self._analyze_binary_feature(feature)
            else:
                results[feature] = self._analyze_continuous_feature(feature)
                
        return results
        
    def _analyze_binary_feature(self, feature: str) -> Dict:
        """Analyze binary feature's relationship with states"""
        feature_results = {}
        
        for group in ['affair', 'paranoia']:
            state_cols = [col for col in self.tr_matrix.columns 
                        if col.startswith(f'{group}_state_')]
            
            group_results = {}
            for state in state_cols:
                try:
                    # Convert feature to numeric and ensure it's a Series
                    X = pd.Series(self.tr_matrix[feature].astype(int), name=feature)
                    y = self.tr_matrix[state]
                    
                    # Add constant term properly
                    X_matrix = sm.add_constant(X, has_constant='add')
                    
                    # Use the new Logit link class
                    family = sm.families.Binomial(link=sm.families.links.Logit())
                    
                    # Fit model with regularization
                    model = sm.GLM(y, X_matrix, family=family)
                    results = model.fit_regularized(alpha=0.1)
                    
                    # Use .iloc[] for positional indexing
                    group_results[state] = {
                        'coefficient': results.params.iloc[1],  # Use .iloc[] instead of [1]
                        'std_err': results.bse.iloc[1] if hasattr(results, 'bse') else None,
                        'p_value': results.pvalues.iloc[1] if hasattr(results, 'pvalues') else None,
                        'mean_when_true': y[X == 1].mean(),
                        'mean_when_false': y[X == 0].mean()
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in binary analysis of {feature} for {state}: {str(e)}")
                    group_results[state] = {
                        'error': str(e),
                        'mean_when_true': y[X == 1].mean() if len(y[X == 1]) > 0 else None,
                        'mean_when_false': y[X == 0].mean() if len(y[X == 0]) > 0 else None
                    }
                    
            feature_results[group] = group_results
            
        return feature_results
        
    def _analyze_continuous_feature(self, feature: str) -> Dict:
        """Analyze continuous feature's relationship with states"""
        feature_results = {}
        
        for group in ['affair', 'paranoia']:
            state_cols = [col for col in self.tr_matrix.columns 
                        if col.startswith(f'{group}_state_')]
            
            group_results = {}
            for state in state_cols:
                try:
                    X = pd.DataFrame({feature: self.tr_matrix[feature]})
                    # Only standardize if there's variation in the data
                    if X[feature].std() > 0:
                        X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()
                    
                    X = sm.add_constant(X)
                    y = self.tr_matrix[state]
                    
                    # Only calculate correlation if we have variation
                    if X[feature].std() > 0 and y.std() > 0:
                        correlation = stats.pearsonr(X[feature], y)[0]
                    else:
                        correlation = None
                    
                    model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
                    results = model.fit_regularized(alpha=0.1)
                    
                    group_results[state] = {
                        'coefficient': results.params[feature],
                        'std_err': getattr(results, 'bse', {}).get(feature),
                        'p_value': getattr(results, 'pvalues', {}).get(feature),
                        'correlation': correlation
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in continuous analysis of {feature} for {state}: {str(e)}")
                    group_results[state] = None
                    
            feature_results[group] = group_results
        
        return feature_results
    
    def _analyze_feature_set(self, features: List[str], set_name: str) -> Dict:
        """Analyze specific feature set interactions"""
        if set_name == 'set1':
            return self._analyze_lee_girl_verb()
        elif set_name == 'set2':
            return self._analyze_arthur_modifier()
        else:
            self.logger.error(f"Unknown feature set: {set_name}")
            return None
            
    def _analyze_lee_girl_verb(self) -> Dict:
        """Analyze lee_girl_together & verb interactions"""
        try:
            results = {
                'regression': {},
                'state_patterns': {},
                'temporal_dynamics': {}
            }
            
            # Create interaction term
            self.tr_matrix['lg_verb_interaction'] = (
                self.tr_matrix['lee_girl_together'] & 
                self.tr_matrix['has_verb']
            ).astype(int)
            
            # Analyze for each group
            for group in ['affair', 'paranoia']:
                state_cols = [col for col in self.tr_matrix.columns 
                            if col.startswith(f'{group}_state_')]
                
                # Regression analysis
                results['regression'][group] = self._run_feature_set1_regression(state_cols)
                
                # State patterns analysis
                results['state_patterns'][group] = self._analyze_feature_set1_patterns(state_cols)
                
                # Temporal dynamics analysis
                results['temporal_dynamics'][group] = self._analyze_feature_set1_dynamics(state_cols)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in lee_girl_verb analysis: {e}")
            raise
            
    def _run_feature_set1_regression(self, state_cols: List[str]) -> Dict:
        """Run regression analysis for feature set 1 with regularization"""
        regression_results = {}
        
        for state in state_cols:
            try:
                # Prepare predictors
                X = pd.DataFrame({
                    'lee_girl': self.tr_matrix['lee_girl_together'].astype(int),
                    'verb': self.tr_matrix['has_verb'].astype(int),
                    'interaction': self.tr_matrix['lg_verb_interaction'].astype(int)
                })
                X = sm.add_constant(X)
                y = self.tr_matrix[state]
                
                # Always use regularization
                family = sm.families.Binomial(link=sm.families.links.Logit())
                model = sm.GLM(y, X, family=family)
                fit_results = model.fit_regularized(alpha=1.0)
                
                # For regularized results, we only store coefficients
                regression_results[state] = {
                    'coefficients': dict(zip(X.columns, fit_results.params)),
                    'regularized': True
                }
                    
            except Exception as e:
                self.logger.error(f"Error in regression for {state}: {str(e)}")
                regression_results[state] = None
                
        return regression_results
            
    def _analyze_feature_set1_patterns(self, state_cols: List[str]) -> Dict:
        """Analyze state patterns for different conditions in feature set 1"""
        patterns = {}
        
        # Define all possible conditions
        conditions = {
            'lg_verb': (self.tr_matrix['lee_girl_together'] & 
                    self.tr_matrix['has_verb']),
            'lg_noverb': (self.tr_matrix['lee_girl_together'] & 
                        ~self.tr_matrix['has_verb']),
            'nlg_verb': (~self.tr_matrix['lee_girl_together'] & 
                        self.tr_matrix['has_verb']),
            'nlg_noverb': (~self.tr_matrix['lee_girl_together'] & 
                        ~self.tr_matrix['has_verb'])
        }
        
        for condition_name, mask in conditions.items():
            condition_patterns = {}
            for state in state_cols:
                try:
                    # Get data for this condition and state
                    state_data = self.tr_matrix.loc[mask, state]
                    
                    # Only calculate statistics if we have valid data
                    if len(state_data) > 0 and not np.isnan(state_data).all():
                        mean = state_data.mean()
                        std = state_data.std()
                        n = len(state_data)
                        
                        # Calculate confidence interval if we have enough valid data
                        if n > 1 and std > 0:
                            ci = stats.t.interval(
                                0.95, 
                                n-1,
                                loc=mean,
                                scale=std/np.sqrt(n)
                            )
                        else:
                            ci = (np.nan, np.nan)
                            
                        condition_patterns[state] = {
                            'mean': mean,
                            'std': std,
                            'n': n,
                            'ci': ci,
                            'raw_data': state_data.values  # Store raw data for potential reanalysis
                        }
                    else:
                        condition_patterns[state] = {
                            'mean': np.nan,
                            'std': np.nan,
                            'n': 0,
                            'ci': (np.nan, np.nan),
                            'raw_data': np.array([])
                        }
                        
                except Exception as e:
                    self.logger.error(
                        f"Error analyzing patterns for {condition_name}, {state}: {str(e)}"
                    )
                    condition_patterns[state] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'n': 0,
                        'ci': (np.nan, np.nan),
                        'raw_data': np.array([]),
                        'error': str(e)
                    }
                    
            patterns[condition_name] = condition_patterns
            
        # Add metadata about the analysis
        patterns['metadata'] = {
            'total_timepoints': len(self.tr_matrix),
            'conditions_analyzed': list(conditions.keys()),
            'states_analyzed': state_cols,
            'timestamp': datetime.now().isoformat()
        }
        
        return patterns
        
    def _analyze_feature_set1_dynamics(self, state_cols: List[str], window_size: int = 5) -> Dict:
        """
        Analyze temporal dynamics for feature set 1
        
        Args:
            state_cols: List of state column names to analyze
            window_size: Number of TRs before/after event to include (default=5)
        
        Returns:
            Dict containing temporal dynamics for different events and conditions
        """
        try:
            dynamics = {
                'window_size': window_size,  # Store window size in results
                'state_cols': state_cols,    # Store analyzed states
                'time_points': np.arange(-window_size, window_size + 1)  # Time axis
            }
            
            # 1. Analyze dynamics around lee_girl_together onsets
            if 'lee_girl_together_onset' in self.tr_matrix:
                dynamics['lee_girl'] = self._compute_temporal_patterns(
                    'lee_girl_together_onset',
                    state_cols,
                    window_size
                )
                
            # 2. Analyze dynamics around verb onsets
            if 'has_verb_onset' in self.tr_matrix:
                dynamics['verb'] = self._compute_temporal_patterns(
                    'has_verb_onset',
                    state_cols,
                    window_size
                )
                
            # 3. Analyze dynamics around co-occurrence (both lee_girl and verb)
            co_occurrence = (
                self.tr_matrix['lee_girl_together_onset'] & 
                self.tr_matrix['has_verb']
            )
            if co_occurrence.any():
                dynamics['co_occurrence'] = self._compute_temporal_patterns(
                    co_occurrence,
                    state_cols,
                    window_size
                )
                
            # 4. Analyze sequential patterns (verb following lee_girl within 2 TRs)
            if 'lee_girl_together_onset' in self.tr_matrix and 'has_verb_onset' in self.tr_matrix:
                seq_window = 2  # Look for verbs within 2 TRs after lee_girl onset
                sequential_events = []
                
                lg_onsets = np.where(self.tr_matrix['lee_girl_together_onset'])[0]
                verb_onsets = set(np.where(self.tr_matrix['has_verb_onset'])[0])
                
                for lg_onset in lg_onsets:
                    # Check if there's a verb onset within the next 2 TRs
                    for i in range(1, seq_window + 1):
                        if lg_onset + i in verb_onsets:
                            sequential_events.append(lg_onset)
                            break
                
                if sequential_events:
                    seq_mask = np.zeros(len(self.tr_matrix), dtype=bool)
                    seq_mask[sequential_events] = True
                    dynamics['sequential'] = self._compute_temporal_patterns(
                        seq_mask,
                        state_cols,
                        window_size
                    )
            
            # 5. Add analysis metadata
            dynamics['metadata'] = {
                'total_timepoints': len(self.tr_matrix),
                'analysis_timestamp': datetime.now().isoformat(),
                'conditions_analyzed': [k for k in dynamics.keys() if k not in 
                                    ['window_size', 'state_cols', 'time_points', 'metadata']],
                'sequence_window': seq_window if 'sequential' in dynamics else None
            }
            
            # 6. Add event counts
            dynamics['event_counts'] = {
                'lee_girl_onsets': self.tr_matrix['lee_girl_together_onset'].sum()
                    if 'lee_girl_together_onset' in self.tr_matrix else 0,
                'verb_onsets': self.tr_matrix['has_verb_onset'].sum()
                    if 'has_verb_onset' in self.tr_matrix else 0,
                'co_occurrences': co_occurrence.sum() if co_occurrence.any() else 0,
                'sequential_events': len(sequential_events) if 'sequential' in dynamics else 0
            }
            
            # 7. Validate results
            for key, value in dynamics.items():
                if key not in ['window_size', 'state_cols', 'time_points', 'metadata', 'event_counts']:
                    if value is None:
                        self.logger.warning(f"No valid patterns found for {key}")
                        
            return dynamics
            
        except Exception as e:
            self.logger.error(f"Error in feature set 1 dynamics analysis: {e}")
            return {
                'window_size': window_size,
                'error': str(e),
                'state_cols': state_cols
            }
        
    def _analyze_arthur_modifier(self) -> Dict:
        """Analyze arthur_speaking & modifier interactions"""
        try:
            results = {
                'regression': {},
                'state_patterns': {},
                'temporal_dynamics': {}
            }
            
            # Create modifier categories
            self.tr_matrix['modifier_cat'] = 'none'
            self.tr_matrix.loc[self.tr_matrix['has_adj'], 'modifier_cat'] = 'adj'
            self.tr_matrix.loc[self.tr_matrix['has_adv'], 'modifier_cat'] = 'adv'
            self.tr_matrix.loc[
                self.tr_matrix['has_adj'] & self.tr_matrix['has_adv'], 
                'modifier_cat'
            ] = 'both'
            
            # Analyze for each group
            for group in ['affair', 'paranoia']:
                state_cols = [col for col in self.tr_matrix.columns 
                            if col.startswith(f'{group}_state_')]
                
                # Regression analysis
                results['regression'][group] = self._run_feature_set2_regression(state_cols)
                
                # State patterns analysis
                results['state_patterns'][group] = self._analyze_feature_set2_patterns(state_cols)
                
                # Temporal dynamics analysis
                results['temporal_dynamics'][group] = self._analyze_feature_set2_dynamics(state_cols)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in arthur_modifier analysis: {e}")
            raise
            
    def _run_feature_set2_regression(self, state_cols: List[str]) -> Dict:
        """Run regression analysis for feature set 2"""
        regression_results = {}
        
        for state in state_cols:
            try:
                # Create design matrix with proper types
                X = pd.get_dummies(
                    pd.DataFrame({
                        'arthur': self.tr_matrix['arthur_speaking'].astype(int),
                        'modifier': self.tr_matrix['modifier_cat']
                    }),
                    drop_first=True
                ).astype(float)  # Ensure float type
                
                X = sm.add_constant(X)
                y = self.tr_matrix[state]
                
                family = sm.families.Binomial(link=sm.families.links.Logit())
                model = sm.GLM(y, X, family=family)
                
                try:
                    fit_results = model.fit()
                    regression_results[state] = {
                        'coefficients': fit_results.params.to_dict(),
                        'p_values': fit_results.pvalues.to_dict()
                    }
                except:
                    fit_results = model.fit_regularized(alpha=0.5)
                    regression_results[state] = {
                        'coefficients': fit_results.params.to_dict(),
                        'regularized': True
                    }
                    
            except Exception as e:
                self.logger.error(f"Error in regression for {state}: {str(e)}")
                regression_results[state] = None
                
        return regression_results
        
    def _analyze_feature_set2_patterns(self, state_cols: List[str]) -> Dict:
        """Analyze state patterns for different conditions in feature set 2"""
        patterns = {}
        conditions = {
            'arthur_adj': (self.tr_matrix['arthur_speaking'] & 
                         self.tr_matrix['has_adj']),
            'arthur_adv': (self.tr_matrix['arthur_speaking'] & 
                         self.tr_matrix['has_adv']),
            'arthur_both': (self.tr_matrix['arthur_speaking'] & 
                          self.tr_matrix['has_adj'] & 
                          self.tr_matrix['has_adv']),
            'arthur_nomod': (self.tr_matrix['arthur_speaking'] & 
                           ~self.tr_matrix['has_adj'] & 
                           ~self.tr_matrix['has_adv']),
            'other': (~self.tr_matrix['arthur_speaking'])
        }
        
        for condition_name, mask in conditions.items():
            condition_patterns = {}
            for state in state_cols:
                state_data = self.tr_matrix.loc[mask, state]
                condition_patterns[state] = {
                    'mean': state_data.mean(),
                    'std': state_data.std(),
                    'n': len(state_data),
                    'ci': stats.t.interval(
                        0.95, 
                        len(state_data)-1,
                        loc=state_data.mean(),
                        scale=state_data.std()/np.sqrt(len(state_data))
                    )
                }
            patterns[condition_name] = condition_patterns
            
        return patterns
        
    def _analyze_feature_set2_dynamics(self, state_cols: List[str], 
                                     window_size: int = 5) -> Dict:
        """Analyze temporal dynamics for feature set 2"""
        dynamics = {}
        
        # Analyze dynamics around arthur_speaking onsets
        if 'arthur_speaking_onset' in self.tr_matrix:
            dynamics['arthur'] = self._compute_temporal_patterns(
                'arthur_speaking_onset',
                state_cols,
                window_size
            )
            
        # Analyze dynamics around modifier onsets
        for modifier in ['adj', 'adv']:
            onset_col = f'has_{modifier}_onset'
            if onset_col in self.tr_matrix:
                dynamics[modifier] = self._compute_temporal_patterns(
                    onset_col,
                    state_cols,
                    window_size
                )
        
        # Analyze dynamics around arthur with modifier
        arthur_with_mod = (
            self.tr_matrix['arthur_speaking_onset'] & 
            (self.tr_matrix['has_adj'] | self.tr_matrix['has_adv'])
        )
        if arthur_with_mod.any():
            dynamics['arthur_with_modifier'] = self._compute_temporal_patterns(
                arthur_with_mod,
                state_cols,
                window_size
            )
            
        return dynamics
        
    def _compute_temporal_patterns(self, event_indicator: Union[str, pd.Series], 
                                 state_cols: List[str], 
                                 window_size: int) -> Dict:
        """
        Compute temporal patterns around events
        
        Args:
            event_indicator: Either column name or boolean series indicating events
            state_cols: List of state column names to analyze
            window_size: Number of TRs before/after event to include
            
        Returns:
            Dict containing temporal patterns statistics
        """
        try:
            # Get events series
            if isinstance(event_indicator, str):
                events = self.tr_matrix[event_indicator]
            else:
                events = event_indicator
            
            # Find event indices
            event_indices = np.where(events)[0]
            patterns = []
            
            # For each event
            for idx in event_indices:
                # Check if we have enough data before/after
                if idx >= window_size and idx < len(self.tr_matrix) - window_size:
                    # Extract window of data around event
                    window_data = self.tr_matrix.loc[
                        idx - window_size:idx + window_size,
                        state_cols
                    ]
                    patterns.append(window_data.values)
                    
            # If we found any valid patterns
            if patterns:
                patterns = np.array(patterns)
                self.logger.info(
                    f"Raw patterns shape: {patterns.shape}\n"
                    f"State cols: {state_cols}"
                )
                
                # Calculate statistics
                mean_pattern = np.mean(patterns, axis=0)
                std_pattern = np.std(patterns, axis=0)
                
                self.logger.info(
                    f"Computed mean shape: {mean_pattern.shape}\n"
                    f"Computed std shape: {std_pattern.shape}"
                )
                return {
                    'mean': mean_pattern,
                    'std': std_pattern,
                    'n': len(patterns),
                    'window_size': window_size,
                    'time_points': np.arange(-window_size, window_size + 1),
                    'state_cols': state_cols
                }
            else:
                self.logger.warning(
                    f"No valid patterns found for given event indicator "
                    f"with window size {window_size}"
                )
                return None
                
        except Exception as e:
            self.logger.error(
                f"Error computing temporal patterns: {e}"
            )
            return None
            
    def _correct_multiple_comparisons(self) -> None:
        """Apply FDR correction across all statistical tests"""
        try:
            # Collect all p-values and their locations
            all_p_values = []
            p_value_locations = []
            
            def collect_p_values(d: Dict, path: List[str] = []):
                """Recursively collect p-values from nested dictionary"""
                for key, value in d.items():
                    if key == 'p_values' and isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, (float, np.floating)):
                                all_p_values.append(v)
                                p_value_locations.append(path + [k])
                    elif isinstance(value, dict):
                        collect_p_values(value, path + [key])
            
            # Collect p-values from all analyses
            collect_p_values(self.results)
            
            if all_p_values:
                # Apply FDR correction
                _, corrected_p_values, _, _ = multipletests(
                    all_p_values,
                    alpha=0.05,
                    method=self.config.correction_method
                )
                
                # Update results with corrected p-values
                for location, corrected_p in zip(p_value_locations, corrected_p_values):
                    current = self.results
                    for key in location[:-1]:
                        current = current[key]
                    if 'p_values_corrected' not in current:
                        current['p_values_corrected'] = {}
                    current['p_values_corrected'][location[-1]] = corrected_p
                    
                self.logger.info(f"Applied {self.config.correction_method} correction to {len(all_p_values)} p-values")
                
        except Exception as e:
            self.logger.error(f"Error in multiple comparison correction: {e}")
            raise

class BrainStateVisualization:
    def __init__(self, tr_matrix: pd.DataFrame, results: Dict, config: Config, logger: logging.Logger):
        self.tr_matrix = tr_matrix
        self.results = results
        self.config = config
        self.logger = logger
        
        # Set style
        plt.style.use('seaborn-v0_8')
        self.colors = self.config.viz_params['colors']
        
        # Create feature set color mappings
        self.feature_colors = {
            'set1': {
                'lee_girl': '#2ecc71',    # green
                'verb': '#e74c3c'         # red
            },
            'set2': {
                'arthur': '#3498db',      # blue
                'adj': '#f1c40f',         # yellow
                'adv': '#9b59b6'          # purple
            }
        }
        
    def plot_state_probabilities(self, group: str) -> None:
        """
        Plot state probability distributions over time
        
        Args:
            group: 'affair' or 'paranoia'
        """
        try:
            # Get state columns
            state_cols = [col for col in self.tr_matrix.columns 
                        if col.startswith(f'{group}_state_')]
            state_cols.sort()  # Ensure proper ordering
            
            if not state_cols:
                self.logger.error(f"No state columns found for group {group}")
                return
                
            # Create figure
            fig, ax = plt.subplots(figsize=self.config.figsize['large'])
            
            # Plot each state
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different color for each state
            for i, state in enumerate(state_cols):
                data = self.tr_matrix[state]
                ax.plot(
                    range(len(data)),
                    data,
                    label=f'State {i}',
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.5
                )
            
            # Customize plot
            ax.set_xlabel('Time (TRs)')
            ax.set_ylabel('State Probability')
            ax.set_title(f'{group.title()} Group State Probabilities')
            
            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits
            ax.set_ylim(-0.05, 1.05)
            
            # Add x-axis ticks
            ax.set_xticks(np.arange(0, len(self.tr_matrix), 50))
            
            plt.tight_layout()
            plt.savefig(
                self.config.fig_dir / f'{group}_state_probabilities.png',
                dpi=self.config.dpi
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting state probabilities: {e}")
            plt.close()
                
    def plot_state_carpet(self, group: str) -> None:
        """
        Create carpet plot of state probabilities
        
        Args:
            group: 'affair' or 'paranoia'
        """
        try:
            # Get state columns
            state_cols = [col for col in self.tr_matrix.columns 
                        if col.startswith(f'{group}_state_')]
            state_cols.sort()  # Ensure proper ordering
            
            if not state_cols:
                self.logger.error(f"No state columns found for group {group}")
                return
                
            # Create figure
            fig, ax = plt.subplots(figsize=self.config.figsize['large'])
            
            # Extract state data
            state_data = self.tr_matrix[state_cols].values.T
            
            # Log data properties
            self.logger.info(f"State data shape: {state_data.shape}")
            self.logger.info(f"Value range: [{state_data.min():.3f}, {state_data.max():.3f}]")
            
            # Create carpet plot
            im = ax.imshow(
                state_data,
                aspect='auto',
                cmap='viridis',
                interpolation='nearest',
                vmin=0,
                vmax=1
            )
            
            # Customize plot
            ax.set_xlabel('Time (TRs)')
            ax.set_ylabel('States')
            ax.set_yticks(range(len(state_cols)))
            ax.set_yticklabels([f'State {i}' for i in range(len(state_cols))])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability')
            
            ax.set_title(f'{group.title()} Group State Patterns')
            
            # Add time markers
            ax.set_xticks(np.arange(0, state_data.shape[1], 50))
            
            plt.tight_layout()
            plt.savefig(
                self.config.fig_dir / f'{group}_state_carpet.png',
                dpi=self.config.dpi
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting state carpet: {e}")
            plt.close()
            
    def plot_regression_results(self, feature_set: str) -> None:
        try:
            if feature_set not in self.results:
                self.logger.error(f"No results found for feature set {feature_set}")
                return
                
            results = self.results[feature_set]['regression']
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=self.config.figsize['double'])
            
            for i, group in enumerate(['affair', 'paranoia']):
                if group not in results:
                    continue
                    
                group_results = results[group]
                
                # Collect coefficients
                coef_data = []
                states = []
                features = []
                
                for state, state_results in group_results.items():
                    if state_results is None or 'coefficients' not in state_results:
                        continue
                        
                    for feature, coef in state_results['coefficients'].items():
                        if feature == 'const':
                            continue
                        coef_data.append(coef)
                        states.append(state.split('_')[-1])
                        features.append(feature)
                
                if not coef_data:  # Skip if no data
                    continue
                    
                # Create coefficient matrix
                df = pd.DataFrame({
                    'State': states,
                    'Feature': features,
                    'Coefficient': coef_data
                })
                
                pivot_data = df.pivot(
                    index='Feature',
                    columns='State',
                    values='Coefficient'
                )
                
                if not pivot_data.empty:  # Only plot if we have data
                    sns.heatmap(
                        pivot_data,
                        ax=axes[i],
                        cmap='RdBu_r',
                        center=0,
                        annot=True,
                        fmt='.2f',
                        cbar_kws={'label': 'Coefficient'}
                    )
                    axes[i].set_title(f'{group.title()} Group')
                
            plt.suptitle(f'Regression Results - Feature Set {feature_set}')
            plt.tight_layout()
            plt.savefig(
                self.config.fig_dir / f'regression_results_set{feature_set}.png',
                dpi=self.config.dpi
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting regression results: {e}")
            plt.close()

    # Continue BrainStateVisualization class
    def plot_state_feature_transitions(self, feature_set: str) -> None:
        """
        Plot state probabilities with feature transitions
        
        Args:
            feature_set: 'set1' or 'set2'
        """
        try:
            # Define features based on set
            if feature_set == 'set1':
                features = {
                    'lee_girl': 'lee_girl_together',
                    'verb': 'has_verb'
                }
                title = 'State Dynamics with Lee-Girl & Verb Features'
            else:  # set2
                features = {
                    'arthur': 'arthur_speaking',
                    'adj': 'has_adj',
                    'adv': 'has_adv'
                }
                title = 'State Dynamics with Arthur & Modifier Features'
            
            # Create figure with subplot grid
            fig = plt.figure(figsize=self.config.figsize['double'])
            gs = plt.GridSpec(4, 1, height_ratios=[3, 3, 0.5, 0.5])
            
            # Plot for each group
            for i, group in enumerate(['affair', 'paranoia']):
                ax_states = fig.add_subplot(gs[i])
                
                # Get state columns
                state_cols = [col for col in self.tr_matrix.columns 
                            if col.startswith(f'{group}_state_')]
                
                # Plot state probabilities
                for state in state_cols:
                    state_num = state.split('_')[-1]
                    ax_states.plot(
                        self.tr_matrix.index,
                        self.tr_matrix[state],
                        label=f'State {state_num}',
                        alpha=0.7
                    )
                
                ax_states.set_ylabel('State Probability')
                ax_states.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_states.set_title(f'{group.title()} Group')
                
                # Remove x-ticks for top plot
                if i == 0:
                    ax_states.set_xticks([])
            
            # Plot feature occurrences
            ax_features = fig.add_subplot(gs[2:])
            
            # Plot each feature as vertical lines
            for feature_name, feature_col in features.items():
                feature_color = self.feature_colors[feature_set][feature_name]
                occurrences = np.where(self.tr_matrix[feature_col])[0]
                
                for idx in occurrences:
                    ax_features.axvline(
                        x=idx,
                        color=feature_color,
                        alpha=0.3,
                        linewidth=1
                    )
            
            # Add feature legend
            feature_patches = [
                plt.Rectangle((0,0), 1, 1,
                            facecolor=self.feature_colors[feature_set][f],
                            label=f.replace('_', ' ').title())
                for f in features.keys()
            ]
            ax_features.legend(
                handles=feature_patches,
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
            
            ax_features.set_xlabel('Time (TRs)')
            ax_features.set_ylabel('Features')
            
            # Set consistent x-axis limits
            xlim = (0, len(self.tr_matrix))
            ax_features.set_xlim(xlim)
            for ax in fig.axes[:-1]:
                ax.set_xlim(xlim)
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(
                self.config.fig_dir / f'state_transitions_set{feature_set}.png',
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting state feature transitions: {e}")
            plt.close()
            
    def plot_temporal_dynamics(self, feature_set: str) -> None:
        """
        Plot temporal dynamics around feature onsets with separate state patterns
        
        Args:
            feature_set: 'set1' or 'set2'
        """
        try:
            if feature_set not in self.results:
                self.logger.error(f"No results found for feature set {feature_set}")
                return
                
            dynamics = self.results[feature_set]['temporal_dynamics']
            
            # Create figure and subplot grid
            for group in ['affair', 'paranoia']:
                if group not in dynamics:
                    continue
                    
                group_dynamics = dynamics[group]
                
                # Get valid features (skip metadata keys)
                features = [f for f in group_dynamics.keys() 
                        if f not in ['window_size', 'time_points', 'state_cols', 
                                    'metadata', 'event_counts']]
                
                if not features:
                    self.logger.warning(f"No features to plot for {group}")
                    continue
                    
                n_features = len(features)
                fig = plt.figure(figsize=(15, 5*n_features))
                gs = plt.GridSpec(n_features, 1)
                
                # Create title based on feature set
                if feature_set == 'set1':
                    title = f'Temporal Dynamics - Lee-Girl & Verb Interactions ({group.title()} Group)'
                else:
                    title = f'Temporal Dynamics - Arthur & Modifier Interactions ({group.title()} Group)'
                
                plt.suptitle(title, y=0.95, fontsize=14)
                
                # Get time points
                time_points = group_dynamics.get('time_points', 
                                            np.arange(-group_dynamics.get('window_size', 5),
                                                    group_dynamics.get('window_size', 5) + 1))
                
                # Plot each feature
                for idx, feature in enumerate(features):
                    pattern = group_dynamics[feature]
                    if pattern is None or 'mean' not in pattern:
                        continue
                        
                    ax = fig.add_subplot(gs[idx])
                    
                    # Convert to numpy arrays
                    mean_data = np.asarray(pattern['mean'])
                    std_data = np.asarray(pattern['std'])
                    
                    if mean_data.ndim != 2 or std_data.ndim != 2:
                        self.logger.warning(
                            f"Unexpected data dimensions for {feature}. "
                            f"mean_shape={mean_data.shape}, std_shape={std_data.shape}"
                        )
                        continue
                    
                    # Plot each state
                    state_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define colors for states
                    for state_idx in range(mean_data.shape[1]):
                        state_color = state_colors[state_idx % len(state_colors)]
                        
                        # Plot mean line
                        ax.plot(
                            time_points,
                            mean_data[:, state_idx],
                            label=f'State {state_idx}',
                            color=state_color,
                            linewidth=2
                        )
                        
                        # Add confidence interval
                        ax.fill_between(
                            time_points,
                            mean_data[:, state_idx] - std_data[:, state_idx],
                            mean_data[:, state_idx] + std_data[:, state_idx],
                            color=state_color,
                            alpha=0.2
                        )
                    
                    # Customize subplot
                    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
                    
                    ax.set_xlabel('Time (TRs)')
                    ax.set_ylabel('State Probability')
                    ax.set_title(f'{feature.replace("_", " ").title()}')
                    
                    # Add legend
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    
                    # Add grid
                    ax.grid(True, alpha=0.3)
                    
                    # Add event count if available
                    if 'event_counts' in group_dynamics:
                        event_count = group_dynamics['event_counts'].get(f'{feature}_events', 
                                                                    pattern.get('n', 0))
                        ax.text(0.02, 0.98, f'Events: {event_count}',
                            transform=ax.transAxes,
                            verticalalignment='top',
                            fontsize=10)
                
                # Adjust layout to prevent overlap
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # Save figure
                plt.savefig(
                    self.config.fig_dir / f'temporal_dynamics_{feature_set}_{group}.png',
                    dpi=self.config.dpi,
                    bbox_inches='tight'
                )
                
                # Close figure
                plt.close()
                
                self.logger.info(
                    f"Successfully created temporal dynamics plot for {group} group in set {feature_set}"
                )
                
        except Exception as e:
            self.logger.error(f"Error plotting temporal dynamics: {e}")
            plt.close()
        
    def plot_state_patterns(self, feature_set: str) -> None:
        """
        Plot state patterns for different conditions
        
        Args:
            feature_set: 'set1' or 'set2'
        """
        try:
            if feature_set not in self.results:
                self.logger.error(f"No results found for feature set {feature_set}")
                return
                
            patterns = self.results[feature_set]['state_patterns']
            self.logger.info(f"Feature set: {feature_set}")
            self.logger.info(f"Available top-level keys: {list(patterns.keys())}")

            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=self.config.figsize['double'])
            
            for i, group in enumerate(['affair', 'paranoia']):
                if group not in patterns:
                    continue
                    
                group_patterns = patterns[group]
                self.logger.info(f"Processing group {group}")
                
                # Filter out metadata and get valid conditions
                valid_conditions = [k for k in group_patterns.keys() if k != 'metadata']
                self.logger.info(f"Valid conditions: {valid_conditions}")
            
                if not valid_conditions:
                    self.logger.warning(f"No valid conditions found for {group}")
                    continue

                # Get states from first condition
                first_condition = valid_conditions[0]
                states = list(group_patterns[first_condition].keys())
                self.logger.info(f"States found: {states}")
                
                # Prepare data for plotting
                data = []
                for cond in valid_conditions:
                    for state in states:
                        if state in group_patterns[cond]:
                            data.append({
                                'Condition': cond,
                                'State': f'State {state.split("_")[-1]}',
                                'Probability': group_patterns[cond][state]['mean']
                            })
                
                if not data:
                    self.logger.warning(f"No data to plot for {group}")
                    continue
                    
                df = pd.DataFrame(data)
                pivot_table = df.pivot(
                    index='Condition',
                    columns='State',
                    values='Probability'
                )
                
                # Plot heatmap
                sns.heatmap(
                    pivot_table,
                    ax=axes[i],
                    cmap='viridis',
                    annot=True,
                    fmt='.3f'
                )
                axes[i].set_title(f'{group.title()} Group')
            
            plt.suptitle(f'State Patterns - Feature Set {feature_set}')
            plt.tight_layout()
            plt.savefig(
                self.config.fig_dir / f'state_patterns_set{feature_set}.png',
                dpi=self.config.dpi
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting state patterns: {e}")
            plt.close()
            
    def create_all_visualizations(self) -> None:
        """Create all visualizations"""
        try:
            self.logger.info("Creating visualizations...")
            
            # Basic state visualizations
            for group in ['affair', 'paranoia']:
                self.plot_state_probabilities(group)
                self.plot_state_carpet(group)
            
            # Feature set visualizations
            for feature_set in ['set1', 'set2']:
                self.plot_regression_results(feature_set)
                self.plot_state_feature_transitions(feature_set)
                self.plot_temporal_dynamics(feature_set)
                self.plot_state_patterns(feature_set)
                
            self.logger.info("Completed all visualizations")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            raise

def main():
    """Main analysis pipeline"""
    load_dotenv()
    base_dir = os.getenv('BASE_DIR')
    try:
        # 1. Setup
        # Load configuration
        config_path = os.path.join(base_dir, 'code', '11_brain_content_config.yml')
        config = Config(config_path)
        config.setup_directories()
        
        # Setup logging
        logger = setup_logging(config.output_dir)
        logger.info("Starting brain state analysis pipeline")
        
        # 2. Load Data
        logger.info("Loading data...")
        data_loader = DataLoader(config, logger)
        annotations_df, affair_states, paranoia_states = data_loader.load_data()
        
        # 3. Create TR Matrix
        logger.info("Creating TR matrix...")
        tr_processor = TRMatrix(
            annotations_df, 
            affair_states, 
            paranoia_states, 
            config,  # Pass config here
            logger
        )
        tr_matrix = tr_processor.create_matrix()
        
        # 4. Run Analysis
        logger.info("Running analysis pipeline...")
        analysis = BrainStateAnalysis(tr_matrix, config, logger)
        results = analysis.run_analysis()
        
        # 5. Create Visualizations
        logger.info("Creating visualizations...")
        visualization = BrainStateVisualization(tr_matrix, results, config, logger)
        visualization.create_all_visualizations()
        
        # 6. Save Results
        logger.info("Saving results...")
        save_results(results, config.output_dir, logger)
        
        logger.info("Analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}", exc_info=True)
        raise

def save_results(results: Dict, output_dir: Path, logger: logging.Logger) -> None:
    """Save analysis results"""
    try:
        # Create results directory
        results_dir = output_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Save full results as pickle
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'brain_state_analysis_results_{timestamp}.pkl'
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
            
        # Create and save summary
        summary = create_summary(results)
        summary_file = results_dir / f'analysis_summary_{timestamp}.md'
        
        with open(summary_file, 'w') as f:
            f.write(summary)
            
        logger.info(f"Results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def create_summary(results: Dict) -> str:
    """Create a comprehensive markdown summary of brain state analysis results"""
    summary = [
        "# Brain State Analysis Summary\n",
        f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        
        "## Executive Summary\n",
        _create_executive_summary(results),
        
        "\n## Detailed Analysis Results\n",
        
        "### 1. General Content Features\n",
        _summarize_general_features(results.get('general', {})),
        
        "\n### 2. Lee-Girl & Verb Interaction Analysis\n",
        _summarize_feature_set(results.get('set1', {}), 'set1'),
        
        "\n### 3. Arthur & Modifier Interaction Analysis\n",
        _summarize_feature_set(results.get('set2', {}), 'set2'),
        
        "\n## Statistical Summary\n",
        _create_statistical_summary(results),
        
        "\n## Limitations and Considerations\n",
        _summarize_limitations(results),
        
        "\n## Technical Details\n",
        _summarize_technical_details(results)
    ]
    
    return "\n".join(summary)

def _create_executive_summary(results: Dict) -> str:
    """Create a high-level executive summary of key findings"""
    summary = []
    
    # Summarize key findings across all analyses
    key_findings = _extract_key_findings(results)
    if key_findings:
        summary.extend([
            "### Key Findings",
            *[f"- {finding}" for finding in key_findings],
            "\n"
        ])
    
    # Add effect size summary
    effect_sizes = _summarize_effect_sizes(results)
    if effect_sizes:
        summary.extend([
            "### Effect Sizes",
            *[f"- {effect}" for effect in effect_sizes],
            "\n"
        ])
    
    return "\n".join(summary)

def _extract_key_findings(results: Dict) -> List[str]:
    """Extract key findings from results"""
    findings = []
    
    # Check for strong state associations
    for group in ['affair', 'paranoia']:
        for feature_set in ['set1', 'set2']:
            if feature_set in results:
                set_results = results[feature_set]
                if 'state_patterns' in set_results and group in set_results['state_patterns']:
                    patterns = set_results['state_patterns'][group]
                    strongest = _find_strongest_associations(patterns)
                    if strongest:
                        findings.extend(strongest)
    
    # Check for significant temporal dynamics
    temporal_findings = _analyze_temporal_patterns(results)
    if temporal_findings:
        findings.extend(temporal_findings)
    
    return findings

def _find_strongest_associations(patterns: Dict) -> List[str]:
    """Find strongest state-condition associations"""
    associations = []
    
    for condition, state_data in patterns.items():
        if isinstance(state_data, dict):
            max_state = None
            max_prob = 0
            
            for state, data in state_data.items():
                if isinstance(data, dict) and 'mean' in data:
                    if data['mean'] > max_prob:
                        max_prob = data['mean']
                        max_state = state
            
            if max_state and max_prob > 0.6:  # Only report strong associations
                associations.append(
                    f"Strong association between {condition} and {max_state} "
                    f"(probability: {max_prob:.2f})"
                )
    
    return associations

def _analyze_temporal_patterns(results: Dict) -> List[str]:
    """Analyze temporal dynamics patterns"""
    findings = []
    
    for feature_set in ['set1', 'set2']:
        if feature_set in results:
            dynamics = results[feature_set].get('temporal_dynamics', {})
            for group in ['affair', 'paranoia']:
                if group in dynamics:
                    group_dynamics = dynamics[group]
                    for feature, pattern in group_dynamics.items():
                        if isinstance(pattern, dict) and 'mean' in pattern:
                            # Look for significant state changes
                            significant_changes = _identify_significant_changes(pattern)
                            if significant_changes:
                                findings.extend(significant_changes)
    
    return findings

def _identify_significant_changes(pattern: Dict) -> List[str]:
    """Identify significant state changes in temporal patterns"""
    changes = []
    
    if 'mean' in pattern and 'std' in pattern:
        mean_data = np.array(pattern['mean'])
        std_data = np.array(pattern['std'])
        
        # Look for significant deviations from baseline
        baseline = mean_data[0]  # Use first timepoint as baseline
        z_scores = (mean_data - baseline) / std_data
        
        significant_timepoints = np.where(np.abs(z_scores) > 2)[0]
        if len(significant_timepoints) > 0:
            changes.append(
                f"Significant state change detected at timepoints: "
                f"{', '.join(map(str, significant_timepoints))}"
            )
    
    return changes

def _summarize_effect_sizes(results: Dict) -> List[str]:
    """Summarize effect sizes across analyses"""
    effects = []
    
    # Analyze regression coefficients
    for feature_set in ['set1', 'set2']:
        if feature_set in results:
            reg_results = results[feature_set].get('regression', {})
            for group in ['affair', 'paranoia']:
                if group in reg_results:
                    large_effects = _find_large_effects(reg_results[group])
                    effects.extend([
                        f"{group.title()} Group - {effect}" for effect in large_effects
                    ])
    
    return effects

def _find_large_effects(regression_results: Dict) -> List[str]:
    """Find large effect sizes in regression results"""
    large_effects = []
    
    for state, results in regression_results.items():
        if results and 'coefficients' in results:
            for feature, coef in results['coefficients'].items():
                if abs(coef) > 0.5:  # Consider effects > 0.5 as large
                    large_effects.append(
                        f"Large effect of {feature} on {state} (β={coef:.2f})"
                    )
    
    return large_effects

def _summarize_general_features(general_results: Dict) -> str:
    """Create detailed summary of general feature results"""
    if not general_results:
        return "No general feature results available."
    
    summary = []
    
    # Analyze dialog effects
    if 'dialog' in general_results:
        summary.extend([
            "#### Dialog Analysis",
            _analyze_feature_category(general_results['dialog'], 'Dialog'),
            "\n"
        ])
    
    # Analyze speaker effects
    if 'speakers' in general_results:
        summary.extend([
            "#### Speaker Analysis",
            _analyze_feature_category(general_results['speakers'], 'Speaker'),
            "\n"
        ])
    
    # Analyze linguistic features
    if 'linguistic' in general_results:
        summary.extend([
            "#### Linguistic Features",
            _analyze_linguistic_features(general_results['linguistic']),
            "\n"
        ])
    
    return "\n".join(summary)

def _analyze_feature_category(results: Dict, category: str) -> str:
    """Analyze a specific feature category"""
    summary = []
    
    for feature, data in results.items():
        if isinstance(data, dict):
            # Analyze effect sizes
            effect_size = _calculate_effect_size(data)
            if effect_size:
                summary.append(
                    f"- {category} feature '{feature}' shows {effect_size} effect"
                )
            
            # Add statistical significance
            if 'p_values_corrected' in data:
                sig_results = [
                    f"{k} (p={v:.3f})" for k, v in data['p_values_corrected'].items()
                    if v < 0.05
                ]
                if sig_results:
                    summary.append(
                        f"  * Significant associations: {', '.join(sig_results)}"
                    )
    
    return "\n".join(summary)

def _analyze_linguistic_features(linguistic_results: Dict) -> str:
    """Analyze linguistic feature results"""
    summary = []
    
    for feature_type, results in linguistic_results.items():
        summary.append(f"\n##### {feature_type.title()} Analysis")
        
        # Analyze frequency
        if 'n_' + feature_type in results:
            freq_data = results['n_' + feature_type]
            summary.append(
                f"- Frequency analysis: mean={freq_data.get('mean', 'N/A'):.2f}, "
                f"std={freq_data.get('std', 'N/A'):.2f}"
            )
        
        # Analyze presence/absence effects
        if 'has_' + feature_type in results:
            presence_data = results['has_' + feature_type]
            effect_size = _calculate_effect_size(presence_data)
            if effect_size:
                summary.append(f"- Presence effect: {effect_size}")
    
    return "\n".join(summary)

def _calculate_effect_size(data: Dict) -> Optional[str]:
    """Calculate and categorize effect size"""
    if 'coefficients' not in data:
        return None
    
    coef = abs(data['coefficients'].get('effect', 0))
    if coef > 0.8:
        return "very large"
    elif coef > 0.5:
        return "large"
    elif coef > 0.3:
        return "moderate"
    elif coef > 0.1:
        return "small"
    else:
        return "minimal"

def _count_statistical_tests(results: Dict) -> int:
    """Count total number of statistical tests performed"""
    count = 0
    
    def count_tests(d: Dict):
        nonlocal count
        if isinstance(d, dict):
            if 'p_values' in d:
                count += len(d['p_values'])
            if 'p_values_corrected' in d:
                count += len(d['p_values_corrected'])
            for v in d.values():
                count_tests(v)
    
    count_tests(results)
    return count

def _summarize_effect_distribution(results: Dict) -> str:
    """Summarize distribution of effect sizes"""
    effect_sizes = []
    
    def collect_effects(d: Dict):
        if isinstance(d, dict):
            if 'coefficients' in d:
                effect_sizes.extend([
                    abs(coef) for coef in d['coefficients'].values()
                    if isinstance(coef, (int, float))
                ])
            for v in d.values():
                collect_effects(v)
    
    collect_effects(results)
    
    if not effect_sizes:
        return "No effect sizes found in results"
        
    summary = [
        f"- Mean effect size: {np.mean(effect_sizes):.3f}",
        f"- Median effect size: {np.median(effect_sizes):.3f}",
        f"- Effect size range: {min(effect_sizes):.3f} to {max(effect_sizes):.3f}",
        "\nEffect size categories:",
        f"- Large effects (>0.5): {sum(1 for e in effect_sizes if e > 0.5)}",
        f"- Medium effects (0.3-0.5): {sum(1 for e in effect_sizes if 0.3 <= e <= 0.5)}",
        f"- Small effects (<0.3): {sum(1 for e in effect_sizes if e < 0.3)}"
    ]
    
    return "\n".join(summary)

def _summarize_model_performance(results: Dict) -> str:
    """Summarize model performance metrics"""
    metrics = []
    
    # Collect regression performance metrics
    for feature_set in ['set1', 'set2']:
        if feature_set in results:
            reg_results = results[feature_set].get('regression', {})
            for group in ['affair', 'paranoia']:
                if group in reg_results:
                    group_metrics = _extract_model_metrics(reg_results[group])
                    if group_metrics:
                        metrics.append(f"\n{group.title()} Group - {feature_set}:")
                        metrics.extend([f"- {m}" for m in group_metrics])
    
    if not metrics:
        return "No model performance metrics available"
        
    return "\n".join(metrics)

def _extract_model_metrics(regression_results: Dict) -> List[str]:
    """Extract model performance metrics from regression results"""
    metrics = []
    
    for state, results in regression_results.items():
        if results and isinstance(results, dict):
            if 'regularized' in results:
                metrics.append(f"{state}: Regularized model")
            
            # Add any available fit metrics
            for metric in ['aic', 'bic', 'pseudo_r2']:
                if metric in results:
                    metrics.append(
                        f"{state}: {metric.upper()}={results[metric]:.3f}"
                    )
    
    return metrics

def _create_statistical_summary(results: Dict) -> str:
    """Create comprehensive statistical summary"""
    summary = [
        "### Multiple Comparison Correction",
        "- Method: FDR (Benjamini-Hochberg)",
        f"- Total tests performed: {_count_statistical_tests(results)}",
        "\n### Effect Size Distribution",
        _summarize_effect_distribution(results),
        "\n### Model Performance Metrics",
        _summarize_model_performance(results)
    ]
    
    return "\n".join(summary)

def _summarize_limitations(results: Dict) -> str:
    """Summarize analysis limitations"""
    limitations = [
        "1. Sample Size Considerations",
        "   - Current analysis based on limited sample size",
        "   - Statistical power may be affected for subtle effects",
        
        "\n2. Temporal Resolution",
        "   - fMRI temporal resolution limitations",
        "   - State transitions may occur between TRs",
        
        "\n3. Model Assumptions",
        "   - State independence assumptions",
        "   - Temporal autocorrelation considerations",
        
        "\n4. Feature Interactions",
        "   - Complex feature interactions may not be fully captured",
        "   - Higher-order effects not included in current analysis"
    ]
    
    return "\n".join(limitations)

def _summarize_technical_details(results: Dict) -> str:
    """Summarize technical analysis details"""
    details = [
        "### Analysis Parameters",
        "- Number of states analyzed",
        "- Window sizes for temporal analysis",
        "- Statistical thresholds applied",
        
        "\n### Data Quality Metrics",
        "- Missing data handling",
        "- Outlier detection results",
        
        "\n### Computational Resources",
        "- Processing time",
        "- Memory usage"
    ]
    
    return "\n".join(details)

def _summarize_general_features(general_results: Dict) -> str:
    """Summarize general feature results"""
    if not general_results:
        return "No general feature results available."
        
    summary = []
    
    # Summarize significant effects
    for feature_type, results in general_results.items():
        sig_effects = _find_significant_effects(results)
        if sig_effects:
            summary.append(f"\n### {feature_type.title()}")
            summary.extend([f"- {effect}" for effect in sig_effects])
            
    return "\n".join(summary)

def _summarize_feature_set(feature_results: Dict, set_name: str) -> str:
    """Summarize feature set results"""
    if not feature_results:
        return f"No results available for feature set {set_name}."
        
    summary = []
    
    # Regression results
    if 'regression' in feature_results:
        summary.append("### Regression Analysis")
        for group in ['affair', 'paranoia']:
            if group in feature_results['regression']:
                sig_effects = _find_significant_regression_effects(
                    feature_results['regression'][group]
                )
                if sig_effects:
                    summary.append(f"\n#### {group.title()} Group")
                    summary.extend([f"- {effect}" for effect in sig_effects])
    
    # State patterns
    if 'state_patterns' in feature_results:
        summary.append("\n### State Patterns")
        for group in ['affair', 'paranoia']:
            if group in feature_results['state_patterns']:
                patterns = _summarize_state_patterns(
                    feature_results['state_patterns'][group]
                )
                if patterns:
                    summary.append(f"\n#### {group.title()} Group")
                    summary.extend([f"- {pattern}" for pattern in patterns])
    
    return "\n".join(summary)

def _find_significant_effects(results: Dict) -> List[str]:
    """Find significant effects in results"""
    significant = []
    
    def process_results(r, path=[]):
        if isinstance(r, dict):
            if 'p_values_corrected' in r:
                for feature, p_val in r['p_values_corrected'].items():
                    if p_val < 0.05:
                        coef = r['coefficients'][feature]
                        significant.append(
                            f"{' -> '.join(path)}: {feature} (β={coef:.3f}, p={p_val:.3f})"
                        )
            else:
                for k, v in r.items():
                    process_results(v, path + [k])
                    
    process_results(results)
    return significant

def _find_significant_regression_effects(results: Dict) -> List[str]:
    """Find significant regression effects"""
    significant = []
    
    for state, state_results in results.items():
        if state_results is None:
            continue
            
        for feature, p_val in state_results.get('p_values_corrected', {}).items():
            if p_val < 0.05:
                coef = state_results['coefficients'][feature]
                significant.append(
                    f"{state}: {feature} effect (β={coef:.3f}, p={p_val:.3f})"
                )
                
    return significant

def _summarize_state_patterns(patterns: Dict) -> List[str]:
    """Summarize state patterns"""
    summary = []
    
    for condition, state_patterns in patterns.items():
        if isinstance(state_patterns, dict):  # Skip metadata
            try:
                # Filter out non-dict values and ensure 'mean' exists
                valid_states = {
                    state: data for state, data in state_patterns.items()
                    if isinstance(data, dict) and 'mean' in data
                }
                
                if valid_states:
                    max_state = max(valid_states.items(), 
                                  key=lambda x: x[1]['mean'])
                    summary.append(
                        f"Condition '{condition}' shows strongest association with "
                        f"{max_state[0]} (p={max_state[1]['mean']:.3f})"
                    )
            except Exception as e:
                logger.error(f"Error summarizing pattern for {condition}: {e}")
                continue
                
    return summary

if __name__ == "__main__":
    main()