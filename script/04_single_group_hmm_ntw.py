import os
import glob
import numpy as np
from hmmlearn import hmm
from scipy.stats import zscore, entropy, ttest_ind
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Tuple, Optional, Generator
import logging
import pickle
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

@dataclass
class HMMConfig:
    """Configuration class for HMM parameters"""
    n_states: int
    random_state: int = 42
    max_tries: int = 5
    tr: float = 1.5
    window_size: int = 20
    n_bootstrap: int = 1000
    convergence_tol: float = 1e-3
    max_iter: int = 1000
    n_jobs: int = 1  # Number of CPU cores to use
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if self.tr <= 0:
            raise ValueError("tr must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.n_bootstrap < 100:
            raise ValueError("n_bootstrap should be at least 100")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be positive")

class SingleGroupHMM:
    """
    Single group Hidden Markov Model analysis for neural state dynamics.
    Designed for comparing state patterns between different experimental groups.
    """
    
    def __init__(self, config: HMMConfig):
        """
        Initialize single group HMM analysis.
        
        Args:
            config: HMMConfig object containing model parameters
        """
        self.config = config
        
        # Initialize storage
        self.model = None
        self.processed_data = None
        self.state_sequence = None
        self.results = {}
        
        # Initialize paths
        self.output_base = None
        self.model_dir = None
        self.plot_dir = None
        self.stats_dir = None
        
        # Analysis parameters
        self.n_timepoints = None
        self.n_features = None
        self.n_subjects = None
        
        # Set numerical stability constant
        self.eps = np.finfo(float).eps
        
        logging.info(f"Initialized SingleGroupHMM with {config.n_states} states")

    def setup_output_directories(self, output_base: str) -> None:
        """
        Create and setup output directory structure.
        
        Args:
            output_base: Base directory for all outputs
        """
        self.output_base = Path(output_base)
        
        # Create directory structure with error handling
        directories = {
            'model': self.output_base / 'models',
            'plot': self.output_base / 'plots',
            'stats': self.output_base / 'statistics',
            'temp': self.output_base / 'temp',
            'logs': self.output_base / 'logs'
        }
        
        try:
            for dir_name, dir_path in directories.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                setattr(self, f"{dir_name}_dir", dir_path)
            logging.info(f"Created output directories in {output_base}")
        except Exception as e:
            logging.error(f"Failed to create directories: {e}")
            raise

    def preprocess_data(self, 
                       data: np.ndarray,
                       normalize: bool = True) -> np.ndarray:
        """
        Preprocess input data for HMM analysis with enhanced validation.
        
        Args:
            data: Input data array [n_subjects, n_features, n_timepoints]
            normalize: Whether to z-score the data
            
        Returns:
            Processed data array [n_timepoints * n_subjects, n_features]
        """
        # Validate input data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
            
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D data array, got shape {data.shape}")
            
        if data.dtype not in [np.float32, np.float64]:
            raise ValueError(f"Expected float data, got {data.dtype}")
            
        if np.any(~np.isfinite(data)):
            raise ValueError("Data contains NaN or Inf values")
            
        self.n_subjects, self.n_features, self.n_timepoints = data.shape
        logging.info(f"Processing data: {self.n_subjects} subjects, "
                    f"{self.n_features} features, {self.n_timepoints} timepoints")
        
        # Process data with memory optimization
        try:
            processed = []
            for subject in tqdm(data, desc="Processing subjects"):
                subject_data = subject.T  # [n_timepoints, n_features]
                if normalize:
                    subject_data = zscore(subject_data, axis=0)
                processed.append(subject_data)
                
            self.processed_data = np.vstack(processed)
            
            # Validate processed data
            if np.any(~np.isfinite(self.processed_data)):
                raise ValueError("Preprocessing resulted in invalid values")
                
            return self.processed_data
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise

    def initialize_model_parameters(self, seed: Optional[int] = None) -> Dict:
        """
        Initialize HMM parameters with informed priors and numerical stability,
        without biasing different states toward different durations.

        Args:
            seed: Random seed for reproducible initialization

        Returns:
            Dictionary of initial parameter values
        """
        try:
            # Initialize transition matrix with uniform temporal constraints
            transmat = np.zeros((self.config.n_states, self.config.n_states))
            
            # Use a consistent average duration across all states
            # Based on typical neural state durations in fMRI (e.g., 5-10 seconds)
            avg_duration = 7.0 / self.config.tr  # 7 seconds is a reasonable expectation
            
            for i in range(self.config.n_states):
                # Calculate self-transition probability based on average duration
                stay_prob = np.exp(-1/avg_duration)
                transmat[i,i] = stay_prob
                
                # Distribute remaining probability uniformly to other states
                remaining_prob = 1 - stay_prob
                transmat[i, [j for j in range(self.config.n_states) if j != i]] = \
                    remaining_prob / (self.config.n_states - 1)
            
            # Ensure exact normalization for numerical stability
            transmat /= transmat.sum(axis=1, keepdims=True)
            
            # Initialize means with k-means for better starting points
            n_features = self.processed_data.shape[1]
            
            # Option 1: Random initialization (can be replaced with k-means)
            # Use provided seed for this attempt, ensuring different initializations across attempts
            if seed is not None:
                np.random.seed(seed)
            means = np.random.randn(self.config.n_states, n_features)
            
            # Option 2: Using covariance structure of the data for more realistic initialization
            data_cov = np.cov(self.processed_data.T)
            
            # Initialize covariance matrices - using scaled data covariance for more realistic priors
            covars = np.tile(np.eye(n_features), (self.config.n_states, 1, 1))
            
            # Add small diagonal term for numerical stability
            covars += np.eye(n_features)[None, :, :] * self.eps
            
            return {
                'transmat': transmat,
                'means': means,
                'covars': covars,
                'startprob': np.ones(self.config.n_states) / self.config.n_states
            }
            
        except Exception as e:
            logging.error(f"Error initializing model parameters: {e}")
            raise

    def fit_model(self, n_states: int) -> Optional[hmm.GaussianHMM]:
        """
        Fit HMM model with parallel initialization attempts.
        
        Args:
            n_states: Number of states for the model
                
        Returns:
            Best fitted model or None if fitting fails
        """
        def single_fit_attempt(seed: int) -> Tuple[float, Optional[hmm.GaussianHMM]]:
            """Single model fitting attempt with specific random seed"""
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type='full',
                    n_iter=self.config.max_iter,
                    tol=self.config.convergence_tol,
                    random_state=seed,
                    init_params=''
                )
                
                # Initialize parameters with the attempt-specific seed
                params = self.initialize_model_parameters(seed=seed)
                model.startprob_ = params['startprob']
                model.means_ = params['means']
                model.covars_ = params['covars']
                model.transmat_ = params['transmat']
                
                # Fit the model once - the HMM implementation will handle
                # convergence internally based on the tolerance parameter
                model.fit(self.processed_data)
                
                # Calculate score on the training data
                score = model.score(self.processed_data)
                
                return score, model
                
            except Exception as e:
                logging.warning(f"Fitting attempt failed: {str(e)}")
                return float('-inf'), None

        try:
            logging.info(f"Starting parallel model fitting with {self.config.max_tries} attempts")
            
            # Generate different random seeds for parallel attempts
            random_seeds = np.random.randint(0, 10000, size=self.config.max_tries)
            
            # Run parallel fitting attempts
            with ThreadPoolExecutor(max_workers=min(self.config.max_tries, self.config.n_jobs)) as executor:
                results = list(executor.map(
                    single_fit_attempt, 
                    random_seeds
                ))
            
            # Find best model
            scores, models = zip(*results)
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best_model = models[best_idx]
            
            if best_model is None:
                raise ValueError("All fitting attempts failed")
                
            logging.info(f"Selected best model with score: {best_score:.2f}")
            return best_model
            
        except Exception as e:
            logging.error(f"Error in model fitting: {e}")
            raise
    
    def analyze_with_loocv(self) -> Dict:
        """
        Perform leave-one-subject-out cross-validation for the current model.
        
        Returns:
            Dictionary containing cross-validation results
        """
        if self.model is None:
            raise ValueError("Must fit model before analyzing with cross-validation")
            
        n_subjects = self.n_subjects
        n_timepoints = self.n_timepoints
        
        # Results storage
        cv_results = {
            'log_likelihood': np.zeros(n_subjects),
            'state_patterns': [],
            'state_predictions': []
        }
        
        # Loop through each subject as a test set
        for test_subj in range(n_subjects):
            # Create training mask (all subjects except test subject)
            train_mask = np.ones(n_subjects, dtype=bool)
            train_mask[test_subj] = False
            
            # Get indices for training data
            train_indices = []
            for i in np.where(train_mask)[0]:
                train_indices.extend(range(i*n_timepoints, (i+1)*n_timepoints))
            
            # Get indices for test data
            test_indices = list(range(test_subj*n_timepoints, (test_subj+1)*n_timepoints))
            
            # Extract training and test data
            train_data = self.processed_data[train_indices]
            test_data = self.processed_data[test_indices]
            
            # Train model on training data
            fold_model = hmm.GaussianHMM(
                n_components=self.config.n_states,
                covariance_type='full',
                n_iter=self.config.max_iter,
                tol=self.config.convergence_tol,
                random_state=self.config.random_state,
                init_params=''
            )
            
            # Initialize parameters with proper regularization
            fold_model.startprob_ = self.model.startprob_.copy()
            fold_model.means_ = self.model.means_.copy()
            
            # Fix the covariance matrices to ensure they are symmetric and positive-definite
            regularized_covars = []
            for cov in self.model.covars_:
                # Make symmetric by averaging with its transpose
                cov_symmetric = (cov + cov.T) / 2
                
                # Add regularization to ensure positive-definiteness
                min_eig = np.min(np.real(np.linalg.eigvals(cov_symmetric)))
                if min_eig < 1e-6:
                    # Add small positive value to diagonal
                    reg_term = abs(min_eig) + 1e-6
                    cov_symmetric += np.eye(cov_symmetric.shape[0]) * reg_term
                    
                regularized_covars.append(cov_symmetric)
                
            fold_model.covars_ = np.array(regularized_covars)
            fold_model.transmat_ = self.model.transmat_.copy()
            
            # Fit model
            fold_model.fit(train_data)
            
            # Evaluate on test data
            cv_results['log_likelihood'][test_subj] = fold_model.score(test_data)
            
            # Store state patterns and predictions
            cv_results['state_patterns'].append(fold_model.means_)
            cv_results['state_predictions'].append(fold_model.predict(test_data))
        
        # Calculate state reliability across folds
        state_reliability = self._calculate_state_pattern_reliability(cv_results['state_patterns'])
        
        cv_results['mean_log_likelihood'] = np.mean(cv_results['log_likelihood'])
        cv_results['state_reliability'] = state_reliability
        
        return cv_results

    def _calculate_state_pattern_reliability(self, state_patterns: List[np.ndarray]) -> float:
        """
        Calculate the reliability of state patterns across cross-validation folds.
        
        Args:
            state_patterns: List of state mean patterns from each fold
            
        Returns:
            Overall state pattern reliability score
        """
        n_folds = len(state_patterns)
        n_states = self.config.n_states
        
        # Calculate correlation between each pair of fold patterns
        pattern_correlations = np.zeros((n_folds, n_folds, n_states, n_states))
        
        for i in range(n_folds):
            for j in range(i+1, n_folds):
                for state_i in range(n_states):
                    for state_j in range(n_states):
                        # Calculate correlation with numerical stability
                        pattern_i = state_patterns[i][state_i]
                        pattern_j = state_patterns[j][state_j]
                        
                        norm_i = np.linalg.norm(pattern_i) + self.eps
                        norm_j = np.linalg.norm(pattern_j) + self.eps
                        
                        correlation = np.dot(pattern_i, pattern_j) / (norm_i * norm_j)
                        pattern_correlations[i, j, state_i, state_j] = correlation
                        pattern_correlations[j, i, state_j, state_i] = correlation
        
        # Find the best matching state between each pair of folds
        from scipy.optimize import linear_sum_assignment
        
        fold_reliabilities = []
        
        for i in range(n_folds):
            for j in range(i+1, n_folds):
                # Create cost matrix (negative correlations for maximization)
                cost_matrix = -pattern_correlations[i, j]
                
                # Find optimal state matching
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Calculate mean correlation with optimal matching
                reliability = np.mean([pattern_correlations[i, j, row_ind[k], col_ind[k]] 
                                    for k in range(n_states)])
                fold_reliabilities.append(reliability)
        
        # Return mean reliability across all fold pairs
        return np.mean(fold_reliabilities)

    def _calculate_bootstrap_uncertainty(self, subject_sequences: List[np.ndarray]) -> Dict:
        """
        Calculate bootstrap-based uncertainty estimates for state metrics.
        
        Args:
            subject_sequences: List of state sequences for each subject
            
        Returns:
            Dictionary containing uncertainty estimates for various metrics
        """
        try:
            uncertainty = {
                'state_occupancy': {},
                'transition_rates': {},
                'temporal_metrics': {}
            }
            
            n_subjects = len(subject_sequences)
            
            # Bootstrap calculations
            bootstrap_results = {
                'occupancy': np.zeros((self.config.n_bootstrap, self.config.n_states)),
                'transitions': np.zeros((self.config.n_bootstrap, self.config.n_states, self.config.n_states)),
                'switching_rate': np.zeros(self.config.n_bootstrap)
            }
            
            for b in range(self.config.n_bootstrap):
                # Sample subjects with replacement
                indices = np.random.randint(0, n_subjects, size=n_subjects)
                boot_sequences = [subject_sequences[i] for i in indices]
                
                # Calculate metrics for this bootstrap sample
                boot_metrics = self._calculate_bootstrap_sample_metrics(boot_sequences)
                
                # Store results
                bootstrap_results['occupancy'][b] = boot_metrics['occupancy']
                bootstrap_results['transitions'][b] = boot_metrics['transitions']
                bootstrap_results['switching_rate'][b] = boot_metrics['switching_rate']
            
            # Calculate confidence intervals
            for state in range(self.config.n_states):
                # State occupancy uncertainty
                occupancy_ci = np.percentile(
                    bootstrap_results['occupancy'][:, state],
                    [2.5, 97.5]
                )
                uncertainty['state_occupancy'][state] = {
                    'ci_lower': occupancy_ci[0],
                    'ci_upper': occupancy_ci[1],
                    'std': np.std(bootstrap_results['occupancy'][:, state])
                }
                
                # Transition rate uncertainty
                for target in range(self.config.n_states):
                    transition_ci = np.percentile(
                        bootstrap_results['transitions'][:, state, target],
                        [2.5, 97.5]
                    )
                    uncertainty['transition_rates'][f'{state}->{target}'] = {
                        'ci_lower': transition_ci[0],
                        'ci_upper': transition_ci[1],
                        'std': np.std(bootstrap_results['transitions'][:, state, target])
                    }
            
            # Temporal metrics uncertainty
            switching_ci = np.percentile(
                bootstrap_results['switching_rate'],
                [2.5, 97.5]
            )
            uncertainty['temporal_metrics']['switching_rate'] = {
                'ci_lower': switching_ci[0],
                'ci_upper': switching_ci[1],
                'std': np.std(bootstrap_results['switching_rate'])
            }
            
            return uncertainty
            
        except Exception as e:
            logging.error(f"Error calculating bootstrap uncertainty: {e}")
            raise

    def _calculate_bootstrap_sample_metrics(self, sequences: List[np.ndarray]) -> Dict:
        """
        Calculate metrics for a single bootstrap sample.
        
        Args:
            sequences: List of state sequences for bootstrap sample
            
        Returns:
            Dictionary containing calculated metrics
        """
        try:
            n_sequences = len(sequences)
            
            # Initialize metrics
            occupancy = np.zeros(self.config.n_states)
            transitions = np.zeros((self.config.n_states, self.config.n_states))
            total_switches = 0
            total_time = 0
            
            # Calculate metrics for each sequence
            for seq in sequences:
                # State occupancy
                for state in range(self.config.n_states):
                    occupancy[state] += np.mean(seq == state)
                
                # Transitions
                for t in range(len(seq)-1):
                    if seq[t] != seq[t+1]:
                        transitions[seq[t], seq[t+1]] += 1
                        total_switches += 1
                total_time += len(seq)
            
            # Normalize metrics
            occupancy /= n_sequences
            transitions = transitions / (transitions.sum(axis=1, keepdims=True) + self.eps)
            switching_rate = total_switches / (total_time * self.config.tr)
            
            return {
                'occupancy': occupancy,
                'transitions': transitions,
                'switching_rate': switching_rate
            }
            
        except Exception as e:
            logging.error(f"Error calculating bootstrap sample metrics: {e}")
            raise

    def analyze_state_metrics(self) -> Dict:
        """
        Calculate comprehensive state-level metrics with enhanced memory efficiency.
        
        Returns:
            Dictionary containing state metrics including durations,
            transitions, and temporal characteristics
        """
        if self.state_sequence is None:
            raise ValueError("Must fit model before analyzing states")
            
        try:
            # Use generator for memory efficiency
            def subject_sequence_generator() -> Generator[np.ndarray, None, None]:
                for i in range(self.n_subjects):
                    yield self.state_sequence[i*self.n_timepoints:(i+1)*self.n_timepoints]
            
            # Initialize storage
            state_metrics = {
                'subject_level': {},
                'group_level': {},
                'temporal': {},
                'uncertainty': {}  # New section for bootstrap results
            }
            
            # Calculate subject-level metrics with progress tracking
            for subject_idx, seq in enumerate(tqdm(subject_sequence_generator(), 
                                                 total=self.n_subjects,
                                                 desc="Analyzing subjects")):
                state_metrics['subject_level'][subject_idx] = \
                    self._calculate_subject_metrics(seq)
                
            # Calculate group-level metrics
            state_metrics['group_level'] = self._aggregate_group_metrics(
                state_metrics['subject_level'])
                
            # Calculate temporal dynamics
            state_metrics['temporal'] = self._analyze_temporal_dynamics(
                list(subject_sequence_generator()))
                
            # Add bootstrap uncertainty estimates
            state_metrics['uncertainty'] = self._calculate_bootstrap_uncertainty(
                list(subject_sequence_generator()))
            
            return state_metrics
            
        except Exception as e:
            logging.error(f"Error in state metrics analysis: {e}")
            raise

    def _calculate_subject_metrics(self, sequence: np.ndarray) -> Dict:
        """
        Calculate state metrics for a single subject with enhanced robustness.
        
        Args:
            sequence: State sequence for one subject
            
        Returns:
            Dictionary of subject-level metrics
        """
        try:
            # Initialize storage
            metrics = {
                'durations': {state: [] for state in range(self.config.n_states)},
                'transitions': np.zeros((self.config.n_states, self.config.n_states)),
                'frequencies': np.zeros(self.config.n_states),
                'fractional_occupancy': np.zeros(self.config.n_states)
            }
            
            # Calculate state durations and transitions
            current_state = sequence[0]
            current_duration = 1
            
            for state in sequence[1:]:
                if state == current_state:
                    current_duration += 1
                else:
                    metrics['durations'][current_state].append(
                        current_duration * self.config.tr)
                    metrics['transitions'][current_state, state] += 1
                    current_state = state
                    current_duration = 1
                    
            # Add final duration
            metrics['durations'][current_state].append(
                current_duration * self.config.tr)
                
            # Calculate frequencies and occupancy with numerical stability
            total_time = self.n_timepoints * self.config.tr + self.eps
            
            for state in range(self.config.n_states):
                metrics['frequencies'][state] = len(metrics['durations'][state])
                metrics['fractional_occupancy'][state] = \
                    sum(metrics['durations'][state]) / total_time
                    
            # Normalize transition matrix with stability
            row_sums = metrics['transitions'].sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, self.eps)  # Avoid division by zero
            metrics['transitions'] = metrics['transitions'] / row_sums
            
            # Calculate additional metrics with error handling
            metrics['duration_stats'] = {}
            for state in range(self.config.n_states):
                durations = metrics['durations'][state]
                if durations:
                    metrics['duration_stats'][state] = {
                        'mean': np.mean(durations),
                        'std': np.std(durations) if len(durations) > 1 else 0,
                        'median': np.median(durations),
                        'max': np.max(durations),
                        'min': np.min(durations),
                        'count': len(durations)
                    }
                else:
                    metrics['duration_stats'][state] = {
                        'mean': 0, 'std': 0, 'median': 0,
                        'max': 0, 'min': 0, 'count': 0
                    }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating subject metrics: {e}")
            raise
    
    def analyze_subject_consistency(self) -> Dict:
        """Analyze consistency of state patterns across subjects."""
        if self.model is None:
            raise ValueError("Must fit model before analyzing subject consistency")
            
        try:
            # Reshape state sequence to [n_subjects, n_timepoints]
            subject_sequences = self.state_sequence.reshape(
                self.n_subjects, self.n_timepoints)
            
            # Reshape processed data to [n_subjects, n_timepoints, n_features]
            subject_data = self.processed_data.reshape(
                self.n_subjects, self.n_timepoints, -1)
            
            consistency = {
                'state_frequency': np.zeros((self.n_subjects, self.config.n_states)),
                'pattern_correlation': np.zeros((self.n_subjects, self.config.n_states)),
                'timing_correlation': np.zeros((self.n_subjects, self.n_subjects))
            }
            
            # Calculate state frequencies and pattern correlations
            for subject in range(self.n_subjects):
                # State frequencies
                for state in range(self.config.n_states):
                    consistency['state_frequency'][subject, state] = \
                        np.mean(subject_sequences[subject] == state)
                
                # Pattern correlations
                subject_patterns = np.mean(subject_data[subject], axis=0)
                for state in range(self.config.n_states):
                    state_mask = (subject_sequences[subject] == state)
                    if np.any(state_mask):
                        state_pattern = np.mean(subject_data[subject][state_mask], axis=0)
                        # Calculate correlation with numerical stability
                        norm1 = np.linalg.norm(state_pattern) + self.eps
                        norm2 = np.linalg.norm(self.model.means_[state]) + self.eps
                        correlation = np.dot(state_pattern, self.model.means_[state]) / (norm1 * norm2)
                        consistency['pattern_correlation'][subject, state] = correlation

            # Calculate timing correlations
            for i in range(self.n_subjects):
                for j in range(i+1, self.n_subjects):
                    corr = np.corrcoef(subject_sequences[i], subject_sequences[j])[0,1]
                    consistency['timing_correlation'][i,j] = corr
                    consistency['timing_correlation'][j,i] = corr

            # Get basic similarity metrics using existing method
            similarity_metrics = self._calculate_similarity_metrics(subject_sequences, subject_data)
            
            # Add temporal coupling to similarity metrics
            temporal_coupling = np.zeros((self.n_subjects, self.n_subjects))
            
            for i in range(self.n_subjects):
                for j in range(i+1, self.n_subjects):
                    seq_i = subject_sequences[i]
                    seq_j = subject_sequences[j]
                    durations_i = []
                    durations_j = []
                    
                    # Calculate state durations for both subjects
                    for state in range(self.config.n_states):
                        runs_i = [len(list(g)) for k, g in itertools.groupby(seq_i) if k == state]
                        runs_j = [len(list(g)) for k, g in itertools.groupby(seq_j) if k == state]
                        durations_i.extend(runs_i)
                        durations_j.extend(runs_j)
                    
                    # Calculate correlation if we have enough data points
                    if len(durations_i) > 1 and len(durations_j) > 1:
                        min_len = min(len(durations_i), len(durations_j))
                        coupling = np.corrcoef(durations_i[:min_len], durations_j[:min_len])[0,1]
                        temporal_coupling[i,j] = coupling
                        temporal_coupling[j,i] = coupling

            # Add temporal coupling to similarity metrics
            similarity_metrics['temporal_coupling'] = temporal_coupling
            
            # Add temporal coupling to summary statistics
            coupling_values = temporal_coupling[np.triu_indices(self.n_subjects, k=1)]
            similarity_metrics['summary']['temporal_coupling'] = {
                'mean': np.mean(coupling_values),
                'std': np.std(coupling_values),
                'ci': self._bootstrap_confidence_interval(coupling_values)
            }
            
            consistency['similarity_metrics'] = similarity_metrics
            
            # Calculate summary statistics for other metrics
            consistency['summary'] = {
                'state_frequency': {
                    'mean': np.mean(consistency['state_frequency'], axis=0),
                    'std': np.std(consistency['state_frequency'], axis=0),
                    'ci': [
                        self._bootstrap_confidence_interval(
                            consistency['state_frequency'][:, state]
                        )
                        for state in range(self.config.n_states)
                    ]
                },
                'pattern_correlation': {
                    'mean': np.mean(consistency['pattern_correlation'], axis=0),
                    'std': np.std(consistency['pattern_correlation'], axis=0),
                    'ci': [
                        self._bootstrap_confidence_interval(
                            consistency['pattern_correlation'][:, state]
                        )
                        for state in range(self.config.n_states)
                    ]
                },
                'timing_correlation': {
                    'mean': np.mean(consistency['timing_correlation'][np.triu_indices_from(
                        consistency['timing_correlation'], k=1)]),
                    'std': np.std(consistency['timing_correlation'][np.triu_indices_from(
                        consistency['timing_correlation'], k=1)]),
                    'ci': self._bootstrap_confidence_interval(
                        consistency['timing_correlation'][np.triu_indices_from(
                            consistency['timing_correlation'], k=1)]
                    )
                }
            }
            
            return consistency
            
        except Exception as e:
            logging.error(f"Error in subject consistency analysis: {e}")
            raise

    def _calculate_similarity_metrics(self,
                                   subject_sequences: np.ndarray,
                                   subject_data: np.ndarray) -> Dict:
        """
        Calculate additional similarity metrics between subjects.
        
        Args:
            subject_sequences: State sequences for all subjects [n_subjects, n_timepoints]
            subject_data: Subject data [n_subjects, n_timepoints, n_features]
            
        Returns:
            Dictionary containing similarity metrics
        """
        try:
            metrics = {
                'mutual_information': np.zeros((self.n_subjects, self.n_subjects)),
                'pattern_similarity': np.zeros((self.n_subjects, self.n_subjects)),
                'state_overlap': np.zeros((self.n_subjects, self.n_subjects))
            }
            
            for i in range(self.n_subjects):
                for j in range(i+1, self.n_subjects):
                    # Calculate mutual information between state sequences
                    mi = mutual_info_score(
                        subject_sequences[i],
                        subject_sequences[j]
                    )
                    metrics['mutual_information'][i,j] = mi
                    metrics['mutual_information'][j,i] = mi
                    
                    # Calculate pattern similarity
                    pattern_i = np.mean(subject_data[i], axis=0)
                    pattern_j = np.mean(subject_data[j], axis=0)
                    norm_i = np.linalg.norm(pattern_i) + self.eps
                    norm_j = np.linalg.norm(pattern_j) + self.eps
                    similarity = np.dot(pattern_i, pattern_j) / (norm_i * norm_j)
                    metrics['pattern_similarity'][i,j] = similarity
                    metrics['pattern_similarity'][j,i] = similarity
                    
                    # Calculate state overlap
                    overlap = np.mean(subject_sequences[i] == subject_sequences[j])
                    metrics['state_overlap'][i,j] = overlap
                    metrics['state_overlap'][j,i] = overlap
            
            # Calculate summary statistics
            metrics['summary'] = {
                metric_name: {
                    'mean': np.mean(metric[np.triu_indices(self.n_subjects, k=1)]),
                    'std': np.std(metric[np.triu_indices(self.n_subjects, k=1)]),
                    'ci': self._bootstrap_confidence_interval(
                        metric[np.triu_indices(self.n_subjects, k=1)]
                    )
                }
                for metric_name, metric in metrics.items()
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating similarity metrics: {e}")
            raise

    def _bootstrap_confidence_interval(self, 
                                     data: np.ndarray, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals with support for multi-dimensional data.
        
        Args:
            data: Input data array (1D or 2D)
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            data = np.asarray(data)
            if len(data) < 2:
                return (data[0], data[0]) if len(data) == 1 else (0.0, 0.0)

            # Handle different data dimensions
            if data.ndim == 1:
                return self._bootstrap_1d(data, confidence)
            elif data.ndim == 2:
                return self._bootstrap_2d(data, confidence)
            else:
                raise ValueError(f"Data with {data.ndim} dimensions not supported")
                
        except Exception as e:
            logging.error(f"Error in bootstrap calculation: {e}")
            raise

    def _bootstrap_1d(self, 
                     data: np.ndarray, 
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap calculation for 1D data."""
        n_samples = len(data)
        bootstrap_means = []
        
        for _ in range(self.config.n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            sample = data[indices]
            bootstrap_means.append(np.mean(sample))
            
        percentiles = [(1 - confidence) / 2, (1 + confidence) / 2]
        return tuple(np.percentile(bootstrap_means, [p * 100 for p in percentiles]))

    def _bootstrap_2d(self, 
                     data: np.ndarray, 
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap calculation for 2D data."""
        n_samples = len(data)
        bootstrap_means = []
        
        for _ in range(self.config.n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            sample = data[indices]
            bootstrap_means.append(np.mean(sample))
            
        bootstrap_means = np.array(bootstrap_means)
        percentiles = [(1 - confidence) / 2, (1 + confidence) / 2]
        return tuple(np.percentile(bootstrap_means, [p * 100 for p in percentiles]))

    def _aggregate_group_metrics(self, subject_metrics: Dict) -> Dict:
        """
        Aggregate metrics across subjects with enhanced statistical calculations.
        
        Args:
            subject_metrics: Dictionary of subject-level metrics
            
        Returns:
            Dictionary of group-level metrics with confidence intervals
        """
        try:
            # Initialize storage for group metrics
            group_metrics = {
                'durations': {
                    state: {
                        'mean': 0.0,
                        'std': 0.0,
                        'ci_lower': 0.0,
                        'ci_upper': 0.0,
                        'distribution': []
                    }
                    for state in range(self.config.n_states)
                },
                'transitions': np.zeros((self.config.n_states, self.config.n_states)),
                'frequencies': np.zeros(self.config.n_states),
                'fractional_occupancy': np.zeros(self.config.n_states)
            }
            
            # Collect all duration data
            for state in range(self.config.n_states):
                all_durations = []
                for subj in subject_metrics.values():
                    all_durations.extend(subj['durations'][state])
                
                if all_durations:
                    # Calculate basic statistics
                    all_durations = np.array(all_durations)
                    group_metrics['durations'][state]['distribution'] = all_durations
                    group_metrics['durations'][state]['mean'] = np.mean(all_durations)
                    group_metrics['durations'][state]['std'] = np.std(all_durations)
                    
                    # Calculate confidence intervals using bootstrap
                    ci = self._bootstrap_confidence_interval(all_durations)
                    group_metrics['durations'][state]['ci_lower'] = ci[0]
                    group_metrics['durations'][state]['ci_upper'] = ci[1]
            
            # Aggregate other metrics
            for metric in ['transitions', 'frequencies', 'fractional_occupancy']:
                values = np.stack([subj[metric] for subj in subject_metrics.values()])
                group_metrics[metric] = np.mean(values, axis=0)
                
                # Add confidence intervals for scalar metrics
                if metric != 'transitions':
                    ci = self._bootstrap_confidence_interval(values)
                    group_metrics[f'{metric}_ci'] = {'lower': ci[0], 'upper': ci[1]}
            
            # Calculate transition entropy with stability
            group_metrics['transition_entropy'] = {}
            for state in range(self.config.n_states):
                probs = group_metrics['transitions'][state]
                probs = np.maximum(probs, self.eps)
                probs = probs / np.sum(probs)
                group_metrics['transition_entropy'][state] = entropy(probs)
            
            return group_metrics
            
        except Exception as e:
            logging.error(f"Error in group metrics aggregation: {e}")
            raise

    def _analyze_temporal_dynamics(self, subject_sequences: List[np.ndarray]) -> Dict:
        """
        Analyze temporal characteristics of state patterns with enhanced metrics.
        
        Args:
            subject_sequences: List of state sequences for each subject
            
        Returns:
            Dictionary containing temporal metrics with uncertainty estimates
        """
        try:
            window_frames = int(self.config.window_size / self.config.tr)
            
            temporal_metrics = {
                'switching_rate': [],
                'state_mixing': [],
                'recurrence_intervals': {state: [] for state in range(self.config.n_states)},
                'state_stability': {state: [] for state in range(self.config.n_states)},
                'entropy_rate': []
            }
            
            # Calculate metrics for each subject
            for subject_seq in tqdm(subject_sequences, desc="Analyzing temporal dynamics"):
                # Calculate switching rate
                switches = np.sum(np.diff(subject_seq) != 0)
                switching_rate = switches / (len(subject_seq) * self.config.tr)
                temporal_metrics['switching_rate'].append(switching_rate)
                
                # Calculate state mixing in windows
                for i in range(0, len(subject_seq) - window_frames + 1, window_frames):
                    window = subject_seq[i:i + window_frames]
                    unique_states = len(np.unique(window))
                    temporal_metrics['state_mixing'].append(
                        unique_states / self.config.n_states)
                
                # Calculate recurrence intervals and stability
                for state in range(self.config.n_states):
                    state_times = np.where(subject_seq == state)[0]
                    
                    # Recurrence intervals
                    if len(state_times) > 1:
                        intervals = np.diff(state_times) * self.config.tr
                        temporal_metrics['recurrence_intervals'][state].extend(intervals)
                    
                    # State stability
                    runs = np.array([len(list(group)) 
                                for key, group in itertools.groupby(subject_seq)
                                if key == state])
                    if len(runs) > 0:
                        temporal_metrics['state_stability'][state].extend(
                            runs * self.config.tr)
                
                # Calculate entropy rate using sliding window
                entropy_rates = []
                for i in range(len(subject_seq) - window_frames):
                    window = subject_seq[i:i + window_frames]
                    counts = np.bincount(window, minlength=self.config.n_states)
                    probs = counts / window_frames + self.eps
                    entropy_rates.append(entropy(probs))
                temporal_metrics['entropy_rate'].append(np.mean(entropy_rates))
            
            # Calculate summary statistics
            switching_rates = np.array(temporal_metrics['switching_rate'])
            state_mixing = np.array(temporal_metrics['state_mixing'])
            entropy_rates = np.array(temporal_metrics['entropy_rate'])
            
            # Calculate confidence intervals
            switching_ci = self._bootstrap_confidence_interval(switching_rates)
            mixing_ci = self._bootstrap_confidence_interval(state_mixing)
            entropy_ci = self._bootstrap_confidence_interval(entropy_rates)
            
            # Structure summary statistics to match plotting function expectations
            summary_stats = {
                'mean_switching_rate': np.mean(switching_rates),
                'switching_rate': {
                    'mean': np.mean(switching_rates),
                    'std': np.std(switching_rates),
                    'ci_lower': switching_ci[0],
                    'ci_upper': switching_ci[1]
                },
                'mean_mixing': np.mean(state_mixing),
                'state_mixing': {
                    'mean': np.mean(state_mixing),
                    'std': np.std(state_mixing),
                    'ci_lower': mixing_ci[0],
                    'ci_upper': mixing_ci[1]
                },
                'mean_entropy_rate': np.mean(entropy_rates),
                'entropy_rate': {
                    'mean': np.mean(entropy_rates),
                    'std': np.std(entropy_rates),
                    'ci_lower': entropy_ci[0],
                    'ci_upper': entropy_ci[1]
                }
            }
            
            # Add state-specific statistics
            for state in range(self.config.n_states):
                # Recurrence intervals
                intervals = temporal_metrics['recurrence_intervals'][state]
                if intervals:
                    intervals = np.array(intervals)
                    interval_ci = self._bootstrap_confidence_interval(intervals)
                    summary_stats[f'recurrence_intervals_state_{state}'] = {
                        'mean': np.mean(intervals),
                        'std': np.std(intervals),
                        'median': np.median(intervals),
                        'ci_lower': interval_ci[0],
                        'ci_upper': interval_ci[1]
                    }
                
                # State stability
                stability = temporal_metrics['state_stability'][state]
                if stability:
                    stability = np.array(stability)
                    stability_ci = self._bootstrap_confidence_interval(stability)
                    summary_stats[f'state_stability_{state}'] = {
                        'mean': np.mean(stability),
                        'std': np.std(stability),
                        'median': np.median(stability),
                        'ci_lower': stability_ci[0],
                        'ci_upper': stability_ci[1]
                    }
            
            temporal_metrics['summary'] = summary_stats
            
            # Add the raw metrics for potential further analysis
            temporal_metrics['raw'] = {
                'switching_rates': switching_rates,
                'state_mixing': state_mixing,
                'entropy_rates': entropy_rates
            }
            
            return temporal_metrics
            
        except Exception as e:
            logging.error(f"Error in temporal dynamics analysis: {e}")
            raise
    
    def _calculate_pattern_stability(self, state_data: np.ndarray) -> float:
        """
        Calculate stability of activation pattern within a state using split-half correlation.
        
        Args:
            state_data: Data points belonging to the state [n_samples, n_features]
            
        Returns:
            Stability score between 0 and 1
        """
        try:
            if len(state_data) < 2:
                return 0.0
            
            # Split data into two halves
            mid = len(state_data) // 2
            pattern1 = state_data[:mid]
            pattern2 = state_data[mid:]
            
            # Calculate mean patterns for each half
            mean_pattern1 = np.mean(pattern1, axis=0)
            mean_pattern2 = np.mean(pattern2, axis=0)
            
            # Calculate correlation with numerical stability
            norm1 = np.linalg.norm(mean_pattern1) + self.eps
            norm2 = np.linalg.norm(mean_pattern2) + self.eps
            correlation = np.dot(mean_pattern1, mean_pattern2) / (norm1 * norm2)
            
            # Ensure correlation is bounded between 0 and 1
            correlation = np.clip(correlation, 0, 1)
            
            # Calculate bootstrap confidence interval for stability estimate
            bootstrap_correlations = []
            for _ in range(100):  # Use fewer iterations for efficiency
                idx1 = np.random.choice(len(pattern1), size=len(pattern1), replace=True)
                idx2 = np.random.choice(len(pattern2), size=len(pattern2), replace=True)
                
                boot_pattern1 = np.mean(pattern1[idx1], axis=0)
                boot_pattern2 = np.mean(pattern2[idx2], axis=0)
                
                norm1 = np.linalg.norm(boot_pattern1) + self.eps
                norm2 = np.linalg.norm(boot_pattern2) + self.eps
                boot_correlation = np.dot(boot_pattern1, boot_pattern2) / (norm1 * norm2)
                bootstrap_correlations.append(np.clip(boot_correlation, 0, 1))
            
            # Calculate confidence interval
            ci_lower, ci_upper = np.percentile(bootstrap_correlations, [2.5, 97.5])
            
            # Store confidence interval in object for potential later use
            if not hasattr(self, 'pattern_stability_ci'):
                self.pattern_stability_ci = {}
            
            stability_stats = {
                'correlation': correlation,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(bootstrap_correlations)
            }
            
            # Store the full statistics
            if not hasattr(self, 'pattern_stability_stats'):
                self.pattern_stability_stats = {}
            self.pattern_stability_stats[len(self.pattern_stability_stats)] = stability_stats
            
            return correlation
            
        except Exception as e:
            logging.error(f"Error calculating pattern stability: {e}")
            raise

    def _calculate_state_separability(self) -> Dict:
        """Calculate how separable the states are from each other."""
        try:
            separability = {
                'pairwise_distances': np.zeros((self.config.n_states, self.config.n_states)),
                'confusion_matrix': np.zeros((self.config.n_states, self.config.n_states)),
                'mahalanobis_distances': np.zeros((self.config.n_states, self.config.n_states)),
                'bootstrap_metrics': {},
                'summary': {}  # Add summary dictionary
            }
            
            # Calculate pairwise Euclidean distances between state means
            for i in range(self.config.n_states):
                for j in range(i+1, self.config.n_states):
                    dist = np.linalg.norm(self.model.means_[i] - self.model.means_[j])
                    separability['pairwise_distances'][i,j] = dist
                    separability['pairwise_distances'][j,i] = dist
            
            # Calculate Mahalanobis distances with improved stability
            for i in range(self.config.n_states):
                for j in range(i+1, self.config.n_states):
                    try:
                        # Calculate pooled covariance
                        pooled_cov = (self.model.covars_[i] + self.model.covars_[j]) / 2
                        
                        # Add regularization term
                        min_eig = np.min(np.real(np.linalg.eigvals(pooled_cov)))
                        if min_eig < 1e-6:
                            reg_term = (1e-6 - min_eig) + 1e-6
                            pooled_cov += np.eye(pooled_cov.shape[0]) * reg_term
                        
                        # Calculate Mahalanobis distance with validation
                        diff = self.model.means_[i] - self.model.means_[j]
                        inv_cov = np.linalg.inv(pooled_cov)
                        mahal_squared = np.abs(diff.dot(inv_cov).dot(diff))
                        mahal_dist = np.sqrt(mahal_squared)
                        
                        if np.isfinite(mahal_dist):
                            separability['mahalanobis_distances'][i,j] = mahal_dist
                            separability['mahalanobis_distances'][j,i] = mahal_dist
                        else:
                            logging.warning(f"Invalid Mahalanobis distance for states {i} and {j}")
                            separability['mahalanobis_distances'][i,j] = np.nan
                            separability['mahalanobis_distances'][j,i] = np.nan
                            
                    except np.linalg.LinAlgError:
                        logging.warning(f"Failed to calculate Mahalanobis distance for states {i} and {j}")
                        separability['mahalanobis_distances'][i,j] = np.nan
                        separability['mahalanobis_distances'][j,i] = np.nan

            # Calculate confusion matrix using state predictions
            state_predictions = self.model.predict(self.processed_data)
            state_probs = self.model.predict_proba(self.processed_data)
            
            for true_state in range(self.config.n_states):
                true_state_indices = (state_predictions == true_state)
                if np.any(true_state_indices):
                    probs_given_true = state_probs[true_state_indices].mean(axis=0)
                    separability['confusion_matrix'][true_state] = probs_given_true

            # Calculate summary statistics
            # Mean distance across all state pairs
            mean_distance = np.mean(separability['pairwise_distances'][np.triu_indices(self.config.n_states, k=1)])
            min_distance = np.min(separability['pairwise_distances'][np.triu_indices(self.config.n_states, k=1)])
            mean_mahalanobis = np.nanmean(separability['mahalanobis_distances'][np.triu_indices(self.config.n_states, k=1)])
            
            # Calculate confusion entropy
            confusion_entropy = -np.sum(
                separability['confusion_matrix'] * np.log2(separability['confusion_matrix'] + self.eps)
            ) / self.config.n_states

            # Add summary statistics
            separability['summary'] = {
                'mean_distance': mean_distance,
                'min_distance': min_distance,
                'mean_mahalanobis': mean_mahalanobis,
                'confusion_entropy': confusion_entropy
            }
            
            return separability
            
        except Exception as e:
            logging.error(f"Error calculating state separability: {e}")
            raise


    def analyze_state_properties(self) -> Dict:
        """
        Analyze spatial/network properties of each state with enhanced statistical rigor.
        
        Returns:
            Dictionary containing state-specific activation patterns and characteristics
        """
        if self.model is None:
            raise ValueError("Must fit model before analyzing state properties")
            
        try:
            state_properties = {}
            
            # Get state-specific data efficiently
            state_data = {}
            for state in range(self.config.n_states):
                state_mask = (self.state_sequence == state)
                if np.any(state_mask):
                    state_data[state] = self.processed_data[state_mask]
                else:
                    state_data[state] = np.array([])
            
            # Calculate properties for each state
            for state in range(self.config.n_states):
                properties = {}
                
                # Mean activation pattern with confidence intervals
                properties['mean_pattern'] = self.model.means_[state]
                if len(state_data[state]) > 0:
                    # Bootstrap confidence intervals for mean pattern
                    pattern_ci = np.array([
                        self._bootstrap_confidence_interval(state_data[state][:, i])
                        for i in range(self.n_features)
                    ])
                    properties['mean_pattern_ci'] = {
                        'lower': pattern_ci[:, 0],
                        'upper': pattern_ci[:, 1]
                    }
                    
                    # Calculate variability measures
                    properties['std_pattern'] = np.std(state_data[state], axis=0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cv = properties['std_pattern'] / np.abs(properties['mean_pattern'])
                        properties['cv_pattern'] = np.where(np.isfinite(cv), cv, 0)
                    
                    # Calculate covariance and correlation structure
                    properties['covariance'] = self.model.covars_[state]
                    properties['correlation'] = self._calculate_correlation_matrix(
                        state_data[state])
                    
                    # Calculate dimensionality metrics
                    eigenvals = np.linalg.eigvals(properties['covariance'])
                    eigenvals = np.maximum(eigenvals.real, 0)  # Ensure non-negative
                    properties['effective_dimension'] = (
                        np.sum(eigenvals)**2 / (np.sum(eigenvals**2) + self.eps))
                    
                    # Calculate feature importance with stability
                    properties['feature_importance'] = np.abs(
                        properties['mean_pattern']) * properties['std_pattern']
                    
                    # Identify top features
                    top_idx = np.argsort(properties['feature_importance'])[::-1]
                    properties['top_features'] = {
                        'indices': top_idx[:10].tolist(),
                        'importance': properties['feature_importance'][top_idx[:10]].tolist(),
                        'mean_values': properties['mean_pattern'][top_idx[:10]].tolist()
                    }
                    
                    # Calculate pattern stability
                    properties['pattern_stability'] = self._calculate_pattern_stability(
                        state_data[state])
                    
                else:
                    # Handle empty states with appropriate defaults
                    self._initialize_empty_state_properties(properties)
                
                state_properties[state] = properties
            
            # Calculate state separability metrics
            state_properties['separability'] = self._calculate_state_separability()
            
            return state_properties
            
        except Exception as e:
            logging.error(f"Error in state properties analysis: {e}")
            raise

    def _calculate_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix with enhanced numerical stability.
        
        Args:
            data: Input data array [n_samples, n_features]
            
        Returns:
            Correlation matrix [n_features, n_features]
        """
        try:
            # Center the data
            centered_data = data - np.mean(data, axis=0)
            
            # Calculate standard deviations with stability
            std = np.std(data, axis=0)
            std = np.maximum(std, self.eps)
            
            # Calculate correlation matrix
            corr_matrix = np.dot(centered_data.T, centered_data) / (
                len(data) * np.outer(std, std))
            
            # Ensure proper bounds
            corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
            
            return corr_matrix
            
        except Exception as e:
            logging.error(f"Error calculating correlation matrix: {e}")
            raise

    def _initialize_empty_state_properties(self, properties: Dict) -> None:
        """Initialize property dictionary for empty states."""
        properties.update({
            'std_pattern': np.zeros_like(properties['mean_pattern']),
            'cv_pattern': np.zeros_like(properties['mean_pattern']),
            'correlation': np.eye(self.n_features),
            'effective_dimension': 0.0,
            'feature_importance': np.zeros_like(properties['mean_pattern']),
            'top_features': {
                'indices': [],
                'importance': [],
                'mean_values': []
            },
            'pattern_stability': 0.0
        })

    def generate_visualizations(self,
                              group_name: str,
                              state_metrics: Dict,
                              state_properties: Dict,
                              subject_consistency: Dict) -> None:
        """Generate comprehensive set of visualizations with error handling."""
        try:
            # Set style for all plots - use a default style instead of seaborn
            plt.style.use('default')
            # Set general plotting parameters
            plt.rcParams.update({
                'figure.figsize': [10.0, 8.0],
                'figure.dpi': 300,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'lines.linewidth': 2,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10
            })
            
            # Create plots with progress tracking
            plot_functions = [
                (self._plot_state_patterns, "State Patterns"),
                (self._plot_duration_distributions, "Duration Distributions"),
                (self._plot_transition_matrix, "Transition Matrix"),
                (self._plot_state_separability, "State Separability"),
                (self._plot_temporal_dynamics, "Temporal Dynamics"),
                (self._plot_subject_consistency, "Subject Consistency")
            ]
            
            for plot_func, desc in tqdm(plot_functions, desc="Generating plots"):
                try:
                    plot_func(
                        group_name=group_name,
                        state_metrics=state_metrics,
                        state_properties=state_properties,
                        subject_consistency=subject_consistency
                    )
                except Exception as e:
                    logging.error(f"Error generating {desc} plot: {e}")
                    continue
                    
            logging.info("Completed visualization generation")
            
        except Exception as e:
            logging.error(f"Error in visualization generation: {e}")
            raise

    def _plot_state_separability(self,
                               group_name: str,
                               state_metrics: Dict,
                               state_properties: Dict,
                               **kwargs) -> None:
        """
        Plot state separability metrics including distances and confusion matrix.
        
        Args:
            group_name: Name identifier for the group
            state_metrics: Dictionary of state metrics
            state_properties: Dictionary of state properties containing separability metrics
        """
        try:
            separability = state_properties['separability']
            
            # Create figure with three subplots
            fig = plt.figure(figsize=(20, 6))
            gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1.2])
            
            # 1. Plot distance matrix
            ax1 = fig.add_subplot(gs[0])
            distances = separability['pairwise_distances']
            im1 = sns.heatmap(
                distances,
                ax=ax1,
                cmap='viridis',
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Euclidean Distance'}
            )
            ax1.set_title('State Distance Matrix')
            ax1.set_xlabel('State')
            ax1.set_ylabel('State')
            
            # 2. Plot Mahalanobis distances
            ax2 = fig.add_subplot(gs[1])
            mahal_dist = separability['mahalanobis_distances']
            # Mask invalid values for visualization
            mask = np.isnan(mahal_dist)
            im2 = sns.heatmap(
                mahal_dist,
                ax=ax2,
                cmap='viridis',
                annot=True,
                fmt='.2f',
                square=True,
                mask=mask,
                cbar_kws={'label': 'Mahalanobis Distance'}
            )
            ax2.set_title('Mahalanobis Distance Matrix')
            ax2.set_xlabel('State')
            ax2.set_ylabel('State')
            
            # 3. Plot confusion matrix
            ax3 = fig.add_subplot(gs[2])
            confusion = separability['confusion_matrix']
            im3 = sns.heatmap(
                confusion,
                ax=ax3,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Confusion Probability'}
            )
            ax3.set_title('State Confusion Matrix')
            ax3.set_xlabel('Assigned State')
            ax3.set_ylabel('True State')
            
            # Add summary statistics
            summary = separability['summary']
            summary_text = (
                f"Mean Distance: {summary['mean_distance']:.2f}\n"
                f"Min Distance: {summary['min_distance']:.2f}\n"
                f"Mean Mahalanobis: {summary['mean_mahalanobis']:.2f}\n"
                f"Confusion Entropy: {summary['confusion_entropy']:.2f}"
            )
            
            # Add text box with summary statistics
            plt.figtext(
                0.98, 0.98,
                summary_text,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8
                )
            )
            
            # Add bootstrap confidence intervals if available
            if 'bootstrap_metrics' in separability and 'distance_ci' in separability['bootstrap_metrics']:
                ci_text = "95% CIs for distances:\n"
                for pair, ci in separability['bootstrap_metrics']['distance_ci'].items():
                    ci_text += f"{pair}: [{ci['lower']:.2f}, {ci['upper']:.2f}]\n"
                
                plt.figtext(
                    0.98, 0.7,
                    ci_text,
                    horizontalalignment='right',
                    verticalalignment='top',
                    bbox=dict(
                        boxstyle='round',
                        facecolor='white',
                        alpha=0.8
                    ),
                    fontsize=8
                )
            
            plt.tight_layout()
            
            # Save figure
            save_path = self.plot_dir / f"{group_name}_state_separability.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting state separability: {e}")
            raise

    def _plot_temporal_dynamics(self,
                          group_name: str,
                          state_metrics: Dict,
                          state_properties: Dict,
                          **kwargs) -> None:
        """
        Plot temporal dynamics metrics including switching rates and state mixing.
        
        Args:
            group_name: Name identifier for the group
            state_metrics: Dictionary of state metrics including temporal statistics
            state_properties: Dictionary of state properties
        """
        try:
            temporal = state_metrics['temporal']
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = plt.GridSpec(2, 2)
            
            # 1. Switching rate distribution
            ax1 = fig.add_subplot(gs[0, 0])
            sns.histplot(
                temporal['switching_rate'],
                ax=ax1,
                bins='auto',
                kde=True,
                color='skyblue'
            )
            mean_rate = temporal['summary']['mean_switching_rate']
            ax1.axvline(
                mean_rate,
                color='r',
                linestyle='--',
                label=f'Mean: {mean_rate:.3f} Hz'
            )
            
            # Add confidence interval
            if 'switching_rate' in temporal['summary']:
                stats = temporal['summary']['switching_rate']
                if 'ci_lower' in stats and 'ci_upper' in stats:
                    ax1.axvline(stats['ci_lower'], color='g', linestyle=':', label='95% CI')
                    ax1.axvline(stats['ci_upper'], color='g', linestyle=':')
                    
                    stats_text = (
                        f"Mean: {mean_rate:.3f} Hz\n"
                        f"95% CI: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
                    )
                    ax1.text(
                        0.95, 0.95,
                        stats_text,
                        transform=ax1.transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
            
            ax1.set_title('Switching Rate Distribution')
            ax1.set_xlabel('Switches per Second')
            ax1.set_ylabel('Count')
            ax1.legend()
            
            # 2. State mixing over time
            ax2 = fig.add_subplot(gs[0, 1])
            window_indices = np.arange(len(temporal['state_mixing'])) * self.config.window_size
            ax2.plot(window_indices, temporal['state_mixing'], 'b-', alpha=0.6)
            
            mean_mixing = temporal['summary']['mean_mixing']
            ax2.axhline(
                mean_mixing,
                color='r',
                linestyle='--',
                label=f"Mean: {mean_mixing:.3f}"
            )
            
            if 'state_mixing' in temporal['summary']:
                stats = temporal['summary']['state_mixing']
                if 'ci_lower' in stats and 'ci_upper' in stats:
                    ax2.fill_between(
                        window_indices,
                        [stats['ci_lower']] * len(window_indices),
                        [stats['ci_upper']] * len(window_indices),
                        color='r',
                        alpha=0.2,
                        label='95% CI'
                    )
            
            ax2.set_title('State Mixing Over Time')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Mixing Ratio')
            ax2.legend()
            
            # 3. Recurrence intervals
            ax3 = fig.add_subplot(gs[1, 0])
            recurrence_data = []
            labels = []
            
            for state in range(self.config.n_states):
                intervals = temporal['recurrence_intervals'].get(state, [])
                if intervals:
                    recurrence_data.append(intervals)
                    labels.append(f'State {state}')
            
            if recurrence_data:
                # Use tick_labels instead of labels (addressing deprecation warning)
                ax3.boxplot(recurrence_data, tick_labels=labels)
                ax3.set_yscale('log')
                ax3.set_title('State Recurrence Intervals')
                ax3.set_xlabel('State')
                ax3.set_ylabel('Interval (s)')
                
                # Add recurrence statistics text
                stats_text = "Mean Intervals (s):\n"
                for state in range(self.config.n_states):
                    stat_key = f'recurrence_intervals_state_{state}'
                    if stat_key in temporal['summary']:
                        stats = temporal['summary'][stat_key]
                        stats_text += f"State {state}: {stats['mean']:.1f}\n"
                
                ax3.text(
                    0.95, 0.95,
                    stats_text,
                    transform=ax3.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            # 4. State stability
            ax4 = fig.add_subplot(gs[1, 1])
            stability_data = []
            stability_labels = []
            
            for state in range(self.config.n_states):
                durations = temporal['state_stability'].get(state, [])
                if durations:
                    stability_data.append(durations)
                    stability_labels.append(f'State {state}')
            
            if stability_data:
                # Use tick_labels instead of labels
                ax4.boxplot(stability_data, tick_labels=stability_labels)
                ax4.set_yscale('log')
                ax4.set_title('State Stability Durations')
                ax4.set_xlabel('State')
                ax4.set_ylabel('Duration (s)')
                
                # Add stability statistics text
                stats_text = "Mean Durations (s):\n"
                for state in range(self.config.n_states):
                    stat_key = f'state_stability_{state}'
                    if stat_key in temporal['summary']:
                        stats = temporal['summary'][stat_key]
                        stats_text += f"State {state}: {stats['mean']:.1f}\n"
                
                ax4.text(
                    0.95, 0.95,
                    stats_text,
                    transform=ax4.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            plt.tight_layout()
            
            # Save figure
            save_path = self.plot_dir / f"{group_name}_temporal_dynamics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting temporal dynamics: {e}")
            raise

    def _plot_state_patterns(self,
                           group_name: str,
                           state_properties: Dict,
                           **kwargs) -> None:
        """Plot state-specific activation patterns with confidence intervals."""
        try:
            fig, axes = plt.subplots(
                self.config.n_states, 1,
                figsize=(15, 5*self.config.n_states),
                squeeze=False
            )
            
            for state in range(self.config.n_states):
                ax = axes[state, 0]
                properties = state_properties[state]
                
                x = np.arange(self.n_features)
                mean_pattern = properties['mean_pattern']
                
                # Plot mean pattern
                ax.plot(x, mean_pattern, 'b-', label='Mean Pattern', linewidth=2)
                
                # Add confidence intervals if available
                if 'mean_pattern_ci' in properties:
                    ci = properties['mean_pattern_ci']
                    ax.fill_between(
                        x, ci['lower'], ci['upper'],
                        color='b', alpha=0.2, label='95% CI'
                    )
                
                # Highlight top features
                top_features = properties['top_features']
                for idx, importance in zip(
                    top_features['indices'][:5],
                    top_features['importance'][:5]
                ):
                    ax.annotate(
                        f'Feature {idx}',
                        xy=(idx, mean_pattern[idx]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        bbox=dict(
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.7
                        )
                    )
                
                # Add dimensionality information
                dim_text = f'Effective Dim: {properties["effective_dimension"]:.1f}'
                stab_text = f'Stability: {properties["pattern_stability"]:.2f}'
                ax.text(
                    0.02, 0.98,
                    f'{dim_text}\n{stab_text}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.7
                    )
                )
                
                ax.set_title(f'State {state} Activation Pattern')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Activation')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            save_path = self.plot_dir / f"{group_name}_state_patterns.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting state patterns: {e}")
            raise

    def _plot_duration_distributions(self,
                               group_name: str,
                               state_metrics: Dict,
                               **kwargs) -> None:
        """Plot state duration distributions with enhanced error handling."""
        try:
            fig, axes = plt.subplots(
                self.config.n_states, 1,
                figsize=(12, 4*self.config.n_states),
                squeeze=False
            )
            
            for state in range(self.config.n_states):
                ax = axes[state, 0]
                
                # Safely get durations with error handling
                durations = []
                if ('group_level' in state_metrics and 
                    'durations' in state_metrics['group_level'] and
                    state in state_metrics['group_level']['durations']):
                    durations = state_metrics['group_level']['durations'][state].get('distribution', [])
                
                if len(durations) > 0:
                    # Plot histogram with KDE
                    sns.histplot(
                        durations, 
                        ax=ax, 
                        bins='auto',
                        kde=True,
                        color='skyblue'
                    )
                    
                    # Safely get statistics
                    stats = state_metrics['group_level']['durations'].get(state, {})
                    mean_val = stats.get('mean', np.nan)
                    ci_lower = stats.get('ci_lower', np.nan)
                    ci_upper = stats.get('ci_upper', np.nan)
                    
                    if not np.isnan(mean_val):
                        ax.axvline(
                            mean_val, 
                            color='r', 
                            linestyle='--',
                            label=f'Mean: {mean_val:.1f}s'
                        )
                    
                    if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                        ax.axvline(
                            ci_lower,
                            color='g',
                            linestyle=':',
                            label=f'95% CI'
                        )
                        ax.axvline(ci_upper, color='g', linestyle=':')
                else:
                    ax.text(
                        0.5, 0.5,
                        'No durations available for this state',
                        ha='center',
                        va='center',
                        transform=ax.transAxes
                    )
                
                ax.set_title(f'State {state} Duration Distribution')
                ax.set_xlabel('Duration (s)')
                ax.set_ylabel('Count')
                if ax.get_legend() is not None:
                    ax.legend()
            
            plt.tight_layout()
            save_path = self.plot_dir / f"{group_name}_duration_distributions.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting duration distributions: {e}")
            raise

    def _plot_transition_matrix(self,
                              group_name: str,
                              state_metrics: Dict,
                              **kwargs) -> None:
        """Plot state transition matrix."""
        try:
            # Create figure with subplot (removed networkx dependency)
            fig, ax1 = plt.subplots(figsize=(10, 8))
            
            # Safely get transitions
            transitions = np.zeros((self.config.n_states, self.config.n_states))
            if ('group_level' in state_metrics and 
                'transitions' in state_metrics['group_level']):
                transitions = state_metrics['group_level']['transitions']
            
            # Plot transition matrix
            mask = np.zeros_like(transitions, dtype=bool)
            np.fill_diagonal(mask, True)  # Mask diagonal for better visualization
            
            sns.heatmap(
                transitions,
                ax=ax1,
                cmap='YlOrRd',
                mask=mask,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Transition Probability'}
            )
            
            # Add diagonal values with different color
            diagonal = np.diag(transitions)
            for i in range(len(diagonal)):
                ax1.text(
                    i, 
                    i, 
                    f'{diagonal[i]:.2f}',
                    ha='center',
                    va='center',
                    color='white',
                    bbox=dict(
                        facecolor='blue',
                        alpha=0.6
                    )
                )
            
            ax1.set_title('State Transition Probabilities')
            ax1.set_xlabel('To State')
            ax1.set_ylabel('From State')
            
            plt.tight_layout()
            save_path = self.plot_dir / f"{group_name}_transition_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting transition matrix: {e}")
            raise

    def _plot_subject_consistency(self,
                               group_name: str,
                               state_metrics: Dict,
                               state_properties: Dict,
                               subject_consistency: Dict,
                               **kwargs) -> None:
        """
        Plot subject consistency metrics including state frequencies and correlations.
        
        Args:
            group_name: Name identifier for the group
            state_metrics: Dictionary of state metrics
            state_properties: Dictionary of state properties
            subject_consistency: Dictionary of subject consistency metrics
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            gs = plt.GridSpec(2, 2)
            
            # 1. State frequency patterns across subjects
            ax1 = fig.add_subplot(gs[0, 0])
            freq_data = pd.DataFrame(
                subject_consistency['state_frequency'],
                columns=[f'State {i}' for i in range(self.config.n_states)]
            )
            
            sns.boxplot(data=freq_data, ax=ax1)
            ax1.set_title('State Frequencies Across Subjects')
            ax1.set_xlabel('State')
            ax1.set_ylabel('Frequency')
            
            # Add summary statistics
            if 'summary' in subject_consistency and 'state_frequency' in subject_consistency['summary']:
                stats = subject_consistency['summary']['state_frequency']
                stats_text = "Mean Frequencies:\n"
                for state in range(self.config.n_states):
                    mean = stats['mean'][state]
                    ci = stats['ci'][state]
                    stats_text += f"State {state}: {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]\n"
                
                ax1.text(
                    1.05, 1.,
                    stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(
                        boxstyle='round',
                        facecolor='white',
                        alpha=0.8
                    )
                )
            
            # 2. Pattern correlations heatmap
            ax2 = fig.add_subplot(gs[0, 1])
            pattern_corr = subject_consistency['pattern_correlation']
            sns.heatmap(
                pattern_corr,
                ax=ax2,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'}
            )
            ax2.set_title('Pattern Correlations by Subject and State')
            ax2.set_xlabel('State')
            ax2.set_ylabel('Subject')
            
            # Add summary statistics if available
            if 'summary' in subject_consistency and 'pattern_correlation' in subject_consistency['summary']:
                stats = subject_consistency['summary']['pattern_correlation']
                stats_text = "Mean Correlations:\n"
                for state in range(self.config.n_states):
                    mean = stats['mean'][state]
                    ci = stats['ci'][state]
                    stats_text += f"State {state}: {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]\n"
                
                ax2.text(
                    1.15, 1.,
                    stats_text,
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    fontsize=8
                )
            
            # 3. Timing correlation matrix
            ax3 = fig.add_subplot(gs[1, 0])
            timing_corr = subject_consistency['timing_correlation']
            mask = np.zeros_like(timing_corr, dtype=bool)
            np.fill_diagonal(mask, True)  # Mask diagonal for better visualization
            
            sns.heatmap(
                timing_corr,
                ax=ax3,
                cmap='RdBu_r',
                center=0,
                mask=mask,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'}
            )
            ax3.set_title('Between-Subject Timing Correlations')
            ax3.set_xlabel('Subject')
            ax3.set_ylabel('Subject')
            
            # Add mean correlation
            if 'summary' in subject_consistency and 'timing_correlation' in subject_consistency['summary']:
                stats = subject_consistency['summary']['timing_correlation']
                mean_corr = stats['mean']
                ci = stats['ci']
                stats_text = (
                    f"Mean Correlation: {mean_corr:.3f}\n"
                    f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
                )
                
                ax3.text(
                    1.15, 1.,
                    stats_text,
                    transform=ax3.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left'
                )
            
            # 4. Additional similarity metrics if available
            ax4 = fig.add_subplot(gs[1, 1])
            if 'similarity_metrics' in subject_consistency:
                metrics = subject_consistency['similarity_metrics']
                
                # Plot boxplots for different metrics
                similarity_data = []
                labels = []
                for metric_name, metric_data in metrics.items():
                    if metric_name != 'summary':
                        # Get upper triangle values (excluding diagonal)
                        triu_indices = np.triu_indices_from(metric_data, k=1)
                        similarity_data.append(metric_data[triu_indices])
                        labels.extend([metric_name] * len(triu_indices[0]))
                
                if similarity_data:
                    df = pd.DataFrame({
                        'Metric': labels,
                        'Value': np.concatenate(similarity_data)
                    })
                    sns.boxplot(data=df, x='Metric', y='Value', ax=ax4)
                    ax4.set_title('Additional Similarity Metrics')
                    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
                    
                    # Add summary statistics
                    if 'summary' in metrics:
                        stats_text = "Mean Values:\n"
                        for metric, stats in metrics['summary'].items():
                            stats_text += (
                                f"{metric}: {stats['mean']:.3f} "
                                f"[{stats['ci'][0]:.3f}, {stats['ci'][1]:.3f}]\n"
                            )
                        
                        ax4.text(
                            1.05, 1.,
                            stats_text,
                            transform=ax4.transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(
                                boxstyle='round',
                                facecolor='white',
                                alpha=0.8
                            ),
                            fontsize=8
                        )
            else:
                ax4.text(
                    0.5, 0.5,
                    'No additional metrics available',
                    horizontalalignment='center',
                    verticalalignment='center'
                )
            
            plt.tight_layout()
            
            # Save figure
            save_path = self.plot_dir / f"{group_name}_subject_consistency.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting subject consistency: {e}")
            raise

    def save_results(self,
                    group_name: str,
                    state_metrics: Dict,
                    state_properties: Dict,
                    subject_consistency: Dict) -> None:
        """
        Save all analysis results to files.
        
        Args:
            group_name: Name identifier for the group
            state_metrics: Dictionary of state metrics
            state_properties: Dictionary of state properties
            subject_consistency: Dictionary of subject consistency metrics
        """
        try:
            # Save model and state sequence
            model_path = self.model_dir / f"{group_name}_hmm_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'state_sequence': self.state_sequence,
                    'parameters': {
                        'n_states': self.config.n_states,
                        'random_state': self.config.random_state,
                        'tr': self.config.tr,
                        'window_size': self.config.window_size
                    }
                }, f)
            logging.info(f"Saved model to {model_path}")
            
            # Save all metrics as pickle for complete data
            metrics_path = self.stats_dir / f"{group_name}_metrics.pkl"
            with open(metrics_path, 'wb') as f:
                pickle.dump({
                    'state_metrics': state_metrics,
                    'state_properties': state_properties,
                    'subject_consistency': subject_consistency
                }, f)
            logging.info(f"Saved detailed metrics to {metrics_path}")
            
            # Create and save summary JSON
            summary = self._create_summary_dict(
                group_name,
                state_metrics,
                state_properties,
                subject_consistency
            )
            
            summary_path = self.stats_dir / f"{group_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4, cls=NumpyJSONEncoder)
            logging.info(f"Saved summary to {summary_path}")
            
            # Save state sequences for each subject
            sequences_path = self.stats_dir / f"{group_name}_state_sequences.npy"
            np.save(sequences_path, 
                   self.state_sequence.reshape(self.n_subjects, self.n_timepoints))
            logging.info(f"Saved state sequences to {sequences_path}")
            
            # Save validation metrics if available
            if hasattr(self, 'pattern_stability_stats'):
                validation_path = self.stats_dir / f"{group_name}_validation_metrics.json"
                with open(validation_path, 'w') as f:
                    json.dump({
                        'pattern_stability': self.pattern_stability_stats
                    }, f, indent=4, cls=NumpyJSONEncoder)
                logging.info(f"Saved validation metrics to {validation_path}")
                
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise
            
    def _create_summary_dict(self,
                           group_name: str,
                           state_metrics: Dict,
                           state_properties: Dict,
                           subject_consistency: Dict) -> Dict:
        """Create a summary dictionary with safe access to metrics."""
        try:
            # Calculate model complexity
            n_samples = len(self.processed_data)
            n_features = self.processed_data.shape[1]
            
            # Calculate number of parameters
            n_parameters = (
                self.config.n_states * self.config.n_states +  # transition matrix
                self.config.n_states +  # initial probabilities
                self.config.n_states * n_features +  # means
                self.config.n_states * n_features * n_features  # covariance matrices
            )
            
            # Safely get temporal metrics
            temporal_metrics = {}
            if 'temporal' in state_metrics and 'summary' in state_metrics['temporal']:
                summary = state_metrics['temporal']['summary']
                if isinstance(summary, dict):
                    temporal_metrics = {
                        'mean_switching_rate': float(summary.get('switching_rate', {}).get('mean', 0.0)),
                        'mean_mixing': float(summary.get('state_mixing', {}).get('mean', 0.0))
                    }
            
            # Create summary dictionary with safe access
            summary = {
                'group_name': group_name,
                'model_info': {
                    'n_states': self.config.n_states,
                    'n_subjects': self.n_subjects,
                    'n_timepoints': self.n_timepoints,
                    'n_features': n_features,
                    'n_parameters': n_parameters
                },
                'model_performance': {
                    'log_likelihood': float(self.model.score(self.processed_data)),
                    'bic': float(-2 * self.model.score(self.processed_data) * n_samples + 
                               np.log(n_samples) * n_parameters)
                },
                'state_metrics': {},
                'temporal_dynamics': temporal_metrics,
                'state_properties': {},
                'subject_consistency': {}
            }
            
            # Safely add state metrics
            if 'group_level' in state_metrics:
                group_level = state_metrics['group_level']
                
                # Add mean durations
                if 'durations' in group_level:
                    summary['state_metrics']['mean_durations'] = {
                        str(state): float(stats.get('mean', 0.0))
                        for state, stats in group_level['durations'].items()
                    }
                
                # Add fractional occupancy
                if 'fractional_occupancy' in group_level:
                    summary['state_metrics']['fractional_occupancy'] = {
                        str(state): float(occ)
                        for state, occ in enumerate(group_level['fractional_occupancy'])
                    }
            
            # Safely add state properties
            for state, props in state_properties.items():
                if state != 'separability' and isinstance(props, dict):
                    summary['state_properties'][str(state)] = {
                        'effective_dimension': float(props.get('effective_dimension', 0.0)),
                        'pattern_stability': float(props.get('pattern_stability', 0.0))
                    }
            
            # Safely add subject consistency
            if 'summary' in subject_consistency:
                sum_data = subject_consistency['summary']
                summary['subject_consistency'] = {
                    'mean_pattern_correlation': float(
                        np.mean(sum_data.get('pattern_correlation', {}).get('mean', [0.0]))
                    ),
                    'mean_timing_correlation': float(
                        sum_data.get('timing_correlation', {}).get('mean', 0.0)
                    )
                }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error creating summary dictionary: {e}")
            raise

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def run_single_group_analysis(data: np.ndarray,
                            selected_indices: Dict,
                            group_name: str,
                            output_dir: str,
                            config: HMMConfig) -> Dict:
    """
    Run complete HMM analysis for a single group with enhanced error handling.
    
    Args:
        data: Input data array [n_subjects, n_features, n_timepoints]
        group_name: Name identifier for the group
        output_dir: Base directory for outputs
        config: HMMConfig object containing model parameters
    
    Returns:
        Dictionary containing all analysis results
    """
    try:
        # Validate input data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
            
        if len(data.shape) != 3:
            raise ValueError(
                f"Data must be 3D array [n_subjects, n_features, n_timepoints], "
                f"got shape {data.shape}"
            )
            
        # Initialize analyzer
        analyzer = SingleGroupHMM(config)
        
        # Setup output directories
        analyzer.setup_output_directories(output_dir)
        if group_name.lower() == "balanced":
            # save selected indices
            with open(analyzer.stats_dir / f"{group_name}_selected_indices.json", 'w') as f:
                json.dump(selected_indices, f, indent=4)
        
        # Process data
        logging.info("Preprocessing data...")
        processed_data = analyzer.preprocess_data(data)
        
        # Fit model with progress tracking
        logging.info("Fitting HMM model...")
        with tqdm(total=1, desc="Model fitting") as pbar:
            analyzer.model = analyzer.fit_model(config.n_states)
            if analyzer.model is None:
                raise ValueError("Model fitting failed")
            pbar.update(1)
            
        # Get state sequence
        analyzer.state_sequence = analyzer.model.predict(processed_data)
        
        # Run analyses with progress tracking
        analysis_steps = [
            ("Analyzing state metrics", analyzer.analyze_state_metrics),
            ("Analyzing state properties", analyzer.analyze_state_properties),
            ("Analyzing subject consistency", analyzer.analyze_subject_consistency)
        ]
    
        
        results = {}
        for desc, func in analysis_steps:
            logging.info(desc)
            with tqdm(total=1, desc=desc) as pbar:
                results[func.__name__] = func()
                pbar.update(1)
        
        logging.info("Performing leave-one-subject-out cross-validation...")
        cv_results = analyzer.analyze_with_loocv()
        
        # Add CV results to overall results
        results['cross_validation'] = cv_results

        # Generate visualizations
        logging.info("Generating visualizations...")
        analyzer.generate_visualizations(
            group_name,
            results['analyze_state_metrics'],
            results['analyze_state_properties'],
            results['analyze_subject_consistency']
        )
        
        # Save results
        logging.info("Saving results...")
        analyzer.save_results(
            group_name,
            results['analyze_state_metrics'],
            results['analyze_state_properties'],
            results['analyze_subject_consistency']
        )
        cv_path = analyzer.stats_dir / f"{group_name}_cv_results.json"
        
        with open(cv_path, 'w') as f:
            json.dump({
                'log_likelihood': cv_results['log_likelihood'].tolist(),
                'mean_log_likelihood': float(cv_results['mean_log_likelihood']),
                'state_reliability': float(cv_results['state_reliability']),
                'n_states': config.n_states
            }, f, indent=4)

        return results
        
    except Exception as e:
        logging.error(f"Error in analysis pipeline: {str(e)}")
        raise

def main():
    """Main execution function with enhanced error handling and logging."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Single Group HMM Analysis for Neural State Dynamics'
    )
    
    parser.add_argument('--n_states', type=int, required=True,
                       help='Number of states to use')
    
    # Optional arguments
    parser.add_argument('--group', type=str, default="combined",
                        choices=["affair", "paranoia", "combined", "constructed"],
                        help='Group to analyze (affair, paranoia, combined, or constructed)')
    parser.add_argument('--res', type=str, default="native",
                       help='Resolution of the atlas')
    parser.add_argument('--trim', type=bool, default=True,
                       help='Trim the data to remove the resting state')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--tr', type=float, default=1.5,
                       help='Repetition time in seconds')
    parser.add_argument('--window_size', type=int, default=20,
                       help='Window size for temporal analyses (seconds)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of CPU cores to use for parallel processing')
    
    args = parser.parse_args()
    group_name = args.group
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Setup paths
        scratch_dir = os.getenv("SCRATCH_DIR")
        if not scratch_dir:
            raise ValueError("SCRATCH_DIR environment variable not set")
            
        if args.trim:
            output_dir = Path(scratch_dir) / "output" / f"04_{group_name}_hmm_{args.n_states}states_ntw_{args.res}_trimmed"
        else:
            output_dir = Path(scratch_dir) / "output" / f"04_{group_name}_hmm_{args.n_states}states_ntw_{args.res}"
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(group_name, output_dir)
        
        logging.info(f"Starting HMM analysis for group: {group_name}")
        logging.info(f"Parameters: n_states={args.n_states}, res={args.res}, trim={args.trim}")
        
        # Create HMMConfig
        config = HMMConfig(
            n_states=args.n_states,
            random_state=args.random_state,
            tr=args.tr,
            window_size=args.window_size,
            n_bootstrap=args.n_bootstrap,
            n_jobs=args.n_jobs 
        )
        
        # Load and prepare data
        logging.info("Loading data...")
        group_data, selected_indices = load_group_data(scratch_dir, args.res, group_name)
        
        if args.trim:
            group_data = group_data[:, :, 17:468]
        
        # Run analysis
        run_single_group_analysis(
            data=group_data,
            selected_indices=selected_indices,
            group_name=group_name,
            output_dir=str(output_dir),
            config=config
        )
        
        logging.info("Analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

def setup_logging(group_name: str, output_dir: Path) -> None:
    """Setup logging with a more robust configuration approach."""
    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = logs_dir / f"{group_name}_hmm_analysis_{timestamp}.log"
    
    # Reset root logger by removing all handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatters
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")

def load_group_data(scratch_dir: str, res: str, group_name: str) -> np.ndarray:
    """
    Load and validate group data based on subject IDs from environment variables.
    
    Args:
        scratch_dir: Scratch directory path
        res: Resolution string (e.g., "native")
        group_name: Group name ("affair", "paranoia", or "combined")
        
    Returns:
        Numpy array of group data with shape (n_subjects, n_networks, n_timepoints)
    """
    try:
        data_path = Path(scratch_dir) / "output" / f"03_network_data_{res}"
        
        # Get subject IDs from environment variables
        affair_subjects = os.getenv("AFFAIR_SUBJECTS", "").split(",")
        paranoia_subjects = os.getenv("PARANOIA_SUBJECTS", "").split(",")
        
        # Determine which subject IDs to load based on group_name
        if group_name.lower() == "affair":
            subjects_to_load = affair_subjects
            logging.info(f"Loading {len(subjects_to_load)} affair subjects")
        elif group_name.lower() == "paranoia":
            subjects_to_load = paranoia_subjects
            logging.info(f"Loading {len(subjects_to_load)} paranoia subjects")
        elif group_name.lower() == "combined":
            subjects_to_load = affair_subjects + paranoia_subjects
            logging.info(f"Loading {len(subjects_to_load)} combined subjects ({len(affair_subjects)} affair, {len(paranoia_subjects)} paranoia)")
        elif group_name.lower() == "balanced":
            # Set a fixed random seed for reproducibility
            np.random.seed(42)
            
            # Randomly select 9 subjects from affair
            affair_indices = np.random.choice(len(affair_subjects), size=9, replace=False)
            selected_affair = [affair_subjects[i] for i in affair_indices]
            
            # Randomly select 10 subjects from paranoia
            paranoia_indices = np.random.choice(len(paranoia_subjects), size=10, replace=False)
            selected_paranoia = [paranoia_subjects[i] for i in paranoia_indices]
            
            # Create dictionary to return subject indices
            selected_indices = {
                'affair': affair_indices.tolist(),
                'paranoia': paranoia_indices.tolist(),
                'selected_subjects': {
                    'affair': selected_affair,
                    'paranoia': selected_paranoia
                }
            }
            
            subjects_to_load = selected_affair + selected_paranoia
            logging.info(f"Balanced group with {len(subjects_to_load)} subjects (9 affair, 10 paranoia)")
        else:
            raise ValueError(f"Unknown group name: {group_name}. Valid options are 'affair', 'paranoia', 'combined', or 'balanced'")
        
        # Find all network files
        all_files = list(data_path.glob("*.npy"))
        if not all_files:
            raise ValueError(f"No .npy files found in {data_path}")
            
        # Extract unique network names from file patterns
        network_names = set()
        for file_path in all_files:
            # Based on file pattern like: sub-023_DefaultA_network_data.npy
            parts = file_path.name.split('_')
            if len(parts) >= 2:
                network_name = parts[1]  # Second part is the network name
                network_names.add(network_name)
        
        # Sort network names for consistent ordering
        network_names = sorted(list(network_names))
        logging.info(f"Discovered {len(network_names)} networks: {', '.join(network_names)}")
        
        # Initialize data dictionary for each subject
        all_subject_data = {}
        for subject_id in subjects_to_load:
            all_subject_data[subject_id] = {network: None for network in network_names}
        
        # Find and load all network files for each subject
        for subject_id in subjects_to_load:
            subject_files = list(data_path.glob(f"{subject_id}_*_*.npy"))
            
            if not subject_files:
                logging.warning(f"No data files found for subject {subject_id}")
                continue
            
            # Process each file for this subject
            for file_path in subject_files:
                # Parse network name from filename (assuming format: {subj_id}_{network_name}_*.npy)
                parts = file_path.name.split('_')
                if len(parts) < 2:
                    logging.warning(f"Unexpected filename format: {file_path.name}")
                    continue
                
                network_name = parts[1]  # Extract network name
                
                # Verify network is in our discovered list
                if network_name not in network_names:
                    logging.warning(f"Network name '{network_name}' not in discovered networks list")
                    continue
                
                # Load data
                try:
                    data = np.load(file_path)
                    
                    # Log the shape of the first network file we encounter
                    if subject_id == subjects_to_load[0] and network_name == network_names[0]:
                        logging.info(f"Network data shape for {file_path.name}: {data.shape}")
                    
                    all_subject_data[subject_id][network_name] = data
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {str(e)}")
        
        # Check that all networks are loaded for each subject
        valid_subjects = []
        for subject_id, networks in all_subject_data.items():
            missing_networks = [network for network, data in networks.items() if data is None]
            
            if missing_networks:
                logging.warning(f"Subject {subject_id} is missing networks: {missing_networks}")
                continue
                
            valid_subjects.append(subject_id)
        
        if not valid_subjects:
            raise ValueError(f"No subjects with complete network data for {group_name} group")
        
        logging.info(f"Found {len(valid_subjects)} subjects with complete data")
        
        # Determine timepoints from first subject's first network
        first_subject = all_subject_data[valid_subjects[0]]
        first_network_data = first_subject[network_names[0]]
        
        # Determine dimensions based on the shape of first network data
        if first_network_data.ndim > 1:
            n_timepoints = first_network_data.shape[1]  # Assuming [n_parcels, n_timepoints]
            logging.info(f"Network data contains multiple parcels per network: {first_network_data.shape}")
        else:
            n_timepoints = len(first_network_data)  # Single time series
            logging.info(f"Network data contains single time series per network: {first_network_data.shape}")
        
        # Initialize output array
        output_data = np.zeros((len(valid_subjects), len(network_names), n_timepoints))
        
        # Fill output array in correct order, averaging across parcels within each network
        for i, subject_id in enumerate(valid_subjects):
            for j, network in enumerate(network_names):
                network_data = all_subject_data[subject_id][network]
                
                # Average across ROIs/parcels within the network
                if network_data.ndim > 1 and network_data.shape[0] > 1:
                    # Assuming shape [n_parcels, n_timepoints]
                    network_avg = np.mean(network_data, axis=0)
                    output_data[i, j, :] = network_avg
                    
                    # Log the first time we do averaging
                    if i == 0 and j == 0:
                        logging.info(f"Averaging {network_data.shape[0]} parcels for each network")
                else:
                    # If there's only one parcel or data is already a single time series
                    if network_data.ndim > 1:
                        output_data[i, j, :] = network_data[0]
                    else:
                        output_data[i, j, :] = network_data
        
        logging.info(f"Final data shape: (n_subjects, n_networks, n_timepoints) = {output_data.shape}")
        
        return output_data, selected_indices
        
    except Exception as e:
        logging.error(f"Error loading group data: {str(e)}")
        raise

if __name__ == "__main__":
    main()