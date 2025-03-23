import os
from dotenv import load_dotenv
import json
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import time
from datetime import datetime
from utils.glmm import BayesianGLMMAnalyzer

warnings.filterwarnings('ignore')

class HierarchicalStateAnalysis:
    def __init__(self, data_type, cluster_id, threshold, data_dir, output_dir, n_cv_folds=5, coding_type="deviation", reference_group="affair", log_level=logging.INFO):
        """Initialize the state analysis with paths and parameters"""
        self.data_type = data_type
        self.cluster_id = cluster_id
        self.threshold = f"{threshold:.2f}".replace('.', '')
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_cv_folds = n_cv_folds
        self.data_validated = False
        self.coding_type = coding_type
        self.reference_group = reference_group
        
        self.folder_name = f"cluster{self.cluster_id}_{self.data_type}_{self.coding_type}_th{self.threshold}"
        # Setup logging
        self.setup_logging(log_level)
        self.logger.info(f"Initializing analysis: data_type={data_type}, cluster_id={cluster_id}")
        self.logger.info(f"Parameters: coding_type={coding_type}, reference_group={reference_group}, n_cv_folds={n_cv_folds}")
        
        # Initialize the GLMM analyzer
        self.glmm_analyzer = BayesianGLMMAnalyzer(
            coding_type=coding_type,
            reference_group=reference_group,
            ar_lags=2  # Set default autoregressive lags
        )
        self.logger.debug("GLMM analyzer initialized")
    
    def setup_logging(self, log_level):
        """Setup logging configuration"""
        # Create folder name for output
        self.log_dir = self.output_dir / "09_brain_content_glmm" / self.folder_name
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamp for log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"analysis_{timestamp}.log"
        
        # Configure logger
        self.logger = logging.getLogger(f"state_analysis_{self.cluster_id}_{self.data_type}")
        self.logger.setLevel(log_level)
        
        # Remove any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def load_data(self):
        """Load all necessary data and prepare for analysis"""
        self.logger.info("Loading data files")
        start_time = time.time()
        
        try:
            # Load content features
            self.logger.debug(f"Loading content features from {self.data_dir / '10_story_annotations_tr.csv'}")
            self.content_df = pd.read_csv(self.data_dir / '10_story_annotations_tr.csv')
            self.logger.info(f"Loaded content features: {len(self.content_df)} timepoints")
            
            if self.data_type == "combined":
                self.logger.debug(f"Loading combined sequence data for cluster {self.cluster_id}")
                sequence_path = self.output_dir / '07_cluster_state_mapping' / f'th_{self.threshold}' / 'combined' / f'cluster_{self.cluster_id}_sequences.npy'
                self.logger.debug(f"Sequence path: {sequence_path}")
                
                self.sequence_data = np.load(sequence_path)
                n_subjects = self.sequence_data.shape[0]
                self.logger.info(f"Loaded combined sequence data: {n_subjects} subjects, {self.sequence_data.shape[1]} timepoints")
                
                self.affair_sequences = self.sequence_data[:n_subjects//2, :]
                self.paranoia_sequences = self.sequence_data[n_subjects//2:, :] 
                self.logger.info(f"Split data: {self.affair_sequences.shape[0]} affair subjects, {self.paranoia_sequences.shape[0]} paranoia subjects")
                
            elif self.data_type == "paired":
                self.logger.debug(f"Loading separate sequence data for cluster {self.cluster_id}")
                
                affair_path = self.output_dir / '07_cluster_state_mapping' / f'th_{self.threshold}' / 'affair' / f'cluster_{self.cluster_id}_sequences.npy'
                paranoia_path = self.output_dir / '07_cluster_state_mapping' / f'th_{self.threshold}' / 'paranoia' / f'cluster_{self.cluster_id}_sequences.npy'
                
                self.logger.debug(f"Affair path: {affair_path}")
                self.logger.debug(f"Paranoia path: {paranoia_path}")
                
                self.affair_sequences = np.load(affair_path)
                self.paranoia_sequences = np.load(paranoia_path)
                
                self.logger.info(f"Loaded affair sequences: {self.affair_sequences.shape[0]} subjects, {self.affair_sequences.shape[1]} timepoints")
                self.logger.info(f"Loaded paranoia sequences: {self.paranoia_sequences.shape[0]} subjects, {self.paranoia_sequences.shape[1]} timepoints")
            
            # Initial data validation
            self.logger.info("Validating data")
            self._validate_data_initial()
            
            # Prepare feature matrices
            self.logger.info("Preparing feature matrices")
            self._prepare_features()
            
            # Validate feature matrix
            self.logger.info("Validating feature matrices")
            self._validate_data_features()
            
            self.data_validated = True
            elapsed_time = time.time() - start_time
            self.logger.info(f"Data loaded and validated successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def _validate_data_initial(self) -> bool:
        """Initial data validation before feature preparation"""
        self.logger.debug("Starting initial data validation")
        
        # Check timepoint alignment
        n_timepoints = len(self.content_df)
        if not (self.affair_sequences.shape[1] == self.paranoia_sequences.shape[1] == n_timepoints):
            error_msg = f"Mismatched timepoints: content={n_timepoints}, affair={self.affair_sequences.shape[1]}, paranoia={self.paranoia_sequences.shape[1]}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            self.logger.debug(f"Timepoints aligned: {n_timepoints}")
        
        # Check for missing values
        missing_counts = self.content_df.isna().sum()
        missing_columns = missing_counts[missing_counts > 0]
        if not missing_columns.empty:
            self.logger.warning(f"Missing values found in columns: {missing_columns.to_dict()}")
        
        # Fill missing values with appropriate defaults
        self.content_df = self.content_df.fillna({
            'onset': 0,
            'onset_TR': 0,
            'adjusted_onset': 0,
            'text': '',
            'segment_onset': 0,
            'segment_text': '',
            'speaker': 'none',
            'main_char': 'none'
        })
        self.logger.debug("Missing values filled with defaults")
        
        # Validate state labels
        unique_states = np.unique(np.concatenate([self.affair_sequences.flatten(), 
                                                self.paranoia_sequences.flatten()]))
        self.logger.debug(f"Unique state labels: {unique_states}")
        
        if not np.all(np.isin(unique_states, [0, 1, 2])):
            error_msg = f"Invalid state labels found: {unique_states}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.debug("Initial data validation passed")
        return True

    def _validate_data_features(self) -> bool:
        """Validate feature matrix after preparation"""
        self.logger.debug("Starting feature matrix validation")
        
        if len(self.feature_matrix) != len(self.content_df):
            error_msg = f"Feature matrix length mismatch: features={len(self.feature_matrix)}, content={len(self.content_df)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            self.logger.debug(f"Feature matrix length matches content: {len(self.feature_matrix)}")
            
        # Check for invalid values
        if np.any(np.isinf(self.feature_matrix)) or np.any(np.isnan(self.feature_matrix)):
            inf_count = np.sum(np.isinf(self.feature_matrix))
            nan_count = np.sum(np.isnan(self.feature_matrix))
            error_msg = f"Invalid values in feature matrix: {inf_count} inf values, {nan_count} NaN values"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate binary values
        for col in self.feature_matrix.columns:
            unique_vals = np.unique(self.feature_matrix[col])
            if not np.all(np.isin(unique_vals, [0, 1])):
                error_msg = f"Non-binary values found in column {col}: {unique_vals}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug(f"Column {col} is properly binary")
        
        self.logger.debug("Feature matrix validation passed")
        return True
        
    def _prepare_features(self):
        """Prepare feature matrices for analysis"""
        self.logger.info("Preparing feature matrices for analysis")
        
        # Main features
        features = [
            'lee_girl_together', 'has_verb', 'lee_speaking', 'girl_speaking',
            'arthur_speaking', 'has_adj', 'has_adv', 'has_noun'
        ]
        self.logger.debug(f"Using base features: {features}")
        
        # Create base feature matrix
        X = self.content_df[features].copy()
        
        # Convert string boolean values to numeric
        for col in X.columns:
            X[col] = (X[col].astype(str).str.lower() == "true").astype(int)
            self.logger.debug(f"Converted {col} to binary: {X[col].value_counts().to_dict()}")
        
        # Add interaction terms
        self.logger.debug("Adding interaction terms")
        X['lee_girl_verb'] = X['lee_girl_together'] * X['has_verb']
        X['arthur_adj'] = X['arthur_speaking'] * X['has_adj']
        
        self.feature_matrix = X
        self.logger.info(f"Prepared feature matrix with {len(X.columns)} features")
        self.logger.debug(f"Feature columns: {X.columns.tolist()}")
    
    def prepare_state_data(self):
        """
        Prepare data for GLMM analysis by aligning state sequences from both groups
        
        Returns:
        --------
        tuple
            (dv_array, feature_matrix, group_labels) ready for GLMM analysis
        """
        self.logger.info("Preparing state data for GLMM analysis")
        
        # Extract dimensions
        n_affair_subjects = self.affair_sequences.shape[0]
        n_paranoia_subjects = self.paranoia_sequences.shape[0]
        n_subjects = n_affair_subjects + n_paranoia_subjects
        n_timepoints = self.affair_sequences.shape[1]
        
        self.logger.debug(f"Data dimensions: affair={n_affair_subjects}, paranoia={n_paranoia_subjects}, timepoints={n_timepoints}")
        
        # Create dependent variable array (subjects × timepoints)
        dv = np.zeros((n_subjects, n_timepoints), dtype=int)
        
        # Fill in state data for affair group
        for i in range(n_affair_subjects):
            dv[i] = self.affair_sequences[i]
        
        # Fill in state data for paranoia group
        for i in range(n_paranoia_subjects):
            dv[i + n_affair_subjects] = self.paranoia_sequences[i]
        
        # Create group labels
        group_labels = ['affair'] * n_affair_subjects + ['paranoia'] * n_paranoia_subjects
        
        # Log state distribution
        state_counts = np.bincount(dv.flatten(), minlength=3)
        state_percentages = state_counts / dv.size * 100
        self.logger.info(f"State distribution: State 0: {state_percentages[0]:.1f}%, State 1: {state_percentages[1]:.1f}%, State 2: {state_percentages[2]:.1f}%")
        
        self.logger.info(f"Prepared data: {n_subjects} subjects, {n_timepoints} timepoints")
        self.logger.info(f"Affair subjects: {n_affair_subjects}, Paranoia subjects: {n_paranoia_subjects}")
        self.logger.debug(f"DV shape: {dv.shape}")
        
        return dv, self.feature_matrix, group_labels
    
    def run_analysis(self):
        """
        Run the full GLMM analysis for comparing states between groups
        
        Returns:
        --------
        dict
            Complete analysis results
        """
        self.logger.info(f"Starting GLMM analysis for cluster {self.cluster_id}, data type {self.data_type}")
        start_time = time.time()
        
        # Prepare data in format needed for GLMM analysis
        self.logger.info("Preparing data for GLMM analysis")
        dv, feature_matrix, group_labels = self.prepare_state_data()
        
        # Run individual feature analyses
        feature_results = {}
        all_features = feature_matrix.columns.tolist()
        
        # First analyze each feature individually
        self.logger.info(f"Analyzing {len(all_features)} features individually")
        for feature in all_features:
            feature_start_time = time.time()
            self.logger.info(f"Analyzing feature: {feature}")
            try:
                # Create single-feature matrix
                feature_df = pd.DataFrame({feature: feature_matrix[feature].values})
                
                # Prepare data for GLMM
                self.logger.debug(f"Preparing GLMM data for feature {feature}")
                model_data = self.glmm_analyzer.prepare_data(
                    dv=dv,
                    feature_matrix=feature_df,
                    group_labels=group_labels,
                    include_ar_terms=True
                )
                
                # Fit model with the GLMM analyzer
                self.logger.debug(f"Fitting GLMM model for feature {feature}")
                result = self.glmm_analyzer.fit_model(
                    model_data=model_data,
                    feature_names=[feature],
                    include_interactions=True,
                    include_ar_terms=True
                )
                
                # Store result
                feature_results[feature] = result
                feature_time = time.time() - feature_start_time
                self.logger.info(f"Successfully analyzed feature: {feature} in {feature_time:.2f} seconds")
                
                # Log key results
                if 'coefficients' in result and 'group_coded' in result['coefficients']:
                    self.logger.debug(f"Feature {feature}: group coefficient = {result['coefficients']['group_coded']}")
                if 'posterior_prob' in result:
                    for param, prob in result['posterior_prob'].items():
                        if 'group' in param:
                            self.logger.debug(f"Feature {feature}: {param} posterior prob = {prob}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing feature {feature}: {str(e)}", exc_info=True)
        
        # Calculate state occupancy statistics
        self.logger.info("Computing state statistics")
        group_stats = self._compute_state_statistics()
        
        # Apply Bayesian multiple comparison correction
        self.logger.info("Applying multiple comparison correction")
        if feature_results:
            # Collect all feature results to correct together
            all_coefficients = {}
            for feature, result in feature_results.items():
                if 'coefficients' in result:
                    for param, value in result['coefficients'].items():
                        if param.startswith('group_') and param.endswith('_interaction'):
                            all_coefficients[f"{feature}:{param}"] = {
                                'feature': feature,
                                'parameter': param,
                                'coefficient': value,
                                'posterior_prob': result['posterior_prob'].get(param, 0)
                            }
            
            # Sort by posterior probability for significance
            sorted_effects = sorted(
                all_coefficients.items(), 
                key=lambda x: x[1]['posterior_prob'], 
                reverse=True
            )
            
            # Apply False Discovery Rate correction
            if sorted_effects:
                self.logger.debug(f"Applying FDR correction to {len(sorted_effects)} interaction effects")
                # Extract posterior probabilities and ensure they're scalar values
                probs = []
                for effect in sorted_effects:
                    prob_value = effect[1]['posterior_prob']
                    
                    # Convert to scalar if it's a list/array
                    if isinstance(prob_value, (list, tuple, np.ndarray)):
                        try:
                            # Try to extract first element if it's a list
                            prob_value = float(prob_value[0])
                        except (IndexError, TypeError, ValueError):
                            # Use 0.5 as default if conversion fails
                            self.logger.warning(f"Could not convert {prob_value} to scalar, using 0.5")
                            prob_value = 0.5
                    else:
                        # Ensure it's a float
                        try:
                            prob_value = float(prob_value)
                        except (TypeError, ValueError):
                            self.logger.warning(f"Could not convert {prob_value} to float, using 0.5")
                            prob_value = 0.5
                            
                    probs.append(prob_value)
                
                # Now calculate FDR with clean scalar values
                fdr_values = []
                for i in range(len(probs)):
                    # Use numpy to ensure proper vectorized operations
                    expected_false = np.sum(1.0 - np.array(probs[:i+1]))
                    fdr = expected_false / (i + 1)
                    fdr_values.append(fdr)
                
                # Find significant effects (FDR < 0.05)
                significant = [
                    sorted_effects[i][0] for i in range(len(fdr_values))
                    if fdr_values[i] < 0.05
                ]
                
                # Add FDR info to results
                fdr_results = {
                    'fdr_threshold': 0.05,
                    'significant_effects': significant,
                    'all_effects': {
                        effect[0]: {
                            'posterior_prob': float(probs[i]),  # Ensure stored as float
                            'fdr': float(fdr_values[i])         # Ensure stored as float
                        }
                        for i, effect in enumerate(sorted_effects)
                    }
                }
                
                self.logger.info(f"FDR correction complete. Found {len(significant)} significant interaction effects")
                for sig in significant:
                    self.logger.debug(f"Significant effect: {sig}")
            else:
                self.logger.info("No effects to apply FDR correction")
                fdr_results = {'significant_effects': []}
        else:
            self.logger.warning("No feature results available for multiple comparison correction")
            fdr_results = {'significant_effects': []}
            
        # Collect results
        analysis_results = {
            'feature_results': feature_results,
            'group_stats': group_stats,
            'multiple_comparison': fdr_results,
            'metadata': {
                'cluster_id': self.cluster_id,
                'coding_type': self.coding_type,
                'reference_group': self.reference_group,
                'timestamp': pd.Timestamp.now()
            }
        }
        
        total_time = time.time() - start_time
        self.logger.info(f"Analysis completed in {total_time:.2f} seconds")
        
        return analysis_results
    
    def _compute_state_statistics(self):
        """Compute basic state statistics for the groups"""
        self.logger.info("Computing state statistics")
        
        # Calculate state occupancy
        affair_occupancy = np.mean(self.affair_sequences)
        paranoia_occupancy = np.mean(self.paranoia_sequences)
        
        self.logger.debug(f"State occupancy - Affair: {affair_occupancy}, Paranoia: {paranoia_occupancy}")
        
        # Calculate transitions
        self.logger.debug("Calculating state transitions")
        affair_transitions = []
        for subj in range(self.affair_sequences.shape[0]):
            # 1 for entry, -1 for exit, 0 for no change
            subj_transitions = np.diff(self.affair_sequences[subj])
            affair_transitions.append(subj_transitions)
        
        paranoia_transitions = []
        for subj in range(self.paranoia_sequences.shape[0]):
            subj_transitions = np.diff(self.paranoia_sequences[subj])
            paranoia_transitions.append(subj_transitions)
        
        # Flatten transitions
        affair_transitions = np.concatenate(affair_transitions)
        paranoia_transitions = np.concatenate(paranoia_transitions)
        
        # Calculate transition rates
        affair_entry_rate = float(np.mean(affair_transitions == 1))
        affair_exit_rate = float(np.mean(affair_transitions == -1))
        paranoia_entry_rate = float(np.mean(paranoia_transitions == 1))
        paranoia_exit_rate = float(np.mean(paranoia_transitions == -1))
        
        self.logger.debug(f"Affair transitions - entry: {affair_entry_rate}, exit: {affair_exit_rate}")
        self.logger.debug(f"Paranoia transitions - entry: {paranoia_entry_rate}, exit: {paranoia_exit_rate}")
        
        return {
            'occupancy': {
                'affair': float(affair_occupancy),
                'paranoia': float(paranoia_occupancy)
            },
            'transitions': {
                'affair': {
                    'entry_rate': affair_entry_rate,
                    'exit_rate': affair_exit_rate
                },
                'paranoia': {
                    'entry_rate': paranoia_entry_rate,
                    'exit_rate': paranoia_exit_rate
                }
            }
        }
    
    def run_cross_validation(self):
        """Run leave-one-subject-out cross-validation"""
        self.logger.info(f"Running cross-validation for cluster {self.cluster_id}")
        start_time = time.time()
        cv_results = []
        
        # Get total number of subjects
        n_affair = self.affair_sequences.shape[0]
        n_paranoia = self.paranoia_sequences.shape[0]
        
        self.logger.info(f"Cross-validation with {n_affair} affair subjects and {n_paranoia} paranoia subjects")
        
        # Run CV for each subject
        for group in ['affair', 'paranoia']:
            group_size = n_affair if group == 'affair' else n_paranoia
            
            for subject_idx in range(group_size):
                cv_start = time.time()
                self.logger.info(f"Cross-validation for {group} subject {subject_idx}")
                
                try:
                    # Prepare leave-one-out data
                    if group == 'affair':
                        # Exclude current affair subject
                        train_affair = np.delete(self.affair_sequences, subject_idx, axis=0)
                        train_paranoia = self.paranoia_sequences
                    else:
                        # Exclude current paranoia subject
                        train_affair = self.affair_sequences
                        train_paranoia = np.delete(self.paranoia_sequences, subject_idx, axis=0)
                    
                    # Create DV array
                    n_train_affair = train_affair.shape[0]
                    n_train_paranoia = train_paranoia.shape[0]
                    n_train_total = n_train_affair + n_train_paranoia
                    n_timepoints = train_affair.shape[1]
                    
                    self.logger.debug(f"CV training data: {n_train_affair} affair, {n_train_paranoia} paranoia subjects")
                    
                    # Create training DV
                    dv_train = np.zeros((n_train_total, n_timepoints), dtype=int)
                    
                    # Fill in state data
                    for i in range(n_train_affair):
                        dv_train[i] = train_affair[i]
                    
                    for i in range(n_train_paranoia):
                        dv_train[i + n_train_affair] = train_paranoia[i]
                    
                    # Create group labels
                    group_labels = ['affair'] * n_train_affair + ['paranoia'] * n_train_paranoia
                    
                    # Prepare data for GLMM
                    self.logger.debug(f"Preparing GLMM data for CV fold")
                    model_data = self.glmm_analyzer.prepare_data(
                        dv=dv_train,
                        feature_matrix=self.feature_matrix,
                        group_labels=group_labels,
                        include_ar_terms=True
                    )
                    
                    # Fit model with key features
                    key_features = ['lee_girl_together', 'has_verb', 'arthur_speaking']
                    self.logger.debug(f"Fitting CV model with key features: {key_features}")
                    
                    result = self.glmm_analyzer.fit_model(
                        model_data=model_data,
                        feature_names=key_features,
                        include_interactions=True,
                        include_ar_terms=True
                    )
                    
                    # Extract relevant coefficients
                    coefficients = {}
                    if 'coefficients' in result:
                        for key, value in result['coefficients'].items():
                            if key in ['const', 'group_coded'] or key.startswith('group_'):
                                coefficients[key] = value
                                self.logger.debug(f"CV coefficient: {key} = {value}")
                    
                    # Add to CV results
                    cv_results.append({
                        'subject': subject_idx,
                        'group': group,
                        'success': True,
                        'coefficients': coefficients,
                        'n_train_subjects': n_train_total
                    })
                    
                    cv_time = time.time() - cv_start
                    self.logger.info(f"CV fold completed in {cv_time:.2f} seconds")
                    
                except Exception as e:
                    self.logger.error(f"Error in CV for {group} subject {subject_idx}: {str(e)}", exc_info=True)
                    cv_results.append({
                        'subject': subject_idx,
                        'group': group,
                        'success': False,
                        'error': str(e)
                    })
        
        total_cv_time = time.time() - start_time
        self.logger.info(f"Cross-validation completed in {total_cv_time:.2f} seconds. {len(cv_results)} CV folds processed.")
        
        return {'subject_cv': cv_results}
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        self.logger.info(f"Starting complete analysis pipeline for cluster {self.cluster_id}, data type {self.data_type}")
        start_time = time.time()
        
        # Ensure data is loaded
        if not self.data_validated:
            self.logger.info("Data not yet validated, loading data")
            self.load_data()
        
        # Run main analysis
        self.logger.info("Running main analysis")
        main_results = self.run_analysis()
        
        # Run cross-validation
        self.logger.info("Running cross-validation")
        cv_results = self.run_cross_validation()
        
        # Combine results
        complete_results = {
            'main_analysis': main_results,
            'cross_validation': cv_results,
            'metadata': {
                'cluster_id': self.cluster_id,
                'n_cv_folds': self.n_cv_folds,
                'timestamp': pd.Timestamp.now(),
                'coding_type': self.coding_type,
                'reference_group': self.reference_group
            }
        }
        
        total_time = time.time() - start_time
        self.logger.info(f"Complete analysis finished in {total_time:.2f} seconds")
        
        return complete_results
    
    def save_results(self, results):
        """Save analysis results to files"""
        # Create output directory
        out_dir = self.output_dir / "09_brain_content_glmm" / self.folder_name
        os.makedirs(out_dir, exist_ok=True)
        
        self.logger.info(f"Saving results to {out_dir}")
        
        # Extract feature results for summary dataframe
        if 'feature_results' in results['main_analysis']:
            # Prepare data for summary CSV
            summary_data = []
            
            for feature, res in results['main_analysis']['feature_results'].items():
                # Basic info
                feature_summary = {'feature': feature}
                
                # Extract group effect
                if 'coefficients' in res and 'group_coded' in res['coefficients']:
                    feature_summary['group_coefficient'] = res['coefficients']['group_coded']
                    self.logger.debug(f"Feature {feature}: group coefficient = {res['coefficients']['group_coded']}")
                    
                    # Get odds ratio if available
                    if 'odds_ratios' in res and 'group_coded' in res['odds_ratios']:
                        feature_summary['odds_ratio'] = res['odds_ratios']['group_coded']['odds_ratio']
                        feature_summary['odds_ratio_lower'] = res['odds_ratios']['group_coded']['lower']
                        feature_summary['odds_ratio_upper'] = res['odds_ratios']['group_coded']['upper']
                        self.logger.debug(f"Feature {feature}: odds ratio = {res['odds_ratios']['group_coded']['odds_ratio']}")
                    
                    # Get posterior probability
                    if 'posterior_prob' in res and 'group_coded' in res['posterior_prob']:
                        feature_summary['posterior_prob'] = res['posterior_prob']['group_coded']
                        self.logger.debug(f"Feature {feature}: posterior probability = {res['posterior_prob']['group_coded']}")
                
                # Get interaction effect
                interaction_key = f'group_{feature}_interaction'
                if 'coefficients' in res and interaction_key in res['coefficients']:
                    feature_summary['interaction_coefficient'] = res['coefficients'][interaction_key]
                    self.logger.debug(f"Feature {feature}: interaction coefficient = {res['coefficients'][interaction_key]}")
                    
                    # Get odds ratio for interaction
                    if 'odds_ratios' in res and interaction_key in res['odds_ratios']:
                        feature_summary['interaction_odds_ratio'] = res['odds_ratios'][interaction_key]['odds_ratio']
                    
                    # Get posterior probability for interaction
                    if 'posterior_prob' in res and interaction_key in res['posterior_prob']:
                        feature_summary['interaction_posterior_prob'] = res['posterior_prob'][interaction_key]
                
                # Check if significant in multiple comparison
                feature_interaction = f"{feature}:{interaction_key}"
                if 'multiple_comparison' in results['main_analysis']:
                    mc_results = results['main_analysis']['multiple_comparison']
                    feature_summary['significant'] = feature_interaction in mc_results.get('significant_effects', [])
                    
                    if feature_summary['significant']:
                        self.logger.info(f"Feature {feature}: SIGNIFICANT interaction effect after FDR correction")
                    
                    # Get FDR if available
                    if 'all_effects' in mc_results and feature_interaction in mc_results['all_effects']:
                        feature_summary['fdr'] = mc_results['all_effects'][feature_interaction]['fdr']
                        self.logger.debug(f"Feature {feature}: FDR = {feature_summary['fdr']}")
                
                # Add group-specific effects if available
                if 'group_specific_effects' in res and feature in res['group_specific_effects']:
                    effects = res['group_specific_effects'][feature]
                    
                    # Add affair group effects
                    if 'affair_group' in effects:
                        affair = effects['affair_group']
                        feature_summary['affair_odds_ratio'] = affair.get('odds_ratio')
                        feature_summary['affair_prob_positive'] = affair.get('prob_positive')
                    
                    # Add paranoia group effects
                    if 'paranoia_group' in effects:
                        paranoia = effects['paranoia_group']
                        feature_summary['paranoia_odds_ratio'] = paranoia.get('odds_ratio')
                        feature_summary['paranoia_prob_positive'] = paranoia.get('prob_positive')
                    
                    # Add difference information
                    if 'diff_between_groups' in effects:
                        diff = effects['diff_between_groups']
                        if 'prob_stronger_in_affair' in diff:
                            feature_summary['prob_stronger_in_affair'] = diff['prob_stronger_in_affair']
                
                summary_data.append(feature_summary)
            
            # Create and save dataframe
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = out_dir / 'feature_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                self.logger.info(f"Saved feature summary with {len(summary_data)} features to {summary_path}")
            else:
                self.logger.warning("No feature summary data to save")
        else:
            self.logger.warning("No feature results found in analysis output")
        
        # Save cross-validation results
        if 'cross_validation' in results and 'subject_cv' in results['cross_validation']:
            cv_data = results['cross_validation']['subject_cv']
            
            # Extract basic CV info for dataframe
            cv_summary = []
            for cv_result in cv_data:
                cv_row = {
                    'subject': cv_result['subject'],
                    'group': cv_result['group'],
                    'success': cv_result['success']
                }
                
                # Add coefficients if available
                if 'coefficients' in cv_result:
                    for key, value in cv_result['coefficients'].items():
                        cv_row[f'coef_{key}'] = value
                
                cv_summary.append(cv_row)
            
            # Save CV summary
            if cv_summary:
                cv_df = pd.DataFrame(cv_summary)
                cv_path = out_dir / 'cv_summary.csv'
                cv_df.to_csv(cv_path, index=False)
                self.logger.info(f"Saved cross-validation summary with {len(cv_summary)} results to {cv_path}")
            else:
                self.logger.warning("No cross-validation summary data to save")
        else:
            self.logger.warning("No cross-validation results found")
        
        # Save metadata
        metadata = results.get('metadata', {}).copy()
        if 'timestamp' in metadata:
            metadata['timestamp'] = metadata['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        metadata_path = out_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata to {metadata_path}")
        
        # Save full results as compressed numpy array
        try:
            # Convert non-serializable objects
            def convert_for_saving(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, pd.Timestamp):
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(obj, dict):
                    return {k: convert_for_saving(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_saving(item) for item in obj]
                else:
                    return obj
            
            # Convert and save results
            save_results = convert_for_saving(results)
            npz_path = out_dir / 'complete_results.npz'
            np.savez_compressed(npz_path, results=save_results)
            self.logger.info(f"Saved complete results as NPZ file to {npz_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving complete results: {str(e)}", exc_info=True)
            # Save simplified version if full save fails
            simplified_path = out_dir / 'results_simplified.npz'
            np.savez_compressed(
                simplified_path,
                metadata=metadata,
                simplified=True
            )
            self.logger.warning(f"Saved simplified results to {simplified_path} due to error")
            
def main():
    import argparse
    from pathlib import Path
    
    # Load environment variables
    load_dotenv()

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run hierarchical state analysis with specified parameters')
    
    # Add arguments
    parser.add_argument('--threshold', type=float, default=0.75, help='Threshold value (default: 0.75)')
    parser.add_argument('--data_type', default="combined", 
                        help='Data type to analyze (default: combined)')
    parser.add_argument('--cluster_id', type=int, default=1,
                        help='Cluster ID to analyze (default: 1)')
    parser.add_argument('--coding_type', default="deviation", 
                        help='Coding type for analysis (default: deviation)')
    parser.add_argument('--reference_group', default="affair", 
                        help='Reference group for analysis (default: affair)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up paths from environment variables
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = Path(scratch_dir) / "output"
    data_dir = Path(scratch_dir) / "data" / "stimuli"
    
    # Setup global logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(output_dir) / "analysis_main.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("main")
    
    logger.info("Starting hierarchical state analysis")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Parameters: threshold={args.threshold}, data_type={args.data_type}, "
               f"cluster_id={args.cluster_id}, coding_type={args.coding_type}, "
               f"reference_group={args.reference_group}")
    
    total_start_time = time.time()
    all_results = {}
    
    # Process single data type and cluster ID
    data_type = args.data_type
    cluster_id = args.cluster_id
    threshold = args.threshold
    
    cluster_start_time = time.time()
    logger.info(f"=== Starting analysis for data_type={data_type}, cluster_id={cluster_id} ===")
            
    # Initialize analysis
    analysis = HierarchicalStateAnalysis(

        data_type=data_type,
        cluster_id=cluster_id,
        threshold=threshold,
        data_dir=data_dir,
        output_dir=output_dir,
        n_cv_folds=5,
        coding_type=args.coding_type,
        reference_group=args.reference_group
    )
    
    try:
        # Load data
        logger.info(f"Loading data for {data_type}, cluster {cluster_id}")
        analysis.load_data()
        
        # Run analysis
        logger.info(f"Running complete analysis for {data_type}, cluster {cluster_id}")
        results = analysis.run_complete_analysis()
        
        # Save results
        logger.info(f"Saving results for {data_type}, cluster {cluster_id}")
        analysis.save_results(results)
        
        # Store results
        all_results[f"{data_type}_{cluster_id}"] = True
        
        cluster_time = time.time() - cluster_start_time
        logger.info(f"Analysis for data_type={data_type}, cluster_id={cluster_id} completed in {cluster_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during analysis for data_type={data_type}, cluster_id={cluster_id}: {str(e)}", exc_info=True)
        logger.error("Continuing with next cluster/data type")
        all_results[f"{data_type}_{cluster_id}"] = False
    
    total_time = time.time() - total_start_time
    logger.info(f"Complete analysis pipeline finished in {total_time:.2f} seconds")
    logger.info(f"Successfully completed: {sum(1 for v in all_results.values() if v)}/{len(all_results)} analyses")

if __name__ == "__main__":
    main()