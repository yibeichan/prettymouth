import os
from dotenv import load_dotenv
import json
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HierarchicalStateAnalysis:
    def __init__(self, data_dir, output_dir, n_cv_folds=5):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_cv_folds = n_cv_folds
        self.data_validated = False
        
    def load_data(self):
        """Load all necessary data and prepare for analysis"""
        try:
            # Load content features
            self.content_df = pd.read_csv(self.data_dir / '10_story_annotations_TR.csv')
            
            # Load state sequences
            self.affair_sequences = np.load(
                self.output_dir / 'affair_hmm_3states_ntw_native_trimmed/statistics/affair_state_sequences.npy'
            )
            self.paranoia_sequences = np.load(
                self.output_dir / 'paranoia_hmm_3states_ntw_native_trimmed/statistics/paranoia_state_sequences.npy'
            )

            # Initial data validation
            self._validate_data_initial()
            
            # Prepare feature matrices
            self._prepare_features()
            
            # Validate feature matrix
            self._validate_data_features()
            
            self.data_validated = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def _validate_data_initial(self) -> bool:
        """Initial data validation before feature preparation"""
        # Check timepoint alignment
        n_timepoints = len(self.content_df)
        if not (self.affair_sequences.shape[1] == self.paranoia_sequences.shape[1] == n_timepoints):
            raise ValueError(f"Mismatched timepoints")
        
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
        
        # Validate state labels
        unique_states = np.unique(np.concatenate([self.affair_sequences.flatten(), 
                                                self.paranoia_sequences.flatten()]))
        if not np.all(np.isin(unique_states, [0, 1, 2])):
            raise ValueError("Invalid state labels found")
        
        return True

    def _validate_data_features(self) -> bool:
        """Validate feature matrix after preparation"""
        if len(self.feature_matrix) != len(self.content_df):
            raise ValueError("Feature matrix length mismatch")
            
        if np.any(np.isinf(self.feature_matrix)) or np.any(np.isnan(self.feature_matrix)):
            raise ValueError("Invalid values in feature matrix")
        
        # Validate binary values
        for col in self.feature_matrix.columns:
            unique_vals = np.unique(self.feature_matrix[col])
            if not np.all(np.isin(unique_vals, [0, 1])):
                raise ValueError(f"Non-binary values found in column {col}: {unique_vals}")
        
        return True
        
    def _prepare_features(self):
        """Prepare feature matrices for analysis"""
        # Main features
        features = [
            'lee_girl_together', 'has_verb', 'lee_speaking', 'girl_speaking',
            'arthur_speaking', 'has_adj', 'has_adv', 'has_noun'
        ]
        
        # Create base feature matrix
        X = self.content_df[features].copy()
        
        # Debug print before conversion
        print("\nBefore conversion:")
        for col in X.columns:
            print(f"\n{col} unique values:", X[col].unique())
        
        # Convert string boolean values to numeric
        for col in X.columns:
            X[col] = (X[col].astype(str).str.lower() == "true").astype(int)
            print(f"\nAfter converting {col}:")
            print(X[col].value_counts())
        
        # Add interaction terms
        X['lee_girl_verb'] = X['lee_girl_together'] * X['has_verb']
        X['arthur_adj'] = X['arthur_speaking'] * X['has_adj']
        
        self.feature_matrix = X
        
        # Debug print after preparing features
        print("\nFeature matrix head:")
        print(self.feature_matrix.head())
        print("\nFeature matrix value counts:")
        for col in self.feature_matrix.columns:
            print(f"\n{col}:")
            print(self.feature_matrix[col].value_counts())
        
    def prepare_subject_data(self, subject_idx, sequences, group):
        """
        Prepare data for a single subject
        
        Parameters:
        -----------
        subject_idx : int
            Subject identifier
        sequences : numpy.ndarray
            Subject's state sequence
        group : str
            'affair' or 'paranoia'
                
        Returns:
        --------
        pandas.DataFrame
            Subject-level data frame with states and features
        """
        # Create DataFrame with state sequences
        n_timepoints = len(sequences)
        data = pd.DataFrame({
            'subject': subject_idx,
            'group': group,
            'timepoint': range(n_timepoints),
            'state': sequences
        })
        
        # Add state indicators for each state
        for state in range(3):  # Assuming 3 states
            data[f'state_{state}'] = (sequences == state).astype(int)
        
        # Debug print before adding features
        print(f"\nPreparing subject {subject_idx} from {group} group")
        print("Feature matrix shape:", self.feature_matrix.shape)
        print("Data shape before features:", data.shape)
                
        # Add features
        for col in self.feature_matrix.columns:
            data[col] = self.feature_matrix[col].values
            print(f"Added feature {col}, unique values:", np.unique(data[col]))
                
        return data
    
    def run_subject_analysis(self, subject_data, state_idx):
        """
        Run analysis for a single subject and state
        
        Parameters:
        -----------
        subject_data : pandas.DataFrame
            Subject-level data
        state_idx : int
            State to analyze
            
        Returns:
        --------
        dict
            Subject-level results
        """
        # Prepare outcome variable
        y = subject_data[f'state_{state_idx}']
        
        # Prepare predictors
        feature_cols = [col for col in self.feature_matrix.columns]
        X = subject_data[feature_cols]
        
        # Add constant
        X = sm.add_constant(X)
        
        # Fit logistic regression (since outcome is binary)
        model = sm.Logit(y, X)
        try:
            results = model.fit(disp=0)
            
            return {
                'subject': subject_data['subject'].iloc[0],
                'group': subject_data['group'].iloc[0],
                'state': state_idx,
                'coefficients': results.params,
                'pvalues': results.pvalues,
                'aic': results.aic,
                'bic': results.bic,
                'success': True
            }
        except:
            return {
                'subject': subject_data['subject'].iloc[0],
                'group': subject_data['group'].iloc[0],
                'state': state_idx,
                'success': False
            }
    def _check_model_convergence(self, model_results, feature_name: str = None) -> Tuple[bool, Dict]:
        """
        Check model convergence and return diagnostic information
        
        Parameters:
        -----------
        model_results : statsmodels.regression.mixed_linear_model.MixedLMResults
            Fitted model results
        feature_name : str, optional
            Name of the feature being analyzed, for logging purposes
        
        Returns:
        --------
        Tuple[bool, Dict]
            - bool: True if model converged successfully
            - Dict: Diagnostic information about convergence
        """
        feature_msg = f" for feature {feature_name}" if feature_name else ""
        
        diagnostics = {
            'converged': getattr(model_results, 'converged', None),
            'scale': getattr(model_results, 'scale', None),
        }
        
        # Check convergence status if available
        if hasattr(model_results, 'converged') and not model_results.converged:
            warnings.warn(f"Model{feature_msg} failed to converge")
            return False, diagnostics
        
        # Check for potential issues with scale parameter
        if hasattr(model_results, 'scale') and model_results.scale > 1e3:
            warnings.warn(f"Large scale parameter detected{feature_msg}: {model_results.scale}")
            diagnostics['warning'] = 'large_scale'
        
        return True, diagnostics
    
    def _prepare_combined_data(self, state_affair: int, state_paranoia: int) -> pd.DataFrame:
        """Prepare combined data for analysis"""
        all_data = []
        
        # Add debug prints
        print("Feature matrix head:")
        print(self.feature_matrix.head())
        print("\nFeature matrix value counts:")
        for col in self.feature_matrix.columns:
            print(f"\n{col}:")
            print(self.feature_matrix[col].value_counts())
        
        # Affair group
        for subj in range(self.affair_sequences.shape[0]):
            subj_data = self.prepare_subject_data(
                subj, self.affair_sequences[subj], 'affair'
            )
            all_data.append(subj_data)
            
        # Paranoia group
        for subj in range(self.paranoia_sequences.shape[0]):
            subj_data = self.prepare_subject_data(
                subj, self.paranoia_sequences[subj], 'paranoia'
            )
            all_data.append(subj_data)
            
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Print debug info about combined data
        print("\nCombined data feature columns:")
        for col in self.feature_matrix.columns:
            print(f"\n{col}:")
            print(combined_data[col].value_counts())
        
        # Add target state column
        combined_data['target_state'] = np.where(
            combined_data['group'] == 'affair',
            combined_data[f'state_{state_affair}'],
            combined_data[f'state_{state_paranoia}']
        )
        
        return combined_data

    def run_hierarchical_analysis(self, state_affair, state_paranoia):
        """Run hierarchical analysis comparing specific states between groups"""
        print(f"Starting hierarchical analysis for states: affair={state_affair}, paranoia={state_paranoia}")
        
        # Prepare data for both groups
        combined_data = self._prepare_combined_data(state_affair, state_paranoia)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Run mixed model for each feature
        feature_results = {}
        for feature in self.feature_matrix.columns:
            try:
                print(f"\nProcessing feature: {feature}")
                
                # Prepare model data
                model_data = combined_data.copy()
                model_data['feature'] = model_data[feature]
                model_data['group_coded'] = (model_data['group'] == 'affair').astype(int)
                
                # Print model data info
                print(f"Target state values:", model_data['target_state'].value_counts())
                print(f"Group distribution:", model_data['group'].value_counts())
                
                # Create design matrix
                design_matrix = sm.add_constant(pd.DataFrame({
                    'feature': model_data['feature'],
                    'group': model_data['group_coded'],
                    'interaction': model_data['feature'] * model_data['group_coded']
                }))
                
                # Fit mixed model
                try:
                    model = MixedLM(
                        model_data['target_state'],
                        design_matrix,
                        groups=model_data['subject']
                    )
                    print("Model created successfully")
                    
                    results = model.fit(reml=True)
                    print("Model fit completed")
                    
                    # Basic convergence check
                    if hasattr(results, 'converged') and not results.converged:
                        print(f"Model failed to converge for feature {feature}")
                        continue
                    
                    print("Model parameters:")
                    print(results.params)
                    print("\nP-values:")
                    print(results.pvalues)
                    
                    feature_results[feature] = {
                        'coefficients': results.params,
                        'pvalues': results.pvalues,
                        'conf_int': results.conf_int(),
                        'aic': results.aic,
                        'bic': results.bic
                    }
                    print(f"Successfully processed feature {feature}")
                    
                except Exception as e:
                    print(f"Error during model fitting: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
            except Exception as e:
                print(f"Error processing feature {feature}: {str(e)}")
                continue
        
        if not feature_results:
            print("No features were successfully processed")
            print("This might be due to convergence issues or data problems.")
            return {'feature_results': {}, 'group_stats': {}}
        
        # Compute group-level statistics
        group_stats = self._compute_group_stats(combined_data, state_affair, state_paranoia)
        
        return {
            'feature_results': feature_results,
            'group_stats': group_stats
        }
        
    def _compute_group_stats(self, data, state_affair, state_paranoia):
        """
        Compute group-level statistics
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Combined data from both groups
        state_affair : int
            State index for affair group
        state_paranoia : int
            State index for paranoia group
            
        Returns:
        --------
        dict
            Group-level statistics
        """
        # Compute state occupancy
        affair_data = data[data['group'] == 'affair']
        paranoia_data = data[data['group'] == 'paranoia']
        
        affair_occupancy = affair_data[f'state_{state_affair}'].mean()
        paranoia_occupancy = paranoia_data[f'state_{state_paranoia}'].mean()
        
        # Compute transition probabilities
        affair_transitions = np.diff(affair_data[f'state_{state_affair}'])
        paranoia_transitions = np.diff(paranoia_data[f'state_{state_paranoia}'])
        
        return {
            'occupancy': {
                'affair': affair_occupancy,
                'paranoia': paranoia_occupancy
            },
            'transitions': {
                'affair': {
                    'entry_rate': (affair_transitions == 1).mean(),
                    'exit_rate': (affair_transitions == -1).mean()
                },
                'paranoia': {
                    'entry_rate': (paranoia_transitions == 1).mean(),
                    'exit_rate': (paranoia_transitions == -1).mean()
                }
            }
        }
    
    # Step 3: Add the cross-validation methods
    def run_cross_validation(self, state_affair: int, state_paranoia: int) -> Dict:
        """Run both subject-level and temporal cross-validation"""
        if not self.data_validated:
            self._validate_input_data()
            self._validate_data_features()
        
        print("Running cross-validation...")
        cv_results = {
            'subject_cv': self._run_subject_cv(state_affair, state_paranoia),
            'temporal_cv': self._run_temporal_cv(state_affair, state_paranoia)
        }
        
        return cv_results
    
    def _run_subject_cv(self, state_affair: int, state_paranoia: int) -> List[Dict]:
        """Implement leave-one-subject-out cross-validation"""
        cv_results = []
        
        for group in ['affair', 'paranoia']:
            sequences = (self.affair_sequences if group == 'affair' 
                       else self.paranoia_sequences)
            target_state = state_affair if group == 'affair' else state_paranoia
            
            for subject_idx in tqdm(range(sequences.shape[0]), 
                                  desc=f"Running subject CV for {group}"):
                cv_results.append(
                    self._run_single_subject_cv(sequences, subject_idx, group, target_state)
                )
        
        return cv_results
    
    def _run_single_subject_cv(self, sequences, subject_idx, group, target_state):
        """Run cross-validation for a single subject"""
        try:
            # Prepare data
            train_idx = [i for i in range(sequences.shape[0]) if i != subject_idx]
            train_sequences = sequences[train_idx]
            
            train_data = pd.concat([
                self.prepare_subject_data(i, train_sequences[i], group)
                for i in range(len(train_idx))
            ])
            
            test_data = self.prepare_subject_data(subject_idx, sequences[subject_idx], group)
            
            train_data['target_state'] = train_data[f'state_{target_state}']
            test_data['target_state'] = test_data[f'state_{target_state}']
            
            # Fit model
            model = MixedLM(
                train_data['target_state'],
                sm.add_constant(train_data[self.feature_matrix.columns]),
                groups=train_data['subject']
            )
            results = model.fit(reml=True)
            
            # Check convergence
            converged, diagnostics = self._check_model_convergence(
                results, 
                f"subject_{subject_idx}_cv"
            )
            if not converged:
                return {
                    'subject': subject_idx,
                    'group': group,
                    'success': False,
                    'error': 'Model failed to converge',
                    'diagnostics': diagnostics
                }
            
            # Predict and evaluate
            test_pred = results.predict(
                sm.add_constant(test_data[self.feature_matrix.columns])
            )
            
            mse = np.mean((test_data['target_state'] - test_pred) ** 2)
            corr = np.corrcoef(test_data['target_state'], test_pred)[0, 1]
            
            return {
                'subject': subject_idx,
                'group': group,
                'mse': mse,
                'correlation': corr,
                'success': True,
                'convergence_diagnostics': diagnostics
            }
            
        except Exception as e:
            return {
                'subject': subject_idx,
                'group': group,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _run_temporal_cv(self, state_affair: int, state_paranoia: int) -> List[Dict]:
        """Implement temporal cross-validation"""
        try:
            cv_results = []
            combined_data = self._prepare_combined_data(state_affair, state_paranoia)
            
            kf = KFold(n_splits=self.n_cv_folds, shuffle=False)
            
            for fold_idx, (train_idx, test_idx) in enumerate(
                kf.split(range(len(self.content_df)))
            ):
                try:
                    # Prepare fold data
                    train_data = combined_data[combined_data['timepoint'].isin(train_idx)]
                    test_data = combined_data[combined_data['timepoint'].isin(test_idx)]
                    
                    # Fit model
                    model = MixedLM(
                        train_data['target_state'],
                        sm.add_constant(train_data[self.feature_matrix.columns]),
                        groups=train_data['subject']
                    )
                    results = model.fit(reml=True)
                    
                    # Check convergence
                    converged, diagnostics = self._check_model_convergence(
                        results, 
                        f"temporal_fold_{fold_idx}"
                    )
                    if not converged:
                        cv_results.append({
                            'fold': fold_idx,
                            'success': False,
                            'error': 'Model failed to converge',
                            'diagnostics': diagnostics
                        })
                        continue
                    
                    # Predict and evaluate
                    predictions = results.predict(
                        sm.add_constant(test_data[self.feature_matrix.columns])
                    )
                    mse = np.mean((test_data['target_state'] - predictions) ** 2)
                    
                    cv_results.append({
                        'fold': fold_idx,
                        'n_train': len(train_data),
                        'n_test': len(test_data),
                        'mse': mse,
                        'train_timepoints': train_idx.tolist(),
                        'test_timepoints': test_idx.tolist(),
                        'success': True,
                        'convergence_diagnostics': diagnostics
                    })
                    
                except Exception as e:
                    cv_results.append({
                        'fold': fold_idx,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    
            return cv_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to run temporal CV: {str(e)}")

    # Step 4: Add statistical enhancement methods
    def enhance_analysis_results(self, results: Dict) -> Dict:
        """Add statistical enhancements to analysis results"""
        enhanced_results = results.copy()
        
        # Add effect sizes
        enhanced_results['effect_sizes'] = self._calculate_effect_sizes(results)
        
        # Add corrected p-values
        enhanced_results['corrected_pvalues'] = self._apply_multiple_comparison_correction(results)
        
        # Add confidence intervals
        enhanced_results['confidence_intervals'] = self._compute_confidence_intervals(results)
        
        return enhanced_results
    
    def _calculate_effect_sizes(self, results: Dict) -> Dict:
        """Calculate effect sizes for model results"""
        effect_sizes = {}
        
        for feature, res in results['feature_results'].items():
            # Calculate Cohen's d
            group_effect = res['coefficients']['group']
            group_se = np.sqrt(res['conf_int'].loc['group', 1] - 
                             res['conf_int'].loc['group', 0]) / 3.92
            
            cohens_d = group_effect / group_se
            
            # Calculate odds ratios
            odds_ratio = np.exp(group_effect)
            odds_ratio_ci = np.exp(res['conf_int'].loc['group'])
            
            effect_sizes[feature] = {
                'cohens_d': cohens_d,
                'odds_ratio': odds_ratio,
                'odds_ratio_ci': odds_ratio_ci
            }
        
        return effect_sizes
    
    def _apply_multiple_comparison_correction(self, results: Dict) -> Dict:
        """
        Apply multiple comparison corrections (FDR and Bonferroni) to p-values
        
        Parameters:
            results (Dict): Dictionary containing analysis results with p-values
            
        Returns:
            Dict: Dictionary with corrected p-values for each feature and comparison
        """
        # Extract p-values for all features and comparisons
        feature_pvals = []
        feature_names = []
        
        # Check if we have feature results
        if not results.get('feature_results'):
            warnings.warn("No feature results found for multiple comparison correction")
            return {
                'feature_names': [],
                'original_pvals': {},
                'fdr_corrected': {},
                'bonferroni_corrected': {}
            }
        
        for feature, res in results['feature_results'].items():
            try:
                # Check if we have the expected p-values
                if not all(key in res.get('pvalues', {}) for key in ['feature', 'group', 'interaction']):
                    warnings.warn(f"Missing p-values for feature {feature}")
                    continue
                    
                # Extract p-values for main effects and interactions
                feature_pvals.extend([
                    res['pvalues']['feature'],
                    res['pvalues']['group'],
                    res['pvalues']['interaction']
                ])
                
                feature_names.extend([
                    f"{feature}_main",
                    f"{feature}_group",
                    f"{feature}_interaction"
                ])
            except Exception as e:
                warnings.warn(f"Error processing feature {feature}: {str(e)}")
                continue
        
        # Check if we have any p-values to correct
        if not feature_pvals:
            warnings.warn("No valid p-values found for multiple comparison correction")
            return {
                'feature_names': [],
                'original_pvals': {},
                'fdr_corrected': {},
                'bonferroni_corrected': {}
            }
        
        # Convert to numpy array for processing
        feature_pvals = np.array(feature_pvals)
        
        try:
            # Apply FDR correction
            _, fdr_pvals, _, _ = multipletests(
                feature_pvals,
                alpha=0.05,
                method='fdr_bh'
            )
            
            # Apply Bonferroni correction
            _, bonf_pvals, _, _ = multipletests(
                feature_pvals,
                alpha=0.05,
                method='bonferroni'
            )
            
            # Organize results
            return {
                'feature_names': feature_names,
                'original_pvals': dict(zip(feature_names, feature_pvals)),
                'fdr_corrected': dict(zip(feature_names, fdr_pvals)),
                'bonferroni_corrected': dict(zip(feature_names, bonf_pvals))
            }
            
        except Exception as e:
            warnings.warn(f"Error in multiple comparison correction: {str(e)}")
            return {
                'feature_names': feature_names,
                'original_pvals': dict(zip(feature_names, feature_pvals)),
                'fdr_corrected': {},
                'bonferroni_corrected': {}
            }

    def _compute_confidence_intervals(self, results: Dict, confidence_level: float = 0.95) -> Dict:
        """
        Compute confidence intervals for model parameters
        
        Parameters:
            results (Dict): Dictionary containing analysis results
            confidence_level (float): Desired confidence level (default: 0.95)
            
        Returns:
            Dict: Dictionary with confidence intervals for each feature and parameter
        """
        ci_results = {}
        
        for feature, res in results['feature_results'].items():
            # Get parameter estimates and standard errors
            params = res['coefficients']
            
            # Calculate critical value for desired confidence level
            crit_val = stats.norm.ppf((1 + confidence_level) / 2)
            
            # Extract confidence intervals from model results
            ci_lower = res['conf_int'].iloc[:, 0]
            ci_upper = res['conf_int'].iloc[:, 1]
            
            # Calculate margin of error
            margin_error = (ci_upper - ci_lower) / 2
            
            # Store results for this feature
            ci_results[feature] = {
                'confidence_level': confidence_level,
                'parameters': {
                    param: {
                        'estimate': params[param],
                        'ci_lower': ci_lower[param],
                        'ci_upper': ci_upper[param],
                        'margin_error': margin_error[param]
                    }
                    for param in params.index
                }
            }
            
            # Add standardized effect sizes with CIs
            if 'group' in params.index:
                # Calculate standardized effect size (Cohen's d)
                effect_size = params['group'] / np.sqrt(margin_error['group'] * 2)
                
                # Calculate CI for effect size
                es_ci_lower = effect_size - crit_val * np.sqrt(1/self.n_cv_folds)
                es_ci_upper = effect_size + crit_val * np.sqrt(1/self.n_cv_folds)
                
                ci_results[feature]['effect_size'] = {
                    'cohens_d': effect_size,
                    'ci_lower': es_ci_lower,
                    'ci_upper': es_ci_upper
                }
        
        # Add summary statistics
        ci_results['summary'] = {
            'confidence_level': confidence_level,
            'n_comparisons': len(results['feature_results']),
            'critical_value': crit_val
        }
        
        return ci_results

    def run_complete_analysis(self, state_affair: int, state_paranoia: int) -> Dict:
        """Run complete analysis pipeline with enhancements"""
        # Ensure data is validated
        if not self.data_validated:
            self._validate_data()
        
        # Run main analysis
        main_results = self.run_hierarchical_analysis(state_affair, state_paranoia)
        
        # Check if we have valid results
        if not main_results.get('feature_results'):
            raise ValueError("No feature results produced by hierarchical analysis")
    
        # Run cross-validation
        cv_results = self.run_cross_validation(state_affair, state_paranoia)
        
        # Enhance results with additional statistics
        enhanced_results = self.enhance_analysis_results(main_results)
        
        # Combine all results
        complete_results = {
            'main_analysis': enhanced_results,
            'cross_validation': cv_results,
            'metadata': {
                'states_compared': {
                    'affair': state_affair,
                    'paranoia': state_paranoia
                },
                'n_cv_folds': self.n_cv_folds,
                'timestamp': pd.Timestamp.now()
            }
        }
        
        return complete_results
    
    def save_results(self, results):
        """
        Save complete analysis results
        """
        out_dir = self.output_dir / "11_brain_content_analysis"
        os.makedirs(out_dir, exist_ok=True)
        
        # Save main analysis results if available
        if 'feature_results' in results.get('main_analysis', {}):
            main_results_df = pd.DataFrame(results['main_analysis']['feature_results'])
            main_results_df.to_csv(out_dir / 'main_analysis_results.csv')
        
        # Save cross-validation results if available
        cv_results = results.get('cross_validation', {})
        if 'subject_cv' in cv_results:
            subject_cv_df = pd.DataFrame(cv_results['subject_cv'])
            subject_cv_df.to_csv(out_dir / 'subject_cv_results.csv')
        
        if 'temporal_cv' in cv_results:
            temporal_cv_df = pd.DataFrame(cv_results['temporal_cv'])
            temporal_cv_df.to_csv(out_dir / 'temporal_cv_results.csv')
        
        # Convert timestamp to string in metadata before saving
        if 'metadata' in results:
            metadata = results['metadata'].copy()
            if 'timestamp' in metadata:
                metadata['timestamp'] = metadata['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            with open(out_dir / 'analysis_metadata.json', 'w') as f:
                json.dump(metadata, f)

        # Prepare output dictionary with available data
        output_dict = {
            'metadata': {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'states_compared': results['metadata']['states_compared'],
                'n_subjects': {
                    'affair': self.affair_sequences.shape[0],
                    'paranoia': self.paranoia_sequences.shape[0]
                }
            },
            'main_analysis': results['main_analysis'],
            'cross_validation': results['cross_validation']
        }
        
        # Save as compressed numpy archive
        np.savez_compressed(
            out_dir / f'hierarchical_analysis_results_{output_dict["metadata"]["timestamp"].replace(" ", "_").replace(":", "-")}.npz',
            **output_dict
        )

def main():
    load_dotenv()

    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = Path(scratch_dir) / "output"
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    
    # Initialize analysis
    analysis = HierarchicalStateAnalysis(
        data_dir=data_dir,
        output_dir=output_dir,
        n_cv_folds=5
    )
    
    try:
        # Load and validate data
        analysis.load_data()
        
        # Run enhanced analysis pipeline
        results = analysis.run_complete_analysis(
            state_affair=0,
            state_paranoia=1
        )
        
        # Check the enhanced results
        print("Effect sizes:", results['main_analysis']['effect_sizes'])
        print("Corrected p-values:", results['main_analysis']['corrected_pvalues'])
        print("Cross-validation results:", results['cross_validation'])
        # Save results
        analysis.save_results(results)
        
        print("Analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()