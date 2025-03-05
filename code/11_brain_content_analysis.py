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
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

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
                
                # Create subject dummies for random effects
                subject_dummies = pd.get_dummies(model_data['subject']).astype(np.float64)
                
                # Create design matrix
                design_matrix = sm.add_constant(pd.DataFrame({
                    'feature': model_data['feature'],
                    'group': model_data['group_coded'],
                    'interaction': model_data['feature'] * model_data['group_coded']
                }))
                
                # Fit model
                try:
                    model = BinomialBayesMixedGLM(
                        model_data['target_state'],
                        design_matrix,
                        exog_vc=subject_dummies,
                        vc_names=['subject'],
                        ident=np.ones(subject_dummies.shape[1], dtype=np.int32)
                    )
                    print("Model created successfully")
                    
                    results = model.fit_map()
                    print("Model fit completed")
                    
                    # Prepare results using helper method
                    feature_results[feature] = self._prepare_glmm_results(results, design_matrix)
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
        
        return {
            'feature_results': feature_results,
            'group_stats': self._compute_group_stats(combined_data, state_affair, state_paranoia)
        }

    def _prepare_glmm_results(self, results, design_matrix):
        """Prepare GLMM results for a single feature analysis"""
        try:
            # Get feature names
            feature_names = design_matrix.columns
            
            # Get fixed effects parameters and standard deviations
            # Ensure we only get parameters for the fixed effects
            n_fe = len(feature_names)  # number of fixed effects
            fe_params = results.params[:n_fe]
            fe_sds = results.fe_sd[:n_fe]
            
            # Calculate z-scores and p-values
            z_scores = fe_params / fe_sds
            prob_nonzero = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            
            # Calculate 95% credible intervals
            ci_lower = fe_params - 1.96 * fe_sds
            ci_upper = fe_params + 1.96 * fe_sds
            
            # Convert everything to dictionaries with proper keys
            coef_dict = {name: val for name, val in zip(feature_names, fe_params)}
            sd_dict = {name: val for name, val in zip(feature_names, fe_sds)}
            zscore_dict = {name: val for name, val in zip(feature_names, z_scores)}
            prob_dict = {name: val for name, val in zip(feature_names, prob_nonzero)}
            ci_dict = {
                name: {'lower': l, 'upper': u} 
                for name, l, u in zip(feature_names, ci_lower, ci_upper)
            }
            
            # Prepare results dictionary
            results_dict = {
                'coefficients': coef_dict,
                'posterior_sds': sd_dict,
                'z_scores': zscore_dict,
                'prob_nonzero': prob_dict,
                'conf_int': ci_dict,
                'model_summary': str(results.summary()),
                'convergence': results.converged if hasattr(results, 'converged') else None,
                'n_obs': len(design_matrix)
            }
            
            # Add random effects variance if available
            if hasattr(results, 're_sd'):
                results_dict['random_effects'] = {
                    'variance': float(results.re_sd ** 2) if results.re_sd is not None else None
                }
            
            return results_dict
            
        except Exception as e:
            print(f"Error preparing GLMM results: {str(e)}")
            print(f"Results object attributes: {dir(results)}")
            print(f"fe_params shape: {fe_params.shape}, fe_sds shape: {fe_sds.shape}")
            print(f"Number of features: {n_fe}")
            print(f"Feature names: {feature_names}")
            print(f"Parameters: {fe_params}")
            print(f"Standard deviations: {fe_sds}")
            raise

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
        """Run cross-validation analysis"""
        subject_cv_results = []
        
        # Run leave-one-out CV for each subject
        for group in ['affair', 'paranoia']:
            sequences = (self.affair_sequences if group == 'affair' 
                        else self.paranoia_sequences)
            
            for subject in range(len(sequences)):
                try:
                    # Prepare data excluding current subject
                    train_data = self._prepare_combined_data(state_affair, state_paranoia)
                    train_data = train_data[
                        ~((train_data['group'] == group) & 
                          (train_data['subject'] == subject))
                    ]
                    
                    # Create subject dummies for random effects
                    subject_dummies = pd.get_dummies(train_data['subject']).astype(np.float64)
                    
                    # Prepare model data - use all features
                    model_data = train_data.copy()
                    feature_matrix = sm.add_constant(model_data[self.feature_matrix.columns])
                    
                    # Fit model
                    model = BinomialBayesMixedGLM(
                        model_data['target_state'],
                        feature_matrix,
                        exog_vc=subject_dummies,
                        vc_names=['subject'],
                        ident=np.ones(subject_dummies.shape[1], dtype=np.int32)
                    )
                    
                    results = model.fit_vb()
                    
                    subject_cv_results.append({
                        'subject': subject,
                        'group': group,
                        'success': True,
                        'coefficients': {
                            name: float(val) 
                            for name, val in zip(feature_matrix.columns, results.params)
                        }
                    })
                    
                except Exception as e:
                    print(f"Error in CV for {group} subject {subject}: {str(e)}")
                    subject_cv_results.append({
                        'subject': subject,
                        'group': group,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
        
        return {
            'subject_cv': subject_cv_results
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
                    model = sm.BinomialBayesMixedGLM(
                        train_data['target_state'],
                        sm.add_constant(train_data[self.feature_matrix.columns]),
                        groups=train_data['subject'],
                        var_names=self.feature_matrix.columns
                    )
                    results = model.fit_map()
                    
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
        """Calculate effect sizes from model results"""
        effect_sizes = {}
        
        for feature, res in results['feature_results'].items():
            try:
                # Get group effect coefficient and CI
                group_effect = res['coefficients']['group']
                group_ci = res['conf_int']['group']
                
                # Calculate standard error from CI
                ci_width = group_ci['upper'] - group_ci['lower']
                se = ci_width / (2 * 1.96)
                
                # Calculate Cohen's d
                cohens_d = group_effect / se
                
                # Calculate odds ratio and CI
                odds_ratio = np.exp(group_effect)
                odds_ratio_ci = {
                    'lower': np.exp(group_ci['lower']),
                    'upper': np.exp(group_ci['upper'])
                }
                
                effect_sizes[feature] = {
                    'cohens_d': float(cohens_d),
                    'odds_ratio': float(odds_ratio),
                    'odds_ratio_ci': odds_ratio_ci
                }
                
            except Exception as e:
                print(f"Error calculating effect size for feature {feature}: {str(e)}")
                print(f"Result structure for feature: {res.keys()}")
                effect_sizes[feature] = None
        
        return effect_sizes
    
    def _apply_multiple_comparison_correction(self, results: Dict) -> Dict:
        """Apply multiple comparison correction to p-values"""
        try:
            # Extract feature names and p-values from results
            feature_names = list(results['feature_results'].keys())
            feature_pvals = []
            
            # Collect p-values for each feature's group effect
            for feature in feature_names:
                pvals = results['feature_results'][feature]['prob_nonzero']
                if 'group' in pvals:
                    feature_pvals.append(pvals['group'])
                else:
                    print(f"Warning: No group p-value for feature {feature}")
                    feature_pvals.append(1.0)  # Conservative
            
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
            print(f"Error in multiple comparison correction: {str(e)}")
            print(f"Results structure: {results.keys()}")
            return {
                'feature_names': [],
                'original_pvals': {},
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
            conf_int = res['conf_int']
            
            # Store results for this feature
            ci_results[feature] = {
                'confidence_level': confidence_level,
                'parameters': {
                    param: {
                        'estimate': params[param],
                        'ci_lower': conf_int[param]['lower'],
                        'ci_upper': conf_int[param]['upper'],
                        'margin_error': (conf_int[param]['upper'] - conf_int[param]['lower']) / 2
                    }
                    for param in params.keys()
                }
            }
            
            # Add standardized effect sizes with CIs if group parameter exists
            if 'group' in params:
                # Calculate standardized effect size (Cohen's d)
                margin_error = (conf_int['group']['upper'] - conf_int['group']['lower']) / 2
                effect_size = params['group'] / np.sqrt(margin_error * 2)
                
                # Calculate CI for effect size
                crit_val = stats.norm.ppf((1 + confidence_level) / 2)
                es_ci_lower = effect_size - crit_val * np.sqrt(1/len(results['feature_results']))
                es_ci_upper = effect_size + crit_val * np.sqrt(1/len(results['feature_results']))
                
                ci_results[feature]['effect_size'] = {
                    'cohens_d': float(effect_size),
                    'ci_lower': float(es_ci_lower),
                    'ci_upper': float(es_ci_upper)
                }
        
        # Add summary statistics
        ci_results['summary'] = {
            'confidence_level': confidence_level,
            'n_comparisons': len(results['feature_results']),
            'critical_value': float(stats.norm.ppf((1 + confidence_level) / 2))
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
    
    def save_results(self, results, state_affair: int, state_paranoia: int):
        """
        Save complete analysis results
        """
        out_dir = self.output_dir / "11_brain_content_analysis" / f"state_affair_{state_affair}_state_paranoia_{state_paranoia}"
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
    
    state_mapping = {"affair_to_paranoia": {0:1, 1:2, 2:0}, "paranoia_to_affair": {1:0, 2:1, 0:2}}
    affair_state = 2
    paranoia_state = state_mapping["affair_to_paranoia"][affair_state]
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
            state_affair=affair_state,
            state_paranoia=paranoia_state
        )
        
        # Check the enhanced results
        print("Effect sizes:", results['main_analysis']['effect_sizes'])
        print("Corrected p-values:", results['main_analysis']['corrected_pvalues'])
        print("Cross-validation results:", results['cross_validation'])
        # Save results
        analysis.save_results(results, state_affair=affair_state, state_paranoia=paranoia_state)
        
        print("Analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()