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
    def __init__(self, data_dir, output_dir, n_cv_folds=5, coding_type="deviation", reference_group="affair"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_cv_folds = n_cv_folds
        self.data_validated = False
        self.coding_type = coding_type
        self.reference_group = reference_group
        
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
                # model_data['group_coded'] = (model_data['group'] == 'affair').astype(int)
                # model_data['group_coded'] = (model_data['group'] == 'paranoia').astype(int)
                if self.coding_type == "deviation":
                    model_data['group_coded'] = np.where(model_data['group'] == 'affair', 0.5, -0.5)
                elif self.coding_type == "treatment":
                    if self.reference_group == "affair":
                        model_data['group_coded'] = np.where(model_data['group'] == 'affair', 0, 1)
                    elif self.reference_group == "paranoia":
                        model_data['group_coded'] = np.where(model_data['group'] == 'paranoia', 0, 1)
                
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
                    feature_results[feature] = self._prepare_glmm_results(results, design_matrix, coding_type=self.coding_type)
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

    def _prepare_glmm_results(self, results, design_matrix, probability_mass=0.95, rope_width=0.1, coding_type="deviation"):
        """
        Process and organize Bayesian GLMM results for deviation coding
        
        Parameters:
        -----------
        results : statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLMResults
            Fitted model results
        design_matrix : pandas.DataFrame
            Design matrix used for model fitting
        probability_mass : float, optional (default=0.95)
            Probability mass for highest density intervals
        rope_width : float, optional (default=0.1)
            Width of Region of Practical Equivalence for effect size interpretation
        coding_type : str, optional (default="deviation")
            Type of coding used ("deviation" or "treatment")
            
        Returns:
        --------
        dict
            Dictionary containing organized Bayesian results and metrics with correct interpretation
            for the coding scheme used
        """
        # Get feature names
        feature_names = design_matrix.columns
        
        # Get fixed effects parameters and standard deviations
        n_fe = len(feature_names)  # number of fixed effects
        fe_params = results.params[:n_fe]
        fe_sds = results.fe_sd[:n_fe]
        
        # Calculate posterior probabilities
        # For each parameter, calculate P(parameter > 0) and P(parameter < 0)
        posterior_prob_positive = 1 - stats.norm.cdf(0, loc=fe_params, scale=fe_sds)
        posterior_prob_negative = stats.norm.cdf(0, loc=fe_params, scale=fe_sds)
        
        # Choose the appropriate probability based on the direction of the effect
        posterior_prob = np.where(fe_params > 0, posterior_prob_positive, posterior_prob_negative)
        
        # Calculate z-score for desired probability mass
        z_score = stats.norm.ppf((1 + probability_mass) / 2)
        
        # Calculate Highest Density Interval (HDI)
        hdi_lower = fe_params - z_score * fe_sds
        hdi_upper = fe_params + z_score * fe_sds
        
        # Calculate ROPE probabilities (Region of Practical Equivalence)
        prob_in_rope = (stats.norm.cdf(rope_width, loc=fe_params, scale=fe_sds) - 
                        stats.norm.cdf(-rope_width, loc=fe_params, scale=fe_sds))
        prob_practical_pos = 1 - stats.norm.cdf(rope_width, loc=fe_params, scale=fe_sds)
        prob_practical_neg = stats.norm.cdf(-rope_width, loc=fe_params, scale=fe_sds)
        
        # Calculate approximate Bayes factors
        null_density = stats.norm.pdf(0, loc=0, scale=1)
        posterior_density = stats.norm.pdf(0, loc=fe_params, scale=fe_sds)
        bayes_factors = np.where(
            posterior_density < 1e-10,
            1000,
            null_density / posterior_density
        )
        
        # Convert parameter dictionaries with proper keys
        coef_dict = {name: float(val) for name, val in zip(feature_names, fe_params)}
        sd_dict = {name: float(val) for name, val in zip(feature_names, fe_sds)}
        posterior_prob_dict = {name: float(val) for name, val in zip(feature_names, posterior_prob)}
        pos_prob_dict = {name: float(val) for name, val in zip(feature_names, posterior_prob_positive)}
        neg_prob_dict = {name: float(val) for name, val in zip(feature_names, posterior_prob_negative)}
        
        hdi_dict = {
            name: {'lower': float(l), 'upper': float(u)} 
            for name, l, u in zip(feature_names, hdi_lower, hdi_upper)
        }
        
        rope_dict = {
            name: {
                'prob_in_rope': float(in_rope),
                'prob_practical_pos': float(p_pos),
                'prob_practical_neg': float(p_neg)
            }
            for name, in_rope, p_pos, p_neg in zip(
                feature_names, prob_in_rope, prob_practical_pos, prob_practical_neg
            )
        }
        
        bayes_factor_dict = {
            name: float(bf) for name, bf in zip(feature_names, bayes_factors)
        }
        
        # Calculate odds ratios differently based on coding scheme
        odds_ratio_dict = {}
        
        # For each parameter, calculate its odds ratio 
        for i, name in enumerate(feature_names):
            param_value = fe_params[i]
            param_lower = hdi_lower[i]
            param_upper = hdi_upper[i]
            
            # For deviation coding, the interpretation is different
            if coding_type == "deviation" and name == "group":
                # In deviation coding (-0.5/+0.5), the group coefficient represents
                # twice the difference between one group and the grand mean
                # To get odds ratio for affair vs paranoia, we need exp(2*group_coef)
                odds_ratio = np.exp(2 * param_value)
                or_lower = np.exp(2 * param_lower)
                or_upper = np.exp(2 * param_upper)
                
                # Store with special note
                odds_ratio_dict[name] = {
                    'odds_ratio': float(odds_ratio), 
                    'lower': float(or_lower),
                    'upper': float(or_upper),
                    'note': 'This represents odds ratio between groups (not vs reference)'
                }
            else:
                # Standard interpretation for other parameters
                odds_ratio = np.exp(param_value)
                or_lower = np.exp(param_lower)
                or_upper = np.exp(param_upper)
                
                odds_ratio_dict[name] = {
                    'odds_ratio': float(odds_ratio),
                    'lower': float(or_lower),
                    'upper': float(or_upper)
                }
        
        # Calculate group-specific effects (different approach for deviation coding)
        group_specific_effects = {}
        
        if coding_type == "deviation":
            # For deviation coding, calculate effects for each group
            # The intercept is the grand mean
            # Group effects are +/- half the group coefficient
            
            # Get coefficients
            intercept = coef_dict.get('const', 0)
            group_coef = coef_dict.get('group', 0)
            feature_coef = coef_dict.get('feature', 0)
            interaction_coef = coef_dict.get('interaction', 0)
            
            # Get SDs for uncertainty calculation
            intercept_sd = sd_dict.get('const', 0)
            group_sd = sd_dict.get('group', 0)
            feature_sd = sd_dict.get('feature', 0)
            interaction_sd = sd_dict.get('interaction', 0)
            
            # Calculate effects for affair group (coded as +0.5)
            affair_effect = feature_coef + (interaction_coef * 0.5)
            # Calculate effects for paranoia group (coded as -0.5)
            paranoia_effect = feature_coef - (interaction_coef * 0.5)
            
            # Calculate approximate SDs for these effects
            # This is approximate and assumes independence between parameters
            affair_sd = np.sqrt(feature_sd**2 + (0.5 * interaction_sd)**2)
            paranoia_sd = np.sqrt(feature_sd**2 + (0.5 * interaction_sd)**2)
            
            # Calculate odds ratios and HDIs
            affair_or = np.exp(affair_effect)
            paranoia_or = np.exp(paranoia_effect)
            
            affair_or_lower = np.exp(affair_effect - z_score * affair_sd)
            affair_or_upper = np.exp(affair_effect + z_score * affair_sd)
            
            paranoia_or_lower = np.exp(paranoia_effect - z_score * paranoia_sd)
            paranoia_or_upper = np.exp(paranoia_effect + z_score * paranoia_sd)
            
            # Calculate probabilities of positive effects
            p_pos_affair = 1 - stats.norm.cdf(0, loc=affair_effect, scale=affair_sd)
            p_pos_paranoia = 1 - stats.norm.cdf(0, loc=paranoia_effect, scale=paranoia_sd)
            
            # Probability that effect is stronger in affair group
            # This is P(affair_effect > paranoia_effect)
            # = P(interaction_coef > 0) since affair_effect - paranoia_effect = interaction_coef
            p_diff = 1 - stats.norm.cdf(0, loc=interaction_coef, scale=interaction_sd)
            
            # Store in group_specific_effects with clear naming for deviation coding
            group_specific_effects['main'] = {
                'affair_group': {
                    'odds_ratio': float(affair_or),
                    'lower': float(affair_or_lower),
                    'upper': float(affair_or_upper),
                    'prob_positive': float(p_pos_affair)
                },
                'paranoia_group': {
                    'odds_ratio': float(paranoia_or),
                    'lower': float(paranoia_or_lower),
                    'upper': float(paranoia_or_upper),
                    'prob_positive': float(p_pos_paranoia)
                },
                'diff_between_groups': {
                    'prob_stronger_in_affair': float(p_diff)
                }
            }
        else:
            # Original treatment coding implementation
            # Get main effect parameters
            feature_coef = coef_dict.get('feature', 0)
            interaction_coef = coef_dict.get('interaction', 0)

            # Get standard deviations
            feature_sd = sd_dict.get('feature', 0)
            interaction_sd = sd_dict.get('interaction', 0)

            # Calculate reference group effect (just the feature coefficient)
            ref_effect = feature_coef
            ref_effect_or = np.exp(ref_effect)

            # Calculate comparison group effect (feature + interaction)
            comp_effect = feature_coef + interaction_coef
            comp_effect_or = np.exp(comp_effect)

            # Calculate approximate combined SD for comparison group
            combined_sd = np.sqrt(feature_sd**2 + interaction_sd**2)

            # Calculate HDIs for odds ratios
            ref_or_lower = np.exp(ref_effect - z_score * feature_sd)
            ref_or_upper = np.exp(ref_effect + z_score * feature_sd)

            # Comparison group HDI 
            comp_or_lower = np.exp(comp_effect - z_score * combined_sd)
            comp_or_upper = np.exp(comp_effect + z_score * combined_sd)

            # Calculate probabilities
            p_pos_ref = 1 - stats.norm.cdf(0, loc=ref_effect, scale=feature_sd)
            p_pos_comp = 1 - stats.norm.cdf(0, loc=comp_effect, scale=combined_sd)

            # Probability that effect is stronger in comparison group
            p_diff = stats.norm.cdf(0, loc=interaction_coef, scale=interaction_sd)
            if interaction_coef > 0:
                p_diff = 1 - p_diff

            # Store in group_specific_effects
            group_specific_effects['main'] = {
                'reference_group': {
                    'odds_ratio': float(ref_effect_or),
                    'lower': float(ref_or_lower),
                    'upper': float(ref_or_upper),
                    'prob_positive': float(p_pos_ref)
                },
                'comparison_group': {
                    'odds_ratio': float(comp_effect_or),
                    'lower': float(comp_or_lower),
                    'upper': float(comp_or_upper),
                    'prob_positive': float(p_pos_comp)
                },
                'diff_between_groups': {
                    'prob_stronger_in_comparison': float(p_diff)
                }
            }
        
        # Prepare results dictionary
        results_dict = {
            # Basic parameter estimates
            'coefficients': coef_dict,
            'posterior_sds': sd_dict,
            
            # Directional probabilities
            'posterior_prob': posterior_prob_dict,
            'prob_positive': pos_prob_dict,
            'prob_negative': neg_prob_dict,
            
            # Interval estimates
            'hdi': hdi_dict,
            'probability_mass': probability_mass,
            
            # Practical significance metrics
            'rope': rope_dict,
            'rope_width': rope_width,
            
            # Evidence metrics
            'bayes_factors': bayes_factor_dict,
            
            # Interpretable effect sizes
            'odds_ratios': odds_ratio_dict,
            'group_specific_effects': group_specific_effects,
            
            # Model diagnostics
            'model_summary': str(results.summary()),
            'convergence': getattr(results, 'converged', None),
            'n_obs': len(design_matrix),
            'effective_samples': getattr(results, 'n_eff', None),
            
            # Coding scheme info
            'coding_scheme': coding_type
        }
        
        # Add evidence categories based on Bayes factors
        evidence_categories = {}
        for name, bf in bayes_factor_dict.items():
            if bf < 1:
                evidence = "Evidence favors null"
            elif bf < 3:
                evidence = "Anecdotal evidence"
            elif bf < 10:
                evidence = "Moderate evidence"
            elif bf < 30:
                evidence = "Strong evidence"
            elif bf < 100:
                evidence = "Very strong evidence"
            else:
                evidence = "Extreme evidence"
            evidence_categories[name] = evidence
        
        results_dict['evidence_categories'] = evidence_categories
        
        return results_dict

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
        """
        Add Bayesian statistical enhancements to analysis results
        
        Parameters:
        -----------
        results : Dict
            Dictionary containing the base analysis results
            
        Returns:
        --------
        Dict
            Enhanced results with additional Bayesian metrics
        """
        enhanced_results = results.copy()
        
        # Add effect sizes (odds ratios and standardized effects)
        enhanced_results['effect_sizes'] = self._calculate_effect_sizes(results)
        
        # Add Bayesian multiple comparison adjustments
        enhanced_results['bayesian_multiple_comparison'] = self._apply_bayesian_multiple_comparison(results)
        
        # Add credible intervals for parameters
        enhanced_results['credible_intervals'] = self._compute_credible_intervals(results)
        
        # Add ROPE (Region of Practical Equivalence) analysis
        enhanced_results['practical_significance'] = self._compute_practical_significance(results)
        
        return enhanced_results
    
    def _calculate_effect_sizes(self, results: Dict) -> Dict:
        """
        Calculate Bayesian effect sizes from model results
        
        Parameters:
        -----------
        results : Dict
            Dictionary containing model results
            
        Returns:
        --------
        Dict
            Dictionary of effect sizes for each feature
        """
        effect_sizes = {}
        
        for feature, res in results['feature_results'].items():
            try:
                # Get group effect coefficient and posterior SD
                if 'group' not in res['coefficients']:
                    print(f"No group effect for feature {feature}, skipping")
                    continue
                    
                group_effect = res['coefficients']['group']
                
                # For Bayesian models, use posterior SD directly
                posterior_sd = res['posterior_sds']['group']
                
                # Get HDI (highest density interval) for the effect
                group_hdi = res['hdi']['group']
                
                # Calculate odds ratio and HDI
                odds_ratio = np.exp(group_effect)
                odds_ratio_hdi = {
                    'lower': np.exp(group_hdi['lower']),
                    'upper': np.exp(group_hdi['upper'])
                }
                
                # Calculate probability of direction (more appropriate than Cohen's d)
                # This is the probability that the effect is in the direction of the mean
                if 'posterior_prob' in res and 'group' in res['posterior_prob']:
                    prob_direction = res['posterior_prob']['group']
                else:
                    # Calculate if not available
                    prob_direction = 1 - stats.norm.cdf(0, loc=group_effect, scale=posterior_sd)
                    if group_effect < 0:
                        prob_direction = 1 - prob_direction
                
                # Calculate standardized effect (better than Cohen's d for this context)
                # This is in log-odds units, appropriate for binary outcomes
                standardized_effect = group_effect / posterior_sd
                
                # Probability of practical significance
                # Probability that effect size is larger than threshold (e.g., 0.2 in standardized units)
                threshold = 0.2
                prob_practical = 1 - stats.norm.cdf(threshold, loc=standardized_effect, scale=1.0)
                
                # Calculate relative risk when possible (more interpretable than odds ratio)
                # Needs base rate information - approximate from model intercept
                if 'const' in res['coefficients']:
                    intercept = res['coefficients']['const']
                    # Convert log odds to probability
                    base_prob = 1 / (1 + np.exp(-intercept))
                    effect_prob = 1 / (1 + np.exp(-(intercept + group_effect)))
                    relative_risk = effect_prob / base_prob
                else:
                    relative_risk = None
                
                effect_sizes[feature] = {
                    'standardized_effect': float(standardized_effect),
                    'odds_ratio': float(odds_ratio),
                    'odds_ratio_hdi': odds_ratio_hdi,
                    'probability_direction': float(prob_direction),
                    'probability_practical': float(prob_practical),
                    'relative_risk': float(relative_risk) if relative_risk is not None else None
                }
                
            except Exception as e:
                print(f"Error calculating effect size for feature {feature}: {str(e)}")
                print(f"Result structure for feature: {list(res.keys())}")
                effect_sizes[feature] = None
        
        return effect_sizes
    
    def _apply_bayesian_multiple_comparison(self, results):
        # Extract posterior probabilities
        feature_names = list(results['feature_results'].keys())
        feature_probs = []
        
        for feature in feature_names:
            probs = results['feature_results'][feature]['posterior_prob']
            if 'interaction' in probs:
                feature_probs.append(probs['interaction'])
        
        # Apply Bayesian FDR
        # Sort probabilities
        sorted_idx = np.argsort(feature_probs)[::-1]  # Descending
        sorted_probs = np.array(feature_probs)[sorted_idx]
        sorted_features = np.array(feature_names)[sorted_idx]
        
        # Calculate cumulative FDR
        cumulative_fdr = []
        for i, p in enumerate(sorted_probs):
            # Sum of expected false discoveries divided by number of discoveries
            expected_false_discoveries = sum(1-sorted_probs[:i+1])
            cumulative_fdr.append(expected_false_discoveries / (i+1))
        
        # Find features passing FDR threshold
        fdr_threshold = 0.05  # Adjust as needed
        passing_idx = np.where(np.array(cumulative_fdr) <= fdr_threshold)[0]
        passing_features = sorted_features[passing_idx].tolist() if len(passing_idx) > 0 else []
        
        return {
            'feature_names': feature_names,
            'posterior_probs': dict(zip(feature_names, feature_probs)),
            'significant_features': passing_features,
            'fdr_threshold': fdr_threshold
        }

    def _compute_credible_intervals(self, results: Dict, probability_mass: float = 0.95) -> Dict:
        """
        Compute Bayesian credible intervals for model parameters
        
        Parameters:
            results (Dict): Dictionary containing analysis results
            probability_mass (float): Desired probability mass within interval (default: 0.95)
            
        Returns:
            Dict: Dictionary with credible intervals for each feature and parameter
        """
        cred_results = {}
        
        for feature, res in results['feature_results'].items():
            # Get parameter estimates and standard deviations from posterior
            params = res['coefficients']
            posterior_sds = res['posterior_sds']
            
            # Compute highest density intervals (HDI)
            cred_int = {}
            for param in params.keys():
                # For normal posterior, HDI is symmetric around mean
                z_score = stats.norm.ppf((1 + probability_mass) / 2)
                cred_int[param] = {
                    'lower': params[param] - z_score * posterior_sds[param],
                    'upper': params[param] + z_score * posterior_sds[param]
                }
            
            # Store results for this feature
            cred_results[feature] = {
                'probability_mass': probability_mass,
                'parameters': {
                    param: {
                        'estimate': params[param],
                        'cred_lower': cred_int[param]['lower'],
                        'cred_upper': cred_int[param]['upper'],
                        'interval_width': cred_int[param]['upper'] - cred_int[param]['lower']
                    }
                    for param in params.keys()
                }
            }
            
            # Add standardized effect sizes with credible intervals if group parameter exists
            if 'group' in params:
                # Calculate standardized effect size (more appropriate for Bayesian)
                posterior_sd_group = posterior_sds['group']
                effect_size = params['group'] / posterior_sd_group
                
                # Calculate credible interval for effect size
                es_cred_lower = effect_size - z_score
                es_cred_upper = effect_size + z_score
                
                cred_results[feature]['effect_size'] = {
                    'standardized_mean': float(effect_size),
                    'cred_lower': float(es_cred_lower),
                    'cred_upper': float(es_cred_upper)
                }
        
        # Add summary statistics
        cred_results['summary'] = {
            'probability_mass': probability_mass,
            'n_comparisons': len(results['feature_results']),
            'z_score': float(stats.norm.ppf((1 + probability_mass) / 2))
        }
        
        return cred_results
    
    def _compute_practical_significance(self, results: Dict, rope_width: float = 0.1) -> Dict:
        """
        Compute practical significance metrics using Region of Practical Equivalence (ROPE)
        
        Parameters:
        -----------
        results : Dict
            Dictionary containing analysis results
        rope_width : float, optional (default=0.1)
            Width of the ROPE interval, representing the range of parameter values
            considered practically equivalent to zero
            
        Returns:
        --------
        Dict
            Dictionary containing practical significance metrics for each feature
        """
        practical_sig = {}
        
        for feature, res in results['feature_results'].items():
            # Skip if no rope information
            if 'rope' not in res:
                continue
                
            feature_practical = {}
            
            # Focus on group effect and interaction
            for param in ['group', 'interaction']:
                if param in res['rope']:
                    # Get ROPE probabilities
                    prob_in_rope = res['rope'][param]['prob_in_rope']
                    prob_practical_pos = res['rope'][param]['prob_practical_pos']
                    prob_practical_neg = res['rope'][param]['prob_practical_neg']
                    
                    # Create clear interpretations
                    if prob_in_rope > 0.95:
                        interpretation = "Practically equivalent to zero"
                    elif prob_practical_pos > 0.95:
                        interpretation = "Practically positive"
                    elif prob_practical_neg > 0.95:
                        interpretation = "Practically negative"
                    else:
                        interpretation = "Uncertain practical significance"
                    
                    feature_practical[param] = {
                        'prob_in_rope': prob_in_rope,
                        'prob_practical_pos': prob_practical_pos,
                        'prob_practical_neg': prob_practical_neg,
                        'interpretation': interpretation
                    }
            
            practical_sig[feature] = feature_practical
        
        # Add summary of practical significance across features
        practical_sig['summary'] = {
            'rope_width': rope_width,
            'n_features': len(results['feature_results']),
            'feature_count_by_interpretation': self._count_interpretations(practical_sig)
        }
        
        return practical_sig

    def _count_interpretations(self, practical_sig: Dict) -> Dict:
        """Helper method to count interpretations across features"""
        counts = {
            'Practically equivalent to zero': 0,
            'Practically positive': 0,
            'Practically negative': 0,
            'Uncertain practical significance': 0
        }
        
        for feature, params in practical_sig.items():
            if feature == 'summary':
                continue
                
            for param, values in params.items():
                if param == 'group' and 'interpretation' in values:
                    counts[values['interpretation']] += 1
        
        return counts

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
    
    def save_results(self, results, state_affair: int, state_paranoia: int, coding_type: str):
        """
        Save complete Bayesian analysis results to files
        
        Parameters:
        -----------
        results : Dict
            Dictionary containing analysis results
        state_affair : int
            State index for affair group
        state_paranoia : int
            State index for paranoia group
        """
        # Determine coding scheme from results
        if 'main_analysis' in results and 'feature_results' in results['main_analysis']:
            # Check first feature's results for coding scheme info
            first_feature = next(iter(results['main_analysis']['feature_results'].values()), {})
            coding_type = first_feature.get('coding_scheme', "treatment")
        
        # Create output directory with coding type
        folder_name = f"state_affair_{state_affair}_state_paranoia_{state_paranoia}_coding_{coding_type}"
        
        out_dir = self.output_dir / "11_brain_content_analysis" / folder_name
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Saving results to {out_dir} with coding scheme: {coding_type}")
        
        # Save main analysis results if available
        if 'feature_results' in results.get('main_analysis', {}):
            # Create dataframe with appropriate columns based on coding scheme
            df_data = {
                'feature': []
            }
            
            # Add standard columns
            for col in ['coefficient', 'probability_direction', 'odds_ratio']:
                df_data[col] = []
            
            # Add coding-specific columns
            if coding_type == "deviation":
                # For deviation coding
                df_data.update({
                    'affair_odds_ratio': [],
                    'paranoia_odds_ratio': [],
                    'affair_prob_positive': [],
                    'paranoia_prob_positive': [],
                    'prob_stronger_in_affair': []
                })
            else:
                # For treatment coding
                df_data.update({
                    'reference_group_odds_ratio': [],
                    'comparison_group_odds_ratio': [],
                    'reference_group_prob_positive': [],
                    'comparison_group_prob_positive': [],
                    'prob_stronger_in_comparison': []
                })
            
            # Extract data from results
            for feature, res in results['main_analysis']['feature_results'].items():
                df_data['feature'].append(feature)
                
                # Get coefficient for group effect if available
                if 'coefficients' in res and 'group' in res['coefficients']:
                    df_data['coefficient'].append(res['coefficients']['group'])
                else:
                    df_data['coefficient'].append(None)
                    
                # Get probability of direction
                if 'posterior_prob' in res and 'group' in res['posterior_prob']:
                    df_data['probability_direction'].append(res['posterior_prob']['group'])
                else:
                    df_data['probability_direction'].append(None)
                    
                # Get odds ratio
                if 'odds_ratios' in res and 'group' in res['odds_ratios']:
                    df_data['odds_ratio'].append(res['odds_ratios']['group']['odds_ratio'])
                else:
                    df_data['odds_ratio'].append(None)
                
                # Extract group-specific effects based on coding scheme
                if 'group_specific_effects' in res and 'main' in res['group_specific_effects']:
                    effects = res['group_specific_effects']['main']
                    
                    if coding_type == "deviation":
                        # Extract deviation coding specific fields
                        if 'affair_group' in effects and 'paranoia_group' in effects:
                            df_data['affair_odds_ratio'].append(effects['affair_group']['odds_ratio'])
                            df_data['paranoia_odds_ratio'].append(effects['paranoia_group']['odds_ratio'])
                            df_data['affair_prob_positive'].append(effects['affair_group']['prob_positive'])
                            df_data['paranoia_prob_positive'].append(effects['paranoia_group']['prob_positive'])
                            df_data['prob_stronger_in_affair'].append(
                                effects['diff_between_groups'].get('prob_stronger_in_affair', None))
                        else:
                            df_data['affair_odds_ratio'].append(None)
                            df_data['paranoia_odds_ratio'].append(None)
                            df_data['affair_prob_positive'].append(None)
                            df_data['paranoia_prob_positive'].append(None)
                            df_data['prob_stronger_in_affair'].append(None)
                    else:
                        # Extract treatment coding specific fields
                        if 'reference_group' in effects and 'comparison_group' in effects:
                            df_data['reference_group_odds_ratio'].append(effects['reference_group']['odds_ratio'])
                            df_data['comparison_group_odds_ratio'].append(effects['comparison_group']['odds_ratio'])
                            df_data['reference_group_prob_positive'].append(effects['reference_group']['prob_positive'])
                            df_data['comparison_group_prob_positive'].append(effects['comparison_group']['prob_positive'])
                            df_data['prob_stronger_in_comparison'].append(
                                effects['diff_between_groups'].get('prob_stronger_in_comparison', None))
                        else:
                            df_data['reference_group_odds_ratio'].append(None)
                            df_data['comparison_group_odds_ratio'].append(None)
                            df_data['reference_group_prob_positive'].append(None)
                            df_data['comparison_group_prob_positive'].append(None)
                            df_data['prob_stronger_in_comparison'].append(None)
                else:
                    # Handle missing effects data
                    if coding_type == "deviation":
                        df_data['affair_odds_ratio'].append(None)
                        df_data['paranoia_odds_ratio'].append(None)
                        df_data['affair_prob_positive'].append(None)
                        df_data['paranoia_prob_positive'].append(None)
                        df_data['prob_stronger_in_affair'].append(None)
                    else:
                        df_data['reference_group_odds_ratio'].append(None)
                        df_data['comparison_group_odds_ratio'].append(None)
                        df_data['reference_group_prob_positive'].append(None)
                        df_data['comparison_group_prob_positive'].append(None)
                        df_data['prob_stronger_in_comparison'].append(None)
            
            # Create and save DataFrame
            main_results_df = pd.DataFrame(df_data)
            main_results_df.to_csv(out_dir / f'main_analysis_results.csv', index=False)
            print(f"Saved main analysis results with columns: {main_results_df.columns.tolist()}")

        # Save cross-validation results if available
        cv_results = results.get('cross_validation', {})
        
        if 'subject_cv' in cv_results:
            # Convert list of dicts to DataFrame
            subject_cv_df = pd.DataFrame(cv_results['subject_cv'])
            subject_cv_df.to_csv(out_dir / f'subject_cv_results.csv', index=False)
        
        if 'temporal_cv' in cv_results:
            # Convert list of dicts to DataFrame
            temporal_cv_df = pd.DataFrame(cv_results['temporal_cv'])
            temporal_cv_df.to_csv(out_dir / f'temporal_cv_results.csv', index=False)
        
        # Save Bayesian comparison results if available
        if 'bayesian_multiple_comparison' in results.get('main_analysis', {}):
            bayes_comp = results['main_analysis']['bayesian_multiple_comparison']
            
            # Extract values to DataFrame
            if 'posterior_probs' in bayes_comp:
                bayes_df = pd.DataFrame({
                    'feature': list(bayes_comp['posterior_probs'].keys()),
                    'posterior_probability': list(bayes_comp['posterior_probs'].values()),
                    'significant': [
                        feat in bayes_comp.get('significant_features', []) 
                        for feat in bayes_comp['posterior_probs'].keys()
                    ]
                })
                bayes_df.to_csv(out_dir / f'bayesian_comparison.csv', index=False)
        
        # Convert metadata timestamp to string before saving
        if 'metadata' in results:
            metadata = results['metadata'].copy()
            if 'timestamp' in metadata:
                metadata['timestamp'] = metadata['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Add analysis type information
            metadata['analysis_type'] = 'bayesian_glmm'
            metadata['coding_scheme'] = coding_type
            metadata['bayes_info'] = {
                'model': 'BinomialBayesMixedGLM',
                'inference': 'MAP (Maximum A Posteriori)',
                'software': 'statsmodels'
            }
            
            # Save metadata as JSON
            with open(out_dir / f'analysis_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        # Prepare complete output dictionary
        output_dict = {
            'metadata': {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'states_compared': {
                    'affair': state_affair,
                    'paranoia': state_paranoia
                },
                'n_subjects': {
                    'affair': self.affair_sequences.shape[0],
                    'paranoia': self.paranoia_sequences.shape[0]
                },
                'analysis_type': 'bayesian_glmm',
                'coding_scheme': coding_type
            },
            'main_analysis': results['main_analysis'],
            'cross_validation': results['cross_validation']
        }
        
        # Save complete results as compressed numpy archive
        try:
            np.savez_compressed(
                out_dir / f'hierarchical_analysis_results.npz',
                **output_dict
            )
        except Exception as e:
            print(f"Error saving complete results: {str(e)}")
            # Try saving a simplified version if the full one fails
            simplified_dict = {
                'metadata': output_dict['metadata'],
                # Include only essential analysis results
                'simplified_results': True
            }
            np.savez_compressed(
                out_dir / f'hierarchical_analysis_results_simplified.npz',
                **simplified_dict
            )
            
        # Also save as JSON for easier inspection
        try:
            # Convert numpy arrays and other non-serializable objects
            def convert_for_json(obj):
                if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(i) for i in obj]
                else:
                    return obj
            
            json_safe_dict = convert_for_json(output_dict)
            with open(out_dir / f'hierarchical_analysis_results.json', 'w') as f:
                json.dump(json_safe_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON results: {str(e)}")

def main():
    load_dotenv()

    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = Path(scratch_dir) / "output"
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    
    state_mapping = {"affair_to_paranoia": {0:1, 1:2, 2:0}, "paranoia_to_affair": {1:0, 2:1, 0:2}}
    # state_mapping = {"affair_to_paranoia": {0:0, 1:1, 2:2}, "paranoia_to_affair": {0:0, 1:1, 2:2}}
    affair_state = 2
    coding_type = "deviation"
    reference_group = "affair"  # This is needed even with deviation coding
    paranoia_state = state_mapping["affair_to_paranoia"][affair_state]
    
    # Initialize analysis with coding_type and reference_group
    analysis = HierarchicalStateAnalysis(
        data_dir=data_dir,
        output_dir=output_dir,
        n_cv_folds=5,
        coding_type=coding_type,
        reference_group=reference_group
    )
    
    try:
        # Load and validate data
        analysis.load_data()
        
        # Run enhanced analysis pipeline - don't pass coding_type and reference_group here
        results = analysis.run_complete_analysis(
            state_affair=affair_state,
            state_paranoia=paranoia_state
        )
        
        # Check the enhanced results
        print("Effect sizes:", results['main_analysis']['effect_sizes'])
        print("Bayesian multiple comparison:", results['main_analysis']['bayesian_multiple_comparison'])
        print("Cross-validation results:", results['cross_validation'])
        # Save results
        analysis.save_results(results, state_affair=affair_state, state_paranoia=paranoia_state, coding_type=coding_type)
        
        print("Analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()