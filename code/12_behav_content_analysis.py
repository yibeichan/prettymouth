import os
from dotenv import load_dotenv
import json
import numpy as np  
import pandas as pd
from pathlib import Path
import pickle
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import warnings
warnings.filterwarnings('ignore')

class BehavioralContentAnalysis:
    def __init__(self, output_dir: Path, data_dir: Path, n_trim_TRs: int = 14, coding_type: str = 'treatment'):
        """
        Initialize behavioral analysis pipeline
        
        Parameters:
        -----------
        output_dir : Path
            Directory containing behavioral data and for saving results
        data_dir : Path
            Directory containing story annotations and other data files
        n_trim_TRs : int
            Number of TRs to trim from the beginning of each sequence
        coding_type : str
            Type of coding to use for categorical variables ('treatment' or 'deviation')
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.n_trim_TRs = n_trim_TRs
        self.data_loaded = False
        self.coding_type = coding_type
        
        if coding_type not in ['treatment', 'deviation']:
            raise ValueError("coding_type must be either 'treatment' or 'deviation'")
        
        # Define content features to use (same as brain analysis)
        self.feature_names = [
            'lee_girl_together', 'has_verb', 'lee_speaking', 'girl_speaking',
            'arthur_speaking', 'has_adj', 'has_adv', 'has_noun'
        ]
        
    def load_data(self):
        """Load behavioral data and content features for both groups"""
        print("\n=== Starting data loading ===")
        try:
            # Load behavioral data
            affair_beh_file = self.output_dir / 'behav_results' / 'individual_response_evidence_affair.pkl'
            paranoia_beh_file = self.output_dir / 'behav_results' / 'individual_response_evidence_paranoia.pkl'
            
            print(f"Loading behavioral data from:\n  {affair_beh_file}\n  {paranoia_beh_file}")
            
            affair_data = pickle.load(open(affair_beh_file, 'rb'))
            paranoia_data = pickle.load(open(paranoia_beh_file, 'rb'))
            print(f"Successfully loaded behavioral data files")
            
            # Extract and process binary responses
            print("Processing binary responses...")
            self.affair_responses = np.array([resp['binary_response'][self.n_trim_TRs:] 
                                        for _, resp in affair_data.iterrows()])
            self.paranoia_responses = np.array([resp['binary_response'][self.n_trim_TRs:] 
                                            for _, resp in paranoia_data.iterrows()])
            
            print(f"Response array shapes - Affair: {self.affair_responses.shape}, Paranoia: {self.paranoia_responses.shape}")
            
            # Store participant IDs
            self.affair_participants = affair_data['participant'].values
            self.paranoia_participants = paranoia_data['participant'].values
            
            # Load content features
            content_file = self.data_dir / '10_story_annotations_TR.csv'
            print(f"\nLoading content features from: {content_file}")
            content_df = pd.read_csv(content_file)
            
            # Prepare feature matrix (no trimming)
            print("Processing content features...")
            self.content_features = pd.DataFrame()
            for feature in self.feature_names:
                self.content_features[feature] = (content_df[feature].astype(str).str.lower() == "true").astype(int)
                print(f"  Processed feature: {feature}")
            
            print(f"Content features shape: {self.content_features.shape}")
            
            self.data_loaded = True
            print("\n=== Data loading complete ===")
            print(f"Loaded data: Affair group n={len(self.affair_participants)}, "
                f"Paranoia group n={len(self.paranoia_participants)}")
            
        except Exception as e:
            print(f"\n!!! ERROR during data loading !!!")
            print(f"Error details: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")
            
    def compute_group_timecourses(self):
        """Compute average response timecourses for each group"""
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        self.affair_timecourse = np.mean(self.affair_responses, axis=0)
        self.paranoia_timecourse = np.mean(self.paranoia_responses, axis=0)
        
        # Compute standard errors
        self.affair_se = stats.sem(self.affair_responses, axis=0)
        self.paranoia_se = stats.sem(self.paranoia_responses, axis=0)
        
        return {
            'affair': {
                'mean': self.affair_timecourse,
                'se': self.affair_se
            },
            'paranoia': {
                'mean': self.paranoia_timecourse,
                'se': self.paranoia_se
            }
        }
        
    def analyze_timepoint_differences(self):
        """Analyze group differences at each timepoint"""
        print("\n=== Starting timepoint difference analysis ===")
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        n_timepoints = self.affair_responses.shape[1]
        print(f"Analyzing {n_timepoints} timepoints")
        
        # Debug input data
        print("\nInput data diagnostics:")
        print(f"Affair responses shape: {self.affair_responses.shape}")
        print(f"Paranoia responses shape: {self.paranoia_responses.shape}")
        print(f"Any NaN in affair responses: {np.any(np.isnan(self.affair_responses))}")
        print(f"Any NaN in paranoia responses: {np.any(np.isnan(self.paranoia_responses))}")
        
        # Initialize results storage
        results = {
            'tvals': np.zeros(n_timepoints),
            'pvals': np.zeros(n_timepoints),
            'effect_sizes': np.zeros(n_timepoints)
        }
        
        # Compute t-test at each timepoint with handling for zero variance
        for t in range(n_timepoints):
            affair_data = self.affair_responses[:, t]
            paranoia_data = self.paranoia_responses[:, t]
            
            # If both groups have zero variance (all same values)
            if np.std(affair_data) == 0 and np.std(paranoia_data) == 0:
                # If means are equal, p = 1 (definitely not different)
                # If means are different (shouldn't happen with binary data), p = 0 (definitely different)
                if np.mean(affair_data) == np.mean(paranoia_data):
                    results['tvals'][t] = 0
                    results['pvals'][t] = 1
                else:
                    results['tvals'][t] = np.inf
                    results['pvals'][t] = 0
            else:
                # Regular t-test when we have variance
                t_stat, p_val = stats.ttest_ind(affair_data, paranoia_data)
                results['tvals'][t] = t_stat
                results['pvals'][t] = p_val
            
            # Calculate effect size (handle zero variance case)
            pooled_std = np.sqrt((np.var(affair_data) + np.var(paranoia_data)) / 2)
            if pooled_std == 0:
                results['effect_sizes'][t] = 0  # No effect when all values are identical
            else:
                d = (np.mean(affair_data) - np.mean(paranoia_data)) / pooled_std
                results['effect_sizes'][t] = d
        
        # Debug p-values before FDR correction
        print("P-value diagnostics:")
        print(f"  Range: {np.min(results['pvals'])} to {np.max(results['pvals'])}")
        print(f"  Any NaN: {np.any(np.isnan(results['pvals']))}")
        print(f"  Number of unique values: {len(np.unique(results['pvals']))}")
        
        # Apply FDR correction with validation
        print("Applying FDR correction...")
        valid_pvals = np.clip(results['pvals'], 0, 1)  # Ensure values are between 0 and 1
        if np.all(np.isnan(valid_pvals)):
            print("WARNING: All p-values are NaN!")
            results['qvals'] = np.full_like(results['pvals'], np.nan)
        else:
            results['qvals'] = multipletests(valid_pvals, method='fdr_bh')[1]
        
        print(f"Found {sum(results['qvals'] < 0.05)} significant timepoints (q < 0.05)")
        print("=== Timepoint analysis complete ===\n")
        return results
        
    def analyze_content_relationships(self, content_features):
        """Analyze relationships between content features and behavioral responses using GLMM"""
        print("Preparing data for GLMM analysis...")
        
        try:
            # Prepare data
            n_total_subjects = len(self.affair_participants) + len(self.paranoia_participants)
            n_timepoints = len(self.affair_responses[0])
            
            # Create model data with explicit types
            model_data = pd.DataFrame(index=range(n_total_subjects * n_timepoints))
            model_data['subject'] = np.repeat(range(n_total_subjects), n_timepoints).astype(np.int32)
            group_list = ['affair'] * len(self.affair_participants) + ['paranoia'] * len(self.paranoia_participants)
            model_data['group'] = np.repeat(group_list, n_timepoints)
            
            # Add features without standardization (they're already binary)
            for col in content_features.columns:
                model_data[col] = np.tile(content_features[col].values, n_total_subjects).astype(np.float64)
            
            # Add interaction features
            model_data['lee_girl_verb'] = model_data['lee_girl_together'] * model_data['has_verb']
            model_data['arthur_adj'] = model_data['arthur_speaking'] * model_data['has_adj']
            
            # Add response data as integer type
            all_responses = np.concatenate([self.affair_responses, self.paranoia_responses])
            model_data['response'] = all_responses.flatten().astype(np.int32)
            
            # Build feature matrix with explicit types based on coding type
            if self.coding_type == 'treatment':
                # Treatment coding (default, one group as reference)
                group_dummies = pd.get_dummies(model_data['group'], drop_first=True)
                group_contrast_col = group_dummies.columns[0]  # Should be 'paranoia'
                print(f"Using treatment coding with reference group: affair")
            else:
                # Deviation coding (compare each group to the grand mean)
                from patsy import dmatrix
                formula = "C(group, Sum)"
                group_dummies = pd.DataFrame(
                    dmatrix(formula, model_data, return_type='dataframe'),
                    index=model_data.index
                ).iloc[:, 1:]  # Drop intercept
                group_contrast_col = group_dummies.columns[0]  # Should be something like 'C(group, Sum)[S.affair]'
                print(f"Using deviation coding with contrast: {group_contrast_col}")
            
            # Create design matrix with constant
            exog = sm.add_constant(group_dummies).astype(np.float64)
            feature_matrix = exog.copy()
            
            # Add all features including interaction features
            feature_cols = list(content_features.columns) + ['lee_girl_verb', 'arthur_adj']
            for feature in feature_cols:
                feature_matrix[feature] = model_data[feature].astype(np.float64)
            
            # Add group interactions
            for feature in feature_cols:
                feature_matrix[f'group_{feature}_interaction'] = (
                    feature_matrix[group_contrast_col] * feature_matrix[feature]
                ).astype(np.float64)
            
            # Add more debug information
            print("\nModel data summary:")
            print("Response values:", np.unique(model_data['response']))
            print("Response range:", model_data['response'].min(), "to", model_data['response'].max())
            print("Group distribution:", model_data['group'].value_counts())
            print(f"Coding type: {self.coding_type}")
            print(f"Group columns: {group_dummies.columns.tolist()}")
            
            # Verify no missing values
            missing_data = model_data.isnull().sum()
            if missing_data.any():
                print("\nWarning: Missing values detected:")
                print(missing_data[missing_data > 0])
            
            # Verify data alignment
            print("\nVerifying data alignment:")
            print(f"Number of subjects × timepoints: {n_total_subjects * n_timepoints}")
            print(f"Actual data length: {len(model_data)}")
            
            # Fit basic model
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            subject_dummies = pd.get_dummies(model_data['subject']).astype(np.float64)
            basic_model = BinomialBayesMixedGLM(
                model_data['response'],
                exog,
                exog_vc=subject_dummies,
                vc_names=['subject'],
                ident=np.ones(subject_dummies.shape[1], dtype=np.int32)
            )
            basic_result = basic_model.fit_vb()
            print("\nBasic GLMM converged successfully!")
            print(basic_result.summary())
            
            # Fit full model - FIX: Ensure feature_matrix is a DataFrame, not ndarray
            print("\nFitting full GLMM...")
            
            # Convert feature_matrix to DataFrame if it's an ndarray
            if isinstance(feature_matrix, np.ndarray):
                print("Converting feature matrix from ndarray to DataFrame...")
                feature_matrix = pd.DataFrame(
                    feature_matrix, 
                    index=model_data.index,
                    columns=[f"col_{i}" for i in range(feature_matrix.shape[1])]
                )
            
            # Ensure subject_dummies is a DataFrame
            if isinstance(subject_dummies, np.ndarray):
                subject_dummies = pd.DataFrame(
                    subject_dummies,
                    index=model_data.index,
                    columns=[f"subject_{i}" for i in range(subject_dummies.shape[1])]
                )
            
            # Print debug info about matrices
            print(f"Feature matrix type: {type(feature_matrix)}")
            print(f"Feature matrix shape: {feature_matrix.shape}")
            print(f"Subject dummies type: {type(subject_dummies)}")
            print(f"Subject dummies shape: {subject_dummies.shape}")
            
            full_model = BinomialBayesMixedGLM(
                model_data['response'],
                feature_matrix,
                exog_vc=subject_dummies,
                vc_names=['subject'],
                ident=np.ones(subject_dummies.shape[1], dtype=np.int32)
            )
            full_result = full_model.fit_vb()
            
            # After fitting the model
            if not getattr(full_result, 'converged', True):
                print("WARNING: Full model did not converge!")

            # Check for extreme parameter values - FIX: Handle NumPy array case
            if hasattr(full_result.params, 'items'):
                # If params is a dictionary-like object with items() method
                extreme_params = [name for name, val in full_result.params.items() if abs(val) > 10]
                if extreme_params:
                    print(f"WARNING: Extreme parameter values found: {extreme_params}")
            else:
                # If params is a NumPy array
                extreme_values = np.where(np.abs(full_result.params) > 10)[0]
                if len(extreme_values) > 0:
                    print(f"WARNING: Extreme parameter values found at indices: {extreme_values}")
                
            print("\nFull GLMM converged successfully!")
            results = self._prepare_glmm_results(full_result, basic_result, feature_matrix, group_contrast_col)
            return results
            
        except Exception as e:
            print(f"\n!!! ERROR fitting GLMM !!!")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            print("\nDebug information:")
            print(f"Feature matrix type: {type(feature_matrix)}")
            if hasattr(feature_matrix, 'shape'):
                print(f"Feature matrix shape: {feature_matrix.shape}")
            print(f"Response shape: {model_data['response'].shape}")
            print(f"Number of subjects: {n_total_subjects}")
            print(f"Number of timepoints: {n_timepoints}")
            return None

    def _prepare_glmm_results(self, full_result, basic_result, feature_matrix, group_contrast_col):
        """Helper to prepare GLMM results"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Get feature names
        feature_names = feature_matrix.columns
        
        # Create parameters with explicit index matching
        params = pd.Series(
            full_result.params[:len(feature_names)],
            index=feature_names,
            dtype=np.float64
        )
        
        # For Bayesian results, use fe_sd (fixed effects standard deviation)
        posterior_sds = pd.Series(
            full_result.fe_sd[:len(feature_names)],
            index=feature_names,
            dtype=np.float64
        )
        
        # Calculate probability of parameter being different from 0
        # using the normal approximation to the posterior
        z_scores = params / posterior_sds
        prob_nonzero = pd.Series(
            2 * (1 - stats.norm.cdf(abs(z_scores))),  # Two-tailed
            index=feature_names,
            dtype=np.float64
        )
        
        # Calculate 95% credible intervals
        conf_int = pd.DataFrame(
            np.column_stack([
                params - 1.96 * posterior_sds,
                params + 1.96 * posterior_sds
            ]),
            index=feature_names,
            columns=['lower', 'upper']
        )
        
        # Calculate VIF only for non-interaction terms
        vif_data = pd.DataFrame()
        main_effect_cols = [col for col in feature_names if 'interaction' not in col]
        for idx, col in enumerate(main_effect_cols):
            X = feature_matrix[main_effect_cols]
            vif_data.loc[col, 'VIF'] = variance_inflation_factor(X.values, idx)
        
        # For Bayesian model comparison, use DIC instead of AIC/BIC
        model_comparison = {}
        for model, name in [(basic_result, 'basic'), (full_result, 'full')]:
            model_comparison[f'{name}_dic'] = getattr(model, 'dic', None)
            model_comparison[f'{name}_vb_elbo'] = getattr(model, 'vb_elbo', None)
        
        # Calculate group-specific effects for each feature
        group_specific_effects = {}
        feature_cols = [col for col in feature_names if col not in ['const', group_contrast_col] and not col.startswith('group_')]
        
        for feature in feature_cols:
            # Check if corresponding interaction term exists
            interaction_col = f'group_{feature}_interaction'
            if interaction_col in params.index:
                # Get main effect parameters
                feature_coef = params.get(feature, 0)
                interaction_coef = params.get(interaction_col, 0)
                
                # Get standard deviations
                feature_sd = posterior_sds.get(feature, 0)
                interaction_sd = posterior_sds.get(interaction_col, 0)
                
                # Calculate effects for each group
                if self.coding_type == 'treatment':
                    # Treatment coding: reference = affair, comparison = paranoia
                    ref_effect = feature_coef
                    comp_effect = feature_coef + interaction_coef
                    ref_group = 'affair'
                    comp_group = 'paranoia'
                else:
                    # Deviation coding: both effects are deviations from grand mean
                    grand_mean = feature_coef
                    ref_effect = grand_mean - interaction_coef/2  # affair
                    comp_effect = grand_mean + interaction_coef/2  # paranoia
                    ref_group = 'affair'
                    comp_group = 'paranoia'
                
                ref_effect_or = np.exp(ref_effect)
                comp_effect_or = np.exp(comp_effect)
                
                # Calculate approximate combined SD for comparison group
                combined_sd = np.sqrt(feature_sd**2 + interaction_sd**2)
                
                # Calculate HDIs
                z_score = stats.norm.ppf((1 + 0.95) / 2)  # 95% HDI
                
                # Reference group HDI
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
                
                # Store results
                group_specific_effects[feature] = {
                    'main': {
                        ref_group: {
                            'odds_ratio': float(ref_effect_or),
                            'lower': float(ref_or_lower),
                            'upper': float(ref_or_upper),
                            'prob_positive': float(p_pos_ref)
                        },
                        comp_group: {
                            'odds_ratio': float(comp_effect_or),
                            'lower': float(comp_or_lower),
                            'upper': float(comp_or_upper),
                            'prob_positive': float(p_pos_comp)
                        },
                        'diff_between_groups': {
                            'prob_stronger_in_comparison': float(p_diff)
                        }
                    }
                }
        
        return {
            'model_results': full_result,
            'summary': str(full_result.summary()),
            'coefficients': params.to_dict(),
            'posterior_sds': posterior_sds.to_dict(),
            'prob_nonzero': prob_nonzero.to_dict(),
            'conf_int': conf_int.to_dict(),
            'vif': vif_data.to_dict(),
            'model_comparison': model_comparison,
            'group_specific_effects': group_specific_effects,
            'coding_type': self.coding_type
        }

    def save_results(self, results, filename):
        """Save analysis results"""
        save_dir = self.output_dir / '12_behavioral_analysis' / self.coding_type
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n=== Saving results to {save_dir / filename}.json ===")

        def convert_to_serializable(obj):
            """Recursively convert objects to JSON-serializable types"""
            if obj is None:
                return None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif pd.api.types.is_categorical_dtype(obj):
                return str(obj)
            elif isinstance(obj, pd.Series):
                return obj.to_list()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj

        # Convert results to saveable format
        save_results = {}
        
        for key, value in results.items():
            if key == 'content_relationships' and value is not None:
                # Special handling for mixed effects model results
                save_results[key] = {
                    'coefficients': convert_to_serializable(value['coefficients']),
                    'posterior_sds': convert_to_serializable(value['posterior_sds']),
                    'prob_nonzero': convert_to_serializable(value['prob_nonzero']),
                    'confidence_intervals': convert_to_serializable(value['conf_int']),
                    'group_specific_effects': convert_to_serializable(value.get('group_specific_effects', {})),
                    'model_summary': str(value['summary']),
                    'coding_type': value['coding_type']
                }
            else:
                save_results[key] = convert_to_serializable(value)
        
        # Save as JSON with pretty printing
        with open(save_dir / f"{filename}.json", 'w') as f:
            json.dump(save_results, f, indent=2)
        print("Results saved successfully")
            
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        if not self.data_loaded:
            self.load_data()
            
        results = {
            'group_timecourses': self.compute_group_timecourses(),
            'timepoint_differences': self.analyze_timepoint_differences(),
            'content_relationships': self.analyze_content_relationships(self.content_features)
        }
        
        return results
    
def main():
    print("\n=== Starting behavioral content analysis ===")
    # Setup paths
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        print("ERROR: SCRATCH_DIR environment variable not set!")
        return
        
    output_dir = Path(scratch_dir) / "output"
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    
    print(f"Output directory: {output_dir}")
    print(f"Data directory: {data_dir}")

    # Initialize analysis
    analysis = BehavioralContentAnalysis(
        output_dir=output_dir,
        data_dir=data_dir,
        coding_type='deviation'
    )

    # Run analysis pipeline
    print("\nStarting analysis pipeline...")
    analysis.load_data()
    results = analysis.run_complete_analysis()
    analysis.save_results(results, 'results')
    print("\n=== Analysis complete ===")

if __name__ == "__main__":
    main()