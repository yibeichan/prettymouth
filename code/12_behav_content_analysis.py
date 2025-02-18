import os
from dotenv import load_dotenv
import json
import numpy as np  
import pandas as pd
from pathlib import Path
import pickle
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

import warnings
warnings.filterwarnings('ignore')

class BehavioralContentAnalysis:
    def __init__(self, output_dir: Path, data_dir: Path, n_trim_TRs: int = 14):
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
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.n_trim_TRs = n_trim_TRs
        self.data_loaded = False
        
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
        """Analyze relationships between content features and behavioral responses"""
        print("Preparing data for mixed effects model...")
        
        # Calculate total number of subjects
        n_total_subjects = len(self.affair_participants) + len(self.paranoia_participants)
        n_timepoints = len(self.affair_responses[0])
        
        # Create model data frame
        model_data = pd.DataFrame(index=range(n_total_subjects * n_timepoints))
        model_data['subject'] = np.repeat(range(n_total_subjects), n_timepoints)
        
        # Add group as categorical and numeric
        group_list = ['affair'] * len(self.affair_participants) + ['paranoia'] * len(self.paranoia_participants)
        model_data['group'] = np.repeat(group_list, n_timepoints)
        model_data['group_numeric'] = np.repeat([0] * len(self.affair_participants) + 
                                            [1] * len(self.paranoia_participants), 
                                            n_timepoints)
        
        # Add timepoint and features
        model_data['timepoint'] = np.tile(range(n_timepoints), n_total_subjects)
        for col in content_features.columns:
            model_data[col] = np.tile(content_features[col].values, n_total_subjects)
            
            # Create interaction terms using numeric group
            model_data[f'group_{col}_interaction'] = model_data['group_numeric'] * model_data[col]
        
        # Add response data
        all_responses = np.concatenate([self.affair_responses, self.paranoia_responses])
        model_data['response'] = all_responses.flatten()
        
        try:
            # Formula with main effects and interactions
            formula = "response ~ group + " + \
                    " + ".join(content_features.columns) + \
                    " + " + \
                    " + ".join([f"group_{col}_interaction" for col in content_features.columns])
            
            print(f"\nFitting mixed effects model with formula:\n{formula}")
            
            model = MixedLM.from_formula(
                formula,
                groups='subject',
                data=model_data
            )
            
            results = model.fit()
            
            # Extract group-specific effects
            group_effects = {}
            for feature in content_features.columns:
                # Effect for affair group (base effect)
                base_effect = results.params[feature]
                base_effect_ci = results.conf_int().loc[feature]
                
                # Additional effect for paranoia group (interaction)
                interaction_effect = results.params[f'group_{feature}_interaction']
                interaction_ci = results.conf_int().loc[f'group_{feature}_interaction']
                
                # Total effect for paranoia group
                paranoia_effect = base_effect + interaction_effect
                
                group_effects[feature] = {
                    'affair': {
                        'effect': base_effect,
                        'ci_lower': base_effect_ci[0],
                        'ci_upper': base_effect_ci[1],
                        'p_value': results.pvalues[feature]
                    },
                    'paranoia': {
                        'effect': paranoia_effect,
                        'ci_lower': base_effect_ci[0] + interaction_ci[0],
                        'ci_upper': base_effect_ci[1] + interaction_ci[1],
                        'interaction_p_value': results.pvalues[f'group_{feature}_interaction']
                    }
                }
            
            return {
                'model_results': results,
                'summary': results.summary(),
                'coefficients': results.params,
                'pvalues': results.pvalues,
                'conf_int': results.conf_int(),
                'group_effects': group_effects
            }
            
        except Exception as e:
            print(f"\n!!! ERROR fitting mixed effects model !!!")
            print(f"Error details: {str(e)}")
            print(f"Model data info:")
            print(f"- Shape: {model_data.shape}")
            print(f"- Columns: {model_data.columns}")
            print(f"- Missing values: {model_data.isnull().sum()}")
            return None
        
    def save_results(self, results, filename):
        """Save analysis results"""
        save_dir = self.output_dir / '12_behavioral_analysis'
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n=== Saving results to {save_dir / filename}.json ===")

        def convert_to_serializable(obj):
            """Recursively convert objects to JSON-serializable types"""
            if isinstance(obj, np.ndarray):
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
                    'coefficients': convert_to_serializable(value['coefficients'].to_dict()),
                    'pvalues': convert_to_serializable(value['pvalues'].to_dict()),
                    'confidence_intervals': convert_to_serializable(value['conf_int'].to_dict()),
                    'model_summary': str(value['summary'])
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
        data_dir=data_dir
    )

    # Run analysis pipeline
    print("\nStarting analysis pipeline...")
    analysis.load_data()
    results = analysis.run_complete_analysis()
    analysis.save_results(results, 'results')
    print("\n=== Analysis complete ===")

if __name__ == "__main__":
    main()