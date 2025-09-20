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
from utils.glmm import BayesianGLMMAnalyzer  # Import the GLMM module we fixed

import warnings
warnings.filterwarnings('ignore')

class BehavioralContentAnalysis:
    def __init__(self, output_dir: Path, data_dir: Path, n_trim_TRs: int = 14, 
                 coding_type: str = 'deviation', reference_group: str = 'affair'):
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
        reference_group : str
            Reference group for treatment coding ('affair' or 'paranoia')
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.n_trim_TRs = n_trim_TRs
        self.data_loaded = False
        self.coding_type = coding_type
        self.reference_group = reference_group
        
        # Initialize the GLMM analyzer
        self.glmm_analyzer = BayesianGLMMAnalyzer(
            coding_type=coding_type,
            reference_group=reference_group,
            ar_lags=2  # Set default autoregressive lags
        )
        
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
            affair_beh_file = self.output_dir / '08_behav_results' / 'individual_response_evidence_affair.pkl'
            paranoia_beh_file = self.output_dir / '08_behav_results' / 'individual_response_evidence_paranoia.pkl'
            
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
            content_file = self.data_dir / '10_story_annotations_tr.csv'
            print(f"\nLoading content features from: {content_file}")
            content_df = pd.read_csv(content_file)
            
            # Prepare feature matrix
            print("Processing content features...")
            self.content_features = pd.DataFrame()
            for feature in self.feature_names:
                self.content_features[feature] = (content_df[feature].astype(str).str.lower() == "true").astype(int)
                print(f"  Processed feature: {feature}")
            
            # Add interaction terms like in the brain analysis
            self.content_features['lee_girl_verb'] = self.content_features['lee_girl_together'] * self.content_features['has_verb']
            self.content_features['arthur_adj'] = self.content_features['arthur_speaking'] * self.content_features['has_adj']
            
            print(f"Content features shape: {self.content_features.shape}")
            print(f"After trimming {self.n_trim_TRs} TRs")
            
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
        
    def prepare_data_for_glmm(self):
        """Prepare data for GLMM analysis"""
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Combine responses into a single array (subjects × timepoints)
        n_affair = len(self.affair_participants)
        n_paranoia = len(self.paranoia_participants)
        n_subjects = n_affair + n_paranoia
        
        # Create DV array (binary responses)
        dv = np.zeros((n_subjects, self.affair_responses.shape[1]), dtype=int)
        
        # Fill in response data for both groups
        for i in range(n_affair):
            dv[i] = self.affair_responses[i]
        
        for i in range(n_paranoia):
            dv[i + n_affair] = self.paranoia_responses[i]
        
        # Create group labels
        group_labels = ['affair'] * n_affair + ['paranoia'] * n_paranoia
        
        print(f"Prepared data: {n_subjects} subjects, {dv.shape[1]} timepoints")
        print(f"Affair subjects: {n_affair}, Paranoia subjects: {n_paranoia}")
        
        return dv, self.content_features, group_labels
        
    def analyze_content_relationships(self):
        """Analyze relationships between content features and behavioral responses using GLMM"""
        print("\n=== Running GLMM analysis for content relationships ===")
        
        try:
            # Prepare data in format needed for GLMM analysis
            dv, feature_matrix, group_labels = self.prepare_data_for_glmm()
            
            # Run individual feature analyses
            feature_results = {}
            all_features = feature_matrix.columns.tolist()
            
            # First analyze each feature individually
            for feature in all_features:
                print(f"Analyzing feature: {feature}")
                try:
                    # Create single-feature matrix
                    feature_df = pd.DataFrame({feature: feature_matrix[feature].values})
                    
                    # Prepare data for GLMM
                    model_data = self.glmm_analyzer.prepare_data(
                        dv=dv,
                        feature_matrix=feature_df,
                        group_labels=group_labels,
                        include_ar_terms=True
                    )
                    
                    # Fit model with the GLMM analyzer
                    result = self.glmm_analyzer.fit_model(
                        model_data=model_data,
                        feature_names=[feature],
                        include_interactions=True,
                        include_ar_terms=True
                    )
                    
                    # Apply Bayesian multiple comparison correction
                    mc_results = self.glmm_analyzer.apply_bayesian_multiple_comparison(result)
                    
                    # Compute standardized effect sizes
                    effect_sizes = self.glmm_analyzer.compute_effect_sizes(result)
                    
                    # Enrich result with additional metrics
                    result['multiple_comparison'] = mc_results
                    result['effect_sizes'] = effect_sizes
                    
                    # Store result
                    feature_results[feature] = result
                    print(f"Successfully analyzed feature: {feature}")
                    
                except Exception as e:
                    print(f"Error analyzing feature {feature}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Run combined analysis with all features
            print("\nRunning combined analysis with all features...")
            try:
                # Prepare data for GLMM with all features
                model_data = self.glmm_analyzer.prepare_data(
                    dv=dv,
                    feature_matrix=feature_matrix,
                    group_labels=group_labels,
                    include_ar_terms=True
                )
                
                # Fit model with all features
                combined_result = self.glmm_analyzer.fit_model(
                    model_data=model_data,
                    feature_names=feature_matrix.columns.tolist(),
                    include_interactions=True,
                    include_ar_terms=True
                )
                
                # Apply Bayesian multiple comparison correction
                combined_mc_results = self.glmm_analyzer.apply_bayesian_multiple_comparison(combined_result)
                
                # Add to results
                print("Combined analysis completed successfully")
                combined_result['multiple_comparison'] = combined_mc_results
                
            except Exception as e:
                print(f"Error in combined analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                combined_result = {"error": str(e)}
            
            # Return complete results
            return {
                'feature_results': feature_results,
                'combined_result': combined_result,
                'metadata': {
                    'coding_type': self.coding_type,
                    'reference_group': self.reference_group,
                    'n_subjects': {
                        'affair': len(self.affair_participants),
                        'paranoia': len(self.paranoia_participants)
                    },
                    'n_timepoints': dv.shape[1],
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            print(f"\n!!! ERROR in GLMM analysis !!!")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
            
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        if not self.data_loaded:
            self.load_data()
            
        results = {
            'group_timecourses': self.compute_group_timecourses(),
            'timepoint_differences': self.analyze_timepoint_differences(),
            'content_relationships': self.analyze_content_relationships()
        }
        
        return results
    
    def save_results(self, results):
        """Save analysis results"""
        save_dir = self.output_dir / '09_behavior_content_glmm' / self.coding_type
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n=== Saving results to {save_dir} ===")

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
            elif hasattr(obj, 'strftime'):  # Handle datetime objects
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            return obj

        # Convert results to saveable format
        save_results = convert_to_serializable(results)
        
        # Save as JSON with pretty printing
        with open(save_dir / "behavioral_content_analysis.json", 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # If the content_relationships analysis was successful, create a summary CSV
        if 'content_relationships' in results and 'feature_results' in results['content_relationships']:
            # Create summary dataframe
            summary_data = []
            for feature, result in results['content_relationships']['feature_results'].items():
                if 'error' in result:
                    continue
                
                row = {'feature': feature}
                
                # Add group effect
                if 'coefficients' in result and 'group_coded' in result['coefficients']:
                    row['group_effect'] = result['coefficients']['group_coded']
                    
                # Add interaction effect
                interaction_key = f'group_{feature}_interaction'
                if 'coefficients' in result and interaction_key in result['coefficients']:
                    row['interaction_effect'] = result['coefficients'][interaction_key]
                
                # Add posterior probabilities
                if 'posterior_prob' in result:
                    if 'group_coded' in result['posterior_prob']:
                        row['group_prob'] = result['posterior_prob']['group_coded']
                    if interaction_key in result['posterior_prob']:
                        row['interaction_prob'] = result['posterior_prob'][interaction_key]
                
                # Add significance from multiple comparison
                if 'multiple_comparison' in result and 'credible_effects' in result['multiple_comparison']:
                    row['fdr_credible'] = interaction_key in result['multiple_comparison']['credible_effects']
                
                # Add odds ratios
                if 'odds_ratios' in result:
                    if 'group_coded' in result['odds_ratios']:
                        row['group_odds_ratio'] = result['odds_ratios']['group_coded']['odds_ratio']
                    if interaction_key in result['odds_ratios']:
                        row['interaction_odds_ratio'] = result['odds_ratios'][interaction_key]['odds_ratio']
                
                # Add group-specific effects
                if 'group_specific_effects' in result and feature in result['group_specific_effects']:
                    effects = result['group_specific_effects'][feature]
                    
                    if 'affair_group' in effects:
                        affair = effects['affair_group']
                        row['affair_odds_ratio'] = affair.get('odds_ratio')
                        row['affair_prob_positive'] = affair.get('prob_positive')
                    
                    if 'paranoia_group' in effects:
                        paranoia = effects['paranoia_group']
                        row['paranoia_odds_ratio'] = paranoia.get('odds_ratio')
                        row['paranoia_prob_positive'] = paranoia.get('prob_positive')
                
                summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(save_dir / 'feature_summary.csv', index=False)
                print(f"Saved feature summary CSV with {len(summary_data)} features")
        
        print("Results saved successfully")

def main():
    print("\n=== Starting behavioral content analysis ===")
    # Setup paths
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        print("ERROR: SCRATCH_DIR environment variable not set!")
        return
        
    output_dir = Path(scratch_dir) / "output_RR"
    data_dir = Path(scratch_dir) / 'data' / 'stimuli'
    
    print(f"Output directory: {output_dir}")
    print(f"Data directory: {data_dir}")

    # Initialize analysis with deviation coding
    analysis = BehavioralContentAnalysis(
        output_dir=output_dir,
        data_dir=data_dir,
        coding_type='deviation',
        reference_group='affair'
    )

    # Run analysis pipeline
    print("\nStarting analysis pipeline...")
    analysis.load_data()
    results = analysis.run_complete_analysis()
    analysis.save_results(results)
    print("\n=== Analysis complete ===")

if __name__ == "__main__":
    main()