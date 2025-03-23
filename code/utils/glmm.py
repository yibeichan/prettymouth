import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

class BayesianGLMMAnalyzer:
    """
    Bayesian Generalized Linear Mixed Model (GLMM) analyzer for brain and behavioral data.
    
    This class provides a standardized approach to fit and analyze Bayesian GLMMs
    for both brain state sequences and behavioral responses in relation to
    content features. Includes autoregressive terms to handle temporal autocorrelation.
    """
    
    def __init__(self, coding_type: str = "deviation", reference_group: str = "affair", ar_lags: int = 2):
        """
        Initialize the GLMM analyzer.
        
        Parameters:
        -----------
        coding_type : str
            Type of categorical variable coding to use ('deviation' or 'treatment')
        reference_group : str
            Reference group for treatment coding (usually 'affair' or 'paranoia')
        ar_lags : int
            Number of autoregressive lags to include for handling temporal autocorrelation
        """
        self.coding_type = coding_type
        self.reference_group = reference_group
        self.ar_lags = ar_lags
        
        if coding_type not in ['deviation', 'treatment']:
            raise ValueError("coding_type must be either 'deviation' or 'treatment'")
    
    def prepare_data(self, 
                 dv: np.ndarray, 
                 feature_matrix: pd.DataFrame,
                 group_labels: List[str] = None,
                 target_state: int = None,
                 include_ar_terms: bool = True) -> pd.DataFrame:
        """
        Prepare data for GLMM analysis with flexible group assignment.
        
        Parameters:
        -----------
        dv : np.ndarray
            Array of state sequences or responses.
            Shape should be (n_subjects, n_timepoints)
        
        feature_matrix : pd.DataFrame
            DataFrame containing content features.
            Shape should be (n_timepoints, n_features)
            
        group_labels : List[str], optional
            List of group labels with length n_subjects.
            If None, assumes first half of subjects are 'affair', second half are 'paranoia'
            
        target_state : int, optional
            State index for brain state analysis.
            If None, assumes binary behavioral data.
            
        include_ar_terms : bool
            Whether to include autoregressive terms to handle temporal autocorrelation
            
        Returns:
        --------
        pd.DataFrame
            Prepared data for GLMM analysis
        """
        # Validate inputs
        if dv.ndim != 2:
            raise ValueError(f"DV must be 2-dimensional (got shape {dv.shape})")
        
        if feature_matrix.shape[0] != dv.shape[1]:
            raise ValueError(f"Feature matrix has {feature_matrix.shape[0]} rows, but DV has {dv.shape[1]} timepoints")
            
        # Get dimensions
        n_subjects, n_timepoints = dv.shape
        
        # Handle group labels
        if group_labels is None:
            # Default: first half affair, second half paranoia
            n_subjects_per_group = n_subjects // 2
            group_labels = ['affair'] * n_subjects_per_group + ['paranoia'] * (n_subjects - n_subjects_per_group)
        
        # Validate group labels
        if len(group_labels) != n_subjects:
            raise ValueError(f"group_labels should have length {n_subjects}, got {len(group_labels)}")
        
        # Check that we have exactly two unique groups
        unique_groups = sorted(set(group_labels))
        if len(unique_groups) != 2:
            raise ValueError(f"Expected exactly 2 unique groups, got {len(unique_groups)}: {unique_groups}")
        
        # Create dataframe for model
        model_data = pd.DataFrame(index=range(n_subjects * n_timepoints))
        
        # Add subject IDs
        model_data['subject'] = np.repeat(np.arange(n_subjects), n_timepoints)
        
        # Add group info from group_labels
        model_data['group'] = np.repeat(group_labels, n_timepoints)
        
        # Add time index
        model_data['time_idx'] = np.tile(np.arange(n_timepoints), n_subjects)
        
        # Add features
        for col in feature_matrix.columns:
            model_data[col] = np.tile(feature_matrix[col].values, n_subjects)
        
        # Add target variable
        if target_state is not None:
            # For brain state analysis: check if state equals target_state
            target_values = (dv == target_state).astype(int).flatten()
        else:
            # For behavioral analysis: use values directly
            target_values = dv.flatten()
                    
        model_data['target'] = target_values
        
        # Create group coding according to specified scheme
        if self.coding_type == "treatment":
            # For treatment coding, reference group is 0
            model_data['group_coded'] = (model_data['group'] != self.reference_group).astype(int)
        else:  # deviation coding
            # For deviation coding, code as +1 and -1
            first_group = unique_groups[0]
            model_data['group_coded'] = np.where(model_data['group'] == first_group, 1, -1)
        
        # Add autoregressive terms if requested
        if include_ar_terms:
            # Initialize AR lag columns
            for lag in range(1, self.ar_lags + 1):
                model_data[f'target_lag{lag}'] = np.nan
            
            # Calculate subject-specific AR terms
            for subject in range(n_subjects):
                subject_mask = model_data['subject'] == subject
                subject_indices = np.where(subject_mask)[0]
                
                # For each lag, calculate shifted values
                for lag in range(1, self.ar_lags + 1):
                    lag_col = f'target_lag{lag}'
                    lagged_values = np.zeros(n_timepoints)
                    lagged_values[lag:] = target_values[subject_indices[:n_timepoints-lag]]
                    model_data.loc[subject_mask, lag_col] = lagged_values
                    
        # Add interaction terms
        for feature in feature_matrix.columns:
            model_data[f'group_{feature}_interaction'] = model_data['group_coded'] * model_data[feature]
                
        return model_data
    
    def check_autocorrelation(self, model_data: pd.DataFrame, max_lag: int = 10) -> Dict[str, Any]:
        """
        Check for temporal autocorrelation in the target variable.
        
        Parameters:
        -----------
        model_data : pd.DataFrame
            Data frame prepared with prepare_data()
        max_lag : int
            Maximum lag to check for autocorrelation
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with autocorrelation diagnostics
        """
        print("\n=== Checking for Temporal Autocorrelation ===")
        
        autocorr_results = {}
        significant_count = 0
        
        # Analyze autocorrelation for each subject
        for subject in sorted(model_data['subject'].unique()):
            # Get data for this subject
            subject_data = model_data[model_data['subject'] == subject]
            
            # Need at least 2*max_lag+5 observations for meaningful test
            if len(subject_data) < 2*max_lag+5:
                print(f"Subject {subject}: Not enough data points for autocorrelation test")
                continue
                
            # Perform Ljung-Box test
            try:
                lb_test = acorr_ljungbox(subject_data['target'], lags=range(1, max_lag+1))
                
                # Check if any lag has significant autocorrelation
                is_significant = any(p < 0.05 for p in lb_test[1])
                if is_significant:
                    significant_count += 1
                
                autocorr_results[f'subject_{subject}'] = {
                    'lb_statistic': lb_test[0].tolist(),
                    'p_values': lb_test[1].tolist(),
                    'significant': is_significant
                }
            except Exception as e:
                print(f"Subject {subject}: Error in autocorrelation test - {str(e)}")
                autocorr_results[f'subject_{subject}'] = {'error': str(e)}
        
        # Calculate autocorrelation function for visual inspection
        acf_avg = np.zeros(max_lag + 1)
        acf_count = 0
        
        for subject in sorted(model_data['subject'].unique()):
            subject_data = model_data[model_data['subject'] == subject]
            if len(subject_data) >= max_lag + 5:
                try:
                    # Calculate autocorrelation function
                    series = pd.Series(subject_data['target'].values)
                    acf = [1.0]  # Lag 0 is always 1
                    for lag in range(1, max_lag + 1):
                        # Calculate autocorrelation for this lag
                        correlation = series.autocorr(lag=lag)
                        if not np.isnan(correlation):
                            acf.append(correlation)
                        else:
                            acf.append(0)
                    
                    if len(acf) == max_lag + 1:
                        acf_avg += np.array(acf)
                        acf_count += 1
                except Exception as e:
                    # More robust error handling
                    print(f"Subject {subject}: Error calculating ACF - {str(e)}")
                    continue
        
        # Average the ACF across subjects
        if acf_count > 0:
            acf_avg /= acf_count
            
        # Summary statistics
        summary = {
            'subjects_tested': len(autocorr_results),
            'subjects_with_autocorr': significant_count,
            'percent_with_autocorr': 100 * significant_count / len(autocorr_results) if autocorr_results else 0,
            'avg_acf': acf_avg.tolist() if acf_count > 0 else None
        }
        
        print(f"{significant_count} out of {len(autocorr_results)} subjects show significant autocorrelation")
        
        # Return both detailed and summary results
        return {
            'subject_results': autocorr_results,
            'summary': summary
        }
    
    def plot_autocorrelation(self, autocorr_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot autocorrelation diagnostics.
        
        Parameters:
        -----------
        autocorr_results : Dict[str, Any]
            Results from check_autocorrelation()
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with autocorrelation plots
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot average ACF if available
        if 'summary' in autocorr_results and autocorr_results['summary']['avg_acf'] is not None:
            acf_values = autocorr_results['summary']['avg_acf']
            lags = range(len(acf_values))
            
            ax.bar(lags, acf_values, width=0.3, alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Add confidence bands (approximate for visualization)
            ci = 1.96 / np.sqrt(len(lags))
            ax.axhline(y=ci, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-ci, color='r', linestyle='--', alpha=0.5)
            
            # Set labels
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.set_title('Average Autocorrelation Function Across Subjects')
            
        plt.tight_layout()
        return fig
    
    def extract_posterior_samples(self, results, design_matrix, ar_terms, n_samples=5000):
        """
        Extract posterior samples from Bayesian model results with multiple fallback methods.
        
        Parameters:
        -----------
        results : statsmodels model results object
            The results from a fitted Bayesian model
        design_matrix : pandas DataFrame
            The design matrix used in the model
        ar_terms : list
            List of autoregressive terms to extract samples for
        n_samples : int
            Number of posterior samples to extract
            
        Returns:
        --------
        dict
            Dictionary of posterior samples for each term
        """
        post_samples = {}
        
        # Get posterior SDs if available
        if hasattr(self, '_get_posterior_sds'):
            sd_dict = self._get_posterior_sds(results, design_matrix.columns)
        else:
            # Create a fallback SD dictionary
            sd_dict = {}
            for i, col in enumerate(design_matrix.columns):
                if hasattr(results, 'bse') and i < len(results.bse):
                    sd_dict[col] = results.bse[i]
                elif hasattr(results, 'std_params') and i < len(results.std_params):
                    sd_dict[col] = results.std_params[i]
                else:
                    # Default to 10% of parameter value as rough estimate
                    param_val = results.params[i] if i < len(results.params) else 0
                    sd_dict[col] = abs(param_val * 0.1) or 0.1  # Avoid 0 SDs
        
        # Try different methods to extract samples from various statsmodels Bayesian objects
        
        # Method 1: Direct access to get_posterior_samples method (newer versions)
        if hasattr(results, 'get_posterior_samples'):
            try:
                return results.get_posterior_samples(n_samples)
            except Exception as e:
                print(f"Could not use get_posterior_samples: {str(e)}")
        
        # Method 2: Access to random_effects/random_state attributes (some versions)
        for attr_name in ['random_effects', 'random_effects_samples', 'random_state', 'posterior_samples']:
            if hasattr(results, attr_name):
                try:
                    samples_obj = getattr(results, attr_name)
                    if isinstance(samples_obj, dict) and any(term in samples_obj for term in ar_terms):
                        # Extract samples for requested terms
                        for term in ar_terms:
                            if term in samples_obj:
                                post_samples[term] = samples_obj[term][:n_samples] if hasattr(samples_obj[term], '__len__') else None
                        return post_samples
                    elif isinstance(samples_obj, np.ndarray) and samples_obj.shape[0] >= n_samples:
                        # Need to map array indices to parameter names
                        param_names = design_matrix.columns
                        for i, term in enumerate(param_names):
                            if term in ar_terms:
                                post_samples[term] = samples_obj[:n_samples, i]
                        return post_samples
                except Exception as e:
                    print(f"Could not extract samples from {attr_name}: {str(e)}")
        
        # Method 3: Specialized handling for BayesMixedGLMResults
        if 'BayesMixedGLMResults' in str(type(results)):
            try:
                # For BayesMixedGLMResults in statsmodels
                # Try to access VB attribute that might contain posterior info
                if hasattr(results, 'vb_output'):
                    vb_out = results.vb_output
                    if hasattr(vb_out, 'posterior_mean') and hasattr(vb_out, 'posterior_sd'):
                        # Extract parameter estimates and SDs from VB output
                        param_names = design_matrix.columns
                        for i, term in enumerate(param_names):
                            if term in ar_terms:
                                # Generate samples from approximate posterior distribution
                                mean = vb_out.posterior_mean[i] if i < len(vb_out.posterior_mean) else 0
                                sd = vb_out.posterior_sd[i] if i < len(vb_out.posterior_sd) else 1
                                post_samples[term] = np.random.normal(mean, sd, size=n_samples)
                        return post_samples
                    
                # Try to access parameter posterior means and SDs directly
                if hasattr(results, 'params'):
                    param_names = design_matrix.columns
                    for i, term in enumerate(param_names):
                        if term in ar_terms and i < len(results.params):
                            # Generate samples from approximate posterior distribution
                            mean = results.params[i]
                            sd = sd_dict.get(term, abs(mean * 0.1) or 0.1)  # Default to 10% of estimate if SD not available
                            post_samples[term] = np.random.normal(mean, sd, size=n_samples)
                    return post_samples
                    
            except Exception as e:
                print(f"Error in specialized BayesMixedGLMResults handling: {str(e)}")
        
        # Method 4: Last resort - generate samples based on parameter estimates and SDs
        print("Warning: Using approximation method to generate posterior samples")
        for term in ar_terms:
            if term in design_matrix.columns:
                # Get index of the term in the parameters
                try:
                    idx = list(design_matrix.columns).index(term)
                    if idx < len(results.params):
                        # Generate approximate posterior samples
                        estimate = results.params[idx]
                        sd = sd_dict.get(term, abs(estimate * 0.1) or 0.1)  # Default to 10% of estimate if SD not available
                        post_samples[term] = np.random.normal(estimate, sd, size=n_samples)
                except (ValueError, IndexError) as e:
                    print(f"Could not find index for {term}: {str(e)}")
        
        return post_samples

    def _get_posterior_sds(self, results, column_names):
        """
        Helper method to extract posterior standard deviations from model results.
        
        Parameters:
        -----------
        results : statsmodels model results object
            The results from a fitted Bayesian model
        column_names : list or Index
            Names of parameters
            
        Returns:
        --------
        dict
            Dictionary mapping parameter names to standard deviations
        """
        sd_dict = {}
        
        # Try different attributes where SDs might be stored
        if hasattr(results, 'bse'):
            # Traditional frequentist models or some Bayesian models
            for i, name in enumerate(column_names):
                if i < len(results.bse):
                    sd_dict[name] = float(results.bse[i])
                    
        elif hasattr(results, 'std_params'):
            # Some Bayesian models
            for i, name in enumerate(column_names):
                if i < len(results.std_params):
                    sd_dict[name] = float(results.std_params[i])
                    
        elif hasattr(results, 'posterior_sd'):
            # Check if posterior_sd is a dictionary or array-like
            if isinstance(results.posterior_sd, dict):
                # Direct dictionary mapping
                for name in column_names:
                    sd_dict[name] = float(results.posterior_sd.get(name, 1.0))
            else:
                # Array-like
                for i, name in enumerate(column_names):
                    if i < len(results.posterior_sd):
                        sd_dict[name] = float(results.posterior_sd[i])
        
        else:
            # As a fallback, try to extract from covariance matrix
            try:
                if hasattr(results, 'cov_params'):
                    # Diagonal of covariance matrix contains variances
                    variances = np.diag(results.cov_params())
                    for i, name in enumerate(column_names):
                        if i < len(variances):
                            sd_dict[name] = float(np.sqrt(variances[i]))
            except Exception as e:
                print(f"Warning: Error extracting standard errors from covariance: {str(e)}. Using rough estimates.")
        
        # For any missing SDs, use rough estimate based on parameter value
        for i, name in enumerate(column_names):
            if name not in sd_dict and i < len(results.params):
                # Default to 10% of absolute parameter value (or 0.1 if parameter is 0)
                param_val = abs(results.params[i])
                sd_dict[name] = float(param_val * 0.1) if param_val > 0 else 0.1
        
        return sd_dict

    def fit_model(self, 
                model_data: pd.DataFrame, 
                feature_names: List[str], 
                include_interactions: bool = True,
                include_ar_terms: bool = True) -> Dict[str, Any]:
        """
        Fit a Bayesian GLMM to the prepared data.
        
        Parameters:
        -----------
        model_data : pd.DataFrame
            Prepared data from prepare_data()
        feature_names : List[str]
            List of feature names to include in the model
        include_interactions : bool
            Whether to include group × feature interactions
        include_ar_terms : bool
            Whether to include autoregressive terms
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing model results and statistics
        """
        print("\n=== Fitting Bayesian GLMM ===")
        
        try:
            # Create design matrix columns
            design_cols = ['const', 'group_coded'] + feature_names
            
            # Add AR terms if requested
            ar_terms = []
            if include_ar_terms:
                ar_terms = [f'target_lag{lag}' for lag in range(1, self.ar_lags + 1)]
                # Check if AR terms exist in the data
                ar_terms = [term for term in ar_terms if term in model_data.columns]
                if ar_terms:
                    print(f"Including autoregressive terms: {', '.join(ar_terms)}")
                    design_cols += ar_terms
                else:
                    print("No autoregressive terms found in data")
            
            # Add interaction terms if requested
            interaction_cols = []
            if include_interactions:
                interaction_cols = [f'group_{feature}_interaction' for feature in feature_names]
                print(f"Including interaction terms: {', '.join(interaction_cols)}")
                design_cols += interaction_cols
                
            # Create design matrix
            design_matrix = sm.add_constant(model_data[design_cols[1:]])
            
            # Create subject dummies for random effects
            subject_dummies = pd.get_dummies(model_data['subject']).astype(np.float64)
            
            # Fit model
            print(f"Fitting model with {len(design_cols)} fixed effects and {subject_dummies.shape[1]} random effects...")
            model = BinomialBayesMixedGLM(
                model_data['target'],
                design_matrix,
                exog_vc=subject_dummies,
                vc_names=['subject'],
                ident=np.ones(subject_dummies.shape[1], dtype=np.int32)
            )
            
            # First try MAP (maximum a posteriori)
            try:
                results = model.fit_map()
                fit_method = "MAP"
            except Exception as e:
                print(f"MAP fitting failed ({str(e)}), trying VB...")
                results = model.fit_vb(verbose=True)
                fit_method = "VB"
                
            print(f"Model fitted successfully using {fit_method}")
            
            # Also fit a simpler model for comparison
            print("Fitting simpler model for comparison...")
            simple_design = sm.add_constant(model_data[['group_coded']])
            simple_model = BinomialBayesMixedGLM(
                model_data['target'],
                simple_design,
                exog_vc=subject_dummies,
                vc_names=['subject'],
                ident=np.ones(subject_dummies.shape[1], dtype=np.int32)
            )
            
            try:
                simple_results = simple_model.fit_map()
                simple_fit_method = "MAP"
            except:
                simple_results = simple_model.fit_vb()
                simple_fit_method = "VB"
                
            print(f"Simple model fitted successfully using {simple_fit_method}")
            
            # Check if AR terms improved the model
            if include_ar_terms and ar_terms:
                # Extract coefficients for AR terms
                ar_coefs = {}
                for term in ar_terms:
                    if term in design_matrix.columns:
                        idx = list(design_matrix.columns).index(term)
                        ar_coefs[term] = results.params[idx]
                
                # Print AR coefficients
                print("\nAutoregressive coefficients:")
                for term, coef in ar_coefs.items():
                    print(f"  {term}: {coef:.4f}")
                    
                # For Bayesian results, use credible intervals and posterior probability
                if hasattr(results, 'fit_bayes') or str(type(results)).find('Bayes') != -1:
                    # Bayesian approach to assess AR terms
                    ar_evidence = False
                    print("\nBayesian assessment of autoregressive terms:")
                    
                    # Get posterior samples using our robust method
                    try:
                        post_samples = self.extract_posterior_samples(results, design_matrix, ar_terms)
                    
                        for term in ar_terms:
                            if term in design_matrix.columns:
                                if term in post_samples and post_samples[term] is not None:
                                    # Extract posterior samples for this AR term
                                    term_samples = post_samples[term]
                                    
                                    # Calculate 95% credible interval
                                    credible_interval = np.percentile(term_samples, [2.5, 97.5])
                                    
                                    # Calculate probability of effect direction
                                    prob_positive = np.mean(term_samples > 0)
                                    prob_negative = np.mean(term_samples < 0)
                                    prob_direction = max(prob_positive, prob_negative)
                                    
                                    # Calculate probability of practical significance (effect size threshold)
                                    effect_threshold = 0.1  # Adjust based on your domain knowledge
                                    prob_meaningful = np.mean(np.abs(term_samples) > effect_threshold)
                                    
                                    print(f"  {term}: 95% CI [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]")
                                    print(f"    Prob(direction): {prob_direction:.4f} ({'positive' if prob_positive > prob_negative else 'negative'})")
                                    print(f"    Prob(|effect| > {effect_threshold}): {prob_meaningful:.4f}")
                                    
                                    # Consider evidence if zero is not in credible interval or high directional probability
                                    if (credible_interval[0] > 0 or credible_interval[1] < 0) or prob_direction > 0.95:
                                        ar_evidence = True
                                else:
                                    # If we can't get samples for this parameter, use parameter estimate and SD
                                    print(f"  {term}: No posterior samples available, using approximation")
                                    param_idx = list(design_matrix.columns).index(term)
                                    if param_idx < len(results.params):
                                        estimate = results.params[param_idx]
                                        # Get SD using our helper method
                                        sd_dict = self._get_posterior_sds(results, design_matrix.columns)
                                        sd = sd_dict.get(term, abs(estimate * 0.1) or 0.1)
                                        
                                        # Approximate using normal distribution
                                        z_score = estimate / sd
                                        prob_dir = stats.norm.cdf(abs(z_score))
                                        print(f"  {term}: estimate {estimate:.4f}, approx. Prob(direction): {prob_dir:.4f}")
                                        if prob_dir > 0.95:
                                            ar_evidence = True
                                    else:
                                        print(f"  {term}: Could not find parameter estimate")
                                        
                    except Exception as e:
                        print(f"Error in Bayesian assessment: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                if 'ar_evidence' in locals() and ar_evidence:
                    print("Strong evidence for temporal autocorrelation")
                else:
                    print("No clear evidence for temporal autocorrelation")
            
            # Prepare and return results
            full_results = self._prepare_results(
                results, 
                simple_results,
                design_matrix, 
                feature_names,
                ar_terms if include_ar_terms else [],
                fit_method,
                interaction_terms=interaction_cols if include_interactions else []
            )
            
            return full_results
            
        except Exception as e:
            print(f"Error fitting GLMM: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _prepare_results(self, 
                   model_results, 
                   simple_results,
                   design_matrix, 
                   feature_names: List[str],
                   ar_terms: List[str],
                   fit_method: str,
                   probability_mass: float = 0.95, 
                   rope_width: float = 0.1,
                   interaction_terms: List[str] = None) -> Dict[str, Any]:
        """
        Process and organize Bayesian GLMM results.
        
        Parameters:
        -----------
        model_results : statsmodels GLMM results object
            Results from the fitted model
        simple_results : statsmodels GLMM results object
            Results from a simpler model for comparison
        design_matrix : pandas.DataFrame
            Design matrix used for model fitting
        feature_names : List[str]
            List of feature names used in the model
        ar_terms : List[str]
            List of autoregressive terms included in the model
        fit_method : str
            Method used to fit the model ('MAP' or 'VB')
        probability_mass : float
            Probability mass for highest density intervals
        rope_width : float
            Width of Region of Practical Equivalence for effect size interpretation
            
        Returns:
        --------
        dict
            Dictionary containing organized Bayesian results and metrics
        """
        # Get feature names
        column_names = design_matrix.columns
        
        # Get fixed effects parameters and standard deviations
        fe_params = model_results.params[:len(column_names)]
    
        # Get standard deviations - fix for missing bse attribute
        if hasattr(model_results, 'bse'):
            # Traditional frequentist models have bse
            fe_sds = model_results.bse[:len(column_names)]
        elif hasattr(model_results, 'std_params'):
            # Some Bayesian models use std_params
            fe_sds = model_results.std_params[:len(column_names)]
        elif hasattr(model_results, 'posterior_sd'):
            # Check if posterior_sd is a dictionary or array-like
            if isinstance(model_results.posterior_sd, dict):
                fe_sds = np.array([model_results.posterior_sd.get(col, 1.0) for col in column_names])
            else:
                fe_sds = model_results.posterior_sd[:len(column_names)]
        else:
            # As a fallback, try to extract from vcov matrix diagonals
            print("Warning: Could not find standard error attribute. Trying to extract from covariance matrix.")
            try:
                if hasattr(model_results, 'cov_params'):
                    # Diagonal of covariance matrix contains variances
                    variances = np.diag(model_results.cov_params())[:len(column_names)]
                    fe_sds = np.sqrt(variances)
                else:
                    # Last resort - use 10% of parameter value as a rough estimate
                    print("Warning: No covariance matrix found. Using rough estimate for standard errors.")
                    fe_sds = np.abs(fe_params) * 0.1
            except Exception as e:
                print(f"Warning: Error extracting standard errors: {str(e)}. Using rough estimates.")
                fe_sds = np.abs(fe_params) * 0.1
        
        # Create parameter dictionaries with proper keys
        coef_dict = {name: float(val) for name, val in zip(column_names, fe_params)}
        sd_dict = {name: float(val) for name, val in zip(column_names, fe_sds)}
        
        # Calculate posterior probabilities
        # For each parameter, calculate P(parameter > 0) and P(parameter < 0)
        posterior_prob_positive = 1 - stats.norm.cdf(0, loc=fe_params, scale=fe_sds)
        posterior_prob_negative = stats.norm.cdf(0, loc=fe_params, scale=fe_sds)
        
        # Choose the appropriate probability based on the direction of the effect
        posterior_prob = np.where(fe_params > 0, posterior_prob_positive, posterior_prob_negative)
        
        posterior_prob_dict = {name: float(val) for name, val in zip(column_names, posterior_prob)}
        pos_prob_dict = {name: float(val) for name, val in zip(column_names, posterior_prob_positive)}
        neg_prob_dict = {name: float(val) for name, val in zip(column_names, posterior_prob_negative)}
        
        # Calculate z-score for desired probability mass
        z_score = stats.norm.ppf((1 + probability_mass) / 2)
        
        # Calculate Highest Density Interval (HDI)
        hdi_lower = fe_params - z_score * fe_sds
        hdi_upper = fe_params + z_score * fe_sds
        
        hdi_dict = {
            name: {'lower': float(l), 'upper': float(u)} 
            for name, l, u in zip(column_names, hdi_lower, hdi_upper)
        }
        
        # Calculate ROPE probabilities (Region of Practical Equivalence)
        prob_in_rope = (stats.norm.cdf(rope_width, loc=fe_params, scale=fe_sds) - 
                        stats.norm.cdf(-rope_width, loc=fe_params, scale=fe_sds))
        prob_practical_pos = 1 - stats.norm.cdf(rope_width, loc=fe_params, scale=fe_sds)
        prob_practical_neg = stats.norm.cdf(-rope_width, loc=fe_params, scale=fe_sds)
        
        rope_dict = {
            name: {
                'prob_in_rope': float(in_rope),
                'prob_practical_pos': float(p_pos),
                'prob_practical_neg': float(p_neg)
            }
            for name, in_rope, p_pos, p_neg in zip(
                column_names, prob_in_rope, prob_practical_pos, prob_practical_neg
            )
        }
        
        # Calculate improved Bayes factors using Savage-Dickey density ratio
        bayes_factors = {}

        for i, name in enumerate(column_names):
            # Skip intercept
            if name == 'const':
                continue
                
            param = fe_params[i]
            sd = fe_sds[i]
            
            # Use standardized parameter for Bayes factor calculation
            standardized_param = param / sd
            
            # Approximate prior density at 0 (using unit normal)
            prior_density = stats.norm.pdf(0, loc=0, scale=1)
            
            # Approximate posterior density at 0
            posterior_density = stats.norm.pdf(0, loc=standardized_param, scale=1)
            
            # Calculate Bayes factor
            if posterior_density < 1e-10:
                bf = 1000  # Cap for numerical stability
            else:
                bf = prior_density / posterior_density
                
            bayes_factors[name] = float(bf)
        
        # Calculate odds ratios
        odds_ratio_dict = {}
        
        # For each parameter, calculate its odds ratio 
        for i, name in enumerate(column_names):
            param_value = fe_params[i]
            param_lower = hdi_lower[i]
            param_upper = hdi_upper[i]
            
            # For deviation coding, the interpretation is different
            if self.coding_type == "deviation" and name == "group_coded":
                # In deviation coding (+1/-1), the group coefficient represents 
                # half the difference between groups when using +1/-1 coding
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
        
        # Calculate group-specific effects based on coding scheme
        group_specific_effects = self._calculate_group_specific_effects(
            coef_dict, 
            sd_dict, 
            feature_names, 
            z_score
        )
        
        # Calculate VIF for multicollinearity diagnosis
        vif_data = {}
        try:
            # Only calculate VIF for main effects (not interactions or AR terms)
            main_effect_cols = [col for col in design_matrix.columns 
                               if not col.startswith('group_') and col != 'const'
                               and col not in ar_terms]
            
            X = design_matrix[main_effect_cols]
            for idx, col in enumerate(main_effect_cols):
                vif_data[col] = float(variance_inflation_factor(X.values, idx))
        except Exception as e:
            print(f"Warning: Could not calculate VIF factors: {str(e)}")
            vif_data = {"error": str(e)}
        
        # Model comparison metrics
        model_comparison = {
            'full_model': {
                'dic': getattr(model_results, 'dic', None),
                'elbo': getattr(model_results, 'vb_elbo', None),
            },
            'simple_model': {
                'dic': getattr(simple_results, 'dic', None),
                'elbo': getattr(simple_results, 'vb_elbo', None),
            }
        }
        
        # Add evidence categories based on Bayes factors
        evidence_categories = {}
        for name, bf in bayes_factors.items():
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
        
        # Prepare final results dictionary
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
            'bayes_factors': bayes_factors,
            'evidence_categories': evidence_categories,
            
            # Interpretable effect sizes
            'odds_ratios': odds_ratio_dict,
            'group_specific_effects': group_specific_effects,
            
            # AR terms information
            'ar_terms': {term: coef_dict.get(term) for term in ar_terms},
            
            # Model diagnostics
            'model_summary': str(model_results.summary()),
            'simple_model_summary': str(simple_results.summary()),
            'convergence': getattr(model_results, 'converged', None),
            'n_obs': len(design_matrix),
            'vif': vif_data,
            'model_comparison': model_comparison,
            
            # Fitting information
            'fit_method': fit_method,
            'coding_scheme': self.coding_type,
            'reference_group': self.reference_group
        }
        
        return results_dict
    
    def _calculate_group_specific_effects(self, 
                                     coef_dict: Dict[str, float], 
                                     sd_dict: Dict[str, float], 
                                     feature_names: List[str],
                                     z_score: float) -> Dict[str, Dict]:
        """
        Calculate group-specific effects for each feature based on coding scheme.
        """
        group_specific_effects = {}
        
        # Calculate group-specific effects for each feature
        for feature in feature_names:
            # Check if feature and its interaction are in the model
            if feature not in coef_dict:
                continue
                
            interaction_term = f'group_{feature}_interaction'
            if interaction_term not in coef_dict:
                continue
                
            # Get coefficients
            feature_coef = coef_dict[feature]
            interaction_coef = coef_dict[interaction_term]
            
            # Get standard deviations
            feature_sd = sd_dict[feature]
            interaction_sd = sd_dict[interaction_term]
            
            if self.coding_type == "deviation":
                # For deviation coding, calculate effects for each group
                # For +1/-1 coding:
                affair_effect = feature_coef + interaction_coef
                paranoia_effect = feature_coef - interaction_coef
                
                # Calculate approximate SDs
                affair_sd = np.sqrt(feature_sd**2 + interaction_sd**2)
                paranoia_sd = np.sqrt(feature_sd**2 + interaction_sd**2)
                
                # Calculate odds ratios and HDIs (FIXED: properly calculate both groups)
                affair_or = np.exp(affair_effect)
                affair_or_lower = np.exp(affair_effect - z_score * affair_sd)
                affair_or_upper = np.exp(affair_effect + z_score * affair_sd)
                
                paranoia_or = np.exp(paranoia_effect)
                paranoia_or_lower = np.exp(paranoia_effect - z_score * paranoia_sd)
                paranoia_or_upper = np.exp(paranoia_effect + z_score * paranoia_sd)
                
                # Calculate probabilities of positive effects
                p_pos_affair = 1 - stats.norm.cdf(0, loc=affair_effect, scale=affair_sd)
                p_pos_paranoia = 1 - stats.norm.cdf(0, loc=paranoia_effect, scale=paranoia_sd)
                
                # Probability that effect is stronger in affair group
                p_diff = 1 - stats.norm.cdf(0, loc=interaction_coef, scale=interaction_sd)
                
                group_specific_effects[feature] = {
                    'affair_group': {
                        'coefficient': float(affair_effect),
                        'std_error': float(affair_sd),
                        'odds_ratio': float(affair_or),
                        'lower': float(affair_or_lower),
                        'upper': float(affair_or_upper),
                        'prob_positive': float(p_pos_affair)
                    },
                    'paranoia_group': {
                        'coefficient': float(paranoia_effect),
                        'std_error': float(paranoia_sd),
                        'odds_ratio': float(paranoia_or),
                        'lower': float(paranoia_or_lower),
                        'upper': float(paranoia_or_upper),
                        'prob_positive': float(p_pos_paranoia)
                    },
                    'diff_between_groups': {
                        'coefficient': float(interaction_coef),
                        'std_error': float(interaction_sd),
                        'prob_stronger_in_affair': float(p_diff),
                        'prob_stronger_in_paranoia': float(1 - p_diff)
                    }
                }
            else:  # treatment coding
                # For treatment coding, reference group is the main effect
                # Fix: add check if interaction term exists
                if self.reference_group == "affair":
                    ref_effect = feature_coef
                    comp_effect = feature_coef + interaction_coef
                    ref_group = 'affair'
                    comp_group = 'paranoia'
                else:
                    ref_effect = feature_coef
                    comp_effect = feature_coef + interaction_coef
                    ref_group = 'paranoia'
                    comp_group = 'affair'
                
                # Calculate standard deviations
                ref_sd = feature_sd
                comp_sd = np.sqrt(feature_sd**2 + interaction_sd**2)
                
                # Calculate odds ratios (FIXED: calculate bounds for both groups correctly)
                ref_or = np.exp(ref_effect)
                ref_or_lower = np.exp(ref_effect - z_score * ref_sd)
                ref_or_upper = np.exp(ref_effect + z_score * ref_sd)
                
                comp_or = np.exp(comp_effect)
                comp_or_lower = np.exp(comp_effect - z_score * comp_sd)
                comp_or_upper = np.exp(comp_effect + z_score * comp_sd)
                
                # Calculate probabilities
                p_pos_ref = 1 - stats.norm.cdf(0, loc=ref_effect, scale=ref_sd)
                p_pos_comp = 1 - stats.norm.cdf(0, loc=comp_effect, scale=comp_sd)
                
                # Probability that effect is stronger in comparison group
                p_diff = 1 - stats.norm.cdf(0, loc=interaction_coef, scale=interaction_sd)
                
                group_specific_effects[feature] = {
                    ref_group + '_group': {
                        'coefficient': float(ref_effect),
                        'std_error': float(ref_sd),
                        'odds_ratio': float(ref_or),
                        'lower': float(ref_or_lower),
                        'upper': float(ref_or_upper),
                        'prob_positive': float(p_pos_ref)
                    },
                    comp_group + '_group': {
                        'coefficient': float(comp_effect),
                        'std_error': float(comp_sd),
                        'odds_ratio': float(comp_or),
                        'lower': float(comp_or_lower),
                        'upper': float(comp_or_upper),
                        'prob_positive': float(p_pos_comp)
                    },
                    'diff_between_groups': {
                        'coefficient': float(interaction_coef),
                        'std_error': float(interaction_sd),
                        f'prob_stronger_in_{comp_group}': float(p_diff),
                        f'prob_stronger_in_{ref_group}': float(1 - p_diff)
                    }
                }
                
        return group_specific_effects
        
    def apply_bayesian_multiple_comparison(self, results: Dict) -> Dict:
        """
        Apply Bayesian multiple comparison adjustment to results.
        
        Parameters:
        -----------
        results : Dict
            Dictionary of GLMM results from fit_model()
            
        Returns:
        --------
        Dict
            Dictionary with multiple comparison adjustments
        """
        # Extract all effects (both main effects and interactions)
        effects = [name for name in results['coefficients'].keys() 
                if name not in ['const'] and not name.startswith('target_lag')]
        
        # Separate main effects and interaction terms
        main_effects = [name for name in effects if not name.startswith('group_') and name != 'group_coded']
        interaction_terms = [name for name in effects if name.startswith('group_') and name.endswith('_interaction')]
        
        # Create lists to store probabilities and names
        all_effects = []
        all_probs = []
        
        # Debug: Print the type of posterior_prob
        print(f"Type of posterior_prob: {type(results.get('posterior_prob', {}))}")
        
        # Add main effects
        for term in main_effects:
            if term in results.get('posterior_prob', {}):
                prob_value = results['posterior_prob'][term]
                
                # Debug: check the type of each probability
                print(f"Term: {term}, Prob type: {type(prob_value)}, Value: {prob_value}")
                
                # Ensure we have a numeric value, not a list or other complex object
                if isinstance(prob_value, (list, tuple, np.ndarray)):
                    # If it's a list-like object, take the first element if possible
                    try:
                        prob_value = float(prob_value[0])
                    except (IndexError, TypeError):
                        print(f"Warning: Skipping term {term} due to non-numeric probability")
                        continue
                
                # Ensure it's a valid floating-point value
                try:
                    prob_value = float(prob_value)
                    all_effects.append(term)
                    all_probs.append(prob_value)
                except (ValueError, TypeError):
                    print(f"Warning: Skipping term {term} due to invalid probability value: {prob_value}")
        
        # Add interaction terms  
        for term in interaction_terms:
            if term in results.get('posterior_prob', {}):
                prob_value = results['posterior_prob'][term]
                
                # Ensure we have a numeric value
                if isinstance(prob_value, (list, tuple, np.ndarray)):
                    try:
                        prob_value = float(prob_value[0])
                    except (IndexError, TypeError):
                        print(f"Warning: Skipping term {term} due to non-numeric probability")
                        continue
                
                # Ensure it's a valid floating-point value
                try:
                    prob_value = float(prob_value)
                    all_effects.append(term)
                    all_probs.append(prob_value)
                except (ValueError, TypeError):
                    print(f"Warning: Skipping term {term} due to invalid probability value: {prob_value}")
        
        # Check if we have any valid effects
        if not all_effects:
            print("Warning: No valid effects found for multiple comparison correction")
            return {
                'all_effects': [],
                'posterior_probs': {},
                'significant_effects': [],
                'significant_main_effects': [],
                'significant_interactions': [],
                'fdr_threshold': 0.05,
                'cumulative_fdr': {}
            }
        
        # Convert all_probs to numpy array to ensure consistent numeric handling
        all_probs = np.array(all_probs, dtype=float)
        
        # Apply Bayesian FDR
        # Sort probabilities
        sorted_idx = np.argsort(all_probs)[::-1]  # Descending
        sorted_probs = all_probs[sorted_idx]
        sorted_effects = np.array(all_effects)[sorted_idx]
        
        # Calculate cumulative FDR
        cumulative_fdr = []
        for i in range(len(sorted_probs)):
            # Sum of expected false discoveries divided by number of discoveries
            # FIXED: Ensure we're working with scalar values, not lists
            expected_false_discoveries = np.sum(1.0 - sorted_probs[:i+1])
            cumulative_fdr.append(float(expected_false_discoveries) / (i+1))
        
        # Find effects passing FDR threshold
        fdr_threshold = 0.05  # Standard threshold
        passing_idx = np.where(np.array(cumulative_fdr) <= fdr_threshold)[0]
        passing_effects = sorted_effects[passing_idx].tolist() if len(passing_idx) > 0 else []
        
        # Separate significant effects by type
        significant_main = [effect for effect in passing_effects if effect in main_effects]
        significant_interactions = [effect for effect in passing_effects if effect in interaction_terms]
        
        return {
            'all_effects': all_effects,
            'posterior_probs': dict(zip(all_effects, all_probs)),
            'significant_effects': passing_effects,
            'significant_main_effects': significant_main,
            'significant_interactions': significant_interactions,
            'fdr_threshold': fdr_threshold,
            'cumulative_fdr': dict(zip(sorted_effects, cumulative_fdr))
        }
        
    def compute_effect_sizes(self, results: Dict) -> Dict:
        """
        Calculate standardized effect sizes from model results.
        
        Parameters:
        -----------
        results : Dict
            Dictionary of GLMM results from fit_model()
            
        Returns:
        --------
        Dict
            Dictionary of effect sizes for each feature
        """
        effect_sizes = {}
        
        # Get feature names
        feature_names = [name for name in results['coefficients'].keys() 
                       if name not in ['const', 'group_coded'] and not name.startswith('group_')
                       and not name.startswith('target_lag')]
        
        for feature in feature_names:
            try:
                # Get feature and interaction coefficients
                feature_coef = results['coefficients'].get(feature, 0)
                interaction_term = f'group_{feature}_interaction'
                interaction_coef = results['coefficients'].get(interaction_term, 0)
                
                # Get posterior SDs
                feature_sd = results['posterior_sds'].get(feature, 1.0)
                interaction_sd = results['posterior_sds'].get(interaction_term, 1.0)
                
                # Calculate standardized effect sizes
                standardized_main = feature_coef / feature_sd
                standardized_interaction = interaction_coef / interaction_sd
                
                # Calculate odds ratios
                main_or = np.exp(feature_coef)
                interaction_or = np.exp(interaction_coef)
                
                # Combine into a result dictionary
                effect_sizes[feature] = {
                    'standardized_main_effect': float(standardized_main),
                    'standardized_interaction': float(standardized_interaction),
                    'main_odds_ratio': float(main_or),
                    'interaction_odds_ratio': float(interaction_or),
                    'probabilistic_relevance': float(results['posterior_prob'].get(feature, 0)),
                }
                
                # Add group-specific effects if available
                if feature in results.get('group_specific_effects', {}):
                    effect_sizes[feature]['group_effects'] = results['group_specific_effects'][feature]
                
            except Exception as e:
                print(f"Error calculating effect size for {feature}: {str(e)}")
                effect_sizes[feature] = {'error': str(e)}
        
        return effect_sizes
    
    def plot_group_effects(self, results: Dict, feature_name: str) -> plt.Figure:
        """
        Visualize the difference in effect between groups for a specific feature.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from fit_model()
        feature_name : str
            Name of the feature to visualize
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with group effects plot
        """
        # Check if feature exists in group-specific effects
        if feature_name not in results.get('group_specific_effects', {}):
            print(f"Feature {feature_name} not found in group-specific effects")
            return None
            
        # Extract group-specific effects
        effects = results['group_specific_effects'][feature_name]
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract group names
        groups = [k for k in effects.keys() if k.endswith('_group')]
        
        # Plot odds ratios with confidence intervals
        x = []
        y = []
        yerr = []
        group_labels = []
        
        for i, group in enumerate(groups):
            group_effect = effects[group]
            x.append(i)
            y.append(group_effect['odds_ratio'])
            yerr.append([
                group_effect['odds_ratio'] - group_effect['lower'],
                group_effect['upper'] - group_effect['odds_ratio']
            ])
            group_labels.append(group.replace('_group', ''))
        
        # Create error bars
        ax.errorbar(x, y, yerr=np.array(yerr).T, fmt='o', capsize=5, 
                    elinewidth=2, markersize=10)
        
        # Add reference line at OR=1 (no effect)
        ax.axhline(y=1, color='r', linestyle='-', alpha=0.3, label='No effect (OR=1)')
        
        # Set labels
        ax.set_ylabel('Odds Ratio (log scale)')
        ax.set_title(f'Effect of {feature_name} by Group')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        
        # Use log scale for y-axis (appropriate for odds ratios)
        ax.set_yscale('log')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add probability information
        if 'diff_between_groups' in effects:
            diff_info = effects['diff_between_groups']
            y_pos = 0.95
            for key, val in diff_info.items():
                if key.startswith('prob_stronger'):
                    # Format probability information
                    prob_text = f"{key.replace('prob_stronger_in_', 'P(stronger in ')}: {val:.3f})"
                    ax.text(0.5, y_pos, prob_text, ha='center', transform=ax.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.8))
                    y_pos -= 0.05
        
        plt.tight_layout()
        return fig
    
    def plot_interaction_summary(self, results: Dict, top_n: int = 10) -> plt.Figure:
        """
        Create a summary plot of the strongest interaction effects.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from fit_model()
        top_n : int
            Number of top interactions to display
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with interaction summary plot
        """
        # Extract interaction terms
        interaction_terms = [k for k in results['coefficients'].keys() if k.startswith('group_') and k.endswith('_interaction')]
        
        if not interaction_terms:
            print("No interaction terms found in results")
            return None
            
        # Get posterior probabilities for interactions
        interaction_probs = [results['posterior_prob'].get(term, 0) for term in interaction_terms]
        
        # Sort by probability
        sorted_idx = np.argsort(interaction_probs)[::-1]  # Descending
        
        # Take top N
        if len(sorted_idx) > top_n:
            sorted_idx = sorted_idx[:top_n]
            
        # Get sorted terms and probabilities
        sorted_terms = [interaction_terms[i] for i in sorted_idx]
        sorted_probs = [interaction_probs[i] for i in sorted_idx]
        
        # Clean up term names for display
        display_terms = [term.replace('group_', '').replace('_interaction', '') for term in sorted_terms]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(display_terms))
        ax.barh(y_pos, sorted_probs, align='center')
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_terms)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Posterior Probability of Group Difference')
        ax.set_title('Top Feature × Group Interactions')
        
        # Add threshold line
        ax.axvline(x=0.95, color='r', linestyle='--', label='95% threshold')
        
        # Add probability values
        for i, prob in enumerate(sorted_probs):
            ax.text(prob + 0.01, i, f"{prob:.3f}", va='center')
            
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        return fig