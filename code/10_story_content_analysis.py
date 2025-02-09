import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from typing import Dict, List, Tuple, Optional
import logging
from dotenv import load_dotenv
from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF

# Set to True to use log-likelihood based BIC (recommended for future compatibility)
SET_USE_BIC_LLF(True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BrainStateAnalysis:
    """
    Analysis of brain states focusing on their relationship with story events and content.
    Implements improved statistical methods for binary time series analysis.
    
    This class provides tools for:
    1. Analyzing state probabilities using appropriate distributions
    2. Testing relationships between brain states and story features
    3. Examining temporal dynamics and state transitions
    4. Visualizing results with statistical rigor
    
    The analysis accounts for:
    - Binary nature of state occurrences
    - Temporal dependencies in the data
    - Multiple comparison corrections
    - Effect size estimations
    """
    
    def __init__(self, 
                 base_dir: str,
                 brain_states_affair: np.ndarray,
                 brain_states_paranoia: np.ndarray,
                 matched_states: List[Tuple[int, int]],
                 state_probs_affair: Optional[np.ndarray] = None,
                 state_probs_paranoia: Optional[np.ndarray] = None):
        """
        Initialize the BrainStateAnalysis class.
        
        Parameters
        ----------
        base_dir : str
            Base directory for data and output files
        brain_states_affair : np.ndarray (n_subjects, n_timepoints)
            State sequences for affair context
        brain_states_paranoia : np.ndarray (n_subjects, n_timepoints)
            State sequences for paranoia context
        matched_states : List[Tuple[int, int]]
            List of matched state pairs between conditions
        state_probs_affair : Optional[np.ndarray] (n_subjects, n_timepoints, n_states)
            State probabilities for affair context
        state_probs_paranoia : Optional[np.ndarray] (n_subjects, n_timepoints, n_states)
            State probabilities for paranoia context
        """
        if brain_states_affair.shape != brain_states_paranoia.shape:
            raise ValueError("Brain state arrays must have the same shape")
            
        self.states_affair = brain_states_affair
        self.states_paranoia = brain_states_paranoia
        self.matched_states = matched_states
        self.state_probs_affair = state_probs_affair
        self.state_probs_paranoia = state_probs_paranoia
        
        # Get total number of unique states
        all_states = set()
        for affair_state, paranoia_state in matched_states:
            all_states.add(affair_state)
            all_states.add(paranoia_state)
        self.n_states = len(all_states)
        
        # Set up output directory
        self.output_dir = Path(base_dir) / "output" / "10_story_state_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized analysis with data shape: {brain_states_affair.shape}")
        logger.info(f"Number of states: {self.n_states}")
        logger.info(f"Output directory: {self.output_dir}")

    def analyze_state_probabilities(self, 
                                  probabilities: np.ndarray,
                                  min_val: float = 1e-10) -> Dict:
        """
        Analyze state probabilities using beta distribution.
        
        Parameters
        ----------
        probabilities : np.ndarray
            Array of state probabilities
        min_val : float
            Minimum value for probability clipping
            
        Returns
        -------
        Dict
            Statistical measures including:
            - Beta distribution parameters (a, b)
            - Mean and variance
            - Mode (if exists)
            - Confidence intervals
        """
        # Remove any exact 0 or 1 values for beta fitting
        prob_adj = np.clip(probabilities, min_val, 1-min_val)
        
        # Fit beta distribution
        a, b, loc, scale = stats.beta.fit(prob_adj)
        
        # Calculate statistics
        mean = a/(a+b)
        variance = (a*b)/((a+b)**2 * (a+b+1))
        
        # Calculate confidence intervals
        ci_low, ci_high = stats.beta.interval(0.95, a, b)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'a': a,
            'b': b,
            'mode': (a-1)/(a+b-2) if a > 1 and b > 1 else None,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high
        }

    def analyze_correlations(self, 
                           binary_var: np.ndarray, 
                           continuous_var: np.ndarray) -> Dict:
        """
        Analyze correlations between binary and continuous variables.
        
        Parameters
        ----------
        binary_var : np.ndarray
            Binary variable (e.g., event indicators)
        continuous_var : np.ndarray
            Continuous variable (e.g., state probabilities)
            
        Returns
        -------
        Dict
            Correlation statistics including:
            - Point-biserial correlation
            - P-value
            - Effect size (Cohen's d)
            - Bootstrap confidence intervals
        """
        # Point-biserial correlation
        correlation, pvalue = stats.pointbiserialr(binary_var, continuous_var)
        
        # Calculate effect size (Cohen's d)
        group1 = continuous_var[binary_var == 1]
        group0 = continuous_var[binary_var == 0]
        
        cohens_d = (np.mean(group1) - np.mean(group0)) / np.sqrt(
            ((len(group1) - 1) * np.var(group1) + 
             (len(group0) - 1) * np.var(group0)) / 
            (len(group1) + len(group0) - 2)
        )
        
        # Bootstrap confidence intervals
        n_boot = 1000
        boot_corrs = np.zeros(n_boot)
        for i in range(n_boot):
            idx = np.random.choice(len(binary_var), len(binary_var), replace=True)
            boot_corr, _ = stats.pointbiserialr(
                binary_var[idx], 
                continuous_var[idx]
            )
            boot_corrs[i] = boot_corr
        
        ci_low, ci_high = np.percentile(boot_corrs, [2.5, 97.5])
        
        return {
            'correlation': correlation,
            'pvalue': pvalue,
            'cohens_d': cohens_d,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'n_observations': len(binary_var)
        }

    def test_state_differences(self, 
                         data: pd.DataFrame, 
                         state_idx: int,
                         context: str = 'affair') -> Dict:
        """
        Perform GLM analysis of brain state patterns with continuous predictors.
        """
        # Get appropriate states
        states = self.states_affair if context == 'affair' else self.states_paranoia
        
        # Create binary state indicator and get mode
        state_indicator = (states == state_idx)
        state_present = np.zeros(states.shape[1])
        for t in range(states.shape[1]):
            values, counts = np.unique(state_indicator[:, t], return_counts=True)
            state_present[t] = values[np.argmax(counts)]
        
        state_present = state_present.astype(float)  # Ensure float type for GLM
        
        # Standardize continuous predictors and ensure numeric types
        data_std = data.copy()
        for col in ['n_verbs', 'n_descriptors']:
            data_std[col] = (data[col].astype(float) - data[col].astype(float).mean()) / data[col].astype(float).std()
        
        # Create design matrix ensuring numeric types
        X = pd.DataFrame({
            'intercept': np.ones(len(data)),
            'lee_girl_together': data['lee_girl_together'].astype(float).values,
            'arthur_speaking': data['arthur_speaking'].astype(float).values,
            'n_verbs': data_std['n_verbs'].values,
            'n_descriptors': data_std['n_descriptors'].values,
            'prev_state': np.roll(state_present, 1)
        })
        
        # Add interaction terms
        X['lee_girl_verbs'] = X['lee_girl_together'] * X['n_verbs']
        X['arthur_desc'] = X['arthur_speaking'] * X['n_descriptors']
        
        # Debug print
        print("Data types in X:")
        print(X.dtypes)
        print("\nShape of state_present:", state_present.shape)
        print("Shape of X:", X.shape)
        
        # Convert to numpy arrays for GLM
        X_array = np.asarray(X).astype(float)
        y_array = np.asarray(state_present).astype(float)
        
        # Fit model
        try:
            model = GLM(y_array, X_array, family=Binomial())
            results = model.fit()
            
            # Extract key statistics using bic_llf instead of bic
            stats_dict = {
                'coefficients': results.params,
                'pvalues': results.pvalues,
                'conf_int': results.conf_int(),
                'aic': results.aic,
                'bic': results.bic_llf,  # Using log-likelihood based BIC
                'deviance': results.deviance,
                'pseudo_rsquared': 1 - (results.deviance / results.null_deviance),
                'n_observations': len(data)
            }
            
            # Add odds ratios and their confidence intervals
            odds_ratios = np.exp(results.params)
            odds_ratio_ci = np.exp(results.conf_int())
            
            stats_dict.update({
                'odds_ratios': odds_ratios,
                'odds_ratio_ci': odds_ratio_ci
            })
            
        except Exception as e:
            logger.error(f"GLM fitting failed: {str(e)}")
            stats_dict = None
            
        return stats_dict

    def save_results(self, results: Dict, analysis_name: str) -> None:
        """
        Save analysis results to files.
        
        Parameters
        ----------
        results : Dict
            Analysis results to save
        analysis_name : str
            Name of the analysis for file naming
        """
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results as pickle
        with open(results_dir / f"{analysis_name}_full_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary DataFrames
        summary_data = []
        for pair_key, pair_results in results.items():
            for context in ['affair', 'paranoia']:
                if 'glm_stats' in pair_results[context]:
                    stats = pair_results[context]['glm_stats']
                    if stats is not None:
                        summary = {
                            'state_pair': pair_key,
                            'context': context,
                            'analysis': analysis_name
                        }
                        # Add GLM coefficients and p-values
                        for name, coef in stats['coefficients'].items():
                            summary[f'coef_{name}'] = coef
                            summary[f'pval_{name}'] = stats['pvalues'][name]
                        
                        # Add model fit statistics
                        summary.update({
                            'pseudo_r2': stats['pseudo_rsquared'],
                            'aic': stats['aic'],
                            'bic': stats['bic']
                        })
                        
                        summary_data.append(summary)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(results_dir / f"{analysis_name}_summary.csv", index=False)
            
        logger.info(f"Saved {analysis_name} results to {results_dir}")

    def compute_state_metrics(self, 
                            states: np.ndarray,
                            state_idx: int) -> Dict:
        """
        Compute comprehensive metrics for a given state.
        
        Parameters
        ----------
        states : np.ndarray (n_subjects, n_timepoints)
            State sequences
        state_idx : int
            Index of the state to analyze
            
        Returns
        -------
        Dict
            Metrics including:
            - Occupation probability
            - Mean dwell time
            - Transition probabilities
            - Temporal statistics
        """
        # State occupation
        occupation = np.mean(states == state_idx)
        
        # Dwell time calculation
        dwell_times = []
        for subj in range(states.shape[0]):
            sequence = states[subj]
            in_state = False
            current_dwell = 0
            
            for t in range(len(sequence)):
                if sequence[t] == state_idx:
                    in_state = True
                    current_dwell += 1
                elif in_state:
                    dwell_times.append(current_dwell)
                    in_state = False
                    current_dwell = 0
            
            if in_state:  # Handle end of sequence
                dwell_times.append(current_dwell)
        
        # Transition probability matrix
        trans_mat = np.zeros((self.n_states, self.n_states))
        for subj in range(states.shape[0]):
            sequence = states[subj]
            for t in range(len(sequence)-1):
                trans_mat[sequence[t], sequence[t+1]] += 1
        
        # Normalize transitions
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        trans_probs = np.divide(trans_mat, row_sums, 
                              where=row_sums!=0)
        
        return {
            'occupation': occupation,
            'dwell_times': {
                'mean': np.mean(dwell_times),
                'std': np.std(dwell_times),
                'median': np.median(dwell_times),
                'raw': dwell_times
            },
            'transition_probs': trans_probs,
            'n_transitions': len(dwell_times)
        }

    def analyze_lee_girl_interactions(self, tr_df: pd.DataFrame) -> Dict:
        """
        Analyze brain states during Lee-Girl interactions with comprehensive metrics.
        
        Parameters
        ----------
        tr_df : pd.DataFrame
            TR-level annotations with interaction indicators
            
        Returns
        -------
        Dict
            Complete analysis results including state metrics and GLM analysis
        """
        logger.info("Analyzing Lee-Girl interactions with comprehensive metrics")
        
        results = {}
        
        # Create binary indicator for Lee-Girl interactions
        lee_girl_events = tr_df['lee_girl_together'].values.astype(bool)
        
        for affair_state, paranoia_state in self.matched_states:
            pair_key = f'state_pair_{affair_state}_{paranoia_state}'
            results[pair_key] = {}
            
            # Compute state metrics for both contexts
            affair_metrics = self.compute_state_metrics(
                self.states_affair, affair_state
            )
            paranoia_metrics = self.compute_state_metrics(
                self.states_paranoia, paranoia_state
            )
            
            # Split metrics by event periods
            affair_event_states = self.states_affair[:, lee_girl_events]
            affair_non_event_states = self.states_affair[:, ~lee_girl_events]
            paranoia_event_states = self.states_paranoia[:, lee_girl_events]
            paranoia_non_event_states = self.states_paranoia[:, ~lee_girl_events]
            
            # GLM analysis
            affair_glm = self.test_state_differences(
                tr_df, affair_state, context='affair'
            )
            paranoia_glm = self.test_state_differences(
                tr_df, paranoia_state, context='paranoia'
            )
            
            # Store results for affair context
            results[pair_key]['affair'] = {
                'overall_metrics': affair_metrics,
                'event_metrics': self.compute_state_metrics(
                    affair_event_states, affair_state
                ),
                'non_event_metrics': self.compute_state_metrics(
                    affair_non_event_states, affair_state
                ),
                'glm_stats': affair_glm,
                'state_probs': self.analyze_state_probabilities(
                    self.state_probs_affair[:, :, affair_state].mean(axis=0)
                ),
                'correlations': self.analyze_correlations(
                    lee_girl_events,
                    self.state_probs_affair[:, :, affair_state].mean(axis=0)
                )
            }
            
            # Store results for paranoia context
            results[pair_key]['paranoia'] = {
                'overall_metrics': paranoia_metrics,
                'event_metrics': self.compute_state_metrics(
                    paranoia_event_states, paranoia_state
                ),
                'non_event_metrics': self.compute_state_metrics(
                    paranoia_non_event_states, paranoia_state
                ),
                'glm_stats': paranoia_glm,
                'state_probs': self.analyze_state_probabilities(
                    self.state_probs_paranoia[:, :, paranoia_state].mean(axis=0)
                ),
                'correlations': self.analyze_correlations(
                    lee_girl_events,
                    self.state_probs_paranoia[:, :, paranoia_state].mean(axis=0)
                )
            }
            
        return results

    def analyze_arthur_descriptive(self, tr_df: pd.DataFrame) -> Dict:
        """
        Analyze brain states during Arthur's descriptive speech with comprehensive metrics.
        
        Parameters
        ----------
        tr_df : pd.DataFrame
            TR-level annotations with speech indicators
            
        Returns
        -------
        Dict
            Complete analysis results including state metrics and GLM analysis
        """
        logger.info("Analyzing Arthur's descriptive speech with comprehensive metrics")
        
        results = {}
        
        # Create binary indicator for Arthur's descriptive speech
        arthur_descriptive = (tr_df['arthur_speaking'] & 
                            (tr_df['n_descriptors'] > 0)).values.astype(bool)
        
        for affair_state, paranoia_state in self.matched_states:
            pair_key = f'state_pair_{affair_state}_{paranoia_state}'
            results[pair_key] = {}
            
            # Compute state metrics for both contexts
            affair_metrics = self.compute_state_metrics(
                self.states_affair, affair_state
            )
            paranoia_metrics = self.compute_state_metrics(
                self.states_paranoia, paranoia_state
            )
            
            # Split metrics by event periods
            affair_event_states = self.states_affair[:, arthur_descriptive]
            affair_non_event_states = self.states_affair[:, ~arthur_descriptive]
            paranoia_event_states = self.states_paranoia[:, arthur_descriptive]
            paranoia_non_event_states = self.states_paranoia[:, ~arthur_descriptive]
            
            # GLM analysis
            affair_glm = self.test_state_differences(
                tr_df, affair_state, context='affair'
            )
            paranoia_glm = self.test_state_differences(
                tr_df, paranoia_state, context='paranoia'
            )
            
            # Store results for affair context
            results[pair_key]['affair'] = {
                'overall_metrics': affair_metrics,
                'event_metrics': self.compute_state_metrics(
                    affair_event_states, affair_state
                ),
                'non_event_metrics': self.compute_state_metrics(
                    affair_non_event_states, affair_state
                ),
                'glm_stats': affair_glm,
                'state_probs': self.analyze_state_probabilities(
                    self.state_probs_affair[:, :, affair_state].mean(axis=0)
                ),
                'correlations': self.analyze_correlations(
                    arthur_descriptive,
                    self.state_probs_affair[:, :, affair_state].mean(axis=0)
                )
            }
            
            # Store results for paranoia context
            results[pair_key]['paranoia'] = {
                'overall_metrics': paranoia_metrics,
                'event_metrics': self.compute_state_metrics(
                    paranoia_event_states, paranoia_state
                ),
                'non_event_metrics': self.compute_state_metrics(
                    paranoia_non_event_states, paranoia_state
                ),
                'glm_stats': paranoia_glm,
                'state_probs': self.analyze_state_probabilities(
                    self.state_probs_paranoia[:, :, paranoia_state].mean(axis=0)
                ),
                'correlations': self.analyze_correlations(
                    arthur_descriptive,
                    self.state_probs_paranoia[:, :, paranoia_state].mean(axis=0)
                )
            }
            
        return results

    def analyze_temporal_dynamics(self,
                                states: np.ndarray,
                                events: np.ndarray,
                                state_idx: int,
                                window_size: int = 5) -> Dict:
        """
        Analyze temporal dynamics of state patterns around events.
        
        Parameters
        ----------
        states : np.ndarray (n_subjects, n_timepoints)
            State sequences
        events : np.ndarray (n_timepoints,)
            Binary event indicators
        state_idx : int
            Index of the state to analyze
        window_size : int
            Size of the window around events to analyze
            
        Returns
        -------
        Dict
            Temporal dynamics metrics including:
            - Pre/post event state probabilities
            - State transition patterns around events
            - Event-triggered averages
        """
        # Find event onsets
        event_onsets = np.where(np.diff(events.astype(int)) == 1)[0] + 1
        
        # Initialize containers for temporal patterns
        pre_event_probs = np.zeros((len(event_onsets), window_size))
        post_event_probs = np.zeros((len(event_onsets), window_size))
        
        # Compute probabilities around events
        for i, onset in enumerate(event_onsets):
            if onset >= window_size and onset < len(events) - window_size:
                # Pre-event
                pre_window = states[:, onset-window_size:onset]
                pre_event_probs[i] = np.mean(pre_window == state_idx, axis=0)
                
                # Post-event
                post_window = states[:, onset:onset+window_size]
                post_event_probs[i] = np.mean(post_window == state_idx, axis=0)
        
        return {
            'pre_event': {
                'mean': np.mean(pre_event_probs, axis=0),
                'std': np.std(pre_event_probs, axis=0),
                'raw': pre_event_probs
            },
            'post_event': {
                'mean': np.mean(post_event_probs, axis=0),
                'std': np.std(post_event_probs, axis=0),
                'raw': post_event_probs
            },
            'n_events': len(event_onsets)
        }

    def plot_state_metrics(self, 
                          results: Dict,
                          event_name: str,
                          pair_key: str) -> None:
        """
        Create comprehensive visualization of state metrics.
        
        Parameters
        ----------
        results : Dict
            Analysis results containing state metrics
        event_name : str
            Name of the event being analyzed
        pair_key : str
            Identifier for the state pair
        """
        fig_dir = self.output_dir / "figures" / event_name / "state_metrics"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Occupation Probability Plot
        ax1 = fig.add_subplot(gs[0, 0])
        contexts = ['affair', 'paranoia']
        metrics = ['event_metrics', 'non_event_metrics']
        
        occ_data = {
            context: [results[pair_key][context][m]['occupation'] 
                     for m in metrics]
            for context in contexts
        }
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, occ_data['affair'], width, label='Affair',
                color='blue', alpha=0.6)
        ax1.bar(x + width/2, occ_data['paranoia'], width, label='Paranoia',
                color='red', alpha=0.6)
        
        ax1.set_ylabel('Occupation Probability')
        ax1.set_title('State Occupation During Events vs Non-Events')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Events', 'Non-Events'])
        ax1.legend()
        
        # 2. Dwell Time Distributions
        ax2 = fig.add_subplot(gs[0, 1])
        for context in contexts:
            dwell_times = results[pair_key][context]['overall_metrics']['dwell_times']['raw']
            sns.kdeplot(data=dwell_times, label=context.capitalize(), ax=ax2)
        
        ax2.set_xlabel('Dwell Time (TRs)')
        ax2.set_ylabel('Density')
        ax2.set_title('Dwell Time Distributions')
        
        # 3. Transition Probability Matrices
        ax3 = fig.add_subplot(gs[1, 0])
        trans_probs_affair = results[pair_key]['affair']['overall_metrics']['transition_probs']
        sns.heatmap(trans_probs_affair, ax=ax3, cmap='viridis',
                    annot=True, fmt='.2f')
        ax3.set_title('Transition Probabilities - Affair')
        
        ax4 = fig.add_subplot(gs[1, 1])
        trans_probs_paranoia = results[pair_key]['paranoia']['overall_metrics']['transition_probs']
        sns.heatmap(trans_probs_paranoia, ax=ax4, cmap='viridis',
                    annot=True, fmt='.2f')
        ax4.set_title('Transition Probabilities - Paranoia')
        
        # 4. GLM Results
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_glm_results(results[pair_key], ax5)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{pair_key}_state_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_glm_results(self, 
                         pair_results: Dict, 
                         ax: plt.Axes) -> None:
        """
        Plot GLM results with confidence intervals.
        
        Parameters
        ----------
        pair_results : Dict
            Results for a specific state pair
        ax : plt.Axes
            Matplotlib axes for plotting
        """
        contexts = ['affair', 'paranoia']
        coef_names = ['lee_girl_together', 'arthur_speaking', 'n_verbs', 
                     'n_descriptors', 'lee_girl_verbs', 'arthur_desc']
        
        x = np.arange(len(coef_names))
        width = 0.35
        
        for i, context in enumerate(contexts):
            glm_stats = pair_results[context]['glm_stats']
            if glm_stats is not None:
                coefs = [glm_stats['coefficients'][name] for name in coef_names]
                ci_low = [glm_stats['conf_int'][name][0] for name in coef_names]
                ci_high = [glm_stats['conf_int'][name][1] for name in coef_names]
                
                ax.bar(x + i*width - width/2, coefs, width,
                      label=context.capitalize(),
                      color='blue' if context == 'affair' else 'red',
                      alpha=0.6)
                
                ax.errorbar(x + i*width - width/2, coefs,
                          yerr=[np.array(coefs) - np.array(ci_low),
                                np.array(ci_high) - np.array(coefs)],
                          fmt='none', color='black', capsize=5)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Coefficient Value')
        ax.set_title('GLM Coefficients with 95% CIs')
        ax.set_xticks(x)
        ax.set_xticklabels(coef_names, rotation=45)
        ax.legend()

    def plot_temporal_dynamics(self,
                             results: Dict,
                             event_name: str,
                             pair_key: str) -> None:
        """
        Create visualization of temporal dynamics around events.
        
        Parameters
        ----------
        results : Dict
            Analysis results containing temporal dynamics
        event_name : str
            Name of the event being analyzed
        pair_key : str
            Identifier for the state pair
        """
        fig_dir = self.output_dir / "figures" / event_name / "temporal"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Time points relative to event
        pre_time = -np.arange(results[pair_key]['affair']['pre_event']['mean'].shape[0])[::-1]
        post_time = np.arange(results[pair_key]['affair']['post_event']['mean'].shape[0])
        
        contexts = ['affair', 'paranoia']
        colors = {'affair': 'blue', 'paranoia': 'red'}
        
        for context in contexts:
            # Pre-event
            mean_pre = results[pair_key][context]['pre_event']['mean']
            std_pre = results[pair_key][context]['pre_event']['std']
            
            # Post-event
            mean_post = results[pair_key][context]['post_event']['mean']
            std_post = results[pair_key][context]['post_event']['std']
            
            # Plot means
            ax1.plot(pre_time, mean_pre, color=colors[context], 
                    label=f'{context.capitalize()} Pre-event')
            ax1.plot(post_time, mean_post, color=colors[context], 
                    linestyle='--', label=f'{context.capitalize()} Post-event')
            
            # Add confidence bands
            ax1.fill_between(pre_time, mean_pre - std_pre, mean_pre + std_pre,
                           color=colors[context], alpha=0.2)
            ax1.fill_between(post_time, mean_post - std_post, mean_post + std_post,
                           color=colors[context], alpha=0.2)
        
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time relative to event (TRs)')
        ax1.set_ylabel('State Probability')
        ax1.set_title('State Dynamics Around Events')
        ax1.legend()
        
        # Plot state distributions before and after events
        for context in contexts:
            pre_dist = results[pair_key][context]['pre_event']['raw'].flatten()
            post_dist = results[pair_key][context]['post_event']['raw'].flatten()
            
            sns.kdeplot(data=pre_dist, ax=ax2, color=colors[context],
                       label=f'{context.capitalize()} Pre-event')
            sns.kdeplot(data=post_dist, ax=ax2, color=colors[context],
                       linestyle='--', label=f'{context.capitalize()} Post-event')
        
        ax2.set_xlabel('State Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of State Probabilities')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{pair_key}_temporal_dynamics.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_visualizations(self,
                                    results: Dict,
                                    event_name: str) -> None:
        """
        Create comprehensive summary visualizations for all state pairs.
        
        Parameters
        ----------
        results : Dict
            Complete analysis results
        event_name : str
            Name of the event being analyzed
        """
        for pair_key in results.keys():
            # Plot state metrics
            self.plot_state_metrics(results, event_name, pair_key)
            
            # Plot temporal dynamics
            self.plot_temporal_dynamics(results, event_name, pair_key)
            
            # Plot additional specific visualizations
            self._plot_specialized_metrics(results, event_name, pair_key)
    
    def _plot_specialized_metrics(self,
                                results: Dict,
                                event_name: str,
                                pair_key: str) -> None:
        """
        Create specialized visualizations based on event type.
        
        Parameters
        ----------
        results : Dict
            Analysis results
        event_name : str
            Name of the event being analyzed
        pair_key : str
            Identifier for the state pair
        """
        fig_dir = self.output_dir / "figures" / event_name / "specialized"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        if event_name == "lee_girl":
            # Create Lee-Girl specific visualizations
            self._plot_verb_relationships(results[pair_key], fig_dir / f'{pair_key}_verb_analysis.png')
        
        elif event_name == "arthur_descriptive":
            # Create Arthur's speech specific visualizations
            self._plot_descriptor_relationships(results[pair_key], fig_dir / f'{pair_key}_descriptor_analysis.png')
    
    def _plot_verb_relationships(self, pair_results: Dict, filename: Path) -> None:
        """
        Plot verb-specific relationships for Lee-Girl interactions.
        
        Parameters
        ----------
        pair_results : Dict
            Results for a specific state pair
        filename : Path
            Output file path
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 3)
        
        # 1. Verb Count Distribution by Context
        ax1 = fig.add_subplot(gs[0, 0])
        contexts = ['affair', 'paranoia']
        colors = {'affair': 'blue', 'paranoia': 'red'}
        
        for context in contexts:
            verb_coef = pair_results[context]['glm_stats']['coefficients']['n_verbs']
            verb_ci = pair_results[context]['glm_stats']['conf_int']['n_verbs']
            
            ax1.bar([context], [verb_coef], 
                   yerr=[[verb_coef - verb_ci[0]], [verb_ci[1] - verb_coef]],
                   color=colors[context], alpha=0.6,
                   capsize=5)
        
        ax1.set_ylabel('Verb Count Coefficient')
        ax1.set_title('Effect of Verb Count on State Probability')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 2. Interaction Effect
        ax2 = fig.add_subplot(gs[0, 1])
        for context in contexts:
            interaction_coef = pair_results[context]['glm_stats']['coefficients']['lee_girl_verbs']
            interaction_ci = pair_results[context]['glm_stats']['conf_int']['lee_girl_verbs']
            
            ax2.bar([context], [interaction_coef],
                    yerr=[[interaction_coef - interaction_ci[0]], 
                          [interaction_ci[1] - interaction_coef]],
                    color=colors[context], alpha=0.6,
                    capsize=5)
        
        ax2.set_ylabel('Interaction Coefficient')
        ax2.set_title('Lee-Girl × Verb Count Interaction')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 3. State Probability by Verb Count
        ax3 = fig.add_subplot(gs[0, 2])
        for context in contexts:
            corr_stats = pair_results[context]['correlations']
            ax3.scatter([], [], color=colors[context], alpha=0.6,
                       label=f"{context.capitalize()}\nr={corr_stats['correlation']:.3f}")
        
        ax3.set_xlabel('Verb Count')
        ax3.set_ylabel('State Probability')
        ax3.set_title('State Probability vs Verb Count')
        ax3.legend()
        
        # 4. Temporal Patterns During High-Verb Periods
        ax4 = fig.add_subplot(gs[1, :])
        time_window = np.arange(-5, 6)  # -5 to +5 TRs around events
        
        for context in contexts:
            # Get temporal dynamics around high-verb events
            temporal = pair_results[context]['temporal']
            if 'high_verb_dynamics' in temporal:
                mean_prob = temporal['high_verb_dynamics']['mean']
                std_prob = temporal['high_verb_dynamics']['std']
                
                ax4.plot(time_window, mean_prob, 
                        color=colors[context],
                        label=context.capitalize())
                ax4.fill_between(time_window,
                               mean_prob - std_prob,
                               mean_prob + std_prob,
                               color=colors[context], alpha=0.2)
        
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Time Relative to High-Verb Events (TRs)')
        ax4.set_ylabel('State Probability')
        ax4.set_title('Temporal Dynamics Around High-Verb Events')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_descriptor_relationships(self, pair_results: Dict, filename: Path) -> None:
        """
        Plot descriptor-specific relationships for Arthur's speech.
        
        Parameters
        ----------
        pair_results : Dict
            Results for a specific state pair
        filename : Path
            Output file path
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 3)
        
        # 1. Descriptor Effect by Context
        ax1 = fig.add_subplot(gs[0, 0])
        contexts = ['affair', 'paranoia']
        colors = {'affair': 'blue', 'paranoia': 'red'}
        
        for context in contexts:
            desc_coef = pair_results[context]['glm_stats']['coefficients']['n_descriptors']
            desc_ci = pair_results[context]['glm_stats']['conf_int']['n_descriptors']
            
            ax1.bar([context], [desc_coef],
                   yerr=[[desc_coef - desc_ci[0]], [desc_ci[1] - desc_coef]],
                   color=colors[context], alpha=0.6,
                   capsize=5)
        
        ax1.set_ylabel('Descriptor Count Coefficient')
        ax1.set_title('Effect of Descriptor Count on State Probability')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 2. Arthur Speech × Descriptor Interaction
        ax2 = fig.add_subplot(gs[0, 1])
        for context in contexts:
            interaction_coef = pair_results[context]['glm_stats']['coefficients']['arthur_desc']
            interaction_ci = pair_results[context]['glm_stats']['conf_int']['arthur_desc']
            
            ax2.bar([context], [interaction_coef],
                    yerr=[[interaction_coef - interaction_ci[0]], 
                          [interaction_ci[1] - interaction_coef]],
                    color=colors[context], alpha=0.6,
                    capsize=5)
        
        ax2.set_ylabel('Interaction Coefficient')
        ax2.set_title('Arthur Speech × Descriptor Count Interaction')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 3. Descriptor Type Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        for context in contexts:
            if 'descriptor_types' in pair_results[context]:
                types = pair_results[context]['descriptor_types']
                ax3.bar(np.arange(len(types)) + (0.4 if context == 'paranoia' else 0),
                       list(types.values()), width=0.4,
                       label=context.capitalize(), alpha=0.6,
                       color=colors[context])
        
        ax3.set_xlabel('Descriptor Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Descriptor Types')
        ax3.legend()
        
        # 4. Temporal Patterns During Descriptive Speech
        ax4 = fig.add_subplot(gs[1, :])
        time_window = np.arange(-5, 6)  # -5 to +5 TRs around events
        
        for context in contexts:
            # Get temporal dynamics around descriptive speech events
            temporal = pair_results[context]['temporal']
            if 'descriptive_dynamics' in temporal:
                mean_prob = temporal['descriptive_dynamics']['mean']
                std_prob = temporal['descriptive_dynamics']['std']
                
                ax4.plot(time_window, mean_prob,
                        color=colors[context],
                        label=context.capitalize())
                ax4.fill_between(time_window,
                               mean_prob - std_prob,
                               mean_prob + std_prob,
                               color=colors[context], alpha=0.2)
        
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Time Relative to Descriptive Speech Events (TRs)')
        ax4.set_ylabel('State Probability')
        ax4.set_title('Temporal Dynamics Around Descriptive Speech')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main execution function for story state analysis
    Implements the complete analysis pipeline with improved statistical methods
    """
    # Setup
    load_dotenv()
    base_dir = os.getenv('SCRATCH_DIR')
    if base_dir is None:
        raise ValueError("SCRATCH_DIR environment variable not set")
    base_dir = Path(base_dir)
    
    try:
        # 1. Load Data
        logger.info("Loading brain state data...")
        # Load state sequences
        affair_states = np.load(
            base_dir / "output" / "affair_hmm_3states_ntw_native_trimmed" / 
            "statistics" / "affair_state_sequences.npy"
        )
        paranoia_states = np.load(
            base_dir / "output" / "paranoia_hmm_3states_ntw_native_trimmed" / 
            "statistics" / "paranoia_state_sequences.npy"
        )
        
        # Load state probabilities
        try:
            logger.info("Loading state probabilities...")
            affair_probs = np.load(
                base_dir / "output" / "affair_hmm_3states_ntw_native_trimmed" / 
                "statistics" / "state_probabilities.npy"
            )
            paranoia_probs = np.load(
                base_dir / "output" / "paranoia_hmm_3states_ntw_native_trimmed" / 
                "statistics" / "state_probabilities.npy"
            )
            logger.info("Successfully loaded state probabilities")
        except FileNotFoundError:
            logger.warning("State probabilities not found. Will compute from state sequences.")
            affair_probs = None
            paranoia_probs = None
        
        # Load state matching information
        logger.info("Loading state matching data...")
        with open(base_dir / "output" / "09_group_HMM_comparison" / "comparisons.pkl", 'rb') as f:
            comparisons = pickle.load(f)
        matched_states = comparisons['state_similarity']['matched_states']
        
        # 2. Load and Process Annotations
        logger.info("Loading story annotations...")
        story_df = pd.read_csv(base_dir / 'data' / 'stimuli' / '10_story_annotations.csv')
        
        # Get TR-level annotations
        n_brain_trs = affair_states.shape[1]
        logger.info(f"Brain state data has {n_brain_trs} TRs")
        
        # Create TR-level annotations
        tr_df = pd.DataFrame()
        for tr in range(n_brain_trs):
            # Find words that occur during this TR
            tr_words = story_df[
                (story_df['onset_TR'] >= tr) & 
                (story_df['onset_TR'] < tr + 1)
            ]
            
            tr_data = {
                'TR': tr,
                'lee_girl_together': any(tr_words['lee_girl_together']),
                'arthur_speaking': any(tr_words['arthur_speaking']),
                'n_verbs': len(tr_words[tr_words['is_verb']]),
                'n_descriptors': len(tr_words[tr_words['is_adj'] | tr_words['is_adv']])
            }
            tr_df = pd.concat([tr_df, pd.DataFrame([tr_data])], ignore_index=True)
        
        # 3. Initialize Analysis
        analysis = BrainStateAnalysis(
            base_dir=str(base_dir),
            brain_states_affair=affair_states,
            brain_states_paranoia=paranoia_states,
            matched_states=matched_states,
            state_probs_affair=affair_probs,
            state_probs_paranoia=paranoia_probs
        )
        
        # 4. Run Analyses
        
        # 4.1 Lee-Girl Interaction Analysis
        logger.info("\nAnalyzing Lee-Girl interactions...")
        lee_girl_results = analysis.analyze_lee_girl_interactions(tr_df)
        
        # Create visualizations
        logger.info("Creating Lee-Girl visualizations...")
        analysis.create_summary_visualizations(lee_girl_results, "lee_girl")
        
        # Save results
        logger.info("Saving Lee-Girl results...")
        analysis.save_results(lee_girl_results, "lee_girl")
        
        # 4.2 Arthur's Descriptive Speech Analysis
        logger.info("\nAnalyzing Arthur's descriptive speech...")
        arthur_results = analysis.analyze_arthur_descriptive(tr_df)
        
        # Create visualizations
        logger.info("Creating Arthur speech visualizations...")
        analysis.create_summary_visualizations(arthur_results, "arthur_descriptive")
        
        # Save results
        logger.info("Saving Arthur speech results...")
        analysis.save_results(arthur_results, "arthur_descriptive")
        
        # 5. Print Summary Statistics
        print("\nAnalysis Summary:")
        
        # 5.1 Lee-Girl Analysis Summary
        print("\nLee-Girl Interactions Summary:")
        for pair_key, pair_results in lee_girl_results.items():
            print(f"\nState Pair: {pair_key}")
            
            for context in ['affair', 'paranoia']:
                print(f"\n{context.capitalize()} Context:")
                
                # GLM Results
                if pair_results[context]['glm_stats']:
                    glm = pair_results[context]['glm_stats']
                    print("\nGLM Coefficients:")
                    for name, coef in glm['coefficients'].items():
                        pval = glm['pvalues'][name]
                        print(f"  {name}: {coef:.3f} (p={pval:.3f})")
                    
                    print(f"\nModel Fit:")
                    print(f"  Pseudo R²: {glm['pseudo_rsquared']:.3f}")
                    print(f"  AIC: {glm['aic']:.1f}")
                
                # State Metrics
                metrics = pair_results[context]['overall_metrics']
                print("\nState Metrics:")
                print(f"  Occupation: {metrics['occupation']:.3f}")
                print(f"  Mean Dwell Time: {metrics['dwell_times']['mean']:.2f} TRs")
        
        # 5.2 Arthur Analysis Summary
        print("\nArthur's Descriptive Speech Summary:")
        for pair_key, pair_results in arthur_results.items():
            print(f"\nState Pair: {pair_key}")
            
            for context in ['affair', 'paranoia']:
                print(f"\n{context.capitalize()} Context:")
                
                # GLM Results
                if pair_results[context]['glm_stats']:
                    glm = pair_results[context]['glm_stats']
                    print("\nGLM Coefficients:")
                    for name, coef in glm['coefficients'].items():
                        pval = glm['pvalues'][name]
                        print(f"  {name}: {coef:.3f} (p={pval:.3f})")
                    
                    print(f"\nModel Fit:")
                    print(f"  Pseudo R²: {glm['pseudo_rsquared']:.3f}")
                    print(f"  AIC: {glm['aic']:.1f}")
                
                # State Metrics
                metrics = pair_results[context]['overall_metrics']
                print("\nState Metrics:")
                print(f"  Occupation: {metrics['occupation']:.3f}")
                print(f"  Mean Dwell Time: {metrics['dwell_times']['mean']:.2f} TRs")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()