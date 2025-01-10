import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats
from scipy.stats import entropy
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import logging
import seaborn as sns
from sklearn.metrics import mutual_info_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_timing_alignment(story_df: pd.DataFrame, n_brain_trs: int) -> None:
    """
    Check alignment between story timing and brain data timing.
    Raises warning if there's a large mismatch.
    
    Parameters:
    -----------
    story_df : pd.DataFrame
        Story annotations with onset_TR
    n_brain_trs : int
        Number of TRs in brain data
    """
    story_max_tr = int(np.ceil(story_df['onset_TR'].max() - story_df['onset_TR'].min()))
    tr_difference = abs(story_max_tr - n_brain_trs)
    
    logger.info(f"Story max TR: {story_max_tr}")
    logger.info(f"Brain data TRs: {n_brain_trs}")
    logger.info(f"Difference: {tr_difference} TRs")
    
    if tr_difference > 10:  # You can adjust this threshold
        logger.warning(
            f"Large timing mismatch detected: {tr_difference} TRs difference between "
            f"story ({story_max_tr} TRs) and brain data ({n_brain_trs} TRs). "
            "This might indicate a problem with the data alignment."
        )

def create_tr_annotations(story_df: pd.DataFrame, n_brain_trs: int, 
                         force_trim: bool = False) -> pd.DataFrame:
    """
    Create TR-level annotations from word-level story annotations using majority vote
    
    Parameters:
    -----------
    story_df : pd.DataFrame
        Word-level story annotations with onset_TR and features
        
    Returns:
    --------
    pd.DataFrame
        TR-level annotations with aggregated features
    """
    # Get the offset from first TR
    tr_offset = story_df['onset_TR'].min()
    
    # Adjust story_max_tr calculation
    story_max_tr = int(np.ceil(story_df['onset_TR'].max() - tr_offset))
    tr_difference = abs(story_max_tr - n_brain_trs)
    
    if tr_difference > 10 and not force_trim:  # Threshold can be adjusted
        raise ValueError(
            f"Large timing mismatch detected ({tr_difference} TRs). "
            "Set force_trim=True to proceed anyway."
        )
    
    # Create annotations up to brain data length
    max_TR = n_brain_trs - 1  # -1 because 0-indexed
    tr_annotations = []

    for tr in range(max_TR + 1):
        # Find words that occur during this TR, accounting for offset
        tr_words = story_df[(story_df['onset_TR'] - tr_offset >= tr) & 
                           (story_df['onset_TR'] - tr_offset < tr + 1)]
        
        # Create TR-level features using majority vote
        tr_data = {
            'TR': tr,
            'speaker': tr_words['speaker'].mode().iloc[0] if len(tr_words) > 0 else 'none',
            'is_dialog': tr_words['is_dialog'].mean() > 0.5 if len(tr_words) > 0 else False,
            'arthur_speaking': tr_words['arthur_speaking'].mean() > 0.5 if len(tr_words) > 0 else False,
            'lee_speaking': tr_words['lee_speaking'].mean() > 0.5 if len(tr_words) > 0 else False,
            'girl_speaking': tr_words['girl_speaking'].mean() > 0.5 if len(tr_words) > 0 else False,
            'lee_girl_together': tr_words['lee_girl_together'].mean() > 0.5 if len(tr_words) > 0 else False,
            
            # Store word lists
            'words': tr_words['text'].tolist(),
            'verbs': tr_words[tr_words['is_verb']]['text'].tolist(),
            'nouns': tr_words[tr_words['is_noun']]['text'].tolist(),
            'descriptors': tr_words[tr_words['is_adj'] | tr_words['is_adv']]['text'].tolist(),
            
            # Count features
            'n_words': len(tr_words),
            'n_verbs': len(tr_words[tr_words['is_verb']]),
            'n_nouns': len(tr_words[tr_words['is_noun']]),
            'n_descriptors': len(tr_words[tr_words['is_adj'] | tr_words['is_adv']]),
            
            # Segment information
            'segment_onset': tr_words['segment_onset'].iloc[0] if len(tr_words) > 0 else None,
            'segment_text': tr_words['segment_text'].iloc[0] if len(tr_words) > 0 else None
        }
        tr_annotations.append(tr_data)
    
    return pd.DataFrame(tr_annotations)

def get_or_create_tr_annotations(story_df: pd.DataFrame, n_brain_trs: int, base_dir: Path) -> pd.DataFrame:
    """
    Get existing TR annotations or create new ones if they don't exist
    
    Parameters:
    -----------
    story_df : pd.DataFrame
        Word-level story annotations
    n_brain_trs : int
        Number of TRs in brain data
    base_dir : Path
        Base directory for data files
        
    Returns:
    --------
    pd.DataFrame
        TR-level annotations
    """
    tr_file = base_dir / 'data' / 'stimuli' / '10_tr_level_annotations.csv'
    
    if tr_file.exists():
        logger.info(f"Loading existing TR-level annotations from {tr_file}")
        tr_df = pd.read_csv(tr_file)
        
        # Verify the loaded annotations match the current data
        if len(tr_df) != n_brain_trs:
            logger.warning(
                f"Existing TR annotations ({len(tr_df)} TRs) don't match "
                f"current brain data ({n_brain_trs} TRs). Creating new annotations."
            )
            tr_df = create_tr_annotations(story_df, n_brain_trs)
            tr_df.to_csv(tr_file, index=False)
    else:
        logger.info("Creating new TR-level annotations")
        tr_df = create_tr_annotations(story_df, n_brain_trs)
        tr_df.to_csv(tr_file, index=False)
        logger.info(f"Saved TR-level annotations to {tr_file}")
    
    return tr_df

class BrainStateAnalysis:
    """
    Analysis of brain states focusing on their relationship with story events and content.
    Uses direct time series analysis without windowing.
    """
    def __init__(self, 
             base_dir: str,
             brain_states_affair: np.ndarray,
             brain_states_paranoia: np.ndarray,
             matched_states: List[Tuple[int, int]],
             state_probs_affair: Optional[np.ndarray] = None,
             state_probs_paranoia: Optional[np.ndarray] = None):
        """
        Initialize the BrainStateAnalysis class
        """
        if brain_states_affair.shape != brain_states_paranoia.shape:
            raise ValueError("Brain state arrays must have the same shape")
            
        self.states_affair = brain_states_affair
        self.states_paranoia = brain_states_paranoia
        self.matched_states = matched_states
        self.state_probs_affair = state_probs_affair
        self.state_probs_paranoia = state_probs_paranoia
        
        # Fix n_states calculation to include all unique states
        all_states = set()
        for affair_state, paranoia_state in matched_states:
            all_states.add(affair_state)
            all_states.add(paranoia_state)
        self.n_states = len(all_states)
        
        self.output_dir = Path(base_dir) / "output" / "10_story_state_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized analysis with data shape: {brain_states_affair.shape}")
        logger.info(f"Number of states: {self.n_states}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _compute_state_occupancy(self, 
                               states: np.ndarray,
                               binary_events: np.ndarray) -> np.ndarray:
        """
        Compute state occupancy during events vs. non-events
        
        Parameters:
        -----------
        states : np.ndarray (n_subjects, n_timepoints)
            State sequences
        binary_events : np.ndarray (n_timepoints,)
            Binary event indicator
            
        Returns:
        --------
        np.ndarray (n_states, 2)
            State occupancy during events and non-events
        """
        occupancy = np.zeros((self.n_states, 2))  # [state, [non-event, event]]
        
        for state in range(self.n_states):
            # Compute occupancy during non-events
            occupancy[state, 0] = np.mean(
                states[:, binary_events == 0] == state
            )
            # Compute occupancy during events
            occupancy[state, 1] = np.mean(
                states[:, binary_events == 1] == state
            )
        
        return occupancy

    def compute_state_metrics(self,
                          states: np.ndarray,
                          state_probs: Optional[np.ndarray],
                          binary_events: np.ndarray,
                          state_idx: int) -> Dict:
        """
        Compute comprehensive metrics for a given state during events vs non-events
        
        Parameters:
        -----------
        states : np.ndarray (n_subjects, n_timepoints)
            State sequences
        state_probs : np.ndarray or None (n_subjects, n_timepoints, n_states)
            State probabilities if available
        binary_events : np.ndarray (n_timepoints,)
            Binary event indicator
        state_idx : int
            Index of the state to analyze
            
        Returns:
        --------
        Dict
            Dictionary of computed metrics
        """
        # Basic occupancy
        event_states = states[:, binary_events == 1]
        non_event_states = states[:, binary_events == 0]
        
        occupancy_event = np.mean(event_states == state_idx)
        occupancy_non_event = np.mean(non_event_states == state_idx)
        
        results = {
            'occupancy_event': occupancy_event,
            'occupancy_non_event': occupancy_non_event,
            'occupancy_ratio': occupancy_event / occupancy_non_event if occupancy_non_event > 0 else np.inf
        }
        
        # State probability metrics (if available)
        if state_probs is not None:
            event_probs = state_probs[:, binary_events == 1, state_idx]
            non_event_probs = state_probs[:, binary_events == 0, state_idx]
            
            # Average probability
            results.update({
                'prob_event_mean': np.mean(event_probs),
                'prob_event_std': np.std(event_probs),
                'prob_non_event_mean': np.mean(non_event_probs),
                'prob_non_event_std': np.std(non_event_probs)
            })
            
            # Statistical comparison of probabilities
            tstat, pval = stats.ttest_ind(
                event_probs.flatten(),
                non_event_probs.flatten()
            )
            results.update({
                'prob_tstat': tstat,
                'prob_pval': pval
            })
        
        return results
    
    def analyze_state_transitions(self,
                            states: np.ndarray,
                            binary_events: np.ndarray) -> Dict:
        """
        Analyze state transition patterns during events vs non-events
        Using group-level state probabilities
        
        Parameters:
        -----------
        states : np.ndarray (n_subjects, n_timepoints)
            State sequences
        binary_events : np.ndarray (n_timepoints,)
            Binary event indicator
        
        Returns:
        --------
        Dict
            Transition matrices and statistics
        """
        # Convert to one-hot encoding and get group probabilities
        one_hot_states = np.eye(self.n_states)[states]  # (n_subjects, n_timepoints, n_states)
        state_probs = one_hot_states.mean(axis=0)  # (n_timepoints, n_states)
        
        # Initialize transition matrices
        n_timepoints = len(binary_events)
        trans_event = np.zeros((self.n_states, self.n_states))
        trans_non_event = np.zeros((self.n_states, self.n_states))
        
        # Count weighted transitions
        for t in range(n_timepoints - 1):
            curr_probs = state_probs[t]
            next_probs = state_probs[t + 1]
            
            # Outer product gives transition probabilities
            trans = np.outer(curr_probs, next_probs)
            
            if binary_events[t]:
                trans_event += trans
            else:
                trans_non_event += trans
        
        # Normalize
        trans_event /= np.maximum(trans_event.sum(axis=1, keepdims=True), 1e-10)
        trans_non_event /= np.maximum(trans_non_event.sum(axis=1, keepdims=True), 1e-10)
        
        # Compute entropy
        entropy_event = np.array([entropy(row) for row in trans_event])
        entropy_non_event = np.array([entropy(row) for row in trans_non_event])
        
        return {
            'transition_matrix_event': trans_event,
            'transition_matrix_non_event': trans_non_event,
            'transition_entropy_event': entropy_event,
            'transition_entropy_non_event': entropy_non_event,
            'entropy_difference': entropy_event - entropy_non_event,
            'state_probabilities': state_probs
        }
    def analyze_temporal_relationships(self,
                                    states: np.ndarray,
                                    binary_events: np.ndarray,
                                    state_idx: int,
                                    max_lag: int = 5) -> Dict:
        """
        Analyze temporal relationships between states and events
        
        Parameters:
        -----------
        states : np.ndarray (n_subjects, n_timepoints)
            State sequences
        binary_events : np.ndarray (n_timepoints,)
            Binary event indicator
        state_idx : int
            Index of the state to analyze
        max_lag : int
            Maximum number of timepoints to lag
            
        Returns:
        --------
        Dict
            Cross-correlation and mutual information at different lags
        """
        results = {'lags': [], 'correlations': [], 'mutual_info': []}
        
        state_indicator = (states == state_idx).mean(axis=0)  # Average across subjects
        
        for lag in range(-max_lag, max_lag + 1):
            results['lags'].append(lag)
            
            # Compute lagged arrays
            if lag < 0:
                s1 = state_indicator[-lag:]
                s2 = binary_events[:lag]
            else:
                s1 = state_indicator[:-lag] if lag > 0 else state_indicator
                s2 = binary_events[lag:] if lag > 0 else binary_events
            
            # Cross-correlation
            correlation = np.corrcoef(s1, s2)[0, 1]
            results['correlations'].append(correlation)
            
            # Mutual information
            # Discretize continuous values for MI calculation
            s1_disc = np.digitize(s1, bins=np.linspace(0, 1, 10))
            mi = mutual_info_score(s1_disc, s2)
            results['mutual_info'].append(mi)
        
        # Convert to numpy arrays
        results['lags'] = np.array(results['lags'])
        results['correlations'] = np.array(results['correlations'])
        results['mutual_info'] = np.array(results['mutual_info'])
        
        # Find optimal lags
        results['max_correlation_lag'] = results['lags'][np.argmax(np.abs(results['correlations']))]
        results['max_mi_lag'] = results['lags'][np.argmax(results['mutual_info'])]
        
        return results

    def analyze_lee_girl_interactions(self, tr_df: pd.DataFrame) -> Dict:
        """
        Analyze brain states during Lee-Girl interactions with focus on verbs
        
        Parameters:
        -----------
        tr_df : pd.DataFrame
            TR-level annotations with interaction indicators
                
        Returns:
        --------
        Dict
            Complete analysis results for Lee-Girl interactions
        """
        logger.info("Analyzing Lee-Girl interactions")
        
        # Create binary event series for Lee-Girl interactions
        lee_girl_events = tr_df['lee_girl_together'].values.astype(bool)
        
        # Fix verb processing
        n_verbs = tr_df['n_verbs'].values
        verb_intensity = np.zeros_like(lee_girl_events, dtype=float)
        verb_intensity[lee_girl_events] = n_verbs[lee_girl_events]
        
        # Fix verb list processing
        interaction_verbs = []
        for verbs in tr_df[lee_girl_events]['verbs']:
            if isinstance(verbs, str):  # If stored as string, evaluate it
                try:
                    verb_list = eval(verbs)
                    if isinstance(verb_list, list):
                        interaction_verbs.extend(verb_list)
                except:
                    continue
            elif isinstance(verbs, list):
                interaction_verbs.extend(verbs)
        
        logger.info(f"Found {np.sum(lee_girl_events)} TRs with Lee-Girl interactions")
        logger.info(f"Total verbs during interactions: {len(interaction_verbs)}")
        logger.info(f"Mean verbs per interaction TR: {np.mean(n_verbs[lee_girl_events]):.2f}")
        
        results = {}
        
        # Convert states to one-hot encoding for both groups
        one_hot_affair = np.eye(self.n_states)[self.states_affair]
        one_hot_paranoia = np.eye(self.n_states)[self.states_paranoia]
        
        # Get group-level probabilities
        affair_probs = one_hot_affair.mean(axis=0)
        paranoia_probs = one_hot_paranoia.mean(axis=0)
        
        # Analyze each matched state pair
        for affair_state, paranoia_state in self.matched_states:
            pair_key = f'state_pair_{affair_state}_{paranoia_state}'
            results[pair_key] = {}
            
            # Separate high-verb and low-verb interactions
            median_verbs = np.median(n_verbs[lee_girl_events])
            high_verb_mask = lee_girl_events & (n_verbs > median_verbs)
            low_verb_mask = lee_girl_events & (n_verbs <= median_verbs)
            non_event_mask = ~lee_girl_events
            
            # Get state probabilities for different conditions
            affair_high_verb = affair_probs[high_verb_mask, affair_state]
            affair_low_verb = affair_probs[low_verb_mask, affair_state]
            affair_non_event = affair_probs[non_event_mask, affair_state]
            
            paranoia_high_verb = paranoia_probs[high_verb_mask, paranoia_state]
            paranoia_low_verb = paranoia_probs[low_verb_mask, paranoia_state]
            paranoia_non_event = paranoia_probs[non_event_mask, paranoia_state]
            
            # Statistical comparison
            # ANOVA for three conditions (high-verb, low-verb, non-event)
            affair_f, affair_p = stats.f_oneway(
                affair_high_verb, affair_low_verb, affair_non_event
            )
            paranoia_f, paranoia_p = stats.f_oneway(
                paranoia_high_verb, paranoia_low_verb, paranoia_non_event
            )
            
            # Store basic probability metrics
            results[pair_key]['affair'] = {
                'mean_prob_high_verb': np.mean(affair_high_verb),
                'mean_prob_low_verb': np.mean(affair_low_verb),
                'mean_prob_non_event': np.mean(affair_non_event),
                'std_prob_high_verb': np.std(affair_high_verb),
                'std_prob_low_verb': np.std(affair_low_verb),
                'std_prob_non_event': np.std(affair_non_event),
                'f_stat': affair_f,
                'p_value': affair_p
            }
            
            results[pair_key]['paranoia'] = {
                'mean_prob_high_verb': np.mean(paranoia_high_verb),
                'mean_prob_low_verb': np.mean(paranoia_low_verb),
                'mean_prob_non_event': np.mean(paranoia_non_event),
                'std_prob_high_verb': np.std(paranoia_high_verb),
                'std_prob_low_verb': np.std(paranoia_low_verb),
                'std_prob_non_event': np.std(paranoia_non_event),
                'f_stat': paranoia_f,
                'p_value': paranoia_p
            }
            
            # Correlation with verb intensity
            affair_verb_corr = np.corrcoef(
                affair_probs[:, affair_state],
                verb_intensity
            )[0,1]
            paranoia_verb_corr = np.corrcoef(
                paranoia_probs[:, paranoia_state],
                verb_intensity
            )[0,1]
            
            results[pair_key]['verb_correlation'] = {
                'affair': affair_verb_corr,
                'paranoia': paranoia_verb_corr
            }
            
            # Store verb information
            results[pair_key]['verb_info'] = {
                'total_verbs': len(interaction_verbs),
                'unique_verbs': len(set(interaction_verbs)),
                'verb_list': list(set(interaction_verbs)),  # List of unique verbs
                'median_verbs_per_tr': median_verbs
            }
            
            # Store temporal information
            results[pair_key]['temporal'] = {
                'affair_prob_timeseries': affair_probs[:, affair_state],
                'paranoia_prob_timeseries': paranoia_probs[:, paranoia_state],
                'verb_intensity': verb_intensity,
                'event_timeseries': lee_girl_events
            }
        
        return results

    def analyze_arthur_descriptive(self, tr_df: pd.DataFrame) -> Dict:
        """
        Analyze brain states during Arthur's descriptive speech using probabilistic state patterns
        
        Parameters:
        -----------
        tr_df : pd.DataFrame
            TR-level annotations with speech indicators
                
        Returns:
        --------
        Dict
            Complete analysis results for Arthur's descriptive speech
        """
        logger.info("Analyzing Arthur's descriptive speech")
        
        # Create binary event series for Arthur's descriptive speech
        arthur_descriptive = (tr_df['arthur_speaking'] & 
                            (tr_df['n_descriptors'] > 0)).values.astype(bool)
        logger.info(f"Found {np.sum(arthur_descriptive)} TRs with Arthur's descriptive speech")
        
        results = {}
        
        # Convert states to one-hot encoding for both groups
        one_hot_affair = np.eye(self.n_states)[self.states_affair]
        one_hot_paranoia = np.eye(self.n_states)[self.states_paranoia]
        
        # Get group-level probabilities
        affair_probs = one_hot_affair.mean(axis=0)
        paranoia_probs = one_hot_paranoia.mean(axis=0)
        
        # Additional descriptive features
        n_descriptors = tr_df['n_descriptors'].values
        descriptor_intensity = np.zeros_like(arthur_descriptive, dtype=float)
        descriptor_intensity[arthur_descriptive] = n_descriptors[arthur_descriptive]
        
        # Analyze each matched state pair
        for affair_state, paranoia_state in self.matched_states:
            pair_key = f'state_pair_{affair_state}_{paranoia_state}'
            results[pair_key] = {}
            
            # Get state probabilities for events vs non-events
            affair_event_probs = affair_probs[arthur_descriptive, affair_state]
            affair_non_event_probs = affair_probs[~arthur_descriptive, affair_state]
            paranoia_event_probs = paranoia_probs[arthur_descriptive, paranoia_state]
            paranoia_non_event_probs = paranoia_probs[~arthur_descriptive, paranoia_state]
            
            # Statistical comparison
            affair_tstat, affair_pval = stats.ttest_ind(
                affair_event_probs, affair_non_event_probs
            )
            paranoia_tstat, paranoia_pval = stats.ttest_ind(
                paranoia_event_probs, paranoia_non_event_probs
            )
            
            # Basic probability metrics
            results[pair_key]['affair'] = {
                'mean_prob_event': np.mean(affair_event_probs),
                'mean_prob_non_event': np.mean(affair_non_event_probs),
                'std_prob_event': np.std(affair_event_probs),
                'std_prob_non_event': np.std(affair_non_event_probs),
                'tstat': affair_tstat,
                'pval': affair_pval,
                'effect_size': (np.mean(affair_event_probs) - np.mean(affair_non_event_probs)) / 
                            np.sqrt(np.var(affair_event_probs) + np.var(affair_non_event_probs))
            }
            
            results[pair_key]['paranoia'] = {
                'mean_prob_event': np.mean(paranoia_event_probs),
                'mean_prob_non_event': np.mean(paranoia_non_event_probs),
                'std_prob_event': np.std(paranoia_event_probs),
                'std_prob_non_event': np.std(paranoia_non_event_probs),
                'tstat': paranoia_tstat,
                'pval': paranoia_pval,
                'effect_size': (np.mean(paranoia_event_probs) - np.mean(paranoia_non_event_probs)) / 
                            np.sqrt(np.var(paranoia_event_probs) + np.var(paranoia_non_event_probs))
            }
            
            # Correlation with descriptor intensity
            affair_corr = np.corrcoef(
                affair_probs[:, affair_state],
                descriptor_intensity
            )[0,1]
            paranoia_corr = np.corrcoef(
                paranoia_probs[:, paranoia_state],
                descriptor_intensity
            )[0,1]
            
            results[pair_key]['descriptor_correlation'] = {
                'affair': affair_corr,
                'paranoia': paranoia_corr
            }
            
            # Transitions using probabilistic approach
            results[pair_key]['transitions'] = {
                'affair': self.analyze_state_transitions(
                    self.states_affair, arthur_descriptive
                ),
                'paranoia': self.analyze_state_transitions(
                    self.states_paranoia, arthur_descriptive
                )
            }
            
            # Temporal dynamics
            results[pair_key]['temporal'] = {
                'affair_prob_timeseries': affair_probs[:, affair_state],
                'paranoia_prob_timeseries': paranoia_probs[:, paranoia_state],
                'event_timeseries': arthur_descriptive,
                'descriptor_intensity': descriptor_intensity
            }
        
        return results
        
    def analyze_all_speech_events(self, tr_df: pd.DataFrame) -> Dict:
        """
        Analyze brain states for all speech-related events
        
        Parameters:
        -----------
        tr_df : pd.DataFrame
            TR-level annotations
            
        Returns:
        --------
        Dict
            Analysis results for all speech events
        """
        results = {}
        
        # Define different speech events
        speech_events = {
            'arthur_speech': tr_df['arthur_speaking'],
            'lee_speech': tr_df['lee_speaking'],
            'girl_speech': tr_df['girl_speaking'],
            'dialog': tr_df['is_dialog'],
            'arthur_descriptive': tr_df['arthur_speaking'] & (tr_df['n_descriptors'] > 0),
            'lee_girl_interaction': tr_df['lee_girl_together']
        }
        
        for event_name, event_series in speech_events.items():
            logger.info(f"Analyzing {event_name}")
            event_binary = event_series.values.astype(bool)
            
            event_results = {}
            for affair_state, paranoia_state in self.matched_states:
                pair_key = f'state_pair_{affair_state}_{paranoia_state}'
                
                # Compute all metrics for this state pair
                event_results[pair_key] = {
                    'affair': self.compute_state_metrics(
                        self.states_affair,
                        self.state_probs_affair,
                        event_binary,
                        affair_state
                    ),
                    'paranoia': self.compute_state_metrics(
                        self.states_paranoia,
                        self.state_probs_paranoia,
                        event_binary,
                        paranoia_state
                    ),
                    'transitions': {
                        'affair': self.analyze_state_transitions(
                            self.states_affair,
                            event_binary
                        ),
                        'paranoia': self.analyze_state_transitions(
                            self.states_paranoia,
                            event_binary
                        )
                    },
                    'temporal': {
                        'affair': self.analyze_temporal_relationships(
                            self.states_affair,
                            event_binary,
                            affair_state
                        ),
                        'paranoia': self.analyze_temporal_relationships(
                            self.states_paranoia,
                            event_binary,
                            paranoia_state
                        )
                    }
                }
            
            results[event_name] = event_results
            
        return results
    
    def plot_lee_girl_interactions(self, results: Dict, event_name: str) -> None:
        """
        Create comprehensive visualizations for Lee-Girl interactions with verb focus
        
        Parameters:
        -----------
        results : Dict
            Analysis results containing state and verb information
        event_name : str
            Name of the event being analyzed
        """
        fig_dir = self.output_dir / "figures" / event_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for pair_key, pair_results in results.items():
            # 1. State Probabilities by Verb Usage Plot
            plt.figure(figsize=(12, 6))
            
            # Set up bar positions
            x = np.array([1, 2, 3, 5, 6, 7])  # Two groups with a gap between
            width = 0.35
            
            # Affair context data
            affair = pair_results['affair']
            affair_data = [
                affair['mean_prob_high_verb'],
                affair['mean_prob_low_verb'],
                affair['mean_prob_non_event']
            ]
            affair_error = [
                affair['std_prob_high_verb'],
                affair['std_prob_low_verb'],
                affair['std_prob_non_event']
            ]
            
            # Paranoia context data
            paranoia = pair_results['paranoia']
            paranoia_data = [
                paranoia['mean_prob_high_verb'],
                paranoia['mean_prob_low_verb'],
                paranoia['mean_prob_non_event']
            ]
            paranoia_error = [
                paranoia['std_prob_high_verb'],
                paranoia['std_prob_low_verb'],
                paranoia['std_prob_non_event']
            ]
            
            # Create bars
            plt.bar([1, 2, 3], affair_data, width, color=['darkblue', 'blue', 'lightblue'],
                    yerr=affair_error, capsize=5, label='Affair Context')
            plt.bar([5, 6, 7], paranoia_data, width, color=['darkred', 'red', 'lightcoral'],
                    yerr=paranoia_error, capsize=5, label='Paranoia Context')
            
            # Customize plot
            plt.xticks([2, 6], ['Affair Context', 'Paranoia Context'])
            plt.ylabel('State Probability')
            plt.title(f'State Probabilities by Verb Usage - {pair_key}\n' + 
                    f'Affair ANOVA: F={affair["f_stat"]:.2f}, p={affair["p_value"]:.3f}\n' +
                    f'Paranoia ANOVA: F={paranoia["f_stat"]:.2f}, p={paranoia["p_value"]:.3f}')
            
            # Add condition labels
            plt.text(1, -0.05, 'High Verb', ha='center')
            plt.text(2, -0.05, 'Low Verb', ha='center')
            plt.text(3, -0.05, 'Non-event', ha='center')
            plt.text(5, -0.05, 'High Verb', ha='center')
            plt.text(6, -0.05, 'Low Verb', ha='center')
            plt.text(7, -0.05, 'Non-event', ha='center')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f'{pair_key}_verb_probabilities.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Temporal Dynamics with Verb Intensity
            temporal = pair_results['temporal']
            plt.figure(figsize=(15, 10))
            
            # Create time points array
            time = np.arange(len(temporal['affair_prob_timeseries']))
            
            # Plot affair context
            plt.subplot(2, 1, 1)
            plt.plot(time, temporal['affair_prob_timeseries'], 'b-', label='State Probability')
            
            # Add verb intensity
            ax2 = plt.gca().twinx()
            ax2.plot(time, temporal['verb_intensity'], 'g--', alpha=0.5, label='Verb Count')
            ax2.set_ylabel('Number of Verbs', color='g')
            
            plt.title(f'Affair Context - {pair_key}\nVerb Correlation: {pair_results["verb_correlation"]["affair"]:.3f}')
            plt.legend()
            
            # Plot paranoia context
            plt.subplot(2, 1, 2)
            plt.plot(time, temporal['paranoia_prob_timeseries'], 'r-', label='State Probability')
            
            # Add verb intensity
            ax2 = plt.gca().twinx()
            ax2.plot(time, temporal['verb_intensity'], 'g--', alpha=0.5, label='Verb Count')
            ax2.set_ylabel('Number of Verbs', color='g')
            
            plt.title(f'Paranoia Context - {pair_key}\nVerb Correlation: {pair_results["verb_correlation"]["paranoia"]:.3f}')
            plt.xlabel('Time (TRs)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(fig_dir / f'{pair_key}_temporal_with_verbs.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Verb Analysis Summary
            verb_info = pair_results['verb_info']
            plt.figure(figsize=(10, 6))
            plt.text(0.1, 0.9, f"Verb Analysis Summary - {pair_key}", fontsize=12, fontweight='bold')
            plt.text(0.1, 0.8, f"Total Verbs: {verb_info['total_verbs']}")
            plt.text(0.1, 0.7, f"Unique Verbs: {verb_info['unique_verbs']}")
            plt.text(0.1, 0.6, f"Median Verbs per TR: {verb_info['median_verbs_per_tr']:.2f}")
            plt.text(0.1, 0.5, "Most Common Verbs:", fontweight='bold')
            
            # List top verbs if available
            if verb_info['verb_list']:
                verb_counts = pd.Series(verb_info['verb_list']).value_counts()
                top_verbs = verb_counts.head(10)
                for i, (verb, count) in enumerate(top_verbs.items()):
                    plt.text(0.1, 0.4 - i*0.05, f"{verb}: {count}")
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fig_dir / f'{pair_key}_verb_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_state_probabilities(self, results: Dict, event_name: str) -> None:
        """
        Create visualizations for state probability patterns using bar plots
        
        Parameters:
        -----------
        results : Dict
            Analysis results containing state probabilities
        event_name : str
            Name of the event being analyzed
        """
        fig_dir = self.output_dir / "figures" / event_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for pair_key, pair_results in results.items():
            plt.figure(figsize=(12, 6))
            
            # Set up bar positions
            x = np.array([1, 2, 4, 5])  # Two groups with a gap between
            width = 0.35
            
            # Affair context data
            affair = pair_results['affair']
            affair_data = [affair['mean_prob_event'], affair['mean_prob_non_event']]
            affair_error = [affair['std_prob_event'], affair['std_prob_non_event']]
            
            # Paranoia context data
            paranoia = pair_results['paranoia']
            paranoia_data = [paranoia['mean_prob_event'], paranoia['mean_prob_non_event']]
            paranoia_error = [paranoia['std_prob_event'], paranoia['std_prob_non_event']]
            
            # Create bars
            plt.bar([1, 2], affair_data, width, color=['lightblue', 'lightblue'],
                    yerr=affair_error, capsize=5, label='Affair Context')
            plt.bar([4, 5], paranoia_data, width, color=['lightpink', 'lightpink'],
                    yerr=paranoia_error, capsize=5, label='Paranoia Context')
            
            # Customize plot
            plt.xticks([1.5, 4.5], ['Affair Context', 'Paranoia Context'])
            plt.ylabel('State Probability')
            plt.title(f'State Probabilities - {pair_key}\n' + 
                    f'Affair p={affair["pval"]:.3f}, d={affair["effect_size"]:.2f}\n' +
                    f'Paranoia p={paranoia["pval"]:.3f}, d={paranoia["effect_size"]:.2f}')
            
            # Add event/non-event labels
            plt.text(1, -0.05, 'Event', ha='center')
            plt.text(2, -0.05, 'Non-event', ha='center')
            plt.text(4, -0.05, 'Event', ha='center')
            plt.text(5, -0.05, 'Non-event', ha='center')
            
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(fig_dir / f'{pair_key}_probabilities.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_temporal_dynamics(self, results: Dict, event_name: str) -> None:
        """
        Create visualizations for temporal dynamics of state probabilities
        
        Parameters:
        -----------
        results : Dict
            Analysis results containing temporal information
        event_name : str
            Name of the event being analyzed
        """
        fig_dir = self.output_dir / "figures" / event_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for pair_key, pair_results in results.items():
            temporal = pair_results['temporal']
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Time points
            time = np.arange(len(temporal['affair_prob_timeseries']))
            
            # Plot affair context
            ax1.plot(time, temporal['affair_prob_timeseries'], 'b-', label='State Probability')
            event_times = time[temporal['event_timeseries']]
            ax1.vlines(event_times, 0, 1, colors='gray', alpha=0.3, label='Events')
            ax1.set_title('Affair Context')
            ax1.set_ylabel('Probability')
            ax1.legend()
            
            # Plot paranoia context
            ax2.plot(time, temporal['paranoia_prob_timeseries'], 'r-', label='State Probability')
            ax2.vlines(event_times, 0, 1, colors='gray', alpha=0.3, label='Events')
            ax2.set_title('Paranoia Context')
            ax2.set_xlabel('Time (TRs)')
            ax2.set_ylabel('Probability')
            ax2.legend()
            
            # Add descriptor intensity if available
            if 'descriptor_intensity' in temporal:
                ax_twin1 = ax1.twinx()
                ax_twin2 = ax2.twinx()
                
                desc_line = ax_twin1.plot(time, temporal['descriptor_intensity'], 
                                        'g--', alpha=0.5, label='Descriptor Intensity')
                ax_twin2.plot(time, temporal['descriptor_intensity'], 
                            'g--', alpha=0.5, label='Descriptor Intensity')
                
                ax_twin1.set_ylabel('Descriptor Count', color='g')
                ax_twin2.set_ylabel('Descriptor Count', color='g')
            
            plt.suptitle(f'Temporal Dynamics - {pair_key}')
            plt.tight_layout()
            
            plt.savefig(fig_dir / f'{pair_key}_temporal.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_transition_patterns(self, results: Dict, event_name: str) -> None:
        """
        Create visualizations for probabilistic state transitions
        
        Parameters:
        -----------
        results : Dict
            Analysis results containing transition information
        event_name : str
            Name of the event being analyzed
        """
        fig_dir = self.output_dir / "figures" / event_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for pair_key, pair_results in results.items():
            transitions = pair_results['transitions']
            
            # Create 2x2 subplot for transition matrices
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # Affair context
            affair_trans = transitions['affair']
            sns.heatmap(affair_trans['transition_matrix_event'], 
                       ax=axes[0,0], cmap='viridis', 
                       annot=True, fmt='.2f')
            axes[0,0].set_title('Affair - During Events')
            
            sns.heatmap(affair_trans['transition_matrix_non_event'], 
                       ax=axes[0,1], cmap='viridis', 
                       annot=True, fmt='.2f')
            axes[0,1].set_title('Affair - Outside Events')
            
            # Paranoia context
            paranoia_trans = transitions['paranoia']
            sns.heatmap(paranoia_trans['transition_matrix_event'], 
                       ax=axes[1,0], cmap='viridis', 
                       annot=True, fmt='.2f')
            axes[1,0].set_title('Paranoia - During Events')
            
            sns.heatmap(paranoia_trans['transition_matrix_non_event'], 
                       ax=axes[1,1], cmap='viridis', 
                       annot=True, fmt='.2f')
            axes[1,1].set_title('Paranoia - Outside Events')
            
            plt.suptitle(f'State Transition Patterns - {pair_key}')
            plt.tight_layout()
            
            plt.savefig(fig_dir / f'{pair_key}_transitions.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_state_metrics(self, results: Dict, event_name: str) -> None:
        """
        Create visualizations for state metrics
        
        Parameters:
        -----------
        results : Dict
            Results from state analysis
        event_name : str
            Name of the event being analyzed
        """
        fig_dir = self.output_dir / "figures" / event_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot occupancy comparisons
        plt.figure(figsize=(12, 6))
        state_pairs = list(results.keys())
        x = np.arange(len(state_pairs))
        width = 0.35
        
        event_occ_affair = [results[pair]['affair']['occupancy_event'] for pair in state_pairs]
        non_event_occ_affair = [results[pair]['affair']['occupancy_non_event'] for pair in state_pairs]
        event_occ_paranoia = [results[pair]['paranoia']['occupancy_event'] for pair in state_pairs]
        non_event_occ_paranoia = [results[pair]['paranoia']['occupancy_non_event'] for pair in state_pairs]
        
        plt.bar(x - width/2, event_occ_affair, width, label='Affair Event', color='blue', alpha=0.6)
        plt.bar(x - width/2, non_event_occ_affair, width, label='Affair Non-event', 
                bottom=event_occ_affair, color='blue', alpha=0.3)
        plt.bar(x + width/2, event_occ_paranoia, width, label='Paranoia Event', color='red', alpha=0.6)
        plt.bar(x + width/2, non_event_occ_paranoia, width, label='Paranoia Non-event',
                bottom=event_occ_paranoia, color='red', alpha=0.3)
        
        plt.xlabel('State Pairs')
        plt.ylabel('Occupancy')
        plt.title(f'State Occupancy During {event_name}')
        plt.xticks(x, [f'States {pair}' for pair in state_pairs])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'state_occupancy.png')
        plt.close()
        
    def plot_temporal_relationships(self, results: Dict, event_name: str) -> None:
        """
        Create visualizations for temporal relationships
        
        Parameters:
        -----------
        results : Dict
            Results from temporal analysis
        event_name : str
            Name of the event being analyzed
        """
        fig_dir = self.output_dir / "figures" / event_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for pair_key in results:
            # Plot cross-correlation
            plt.figure(figsize=(10, 5))
            
            lags_affair = results[pair_key]['temporal']['affair']['lags']
            corr_affair = results[pair_key]['temporal']['affair']['correlations']
            lags_paranoia = results[pair_key]['temporal']['paranoia']['lags']
            corr_paranoia = results[pair_key]['temporal']['paranoia']['correlations']
            
            plt.plot(lags_affair, corr_affair, 'b-', label='Affair Context')
            plt.plot(lags_paranoia, corr_paranoia, 'r-', label='Paranoia Context')
            
            plt.axhline(y=0, color='k', linestyle=':')
            plt.axvline(x=0, color='k', linestyle=':')
            
            plt.xlabel('Lag (TRs)')
            plt.ylabel('Cross-correlation')
            plt.title(f'Temporal Relationship - {pair_key}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(fig_dir / f'{pair_key}_temporal.png')
            plt.close()
            
            # Plot mutual information
            plt.figure(figsize=(10, 5))
            
            mi_affair = results[pair_key]['temporal']['affair']['mutual_info']
            mi_paranoia = results[pair_key]['temporal']['paranoia']['mutual_info']
            
            plt.plot(lags_affair, mi_affair, 'b-', label='Affair Context')
            plt.plot(lags_paranoia, mi_paranoia, 'r-', label='Paranoia Context')
            
            plt.axvline(x=0, color='k', linestyle=':')
            
            plt.xlabel('Lag (TRs)')
            plt.ylabel('Mutual Information')
            plt.title(f'Mutual Information - {pair_key}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(fig_dir / f'{pair_key}_mutual_info.png')
            plt.close()
    
    def plot_transition_matrices(self, results: Dict, event_name: str) -> None:
        """
        Create visualizations for transition matrices
        
        Parameters:
        -----------
        results : Dict
            Results containing transition matrices
        event_name : str
            Name of the event being analyzed
        """
        fig_dir = self.output_dir / "figures" / event_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for pair_key in results:
            # Plot transition matrices for both conditions
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # Affair context
            trans_event = results[pair_key]['transitions']['affair']['transition_matrix_event']
            trans_non_event = results[pair_key]['transitions']['affair']['transition_matrix_non_event']
            
            sns.heatmap(trans_event, ax=axes[0,0], cmap='viridis', annot=True)
            axes[0,0].set_title('Affair Context - During Events')
            
            sns.heatmap(trans_non_event, ax=axes[0,1], cmap='viridis', annot=True)
            axes[0,1].set_title('Affair Context - Outside Events')
            
            # Paranoia context
            trans_event = results[pair_key]['transitions']['paranoia']['transition_matrix_event']
            trans_non_event = results[pair_key]['transitions']['paranoia']['transition_matrix_non_event']
            
            sns.heatmap(trans_event, ax=axes[1,0], cmap='viridis', annot=True)
            axes[1,0].set_title('Paranoia Context - During Events')
            
            sns.heatmap(trans_non_event, ax=axes[1,1], cmap='viridis', annot=True)
            axes[1,1].set_title('Paranoia Context - Outside Events')
            
            plt.suptitle(f'State Transition Matrices - {pair_key}')
            plt.tight_layout()
            plt.savefig(fig_dir / f'{pair_key}_transitions.png')
            plt.close()
    
    def save_results(self, results: Dict, event_name: str) -> None:
        """
        Save analysis results to files
        
        Parameters:
        -----------
        results : Dict
            Analysis results to save
        event_name : str
            Name of the event being analyzed
        """
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results as pickle
        with open(results_dir / f"{event_name}_full_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary DataFrames
        summary_data = []
        for pair_key in results:
            pair_summary = {
                'state_pair': pair_key,
                'event_name': event_name
            }
            
            # Add metrics for both conditions
            for condition in ['affair', 'paranoia']:
                metrics = results[pair_key][condition]
                pair_summary.update({
                    f'{condition}_occupancy_ratio': metrics['occupancy_ratio'],
                    f'{condition}_occupancy_event': metrics['occupancy_event'],
                    f'{condition}_occupancy_non_event': metrics['occupancy_non_event']
                })
                
                if 'prob_tstat' in metrics:
                    pair_summary.update({
                        f'{condition}_prob_tstat': metrics['prob_tstat'],
                        f'{condition}_prob_pval': metrics['prob_pval']
                    })
            
            # Add temporal metrics
            for condition in ['affair', 'paranoia']:
                temporal = results[pair_key]['temporal'][condition]
                pair_summary.update({
                    f'{condition}_max_correlation_lag': temporal['max_correlation_lag'],
                    f'{condition}_max_correlation': np.max(np.abs(temporal['correlations'])),
                    f'{condition}_max_mi_lag': temporal['max_mi_lag'],
                    f'{condition}_max_mi': np.max(temporal['mutual_info'])
                })
                
            summary_data.append(pair_summary)
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_dir / f"{event_name}_summary.csv", index=False)

def main():
    """Main execution function for story state analysis"""
    load_dotenv()
    base_dir = os.getenv('SCRATCH_DIR')
    if base_dir is None:
        raise ValueError("SCRATCH_DIR environment variable not set")
    base_dir = Path(base_dir)
    
    try:
        # Load brain state data
        logger.info("Loading brain state data...")
        affair_states = np.load(
            base_dir / "output" / "affair_hmm_3states_ntw_native_trimmed" / 
            "statistics" / "affair_state_sequences.npy"
        )
        paranoia_states = np.load(
            base_dir / "output" / "paranoia_hmm_3states_ntw_native_trimmed" / 
            "statistics" / "paranoia_state_sequences.npy"
        )
        
        # Try to load state probabilities
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
        
        # Load story annotations
        logger.info("Loading story annotations...")
        story_df = pd.read_csv(base_dir / 'data' / 'stimuli' / '10_story_annotations.csv')
        segments_df = pd.read_csv(
            base_dir / 'data' / 'stimuli' / 'segments_speaker.csv',
            encoding='cp1252'
        )
        
        # Get TR-level annotations
        n_brain_trs = affair_states.shape[1]
        logger.info(f"Brain state data has {n_brain_trs} TRs")
        tr_df = get_or_create_tr_annotations(story_df, n_brain_trs, base_dir)
        
        # Initialize analysis
        analysis = BrainStateAnalysis(
            base_dir=str(base_dir),
            brain_states_affair=affair_states,
            brain_states_paranoia=paranoia_states,
            matched_states=matched_states,
            state_probs_affair=affair_probs,
            state_probs_paranoia=paranoia_probs
        )
        
        # Analyze Lee-Girl interactions
        logger.info("\nAnalyzing Lee-Girl interactions...")
        lee_girl_results = analysis.analyze_lee_girl_interactions(tr_df)
        
        # Create visualizations for Lee-Girl analysis
        logger.info("Creating Lee-Girl visualizations...")
        # analysis.plot_state_probabilities(lee_girl_results, "lee_girl")
        # analysis.plot_temporal_dynamics(lee_girl_results, "lee_girl")
        # analysis.plot_transition_patterns(lee_girl_results, "lee_girl")
        analysis.plot_lee_girl_interactions(lee_girl_results, "lee_girl")  # Use new plotting function
    
        
        # Save Lee-Girl results
        results_dir = analysis.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "lee_girl_results.pkl", 'wb') as f:
            pickle.dump(lee_girl_results, f)
        
        # Analyze Arthur's descriptive speech
        logger.info("\nAnalyzing Arthur's descriptive speech...")
        arthur_results = analysis.analyze_arthur_descriptive(tr_df)
        
        # Create visualizations for Arthur analysis
        logger.info("Creating Arthur speech visualizations...")
        analysis.plot_state_probabilities(arthur_results, "arthur_descriptive")
        analysis.plot_temporal_dynamics(arthur_results, "arthur_descriptive")
        analysis.plot_transition_patterns(arthur_results, "arthur_descriptive")
        
        # Save Arthur results
        with open(results_dir / "arthur_results.pkl", 'wb') as f:
            pickle.dump(arthur_results, f)
        
        print("\nAnalysis Summary:")
    
        print("\nLee-Girl Interactions:")
        for pair_key, pair_results in lee_girl_results.items():
            print(f"\n{pair_key}:")
            for context in ['affair', 'paranoia']:
                res = pair_results[context]
                print(f"  {context.capitalize()} Context:")
                print(f"    High-verb probability: {res['mean_prob_high_verb']:.3f} ± {res['std_prob_high_verb']:.3f}")
                print(f"    Low-verb probability: {res['mean_prob_low_verb']:.3f} ± {res['std_prob_low_verb']:.3f}")
                print(f"    Non-event probability: {res['mean_prob_non_event']:.3f} ± {res['std_prob_non_event']:.3f}")
                print(f"    ANOVA F-stat: {res['f_stat']:.3f}")
                print(f"    ANOVA p-value: {res['p_value']:.3f}")
                
        # Print verb correlations
        corr = pair_results['verb_correlation']
        print(f"  Verb Correlations:")
        print(f"    Affair: {corr['affair']:.3f}")
        print(f"    Paranoia: {corr['paranoia']:.3f}")
        
        # Print verb info
        verb_info = pair_results['verb_info']
        print(f"  Verb Statistics:")
        print(f"    Total verbs: {verb_info['total_verbs']}")
        print(f"    Unique verbs: {verb_info['unique_verbs']}")
        print(f"    Median verbs per TR: {verb_info['median_verbs_per_tr']:.2f}")
    
        print("\nArthur's Descriptive Speech:")
        for pair_key, pair_results in arthur_results.items():
            print(f"\n{pair_key}:")
            for context in ['affair', 'paranoia']:
                res = pair_results[context]
                print(f"    Event probability: {res['mean_prob_event']:.3f} ± {res['std_prob_event']:.3f}")
                print(f"    Non-event probability: {res['mean_prob_non_event']:.3f} ± {res['std_prob_non_event']:.3f}")
                print(f"    Effect size: {res['effect_size']:.3f}")
                print(f"    P-value: {res['pval']:.3f}")
                
        # Print descriptor correlations
        corr = pair_results['descriptor_correlation']
        print(f"  Descriptor Correlations:")
        print(f"    Affair: {corr['affair']:.3f}")
        print(f"    Paranoia: {corr['paranoia']:.3f}")
            
        logger.info("Analysis completed successfully!")
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()