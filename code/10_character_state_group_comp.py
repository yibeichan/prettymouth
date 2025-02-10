#!/usr/bin/env python3
"""
Script for analyzing brain state dynamics in relation to story content,
focusing on group comparisons between different contextual conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from statsmodels.stats.multitest import multipletests
import warnings
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StateCharacterAnalysis:
    """
    Refined analysis of brain states with character-specific focus and adaptive windows
    """
    def __init__(self, 
                 base_dir: Union[str, Path],
                 brain_states_affair: np.ndarray, 
                 brain_states_paranoia: np.ndarray, 
                 matched_states: List[Tuple[int, int]], 
                 tr_duration: float = 1.5):
        """Initialize with same parameters as before"""
        # Validate inputs
        if brain_states_affair.shape != brain_states_paranoia.shape:
            raise ValueError("Brain state arrays must have the same shape")
            
        self.states_affair = brain_states_affair
        self.states_paranoia = brain_states_paranoia
        self.matched_states = matched_states
        self.tr_duration = tr_duration
        self.base_dir = Path(base_dir)
        
        # Setup output directory
        self.output_dir = self.base_dir / "output" / "10_character_state_group_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'character_specific': {},
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'data_shape': brain_states_affair.shape,
                'tr_duration': tr_duration
            }
        }
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized analysis with data shape: {brain_states_affair.shape}")
        logger.info(f"Output directory: {self.output_dir}")
        
    def _setup_logging(self):
        """Setup logging to file"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def determine_adaptive_window(self, segments_df: pd.DataFrame) -> Dict[str, int]:
        """
        Determine appropriate window sizes based on event timing
        """
        logger.info("Determining adaptive window sizes")
        windows = {}
        
        # For each character's dialogue
        for char in ['lee', 'arthur', 'girl']:
            char_segments = segments_df[segments_df['speaker'] == char]['onset_sec'].values
            if len(char_segments) > 1:
                intervals = np.diff(char_segments)
                # Use 25th percentile of intervals to avoid overlap
                window_sec = np.percentile(intervals, 25) / 2  # half of typical interval
                windows[f'{char}_dialogue'] = max(1, int(window_sec / self.tr_duration))
                logger.info(f"Window size for {char}: {windows[f'{char}_dialogue']} TRs")
        
        # For character interactions
        interaction_segments = segments_df[segments_df['actor'].str.contains(',', na=False)]['onset_sec'].values
        if len(interaction_segments) > 1:
            interaction_intervals = np.diff(interaction_segments)
            window_sec = np.percentile(interaction_intervals, 25) / 2
            windows['interaction'] = max(1, int(window_sec / self.tr_duration))
            logger.info(f"Window size for interactions: {windows['interaction']} TRs")
        
        return windows
    
    def analyze_character_specific_states(self, segments_df: pd.DataFrame) -> Dict:
        """
        Analyze state patterns specific to each character
        """
        logger.info("Starting character-specific state analysis")
        
        # Get adaptive windows
        windows = self.determine_adaptive_window(segments_df)
        self.results['metadata']['windows'] = windows
        
        # Analyze each character separately
        for char in ['lee', 'arthur', 'girl']:
            logger.info(f"Analyzing patterns for character: {char}")
            char_mask = (segments_df['speaker'] == char) | (segments_df['actor'].str.contains(char, na=False))
            char_segments = segments_df[char_mask]
            
            if len(char_segments) == 0:
                logger.warning(f"No segments found for character: {char}")
                continue
                
            # Get window size for this character
            window_size = windows.get(f'{char}_dialogue', 2)
            
            # Analyze state patterns
            char_results = self._analyze_character_patterns(
                char_segments,
                window_size,
                char_name=char
            )
            
            self.results['character_specific'][char] = char_results
            
        return self.results

    def _analyze_character_patterns(self, 
                                  char_segments: pd.DataFrame,
                                  window_size: int,
                                  char_name: str) -> Dict:
        """Analyze state patterns for a specific character"""
        logger.info(f"Analyzing patterns for {char_name} with window size {window_size}")
        
        n_timepoints = self.states_affair.shape[1]
        
        # Convert segment onsets to TR indices
        tr_onsets = [int(onset / self.tr_duration) for onset in char_segments['onset_sec']]
        tr_onsets = [onset for onset in tr_onsets if window_size <= onset < n_timepoints - window_size]
        
        results = {}
        # Analyze each matched state pair
        for affair_state, paranoia_state in self.matched_states:
            affair_patterns = []
            paranoia_patterns = []
            
            # Get state probabilities
            affair_prob = (self.states_affair == affair_state).mean(axis=0)
            paranoia_prob = (self.states_paranoia == paranoia_state).mean(axis=0)
            
            # Collect patterns around character events
            for onset in tr_onsets:
                affair_patterns.append(
                    affair_prob[onset-window_size:onset+window_size+1]
                )
                paranoia_patterns.append(
                    paranoia_prob[onset-window_size:onset+window_size+1]
                )
            
            if not affair_patterns:
                logger.warning(f"No valid patterns found for state pair {affair_state}_{paranoia_state}")
                continue
                
            # Convert to arrays
            affair_patterns = np.array(affair_patterns)
            paranoia_patterns = np.array(paranoia_patterns)
            
            # Calculate statistics
            stats_results = self._calculate_pattern_statistics(
                affair_patterns,
                paranoia_patterns,
                f"{char_name} - State Pair {affair_state}_{paranoia_state}"
            )
            
            results[f'state_pair_{affair_state}_{paranoia_state}'] = stats_results
            
        return results

    def _calculate_pattern_statistics(self,
                                    affair_patterns: np.ndarray,
                                    paranoia_patterns: np.ndarray,
                                    label: str) -> Dict:
        """Calculate statistics for pattern comparison"""
        affair_mean = np.mean(affair_patterns, axis=0)
        paranoia_mean = np.mean(paranoia_patterns, axis=0)
        affair_sem = stats.sem(affair_patterns, axis=0)
        paranoia_sem = stats.sem(paranoia_patterns, axis=0)
        
        # Statistical testing with FDR correction
        tstats = []
        pvals = []
        for t in range(affair_patterns.shape[1]):
            tstat, pval = stats.ttest_ind(
                affair_patterns[:, t],
                paranoia_patterns[:, t]
            )
            tstats.append(float(tstat))
            pvals.append(float(pval))
            
        # FDR correction
        _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
        
        logger.info(f"Calculated statistics for {label} with {len(affair_patterns)} events")
        
        return {
            'label': label,
            'affair_mean': affair_mean,
            'paranoia_mean': paranoia_mean,
            'affair_sem': affair_sem,
            'paranoia_sem': paranoia_sem,
            'tstats': np.array(tstats),
            'pvals': np.array(pvals),
            'pvals_fdr': pvals_fdr,
            'n_events': len(affair_patterns)
        }

    def visualize_results(self) -> None:
        """Create and save visualization plots for all analyses"""
        logger.info("Creating visualization plots")
        
        try:
            # Create figures directory
            fig_dir = self.output_dir / "figures"
            fig_dir.mkdir(exist_ok=True)
            
            # Create plots for each character
            for char, char_results in self.results['character_specific'].items():
                self._create_character_plots(
                    char=char,
                    results=char_results,
                    fig_dir=fig_dir
                )
                
            logger.info(f"Plots saved to {fig_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def _create_character_plots(self, char: str, results: Dict, fig_dir: Path) -> None:
        """Create plots for a specific character's results"""
        for state_pair, state_results in results.items():
            if not isinstance(state_results, dict) or 'affair_mean' not in state_results:
                continue
                
            window_size = (len(state_results['affair_mean']) - 1) // 2
            timepoints = np.arange(-window_size, window_size + 1)
            
            plt.figure(figsize=(10, 6))
            
            # Plot means and error bands
            plt.plot(timepoints, state_results['affair_mean'],
                    label='Affair', color='blue')
            plt.fill_between(timepoints,
                           state_results['affair_mean'] - state_results['affair_sem'],
                           state_results['affair_mean'] + state_results['affair_sem'],
                           color='blue', alpha=0.2)
            
            plt.plot(timepoints, state_results['paranoia_mean'],
                    label='Paranoia', color='red')
            plt.fill_between(timepoints,
                           state_results['paranoia_mean'] - state_results['paranoia_sem'],
                           state_results['paranoia_mean'] + state_results['paranoia_sem'],
                           color='red', alpha=0.2)
            
            # Mark significant timepoints (using FDR-corrected p-values)
            sig_points = timepoints[state_results['pvals_fdr'] < 0.05]
            if len(sig_points) > 0:
                plt.plot(sig_points,
                        np.ones_like(sig_points) * plt.ylim()[1],
                        'k*', label='p < 0.05 (FDR)')
            
            plt.title(f'{state_results["label"]}\n(n={state_results["n_events"]} events)')
            plt.xlabel('Time relative to event (TRs)')
            plt.ylabel('State Probability')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(fig_dir / f'{char}_{state_pair}.png')
            logger.info(f"Saved plot to {fig_dir / f'{char}_{state_pair}.png'}")
            plt.close()

    def save_results(self) -> None:
        """Save analysis results and metadata to files"""
        logger.info("Saving analysis results")
        
        try:
            # Save results dictionary
            results_file = self.output_dir / "character_state_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(self.results, f)
            
            # Save metadata as JSON
            metadata = {
                'analysis_time': self.results['metadata']['analysis_time'],
                'data_shape': list(self.results['metadata']['data_shape']),
                'tr_duration': self.results['metadata']['tr_duration'],
                'windows': self.results['metadata'].get('windows', {}),
                'output_directory': str(self.output_dir)
            }
            
            meta_file = self.output_dir / "metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


class CharacterInteractionAnalysis:
    """Analysis focused on interactions between specific characters"""
    
    def __init__(self, 
                 base_dir: Union[str, Path],
                 brain_states_affair: np.ndarray, 
                 brain_states_paranoia: np.ndarray, 
                 matched_states: List[Tuple[int, int]], 
                 tr_duration: float = 1.5):
        self.states_affair = brain_states_affair
        self.states_paranoia = brain_states_paranoia
        self.matched_states = matched_states
        self.tr_duration = tr_duration
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "output" / "10_character_state_group_comparison"
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Initialized CharacterInteractionAnalysis")

    def _setup_logging(self):
        """Setup logging to file"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"interaction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def analyze_character_interaction(self, segments_df: pd.DataFrame,
                                   char1: str = 'lee',
                                   char2: str = 'girl',
                                   window_size: int = 1) -> Dict:
        """
        Analyze state patterns during interactions between two characters
        
        Parameters:
        -----------
        segments_df : DataFrame with columns 'actor', 'onset_sec', 'speaker'
        char1, char2 : names of characters to analyze interaction
        window_size : number of TRs before/after interaction to analyze
        """
        # Find interactions between the two characters
        interaction_mask = (segments_df['actor'].fillna('').str.contains(char1, case=False) & 
                          segments_df['actor'].fillna('').str.contains(char2, case=False))
        interaction_segments = segments_df[interaction_mask]
        
        n_timepoints = self.states_affair.shape[1]
        results = {}
        
        # Convert segment onsets to TR indices
        tr_onsets = [int(onset / self.tr_duration) for onset in interaction_segments['onset_sec']]
        tr_onsets = [onset for onset in tr_onsets if window_size <= onset < n_timepoints - window_size]
        
        print(f"Found {len(tr_onsets)} interaction events between {char1} and {char2}")
        
        # Analyze each matched state pair
        for affair_state, paranoia_state in self.matched_states:
            affair_patterns = []
            paranoia_patterns = []
            
            # Get state probabilities
            affair_prob = (self.states_affair == affair_state).mean(axis=0)
            paranoia_prob = (self.states_paranoia == paranoia_state).mean(axis=0)
            
            # Collect patterns around interactions
            for onset in tr_onsets:
                affair_patterns.append(
                    affair_prob[onset-window_size:onset+window_size+1]
                )
                paranoia_patterns.append(
                    paranoia_prob[onset-window_size:onset+window_size+1]
                )
            
            if not affair_patterns:
                continue
                
            # Convert to arrays
            affair_patterns = np.array(affair_patterns)
            paranoia_patterns = np.array(paranoia_patterns)
            
            # Calculate statistics
            affair_mean = np.mean(affair_patterns, axis=0)
            paranoia_mean = np.mean(paranoia_patterns, axis=0)
            affair_sem = stats.sem(affair_patterns, axis=0)
            paranoia_sem = stats.sem(paranoia_patterns, axis=0)
            
            # Statistical testing with FDR correction
            tstats = []
            pvals = []
            for t in range(affair_patterns.shape[1]):
                tstat, pval = stats.ttest_ind(
                    affair_patterns[:, t],
                    paranoia_patterns[:, t]
                )
                tstats.append(float(tstat))
                pvals.append(float(pval))
                
            # FDR correction using multipletests
            _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.05)
            
            pair_key = f'state_pair_{affair_state}_{paranoia_state}'
            results[pair_key] = {
                'label': f'Interaction: {char1}-{char2}',
                'affair_mean': affair_mean,
                'paranoia_mean': paranoia_mean,
                'affair_sem': affair_sem,
                'paranoia_sem': paranoia_sem,
                'tstats': np.array(tstats),
                'pvals': np.array(pvals),
                'pvals_fdr': pvals_fdr,
                'n_events': len(affair_patterns)
            }
            
        return results
    
    def save_interaction_results(self, results: Dict) -> None:
        """Save the interaction analysis results to a file"""
        results_file = self.output_dir / "interaction_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

    def visualize_interaction_results(self, results: Dict, 
                                    char1: str = 'lee',
                                    char2: str = 'girl') -> None:
        """
        Visualize and save the interaction analysis results
        """
        # Create figures directory if it doesn't exist
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        for state_pair, state_results in results.items():
            window_size = (len(state_results['affair_mean']) - 1) // 2
            timepoints = np.arange(-window_size, window_size + 1)
            
            plt.figure(figsize=(10, 6))
            
            # Plot means and error bands
            plt.plot(timepoints, state_results['affair_mean'],
                    label='Affair', color='blue')
            plt.fill_between(timepoints,
                           state_results['affair_mean'] - state_results['affair_sem'],
                           state_results['affair_mean'] + state_results['affair_sem'],
                           color='blue', alpha=0.2)
            
            plt.plot(timepoints, state_results['paranoia_mean'],
                    label='Paranoia', color='red')
            plt.fill_between(timepoints,
                           state_results['paranoia_mean'] - state_results['paranoia_sem'],
                           state_results['paranoia_mean'] + state_results['paranoia_sem'],
                           color='red', alpha=0.2)
            
            # Mark significant timepoints
            sig_points = timepoints[state_results['pvals_fdr'] < 0.05]
            if len(sig_points) > 0:
                plt.plot(sig_points,
                        np.ones_like(sig_points) * plt.ylim()[1],
                        'k*', label='p < 0.05 (FDR)')
            
            plt.title(f'{char1.capitalize()}-{char2.capitalize()} Interaction: {state_pair}\n(n={state_results["n_events"]} events)')
            plt.xlabel('Time relative to interaction (TRs)')
            plt.ylabel('State Probability')
            plt.legend()
            
            plt.tight_layout()
            
            # Save the plot
            filename = f'interaction_{char1}_{char2}_{state_pair}.png'
            plt.savefig(fig_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {fig_dir / filename}")
            plt.close()  # Close the figure to free memory

def load_brain_states(base_dir: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Load brain state data and matched states from files
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing brain state analysis results
    
    Returns:
    --------
    tuple : (affair_states, paranoia_states, matched_states)
    """
    base_dir = Path(base_dir)
    
    # Load affair states
    affair_file = base_dir / "output" / "affair_hmm_3states_ntw_native_trimmed" / "statistics" / "affair_state_sequences.npy"
    affair_states = np.load(affair_file)
    
    # Load paranoia states
    paranoia_file = base_dir / "output" / "paranoia_hmm_3states_ntw_native_trimmed" / "statistics" / "paranoia_state_sequences.npy"
    paranoia_states = np.load(paranoia_file)
    
    # Load state matching results
    matching_file = base_dir / "output" / "09_group_HMM_comparison" / "comparisons.pkl"
    with open(matching_file, 'rb') as f:
        comparisons = pickle.load(f)
    matched_states = comparisons['state_similarity']['matched_states']
    
    return affair_states, paranoia_states, matched_states

def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Get base directory with more explicit error handling
    base_dir = os.getenv('SCRATCH_DIR')
    if base_dir is None:
        # Check if environment variable exists in .env file
        env_path = Path('.env')
        if not env_path.exists():
            raise FileNotFoundError(
                "No .env file found. Please create a .env file with SCRATCH_DIR=/path/to/your/directory"
            )
        
        # If .env exists but SCRATCH_DIR is not set
        raise ValueError(
            "SCRATCH_DIR environment variable is not set. "
            "Please set it in your .env file or use: "
            "export SCRATCH_DIR=/path/to/your/directory"
        )
    
    # Verify the directory exists
    base_dir_path = Path(base_dir)
    if not base_dir_path.exists():
        raise NotADirectoryError(
            f"Directory does not exist: {base_dir}. "
            "Please check your SCRATCH_DIR path."
        )
    
    logger.info(f"Using base directory: {base_dir}")
    
    try:
        # Load data
        affair_states, paranoia_states, matched_states = load_brain_states(base_dir)
        segments_df = pd.read_csv(
            Path(base_dir) / 'data' / 'stimuli' / 'segments_speaker.csv',
            encoding='cp1252'
        )
        
        # Character analysis
        char_analysis = StateCharacterAnalysis(
            base_dir=base_dir,
            brain_states_affair=affair_states,
            brain_states_paranoia=paranoia_states,
            matched_states=matched_states
        )
        
        # Analyze all characters at once and visualize
        results = char_analysis.analyze_character_specific_states(segments_df)
        char_analysis.visualize_results()  # This will handle all the plotting
        
        # Interaction analysis
        interaction_analysis = CharacterInteractionAnalysis(
            base_dir=base_dir,
            brain_states_affair=affair_states,
            brain_states_paranoia=paranoia_states,
            matched_states=matched_states
        )
        
        # Analyze specific interactions
        interaction_pairs = [('lee', 'girl'), ('lee', 'arthur'), ('arthur', 'girl')]
        for char1, char2 in interaction_pairs:
            results = interaction_analysis.analyze_character_interaction(
                segments_df, char1, char2, window_size=1
            )
            
            # Visualize results
            for state_pair, state_results in results.items():
                interaction_analysis.visualize_interaction_results(results)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()