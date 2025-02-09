import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path

class ContentBehaviorAnalyzer:
    def __init__(self, data_dir, output_dir):
        """
        Initialize the analyzer with data directory path.
        
        Args:
            data_dir (str): Path to directory containing processed data files
            output_dir (str): Path to directory to save analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_data(self):
        """
        Load processed behavioral data and story annotations.
        """
        try:
            # Load behavioral data
            self.eventseg_agreement = pd.read_pickle(
                self.output_dir / "agreement_eventseg_filtered.pkl"
            )
            self.evidence_affair = pd.read_pickle(
                self.output_dir / "agreement_evidence_affair.pkl"
            )
            self.evidence_paranoia = pd.read_pickle(
                self.output_dir / "agreement_evidence_paranoia.pkl"
            )
            
            # Load content features
            self.content_df = pd.read_csv(
                self.data_dir / "10_story_annotations_TR.csv"
            )
            
            self.logger.info("Data loaded successfully")
            
            print(self.eventseg_agreement.head())
            print(self.evidence_affair.head())
            print(self.evidence_paranoia.head())    

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def preprocess_content_features(self):
        """
        Preprocess content features for analysis.
        """
        # Create feature groups
        self.pos_features = {
            'verb': 'has_verb',
            'noun': 'has_noun',
            'adj': 'has_adj',
            'adv': 'has_adv'
        }
        
        self.speaker_features = {
            'arthur': 'arthur_speaking',
            'lee': 'lee_speaking',
            'girl': 'girl_speaking',
            'lee_girl': 'lee_girl_together'
        }
        
        # Aggregate features by TR
        self.tr_features = self._aggregate_features_by_tr()
        
    def extract_narrative_shifts(self):
        """Extract points of transition in narrative features"""
        shifts = pd.DataFrame()
        shifts['TR'] = self.tr_features['TR']
        
        # Detect dialogue shifts (0->1 or 1->0)
        shifts['dialogue_shift'] = self.tr_features['dialog_freq'].diff().abs()
        
        # Detect speaker shifts
        for speaker in self.speaker_features:
            shifts[f'{speaker}_shift'] = self.tr_features[f'{speaker}_freq'].diff().abs()
        
        # Detect any speaker change (regardless of specific speakers)
        speaker_cols = [col for col in self.tr_features.columns if 'speaking' in col]
        prev_speakers = self.tr_features[speaker_cols].apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
        shifts['any_speaker_shift'] = (prev_speakers != prev_speakers.shift()).astype(float)
        
        return shifts
    
    def analyze_event_boundaries(self):
        """Analyze relationship between narrative shifts and event boundaries"""
        # Get event boundaries (peaks in event agreement)
        event_peaks = self._get_peaks(
            self.feature_response_df['event_agreement'],
            prominence=0.23,
            width=1
        )
        
        # Calculate narrative shifts
        shifts_df = pd.DataFrame()
        shifts_df['TR'] = self.tr_features['TR']
        
        # Dialogue shifts
        shifts_df['dialogue_active'] = self.tr_features['dialog_freq']
        shifts_df['dialogue_shift'] = shifts_df['dialogue_active'].diff().abs()
        
        # Speaker shifts
        speaker_cols = [col for col in self.tr_features.columns if 'speaking' in col]
        prev_speakers = self.tr_features[speaker_cols].apply(
            lambda x: ''.join(x.astype(int).astype(str)), axis=1
        )
        shifts_df['speaker_shift'] = (prev_speakers != prev_speakers.shift()).astype(float)
        
        return event_peaks, shifts_df

    def _plot_event_boundary_analysis(self, output_dir):
        """Create static visualizations for event boundary analysis"""
        event_peaks, shifts_df = self.analyze_event_boundaries()
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[1, 1, 0.5])
        fig.suptitle('Event Boundaries and Narrative Shifts Analysis', fontsize=14)
        
        # Plot 1: Event Agreement Timeline 
        ax1.plot(self.feature_response_df['TR'], 
                self.feature_response_df['event_agreement'],
                label='Event Agreement', color='black')
        for peak in event_peaks:
            ax1.axvline(x=peak, color='red', alpha=0.3, linestyle='--')
        ax1.set_title('Event Segmentation Agreement')
        ax1.set_ylabel('Agreement Score')
        ax1.legend()
        
        # Plot 2: Speaker States
        # Fixed column names to match what's in tr_features
        speaker_cols = ['arthur', 'lee', 'girl']  # Remove '_speaking' suffix
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for col, color in zip(speaker_cols, colors):
            ax2.fill_between(self.tr_features['TR'], 
                            self.tr_features[f'{col}_freq'],  # Using the correct column names
                            alpha=0.3, color=color,
                            label=f'{col.title()} Speaking')
        for peak in event_peaks:
            ax2.axvline(x=peak, color='red', alpha=0.3, linestyle='--')
        ax2.set_title('Speaker States')
        ax2.set_ylabel('Active State')
        ax2.legend()
        
        # Plot 3: Shifts
        ax3.plot(shifts_df['TR'], shifts_df['dialogue_shift'], 
                label='Dialogue Shifts', color='blue', alpha=0.6)
        ax3.plot(shifts_df['TR'], shifts_df['speaker_shift'],
                label='Speaker Shifts', color='green', alpha=0.6)
        for peak in event_peaks:
            ax3.axvline(x=peak, color='red', alpha=0.3, linestyle='--')
        ax3.set_title('Narrative Shifts')
        ax3.set_xlabel('Time (TR)')
        ax3.set_ylabel('Shift Magnitude')
        ax3.legend()
        
        # Add statistics text
        stats = self._calculate_shift_statistics(shifts_df, event_peaks, window_size=2)
        stats_text = (
            f"Dialogue Shifts: {stats['dialogue_shifts']['total']} total, "
            f"{stats['dialogue_shifts']['aligned']} aligned with boundaries "
            f"({stats['dialogue_shifts']['percent_aligned']:.1f}%)\n"
            f"Speaker Shifts: {stats['speaker_shifts']['total']} total, "
            f"{stats['speaker_shifts']['aligned']} aligned with boundaries "
            f"({stats['speaker_shifts']['percent_aligned']:.1f}%)"
        )
        fig.text(0.1, 0.02, stats_text, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'event_boundary_analysis.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()

    def _calculate_shift_statistics(self, shifts_df, event_peaks, window_size):
        """Calculate statistics about shift-boundary alignment"""
        stats = {
            'dialogue_shifts': {'total': 0, 'aligned': 0},
            'speaker_shifts': {'total': 0, 'aligned': 0}
        }
        
        # Count dialogue shifts
        dialogue_shifts = shifts_df[shifts_df['dialogue_shift'] > 0]['TR'].values
        stats['dialogue_shifts']['total'] = len(dialogue_shifts)
        
        # Count speaker shifts
        speaker_shifts = shifts_df[shifts_df['speaker_shift'] > 0]['TR'].values
        stats['speaker_shifts']['total'] = len(speaker_shifts)
        
        # Count alignments
        for peak in event_peaks:
            # Check dialogue shifts
            aligned_dialogue = np.any(np.abs(dialogue_shifts - peak) <= window_size)
            if aligned_dialogue:
                stats['dialogue_shifts']['aligned'] += 1
                
            # Check speaker shifts
            aligned_speaker = np.any(np.abs(speaker_shifts - peak) <= window_size)
            if aligned_speaker:
                stats['speaker_shifts']['aligned'] += 1
        
        # Calculate percentages
        for shift_type in stats:
            stats[shift_type]['percent_aligned'] = (
                stats[shift_type]['aligned'] / len(event_peaks) * 100
                if len(event_peaks) > 0 else 0
            )
        
        return stats

    def _aggregate_features_by_tr(self, window_size=3):
        """
        Aggregate content features by TR using sliding window.
        
        Args:
            window_size (int): Size of sliding window in TRs
        
        Returns:
            DataFrame with aggregated features by TR
        """
        tr_features = []
        
        for tr in range(14, 466):  # TR range from your script
            # Get window of content
            mask = (self.content_df['onset_TR'] >= tr - window_size/2) & \
                   (self.content_df['onset_TR'] < tr + window_size/2)
            window_data = self.content_df[mask]
            
            # Calculate feature frequencies
            features = {'TR': tr}
            
            # POS features
            for pos, col in self.pos_features.items():
                features[f'{pos}_freq'] = window_data[col].any()
            
            # Speaker features
            for speaker, col in self.speaker_features.items():
                features[f'{speaker}_freq'] = window_data[col].mean()
            
            # Dialog feature
            features['dialog_freq'] = window_data['is_dialog'].mean()
            
            tr_features.append(features)
        
        return pd.DataFrame(tr_features)

    def analyze_temporal_patterns(self):
        """
        New method for analyzing temporal patterns
        """
        results = {}
        for response in ['event_agreement', 'affair_agreement', 'paranoia_agreement']:
            # Calculate time-lagged associations
            max_lag = 5  # TRs
            lag_correlations = {}
            
            for feature in [col for col in self.feature_response_df.columns if col.endswith('_freq')]:
                feature_series = self.feature_response_df[feature]
                response_series = self.feature_response_df[response]
                
                # Calculate cross-correlation at different lags
                lags = range(-max_lag, max_lag + 1)
                correlations = [stats.spearmanr(
                    feature_series.iloc[max(0, -lag):min(len(feature_series), len(feature_series)-lag)],
                    response_series.iloc[max(0, lag):min(len(response_series), len(response_series)+lag)]
                )[0] for lag in lags]
                
                lag_correlations[feature] = {
                    'lags': list(lags),
                    'correlations': correlations,
                    'optimal_lag': lags[np.argmax(np.abs(correlations))]
                }
                
            results[response] = lag_correlations
        
        return results

    def analyze_feature_response_relationship(self):
        """
        Analyze relationship between content features and behavioral responses.
        """
        # Merge features with responses
        self.feature_response_df = self._merge_features_responses()
        
        # Calculate standard correlations
        self.correlations = self._calculate_correlations()
        
        # Analyze peak alignment
        self.peak_alignment = self._analyze_peak_alignment()
        
        # Add temporal pattern analysis
        self.temporal_patterns = self.analyze_temporal_patterns()
        
        return {
            'correlations': self.correlations,
            'peak_alignment': self.peak_alignment,
            'temporal_patterns': self.temporal_patterns
        }
    
    def _merge_features_responses(self):
        """
        Merge content features with behavioral responses.
        """
        merged_df = self.tr_features.copy()
        
        # Add behavioral responses
        merged_df['event_agreement'] = merged_df['TR'].map(
            self.eventseg_agreement.set_index('TR')['agreement']
        )
        merged_df['affair_agreement'] = merged_df['TR'].map(
            self.evidence_affair.set_index('TR')['agreement']
        )
        merged_df['paranoia_agreement'] = merged_df['TR'].map(
            self.evidence_paranoia.set_index('TR')['agreement']
        )
        
        return merged_df
    
    def _calculate_correlations(self):
        """
        Modified correlation analysis with proper type handling
        """
        response_cols = ['event_agreement', 'affair_agreement', 'paranoia_agreement']
        feature_cols = [col for col in self.feature_response_df.columns 
                    if col.endswith('_freq')]
        
        # Calculate shifts for event agreement analysis
        shifts_df = pd.DataFrame()
        shifts_df['TR'] = self.feature_response_df['TR']
        
        for feature in feature_cols:
            base_name = feature.replace('_freq', '')
            # Ensure numeric type when calculating shifts
            shifts_df[f'{base_name}_shift'] = pd.to_numeric(
                self.feature_response_df[feature], errors='coerce'
            ).diff().abs()
        
        # Merge shifts with original data
        merged_df = pd.concat([self.feature_response_df, shifts_df.set_index('TR')], axis=1)
        
        correlations = {}
        for response in response_cols:
            corr_values = {}
            
            # Choose features based on response type
            if response == 'event_agreement':
                features_to_use = [col for col in shifts_df.columns if col.endswith('_shift')]
            else:
                features_to_use = feature_cols
            
            for feature in features_to_use:
                # Convert both series to numeric, handling any non-numeric values
                feature_series = pd.to_numeric(merged_df[feature], errors='coerce').fillna(0)
                response_series = pd.to_numeric(merged_df[response], errors='coerce').fillna(0)
                
                if feature_series.nunique() <= 2:
                    try:
                        correlation, p_value = stats.pointbiserialr(
                            feature_series.astype(float),
                            response_series.astype(float)
                        )
                    except (ValueError, TypeError):
                        correlation, p_value = np.nan, np.nan
                else:
                    try:
                        correlation, p_value = stats.spearmanr(
                            feature_series.astype(float),
                            response_series.astype(float)
                        )
                    except (ValueError, TypeError):
                        correlation, p_value = np.nan, np.nan
                
                corr_values[feature] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'method': 'pointbiserial' if feature_series.nunique() <= 2 else 'spearman'
                }
            correlations[response] = corr_values
        
        return correlations
    
    def _analyze_peak_alignment(self, threshold=0.8):
        """
        Analyze alignment between feature peaks and response peaks.
        
        Args:
            threshold (float): Threshold for peak detection
        """
        alignment = {}
        
        # Get peaks for each response type
        for response in ['event_agreement', 'affair_agreement', 'paranoia_agreement']:
            if response == 'event_agreement':
                prominence = 0.23
                width = 1
            else:
                prominence = 0.1
                width = 1
            response_peaks = self._get_peaks(
                self.feature_response_df[response],
                prominence=prominence,
                width=width
            )
            
            # Check feature values around peaks
            feature_peaks = {}
            for feature in [col for col in self.feature_response_df.columns 
                          if col.endswith('_freq')]:
                feature_values = self._get_feature_values_at_peaks(
                    self.feature_response_df[feature],
                    response_peaks
                )
                feature_peaks[feature] = feature_values
                
            alignment[response] = feature_peaks
            
        return alignment
    
    def _get_peaks(self, series, prominence=0.1, width=1):
        """
        Improved peak detection using scipy.signal
        """
        try:
            clean_series = series.fillna(0).values
            
            # Use scipy.signal.find_peaks with multiple criteria
            peaks, properties = signal.find_peaks(
                clean_series,
                prominence=prominence,            # Minimum prominence of peaks
                width=width                    # Minimum peak width
            )
            
            self.logger.info(f"Found {len(peaks)} peaks with properties:")
            self.logger.info(f"Peak prominences: {properties['prominences']}")
            
            return series.index[peaks]
            
        except Exception as e:
            self.logger.error(f"Error finding peaks: {str(e)}")
            return []
    
    def _get_feature_values_at_peaks(self, feature_series, peak_indices, window=1):
        """
        Get feature values around peak locations with error handling.
        
        Args:
            feature_series (pd.Series): Feature time series
            peak_indices (list): List of peak indices
            window (int): Window size around peaks
        
        Returns:
            list: Feature values at peaks
        """
        try:
            values = []
            for peak in peak_indices:
                # Get window of values around peak
                window_mask = (feature_series.index >= peak - window) & \
                            (feature_series.index <= peak + window)
                window_values = feature_series[window_mask]
                
                if len(window_values) > 0:
                    values.append(window_values.mean())
            
            return values
        except Exception as e:
            self.logger.error(f"Error getting feature values: {str(e)}")
            return []
    
    def _plot_peak_alignment(self, output_dir):
        """
        Create visualization of peak alignment analysis.
        
        Args:
            output_dir (Path): Directory to save the plot
        """
        # Get unique response types and features
        response_types = ['event_agreement', 'affair_agreement', 'paranoia_agreement']
        feature_types = [col for col in self.feature_response_df.columns 
                        if col.endswith('_freq')]
        
        # Create subplots for each response type
        fig, axes = plt.subplots(len(response_types), 1, figsize=(15, 5*len(response_types)))
        if len(response_types) == 1:
            axes = [axes]
        
        for idx, response in enumerate(response_types):
            ax = axes[idx]
            
            try:
                # Plot response time series
                response_series = self.feature_response_df[response].fillna(0)
                ax.plot(self.feature_response_df['TR'], 
                    response_series,
                    label=response, color='black', alpha=0.5)
                if response == 'event_agreement':
                    # Get peaks for this response type
                    response_peaks = self._get_peaks(response_series, prominence=0.23, width=1)

                else:
                    response_peaks = self._get_peaks(response_series, prominence=0.1, width=1)
                    
                if len(response_peaks) > 0:
                    # Plot vertical lines at peak locations
                    for peak in response_peaks:
                        ax.axvline(x=peak, color='gray', linestyle='--', alpha=0.3)
                    
                    # Plot feature values at peaks
                    for feature in feature_types:
                        feature_series = self.feature_response_df[feature].fillna(0)
                        peak_values = []
                        
                        # Get feature values at each peak
                        for peak in response_peaks:
                            # Get window around peak
                            window_mask = (self.feature_response_df['TR'] >= peak - 1) & \
                                        (self.feature_response_df['TR'] <= peak + 1)
                            window_values = feature_series[window_mask]
                            if len(window_values) > 0:
                                peak_values.append(window_values.mean())
                        
                        if len(peak_values) > 0:
                            # Make sure response_peaks and peak_values are the same length
                            x_values = response_peaks[:len(peak_values)]
                            y_values = peak_values[:len(x_values)]
                            
                            ax.scatter(x_values, y_values, 
                                    label=feature.replace('_freq', ''),
                                    alpha=0.7)
                
                ax.set_title(f'Peak Alignment - {response}')
                ax.set_xlabel('TR')
                ax.set_ylabel('Value')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            except Exception as e:
                self.logger.error(f"Error plotting {response}: {str(e)}")
                continue
        
        plt.tight_layout()
        try:
            plt.savefig(output_dir / 'peak_alignment.png', bbox_inches='tight')
        except Exception as e:
            self.logger.error(f"Error saving plot: {str(e)}")
        finally:
            plt.close()

    def plot_results(self):
        """
        Create visualizations of the analysis results.
        """
        output_dir = Path(self.output_dir / "02_behav_content_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Existing plots
        self._plot_correlation_heatmap(output_dir)
        self._plot_feature_response_timeseries(output_dir)
        self._plot_peak_alignment(output_dir)
        
        # Add temporal patterns visualization
        self._plot_temporal_patterns(output_dir)

    def _plot_temporal_patterns(self, output_dir):
        """
        Create visualization of temporal pattern analysis.
        """
        response_types = ['event_agreement', 'affair_agreement', 'paranoia_agreement']
        feature_types = [col for col in self.feature_response_df.columns 
                        if col.endswith('_freq')]
        
        fig, axes = plt.subplots(len(response_types), 1, 
                                figsize=(15, 5*len(response_types)))
        if len(response_types) == 1:
            axes = [axes]
        
        for idx, response in enumerate(response_types):
            ax = axes[idx]
            temporal_data = self.temporal_patterns[response]
            
            for feature in feature_types:
                data = temporal_data[feature]
                ax.plot(data['lags'], data['correlations'], 
                    label=feature.replace('_freq', ''),
                    marker='o')
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'Time-Lagged Correlations - {response}')
            ax.set_xlabel('Lag (TRs)')
            ax.set_ylabel('Correlation Coefficient')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_patterns.png', bbox_inches='tight')
        plt.close()
    
    def explore_agreement_distributions(self):
        """
        Explore the distributions of agreement values for each response type.
        """
        response_types = ['event_agreement', 'affair_agreement', 'paranoia_agreement']
        
        for response in response_types:
            data = self.feature_response_df[response].dropna()
            self.logger.info(f"\nDistribution for {response}:")
            self.logger.info(f"Mean: {data.mean():.3f}")
            self.logger.info(f"Median: {data.median():.3f}")
            self.logger.info(f"Max: {data.max():.3f}")
            self.logger.info(f"75th percentile: {data.quantile(0.75):.3f}")
            self.logger.info(f"90th percentile: {data.quantile(0.90):.3f}")
            
            # Calculate suggested threshold
            suggested_threshold = data.quantile(0.75)
            self.logger.info(f"Suggested threshold (75th percentile): {suggested_threshold:.3f}")
            
    def _plot_correlation_heatmap(self, output_dir):
        """Create correlation heatmap with appropriate features per response type"""
        response_types = ['event_agreement', 'affair_agreement', 'paranoia_agreement']
        
        # Get base features
        state_features = [col for col in self.feature_response_df.columns 
                        if col.endswith('_freq')]
        shift_features = [f"{col.replace('_freq', '_shift')}" for col in state_features]
        
        # Create feature lists for each response type
        feature_lists = {
            'event_agreement': shift_features,
            'affair_agreement': state_features,
            'paranoia_agreement': state_features
        }
        
        # Get all unique features
        all_features = list(dict.fromkeys(shift_features + state_features))
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(all_features), len(response_types)))
        corr_matrix[:] = np.nan  # Fill with NaN for features we don't want to show
        
        for i, feature in enumerate(all_features):
            for j, response in enumerate(response_types):
                if feature in feature_lists[response]:
                    corr_matrix[i, j] = self.correlations[response][feature]['correlation']
        
        plt.figure(figsize=(12, 15))
        sns.heatmap(corr_matrix, 
                    xticklabels=response_types,
                    yticklabels=all_features,
                    cmap='RdBu_r',
                    center=0,
                    mask=np.isnan(corr_matrix))  # Mask NaN values
        plt.title('Feature-Response Correlations\n(Shifts for Event Agreement, States for Others)')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png')
        plt.close()
    
    def _plot_feature_response_timeseries(self, output_dir):
        """Create time series plots"""
        response_types = ['event_agreement', 'affair_agreement', 'paranoia_agreement']
        feature_groups = {
            'POS': [col for col in self.feature_response_df.columns 
                   if any(pos in col for pos in ['verb', 'noun', 'adj', 'adv'])],
            'Speakers': [col for col in self.feature_response_df.columns 
                        if any(spk in col for spk in ['arthur', 'lee', 'girl'])]
        }
        
        for group_name, features in feature_groups.items():
            plt.figure(figsize=(15, 8))
            
            # Plot features
            for feature in features:
                plt.plot(self.feature_response_df['TR'], 
                        self.feature_response_df[feature],
                        label=feature, alpha=0.5)
            
            # Plot responses
            for response in response_types:
                plt.plot(self.feature_response_df['TR'],
                        self.feature_response_df[response],
                        label=response, linestyle='--')
            
            plt.title(f'{group_name} Features and Responses Over Time')
            plt.xlabel('TR')
            plt.ylabel('Frequency/Agreement')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_dir / f'{group_name.lower()}_timeseries.png')
            plt.close()

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set up paths (keeping existing setup)
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, "data", "stimuli")
    output_dir = os.path.join(scratch_dir, "output", "behav_results")
    
    # Initialize analyzer
    analyzer = ContentBehaviorAnalyzer(data_dir, output_dir)
    
    # Load and process data (keeping existing steps)
    analyzer.load_data()
    analyzer.preprocess_content_features()
    
    # Create output directory for event boundary analysis
    event_boundary_dir = Path(output_dir) / "03_event_boundary_analysis"
    event_boundary_dir.mkdir(exist_ok=True)
    
    # Run existing analysis
    results = analyzer.analyze_feature_response_relationship()
    
    # Run new event boundary analysis and create visualizations
    analyzer._plot_event_boundary_analysis(event_boundary_dir)
    
    # Save all results
    with open(os.path.join(output_dir, "02_behav_content_analysis", 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()