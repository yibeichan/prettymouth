import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from scipy.interpolate import interp1d

def extract_dimensional_emotions(audio_file, window_size=5.0, hop_length=2.5, sample_rate=16000):
    """
    Extract emotional dimensions (arousal, valence, dominance) from audio using 
    the audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim model.
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    window_size : float
        Size of the sliding window in seconds
    hop_length : float
        Hop length between windows in seconds
    sample_rate : int
        Sample rate for audio processing (model expects 16kHz)
        
    Returns:
    --------
    emotions_df : pandas.DataFrame
        DataFrame with time and emotional dimensions
    """
    # Load the feature extractor and model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    )
    model = AutoModel.from_pretrained(
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    )
    
    # Load and resample audio if needed
    y, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Calculate window and hop length in samples
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_length * sample_rate)
    
    # Initialize lists for results
    times = []
    arousal_values = []
    valence_values = []
    dominance_values = []
    
    # Process audio in windows
    for i in range(0, len(y) - window_samples + 1, hop_samples):
        # Extract window
        window = y[i:i + window_samples]
        
        # Calculate center time of the window
        center_time = (i + window_samples/2) / sample_rate
        times.append(center_time)
        
        # Process with model
        inputs = feature_extractor(window, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract emotion dimensions
        # The model outputs are in the format [batch_size, sequence_length, 3]
        # where the last dimension contains [arousal, valence, dominance]
        emotion_preds = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # The model is trained to output values in range [-1, 1] for each dimension
        arousal_values.append(emotion_preds[0])    # First dimension is arousal
        valence_values.append(emotion_preds[1])    # Second dimension is valence
        dominance_values.append(emotion_preds[2])  # Third dimension is dominance
    
    # Create DataFrame
    emotions_df = pd.DataFrame({
        'time': times,
        'arousal': arousal_values,
        'valence': valence_values,
        'dominance': dominance_values
    })
    
    return emotions_df

def align_emotions_with_word_onsets(emotions_df, word_onset_df):
    """
    Align emotional dimensions with word onsets
    
    Parameters:
    -----------
    emotions_df : pandas.DataFrame
        DataFrame with times and emotional dimensions
    word_onset_df : pandas.DataFrame
        DataFrame with word onsets from word_onset_tagged_zz.csv
        
    Returns:
    --------
    aligned_df : pandas.DataFrame
        Word-level DataFrame with emotional dimensions
    """
    # Create a copy of the word onset dataframe
    aligned_df = word_onset_df.copy()
    
    # Create interpolation functions for each emotional dimension
    arousal_interp = interp1d(
        emotions_df['time'], 
        emotions_df['arousal'],
        bounds_error=False,
        fill_value=(emotions_df['arousal'].iloc[0], emotions_df['arousal'].iloc[-1])
    )
    
    valence_interp = interp1d(
        emotions_df['time'], 
        emotions_df['valence'],
        bounds_error=False,
        fill_value=(emotions_df['valence'].iloc[0], emotions_df['valence'].iloc[-1])
    )
    
    dominance_interp = interp1d(
        emotions_df['time'], 
        emotions_df['dominance'],
        bounds_error=False,
        fill_value=(emotions_df['dominance'].iloc[0], emotions_df['dominance'].iloc[-1])
    )
    
    # Interpolate values for each word onset time
    word_times = aligned_df['onset'].values
    aligned_df['arousal'] = arousal_interp(word_times)
    aligned_df['valence'] = valence_interp(word_times)
    aligned_df['dominance'] = dominance_interp(word_times)
    
    return aligned_df

def align_emotions_with_fmri(emotions_df, tr=1.5, n_timepoints=475):
    """
    Align emotional dimensions with fMRI TRs
    
    Parameters:
    -----------
    emotions_df : pandas.DataFrame
        DataFrame with times and emotional dimensions
    tr : float
        TR (repetition time) in seconds
    n_timepoints : int
        Number of TRs in the fMRI data
        
    Returns:
    --------
    tr_emotions : pandas.DataFrame
        DataFrame with emotional dimensions at each TR
    """
    # Create array of TR timepoints
    tr_times = np.arange(n_timepoints) * tr
    
    # Create interpolation functions for each dimension
    arousal_interp = interp1d(
        emotions_df['time'], 
        emotions_df['arousal'],
        bounds_error=False,
        fill_value=(emotions_df['arousal'].iloc[0], emotions_df['arousal'].iloc[-1])
    )
    
    valence_interp = interp1d(
        emotions_df['time'], 
        emotions_df['valence'],
        bounds_error=False,
        fill_value=(emotions_df['valence'].iloc[0], emotions_df['valence'].iloc[-1])
    )
    
    dominance_interp = interp1d(
        emotions_df['time'], 
        emotions_df['dominance'],
        bounds_error=False,
        fill_value=(emotions_df['dominance'].iloc[0], emotions_df['dominance'].iloc[-1])
    )
    
    # Interpolate values for each TR
    tr_emotions = pd.DataFrame({
        'TR': np.arange(n_timepoints),
        'time': tr_times,
        'arousal': arousal_interp(tr_times),
        'valence': valence_interp(tr_times),
        'dominance': dominance_interp(tr_times)
    })
    
    return tr_emotions

def analyze_emotions_by_speaker(emotions_df, segments_df):
    """
    Analyze emotional dimensions by speaker
    
    Parameters:
    -----------
    emotions_df : pandas.DataFrame
        DataFrame with times and emotional dimensions
    segments_df : pandas.DataFrame
        DataFrame with segment information from segments_speaker.csv
        
    Returns:
    --------
    speaker_emotions : pandas.DataFrame
        DataFrame with aggregated emotional dimensions by speaker
    """
    # Create a list to store segment-level data
    segment_data = []
    
    # Process each segment
    for _, segment in segments_df.iterrows():
        # Get start time of segment
        start_time = segment['onset_sec']
        
        # If this is not the last segment, get end time from next segment
        if _ < len(segments_df) - 1:
            end_time = segments_df.iloc[_ + 1]['onset_sec']
        else:
            # For the last segment, use the last time in emotions_df
            end_time = emotions_df['time'].max()
        
        # Get emotions data for this segment
        segment_emotions = emotions_df[
            (emotions_df['time'] >= start_time) & 
            (emotions_df['time'] < end_time)
        ]
        
        # Calculate mean emotional dimensions for this segment
        if len(segment_emotions) > 0:
            segment_data.append({
                'segment': segment['Seg'],
                'speaker': segment['speaker'],
                'actor': segment['actor'],
                'main_char': segment['main_char'],
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'mean_arousal': segment_emotions['arousal'].mean(),
                'std_arousal': segment_emotions['arousal'].std(),
                'mean_valence': segment_emotions['valence'].mean(),
                'std_valence': segment_emotions['valence'].std(),
                'mean_dominance': segment_emotions['dominance'].mean(),
                'std_dominance': segment_emotions['dominance'].std()
            })
    
    # Create DataFrame with segment-level data
    segments_with_emotions = pd.DataFrame(segment_data)
    
    # Aggregate by speaker
    speaker_emotions = segments_with_emotions.groupby('speaker').agg({
        'mean_arousal': 'mean',
        'std_arousal': 'mean',
        'mean_valence': 'mean',
        'std_valence': 'mean',
        'mean_dominance': 'mean',
        'std_dominance': 'mean',
        'duration': 'sum'
    }).reset_index()
    
    return speaker_emotions, segments_with_emotions

def plot_emotion_timelines(emotions_df, segments_df=None, word_df=None, tr_emotions=None):
    """
    Create visualizations of emotional dimensions over time
    
    Parameters:
    -----------
    emotions_df : pandas.DataFrame
        DataFrame with times and emotional dimensions
    segments_df : pandas.DataFrame
        Optional DataFrame with segment information
    word_df : pandas.DataFrame
        Optional DataFrame with word-level emotions
    tr_emotions : pandas.DataFrame
        Optional DataFrame with TR-level emotions
        
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    figs = []
    
    # 1. Create timeline of emotional dimensions
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Plot arousal
    axs[0].plot(emotions_df['time'], emotions_df['arousal'], 'r-', linewidth=2)
    axs[0].set_ylabel('Arousal')
    axs[0].set_title('Emotional Arousal Timeline')
    axs[0].grid(True, alpha=0.3)
    
    # Plot valence
    axs[1].plot(emotions_df['time'], emotions_df['valence'], 'g-', linewidth=2)
    axs[1].set_ylabel('Valence')
    axs[1].set_title('Emotional Valence Timeline')
    axs[1].grid(True, alpha=0.3)
    
    # Plot dominance
    axs[2].plot(emotions_df['time'], emotions_df['dominance'], 'b-', linewidth=2)
    axs[2].set_ylabel('Dominance')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_title('Emotional Dominance Timeline')
    axs[2].grid(True, alpha=0.3)
    
    # Add segment boundaries if provided
    if segments_df is not None:
        for _, segment in segments_df.iterrows():
            for ax in axs:
                ax.axvline(x=segment['onset_sec'], color='k', linestyle='--', alpha=0.5)
                # Add speaker annotation
                ax.text(
                    segment['onset_sec'], 
                    ax.get_ylim()[1] * 0.9,
                    segment['speaker'],
                    rotation=90,
                    fontsize=8
                )
    
    plt.tight_layout()
    figs.append(fig)
    
    # 2. Create speaker comparison plot if segments provided
    if segments_df is not None:
        speaker_emotions, _ = analyze_emotions_by_speaker(emotions_df, segments_df)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sort speakers by arousal
        speaker_emotions = speaker_emotions.sort_values('mean_arousal', ascending=False)
        
        # Plot arousal by speaker
        axs[0].bar(speaker_emotions['speaker'], speaker_emotions['mean_arousal'], yerr=speaker_emotions['std_arousal'])
        axs[0].set_ylabel('Mean Arousal')
        axs[0].set_title('Arousal by Speaker')
        axs[0].tick_params(axis='x', rotation=45)
        
        # Plot valence by speaker
        axs[1].bar(speaker_emotions['speaker'], speaker_emotions['mean_valence'], yerr=speaker_emotions['std_valence'])
        axs[1].set_ylabel('Mean Valence')
        axs[1].set_title('Valence by Speaker')
        axs[1].tick_params(axis='x', rotation=45)
        
        # Plot dominance by speaker
        axs[2].bar(speaker_emotions['speaker'], speaker_emotions['mean_dominance'], yerr=speaker_emotions['std_dominance'])
        axs[2].set_ylabel('Mean Dominance')
        axs[2].set_title('Dominance by Speaker')
        axs[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        figs.append(fig)
    
    # 3. Create TR-aligned plot if provided
    if tr_emotions is not None:
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Plot arousal
        axs[0].plot(tr_emotions['TR'], tr_emotions['arousal'], 'r-', linewidth=2)
        axs[0].set_ylabel('Arousal')
        axs[0].set_title('TR-Aligned Emotional Arousal')
        axs[0].grid(True, alpha=0.3)
        
        # Plot valence
        axs[1].plot(tr_emotions['TR'], tr_emotions['valence'], 'g-', linewidth=2)
        axs[1].set_ylabel('Valence')
        axs[1].set_title('TR-Aligned Emotional Valence')
        axs[1].grid(True, alpha=0.3)
        
        # Plot dominance
        axs[2].plot(tr_emotions['TR'], tr_emotions['dominance'], 'b-', linewidth=2)
        axs[2].set_ylabel('Dominance')
        axs[2].set_xlabel('TR')
        axs[2].set_title('TR-Aligned Emotional Dominance')
        axs[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        figs.append(fig)
    
    return figs

def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    scratch_dir = os.getenv('SCRATCH_DIR')
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    output_dir = os.path.join(scratch_dir, 'data', 'audio_arousal')

    # 1. Extract emotional dimensions
    emotions_df = extract_dimensional_emotions(os.path.join(data_dir, "reduced_audio.mp3"), window_size=3.0, hop_length=1.0)

    # 2. Load word onset and segment data
    word_df = pd.read_csv(os.path.join(data_dir, "word_onset_tagged_zz.csv"))
    segments_df = pd.read_csv(os.path.join(data_dir, "segments_speaker.csv"))

    # 3. Align with words and TRs
    word_emotions_df = align_emotions_with_word_onsets(emotions_df, word_df)
    tr_emotions_df = align_emotions_with_fmri(emotions_df, tr=1.5, n_timepoints=475)

    # 4. Analyze by speaker
    speaker_emotions, segment_emotions = analyze_emotions_by_speaker(emotions_df, segments_df)

    # 5. Create visualizations
    figs = plot_emotion_timelines(emotions_df, segments_df, word_emotions_df, tr_emotions_df)
    for i, fig in enumerate(figs):
        fig.savefig(f"emotion_dimensions_fig_{i+1}.png", dpi=300)

    # 6. Save data
    emotions_df.to_csv(os.path.join(output_dir, "emotions_df.csv"), index=False)
    word_emotions_df.to_csv(os.path.join(output_dir, "word_emotions_df.csv"), index=False)
    tr_emotions_df.to_csv(os.path.join(output_dir, "tr_emotions_df.csv"), index=False)
    

if __name__ == "__main__":
    main()