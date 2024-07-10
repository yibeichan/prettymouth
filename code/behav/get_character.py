# ignore warnings 
import warnings
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import ast

def main(stimuli_dir):
    """Create binary time series arrays for two (sets of) characters: Lee+Girl and Arthur."""
    speaker_df = pd.read_csv(os.path.join(stimuli_dir, "segments_speaker.csv"), encoding="latin-1")
    
    # Define the TR duration
    TR = 1.5

    # Calculate the number of TRs
    max_time = 676
    n_TRs = int(np.round(max_time / TR))

    # Initialize the binary time series arrays
    lee_girl_series_TR = np.zeros(n_TRs, dtype=int)
    arthur_series_TR = np.zeros(n_TRs, dtype=int)

    for i, row in speaker_df.iterrows():
        start = int(np.round(row['onset_sec']/TR))
        if i < len(speaker_df) - 1:
            end = int(np.round(speaker_df.iloc[i + 1]['onset_sec']/TR))
        else:
            end = n_TRs  # For the last segment

        main_char = str(row['main_char']) if not pd.isna(row['main_char']) else ''

        if 'lee' in main_char or 'girl' in main_char:
            lee_girl_series_TR[start:end] = 1

        if 'arthur' in main_char:
            arthur_series_TR[start:end] = 1

    # save the binary time series arrays
    np.save(os.path.join(stimuli_dir, "lee_girl_series_TR.npy"), lee_girl_series_TR)
    np.save(os.path.join(stimuli_dir, "arthur_series_TR.npy"), arthur_series_TR)

if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(base_dir, "data")
    stimuli_dir = os.path.join(data_dir, "stimuli")

    main(stimuli_dir)
