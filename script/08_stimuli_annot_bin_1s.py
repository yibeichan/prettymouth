"""
08_stimuli_annot_bin_1s.py

Same logic as 08_stimuli_annot.py but using 1-second binning instead of TR (1.5s) binning.
This addresses reviewer concerns about linguistic features that may occur at sub-TR resolution.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

def load_data(segments_file: str, words_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the segments and words files.
    Handles different file encodings.
    """
    # Try different encodings
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

    # Load segments file
    for encoding in encodings_to_try:
        try:
            segments_df = pd.read_csv(segments_file, encoding=encoding)
            print(f"Successfully loaded segments file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not read segments file with any of the attempted encodings: {encodings_to_try}")

    # Load words file
    for encoding in encodings_to_try:
        try:
            words_df = pd.read_csv(words_file, encoding=encoding)
            print(f"Successfully loaded words file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not read words file with any of the attempted encodings: {encodings_to_try}")

    # Calculate timing offset (onset is already in seconds)
    first_word_onset = words_df['onset'].min()
    words_df['adjusted_onset'] = words_df['onset'] - first_word_onset
    words_df['adjusted_onset_seconds'] = words_df['adjusted_onset']  # Already in seconds
    words_df['onset_seconds'] = words_df['onset']  # Already in seconds

    return segments_df, words_df

def find_segment(word_onset: float, segments_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Find which segment a word belongs to based on its onset time (in seconds).
    """
    for i in range(len(segments_df)-1):
        current_onset = segments_df.iloc[i]['onset_sec']
        next_onset = segments_df.iloc[i+1]['onset_sec']
        if current_onset <= word_onset < next_onset:
            return segments_df.iloc[i]

    if word_onset >= segments_df.iloc[-1]['onset_sec']:
        return segments_df.iloc[-1]

    return None

def create_annotations(segments_df: pd.DataFrame, words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create annotations with basic features (using seconds instead of TRs).
    """
    annotations = []
    prev_segment = None

    for _, word_row in words_df.iterrows():
        current_segment = find_segment(word_row['adjusted_onset_seconds'], segments_df)
        if current_segment is None:
            continue

        # Create annotation
        annotation = {
            # Timing in seconds
            'onset': word_row['onset'],
            'onset_seconds': word_row['onset_seconds'],
            'onset_TR': word_row.get('onset_TR', word_row['onset_seconds'] / 1.5),  # For comparison
            'adjusted_onset': word_row['adjusted_onset'],
            'adjusted_onset_seconds': word_row['adjusted_onset_seconds'],

            # Word info
            'text': word_row['text'],
            'pos': word_row['pos'],

            # Basic binary features
            'is_dialog': current_segment['speaker'] != 'narrator',
            'arthur_speaking': current_segment['speaker'] == 'arthur',
            'lee_speaking': current_segment['speaker'] == 'lee',
            'girl_speaking': current_segment['speaker'] == 'girl',
            'lee_girl_together': (
                'lee' in str(current_segment['actor']).lower() and
                'girl' in str(current_segment['actor']).lower()
            ),

            # POS features
            'is_verb': word_row['pos'] == 'VERB',
            'is_noun': word_row['pos'] == 'NOUN',
            'is_adj': word_row['pos'] == 'ADJ',
            'is_adv': word_row['pos'] == 'ADV',

            # Context
            'segment_onset': current_segment['onset_sec'],
            'segment_text': current_segment['seg'],
            'speaker': current_segment['speaker'],
            'main_char': current_segment['main_char']
        }

        annotations.append(annotation)
        prev_segment = current_segment

    return pd.DataFrame(annotations)

def create_second_annotations(story_df: pd.DataFrame, n_brain_seconds: int,
                             force_trim: bool = False) -> pd.DataFrame:
    """
    Create 1-second binned annotations from word-level story annotations using majority vote

    Parameters:
    -----------
    story_df : pd.DataFrame
        Word-level story annotations with onset_seconds and features
    n_brain_seconds : int
        Number of seconds in the brain data
    force_trim : bool, optional
        Whether to force trimming if there's a large timing mismatch, by default False

    Returns:
    --------
    pd.DataFrame
        1-second binned annotations with aggregated features
    """
    # Get the offset from first second
    second_offset = story_df['onset_seconds'].min()

    # Adjust story_max_second calculation
    story_max_second = int(np.ceil(story_df['onset_seconds'].max() - second_offset))
    second_difference = abs(story_max_second - n_brain_seconds)

    if second_difference > 15 and not force_trim:  # Threshold in seconds (15s = 10 TRs)
        raise ValueError(
            f"Large timing mismatch detected ({second_difference} seconds). "
            "Set force_trim=True to proceed anyway."
        )

    # Create annotations up to brain data length
    max_second = n_brain_seconds - 1  # -1 because 0-indexed
    second_annotations = []

    for sec in range(max_second + 1):
        # Find words that occur during this second, accounting for offset
        second_words = story_df[(story_df['onset_seconds'] - second_offset >= sec) &
                               (story_df['onset_seconds'] - second_offset < sec + 1)]

        # Create second-level features using majority vote
        second_data = {
            'second': sec,
            'is_dialog': second_words['is_dialog'].mean() > 0.5 if len(second_words) > 0 else False,
            'arthur_speaking': second_words['arthur_speaking'].mean() > 0.5 if len(second_words) > 0 else False,
            'lee_speaking': second_words['lee_speaking'].mean() > 0.5 if len(second_words) > 0 else False,
            'girl_speaking': second_words['girl_speaking'].mean() > 0.5 if len(second_words) > 0 else False,
            'lee_girl_together': second_words['lee_girl_together'].mean() > 0.5 if len(second_words) > 0 else False,

            # Store word lists
            'words': second_words['text'].tolist(),
            'verbs': second_words[second_words['is_verb']]['text'].tolist(),
            'nouns': second_words[second_words['is_noun']]['text'].tolist(),
            'descriptors': second_words[(second_words['is_adj'] | second_words['is_adv'])]['text'].tolist(),

            # Count features
            'n_words': len(second_words),
            'n_verbs': len(second_words[second_words['is_verb']]),
            'n_nouns': len(second_words[second_words['is_noun']]),
            'n_adjectives': len(second_words[second_words['is_adj']]),
            'n_adverbs': len(second_words[second_words['is_adv']]),
            'n_descriptors': len(second_words[(second_words['is_adj'] | second_words['is_adv'])]),

            # Binary presence indicators
            'has_verb': len(second_words[second_words['is_verb']]) > 0,
            'has_noun': len(second_words[second_words['is_noun']]) > 0,
            'has_adj': len(second_words[second_words['is_adj']]) > 0,
            'has_adv': len(second_words[second_words['is_adv']]) > 0,

            # Segment information
            'segment_onset': second_words['segment_onset'].iloc[0] if len(second_words) > 0 else None,
            'segment_text': second_words['segment_text'].iloc[0] if len(second_words) > 0 else None,
        }

        # Handle categorical features with robust mode calculation
        if len(second_words) > 0:
            # Handle speaker mode
            speaker_mode = second_words['speaker'].mode()
            second_data['speaker'] = speaker_mode.iloc[0] if not speaker_mode.empty else 'none'

            # Handle main_char mode
            main_char_mode = second_words['main_char'].mode()
            second_data['main_char'] = main_char_mode.iloc[0] if not main_char_mode.empty else 'none'
        else:
            second_data['speaker'] = 'none'
            second_data['main_char'] = 'none'
        second_annotations.append(second_data)

    return pd.DataFrame(second_annotations)

def main():
    load_dotenv()
    scratch_dir = os.getenv('SCRATCH_DIR')
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    output_dir = os.path.join(scratch_dir, 'output_RR', '08_stimuli_annot_bin_1s')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    segments_df, words_df = load_data(os.path.join(data_dir, 'segments_speaker.csv'),
                                      os.path.join(data_dir, 'word_onset_tagged_zz.csv'))

    # Create word-level annotations
    annotations_df = create_annotations(segments_df, words_df)

    # Save word-level annotations
    annotations_df.to_csv(os.path.join(output_dir, '10_story_annotations_word_1s.csv'), index=False)

    # Create 1-second binned annotations
    n_brain_seconds = 677  # 451 TRs * 1.5 = 676.5, rounded up

    try:
        # First attempt without forcing trim
        second_annotations_df = create_second_annotations(annotations_df, n_brain_seconds, force_trim=False)
        print("\nSuccessfully created 1-second binned annotations with natural alignment.")
    except ValueError as e:
        print(f"\nWarning: {e}")
        print("Proceeding with forced trimming...")
        second_annotations_df = create_second_annotations(annotations_df, n_brain_seconds, force_trim=True)
        print("Successfully created 1-second binned annotations with forced trimming.")

    # Save 1-second binned annotations
    second_annotations_df.to_csv(os.path.join(output_dir, '10_story_annotations_1s.csv'), index=False)

    # Print word-level statistics
    print("\nWord-Level Annotation Statistics:")
    print(f"Total words: {len(annotations_df)}")
    print(f"Dialog words: {annotations_df['is_dialog'].sum()}")
    print(f"Arthur speaking: {annotations_df['arthur_speaking'].sum()}")
    print(f"Lee speaking: {annotations_df['lee_speaking'].sum()}")
    print(f"Girl speaking: {annotations_df['girl_speaking'].sum()}")
    print(f"Lee-girl interactions: {annotations_df['lee_girl_together'].sum()}")
    print("\nPOS counts:")
    print(f"Verbs: {annotations_df['is_verb'].sum()}")
    print(f"Nouns: {annotations_df['is_noun'].sum()}")
    print(f"Adjectives: {annotations_df['is_adj'].sum()}")
    print(f"Adverbs: {annotations_df['is_adv'].sum()}")

    # Print 1-second binned statistics
    print("\n1-Second Binned Annotation Statistics:")
    print(f"Total seconds: {len(second_annotations_df)}")
    print(f"Dialog seconds: {second_annotations_df['is_dialog'].sum()}")
    print(f"Arthur speaking seconds: {second_annotations_df['arthur_speaking'].sum()}")
    print(f"Lee speaking seconds: {second_annotations_df['lee_speaking'].sum()}")
    print(f"Girl speaking seconds: {second_annotations_df['girl_speaking'].sum()}")
    print(f"Lee-girl interaction seconds: {second_annotations_df['lee_girl_together'].sum()}")
    print(f"Average words per second: {second_annotations_df['n_words'].mean():.2f}")

    # Print POS statistics for seconds
    print("\nPart of Speech statistics for 1-second bins:")
    print(f"Seconds with verbs: {second_annotations_df['has_verb'].sum()} ({second_annotations_df['has_verb'].mean()*100:.1f}%)")
    print(f"Seconds with nouns: {second_annotations_df['has_noun'].sum()} ({second_annotations_df['has_noun'].mean()*100:.1f}%)")
    print(f"Seconds with adjectives: {second_annotations_df['has_adj'].sum()} ({second_annotations_df['has_adj'].mean()*100:.1f}%)")
    print(f"Seconds with adverbs: {second_annotations_df['has_adv'].sum()} ({second_annotations_df['has_adv'].mean()*100:.1f}%)")

    # Print distribution of speakers across seconds
    print("\nSpeaker distribution across seconds:")
    print(second_annotations_df['speaker'].value_counts())

    # Print distribution of main characters across seconds
    print("\nMain character distribution across seconds:")
    print(second_annotations_df['main_char'].value_counts())

if __name__ == "__main__":
    main()