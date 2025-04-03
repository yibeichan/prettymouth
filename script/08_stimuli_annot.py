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
    
    # Calculate timing offset
    first_word_onset = words_df['onset'].min()
    words_df['adjusted_onset'] = words_df['onset'] - first_word_onset
    
    return segments_df, words_df

def find_segment(word_onset: float, segments_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Find which segment a word belongs to based on its onset time.
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
    Create annotations with basic features.
    """
    annotations = []
    prev_segment = None
    
    for _, word_row in words_df.iterrows():
        current_segment = find_segment(word_row['adjusted_onset'], segments_df)
        if current_segment is None:
            continue
            
        # Create annotation
        annotation = {
            # Timing
            'onset': word_row['onset'],
            'onset_TR': word_row['onset_TR'],
            'adjusted_onset': word_row['adjusted_onset'],
            
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
            'segment_text': current_segment['Seg'],
            'speaker': current_segment['speaker'],
            'main_char': current_segment['main_char']
        }
        
        annotations.append(annotation)
        prev_segment = current_segment
    
    return pd.DataFrame(annotations)

def create_tr_annotations(story_df: pd.DataFrame, n_brain_trs: int, 
                         force_trim: bool = False) -> pd.DataFrame:
    """
    Create TR-level annotations from word-level story annotations using majority vote
    
    Parameters:
    -----------
    story_df : pd.DataFrame
        Word-level story annotations with onset_TR and features
    n_brain_trs : int
        Number of TRs in the brain data
    force_trim : bool, optional
        Whether to force trimming if there's a large timing mismatch, by default False
        
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
            'is_dialog': tr_words['is_dialog'].mean() > 0.5 if len(tr_words) > 0 else False,
            'arthur_speaking': tr_words['arthur_speaking'].mean() > 0.5 if len(tr_words) > 0 else False,
            'lee_speaking': tr_words['lee_speaking'].mean() > 0.5 if len(tr_words) > 0 else False,
            'girl_speaking': tr_words['girl_speaking'].mean() > 0.5 if len(tr_words) > 0 else False,
            'lee_girl_together': tr_words['lee_girl_together'].mean() > 0.5 if len(tr_words) > 0 else False,
            
            # Store word lists
            'words': tr_words['text'].tolist(),
            'verbs': tr_words[tr_words['is_verb']]['text'].tolist(),
            'nouns': tr_words[tr_words['is_noun']]['text'].tolist(),
            'descriptors': tr_words[(tr_words['is_adj'] | tr_words['is_adv'])]['text'].tolist(),
            
            # Count features
            'n_words': len(tr_words),
            'n_verbs': len(tr_words[tr_words['is_verb']]),
            'n_nouns': len(tr_words[tr_words['is_noun']]),
            'n_adjectives': len(tr_words[tr_words['is_adj']]),
            'n_adverbs': len(tr_words[tr_words['is_adv']]),
            'n_descriptors': len(tr_words[(tr_words['is_adj'] | tr_words['is_adv'])]),
            
            # Binary presence indicators
            'has_verb': len(tr_words[tr_words['is_verb']]) > 0,
            'has_noun': len(tr_words[tr_words['is_noun']]) > 0,
            'has_adj': len(tr_words[tr_words['is_adj']]) > 0,
            'has_adv': len(tr_words[tr_words['is_adv']]) > 0,
            
            # Segment information
            'segment_onset': tr_words['segment_onset'].iloc[0] if len(tr_words) > 0 else None,
            'segment_text': tr_words['segment_text'].iloc[0] if len(tr_words) > 0 else None,
        }
        
        # Handle categorical features with robust mode calculation
        if len(tr_words) > 0:
            # Handle speaker mode
            speaker_mode = tr_words['speaker'].mode()
            tr_data['speaker'] = speaker_mode.iloc[0] if not speaker_mode.empty else 'none'
            
            # Handle main_char mode
            main_char_mode = tr_words['main_char'].mode()
            tr_data['main_char'] = main_char_mode.iloc[0] if not main_char_mode.empty else 'none'
        else:
            tr_data['speaker'] = 'none'
            tr_data['main_char'] = 'none'
        tr_annotations.append(tr_data)
    
    return pd.DataFrame(tr_annotations)

def main():
    load_dotenv()
    scratch_dir = os.getenv('SCRATCH_DIR')
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    
    # Load data
    segments_df, words_df = load_data(os.path.join(data_dir, 'segments_speaker.csv'), 
                                      os.path.join(data_dir, 'word_onset_tagged_zz.csv'))
    
    # Create word-level annotations
    annotations_df = create_annotations(segments_df, words_df)
    
    # Save word-level annotations
    annotations_df.to_csv(os.path.join(data_dir, '10_story_annotations_word.csv'), index=False)
    
    # Create TR-level annotations
    n_brain_trs = 451  # matching HMM results
    
    try:
        # First attempt without forcing trim
        tr_annotations_df = create_tr_annotations(annotations_df, n_brain_trs, force_trim=False)
        print("\nSuccessfully created TR-level annotations with natural alignment.")
    except ValueError as e:
        print(f"\nWarning: {e}")
        print("Proceeding with forced trimming...")
        tr_annotations_df = create_tr_annotations(annotations_df, n_brain_trs, force_trim=True)
        print("Successfully created TR-level annotations with forced trimming.")
    
    # Save TR-level annotations
    tr_annotations_df.to_csv(os.path.join(data_dir, '10_story_annotations_tr.csv'), index=False)
    
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
    
    # Print TR-level statistics
    print("\nTR-Level Annotation Statistics:")
    print(f"Total TRs: {len(tr_annotations_df)}")
    print(f"Dialog TRs: {tr_annotations_df['is_dialog'].sum()}")
    print(f"Arthur speaking TRs: {tr_annotations_df['arthur_speaking'].sum()}")
    print(f"Lee speaking TRs: {tr_annotations_df['lee_speaking'].sum()}")
    print(f"Girl speaking TRs: {tr_annotations_df['girl_speaking'].sum()}")
    print(f"Lee-girl interaction TRs: {tr_annotations_df['lee_girl_together'].sum()}")
    print(f"Average words per TR: {tr_annotations_df['n_words'].mean():.2f}")
    
    # Print POS statistics for TRs
    print("\nPart of Speech statistics for TRs:")
    print(f"TRs with verbs: {tr_annotations_df['has_verb'].sum()} ({tr_annotations_df['has_verb'].mean()*100:.1f}%)")
    print(f"TRs with nouns: {tr_annotations_df['has_noun'].sum()} ({tr_annotations_df['has_noun'].mean()*100:.1f}%)")
    print(f"TRs with adjectives: {tr_annotations_df['has_adj'].sum()} ({tr_annotations_df['has_adj'].mean()*100:.1f}%)")
    print(f"TRs with adverbs: {tr_annotations_df['has_adv'].sum()} ({tr_annotations_df['has_adv'].mean()*100:.1f}%)")
    
    # Print distribution of speakers across TRs
    print("\nSpeaker distribution across TRs:")
    print(tr_annotations_df['speaker'].value_counts())
    
    # Print distribution of main characters across TRs
    print("\nMain character distribution across TRs:")
    print(tr_annotations_df['main_char'].value_counts())

if __name__ == "__main__":
    main()