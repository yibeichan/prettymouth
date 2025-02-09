import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

def load_data(segments_file: str, words_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    
    # Load words file with word_idx
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
    
    # Calculate adjusted_onset_TR
    first_word_onset_TR = words_df['onset_TR'].min()
    words_df['adjusted_onset_TR'] = words_df['onset_TR'] - first_word_onset_TR
    
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

def consolidate_tokens_to_word(tokens_group: pd.DataFrame, segments_df: pd.DataFrame) -> Dict:
    """
    Consolidate multiple tokens into a single word annotation.
    """
    # Get the first non-punctuation token for main properties
    main_token = tokens_group[tokens_group['pos'] != 'PUNCT'].iloc[0]
    
    # Get all non-punctuation POS tags
    non_punct_tokens = tokens_group[tokens_group['pos'] != 'PUNCT']
    pos_tags = set(non_punct_tokens['pos'])
    
    # Reconstruct full word text from word_with_ws
    full_text = main_token['word_with_ws'].strip()
    
    # Get segment info
    current_segment = find_segment(main_token['adjusted_onset'], segments_df)
    
    return {
        # Timing from first token
        'onset': main_token['onset'],
        'onset_TR': main_token['onset_TR'],
        'adjusted_onset': main_token['adjusted_onset'],
        'adjusted_onset_TR': main_token['adjusted_onset_TR'],
        
        # Word info
        'text': full_text,
        'pos': main_token['pos'],  # POS of main token
        
        # Binary features from segment
        'is_dialog': current_segment['speaker'] != 'narrator',
        'arthur_speaking': current_segment['speaker'] == 'arthur',
        'lee_speaking': current_segment['speaker'] == 'lee',
        'girl_speaking': current_segment['speaker'] == 'girl',
        'lee_girl_together': (
            'lee' in str(current_segment['actor']).lower() and 
            'girl' in str(current_segment['actor']).lower()
        ),
        
        # POS features (true if any non-punctuation token has this POS)
        'is_verb': 'VERB' in pos_tags,
        'is_noun': 'NOUN' in pos_tags,
        'is_adj': 'ADJ' in pos_tags,
        'is_adv': 'ADV' in pos_tags,
        
        # Context
        'segment_onset': current_segment['onset_sec'],
        'segment_text': current_segment['Seg'],
        'speaker': current_segment['speaker'],
        'main_char': current_segment['main_char']
    }

def create_word_level_annotations(segments_df: pd.DataFrame, words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create word-level annotations by consolidating tokens.
    """
    word_annotations = []
    
    # Group by word_idx to consolidate tokens into words
    for word_idx, tokens_group in words_df.groupby('word_idx'):
        word_annotation = consolidate_tokens_to_word(tokens_group, segments_df)
        word_annotations.append(word_annotation)
    
    return pd.DataFrame(word_annotations)

def create_TR_level_annotations(word_annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create TR-level annotations by aggregating word-level annotations.
    Creates a row for every integer TR using adjusted_onset_TR.
    """
    # Get min and max TR from adjusted_onset_TR to create full range
    min_TR = int(np.floor(word_annotations_df['adjusted_onset_TR'].min()))
    max_TR = int(np.ceil(word_annotations_df['adjusted_onset_TR'].max()))
    print(f"TR range: {min_TR} to {max_TR}")
    
    all_TRs = range(min_TR, max_TR + 1)
    tr_annotations = []
    
    for tr in all_TRs:
        # Print progress every 50 TRs
        if tr % 50 == 0:
            print(f"Processing TR {tr}")
            
        # Create base annotation dictionary with default values
        tr_annotation = {
            'onset': None,
            'onset_TR': None,
            'adjusted_onset': None,
            'adjusted_onset_TR': tr,
            'text': '',
            'is_dialog': False,
            'arthur_speaking': False,
            'lee_speaking': False,
            'girl_speaking': False,
            'lee_girl_together': False,
            'has_verb': False,
            'n_verb': 0,
            'has_noun': False,
            'n_noun': 0,
            'has_adj': False,
            'n_adj': 0,
            'has_adv': False,
            'n_adv': 0,
            'segment_onset': None,
            'segment_text': '',
            'speaker': '',
            'main_char': ''
        }
        
        # Get all words in this TR using adjusted_onset_TR
        tr_words = word_annotations_df[
            (word_annotations_df['adjusted_onset_TR'] >= tr) & 
            (word_annotations_df['adjusted_onset_TR'] < tr + 1)
        ]
        
        # If we have words in this TR, update the annotation
        if not tr_words.empty:
            first_word = tr_words.iloc[0]
            total_words = len(tr_words)
            
            # Update timing information
            tr_annotation['onset'] = first_word['onset']
            tr_annotation['onset_TR'] = first_word['onset_TR']
            tr_annotation['adjusted_onset'] = first_word['adjusted_onset']
            
            # Update text
            tr_annotation['text'] = ' '.join(tr_words['text'])
            
            # Update binary features using majority rule
            tr_annotation['is_dialog'] = (tr_words['is_dialog'].sum() / total_words) > 0.5
            tr_annotation['arthur_speaking'] = (tr_words['arthur_speaking'].sum() / total_words) > 0.5
            tr_annotation['lee_speaking'] = (tr_words['lee_speaking'].sum() / total_words) > 0.5
            tr_annotation['girl_speaking'] = (tr_words['girl_speaking'].sum() / total_words) > 0.5
            tr_annotation['lee_girl_together'] = (tr_words['lee_girl_together'].sum() / total_words) > 0.5
            
            # Update POS features
            tr_annotation['has_verb'] = tr_words['is_verb'].any()
            tr_annotation['n_verb'] = tr_words['is_verb'].sum()
            tr_annotation['has_noun'] = tr_words['is_noun'].any()
            tr_annotation['n_noun'] = tr_words['is_noun'].sum()
            tr_annotation['has_adj'] = tr_words['is_adj'].any()
            tr_annotation['n_adj'] = tr_words['is_adj'].sum()
            tr_annotation['has_adv'] = tr_words['is_adv'].any()
            tr_annotation['n_adv'] = tr_words['is_adv'].sum()
            
            # Update context information
            tr_annotation['segment_onset'] = first_word['segment_onset']
            tr_annotation['segment_text'] = first_word['segment_text']
            tr_annotation['speaker'] = first_word['speaker']
            tr_annotation['main_char'] = first_word['main_char']
        
        tr_annotations.append(tr_annotation)
    
    # Create DataFrame from annotations
    result_df = pd.DataFrame(tr_annotations)
    print(f"\nCreated DataFrame with {len(result_df)} rows")
    return result_df

def main():
    load_dotenv()
    scratch_dir = os.getenv('SCRATCH_DIR')
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    
    # Load data
    segments_df, words_df = load_data(
        os.path.join(data_dir, 'segments_speaker.csv'),
        os.path.join(data_dir, 'word_onset_tagged_zz.csv')
    )
    
    # Create word-level annotations
    word_annotations_df = create_word_level_annotations(segments_df, words_df)
    
    # Create TR-level annotations
    tr_annotations_df = create_TR_level_annotations(word_annotations_df)
    
    # Save annotations
    word_annotations_df.to_csv(os.path.join(data_dir, '10_story_annotations.csv'), index=False)
    tr_annotations_df.to_csv(os.path.join(data_dir, '10_story_annotations_TR.csv'), index=False)
    
    # Print statistics for both word-level and TR-level annotations
    print("\nWord-Level Annotation Statistics:")
    print(f"Total words: {len(word_annotations_df)}")
    print(f"Dialog words: {word_annotations_df['is_dialog'].sum()}")
    print(f"Arthur speaking: {word_annotations_df['arthur_speaking'].sum()}")
    print(f"Lee speaking: {word_annotations_df['lee_speaking'].sum()}")
    print(f"Girl speaking: {word_annotations_df['girl_speaking'].sum()}")
    print(f"Lee-girl interactions: {word_annotations_df['lee_girl_together'].sum()}")
    print("\nPOS counts (word-level):")
    print(f"Verbs: {word_annotations_df['is_verb'].sum()}")
    print(f"Nouns: {word_annotations_df['is_noun'].sum()}")
    print(f"Adjectives: {word_annotations_df['is_adj'].sum()}")
    print(f"Adverbs: {word_annotations_df['is_adv'].sum()}")
    
    print("\nTR-Level Annotation Statistics:")
    print(f"Total TRs: {len(tr_annotations_df)}")
    print(f"Dialog TRs: {tr_annotations_df['is_dialog'].sum()}")
    print(f"Arthur speaking TRs: {tr_annotations_df['arthur_speaking'].sum()}")
    print(f"Lee speaking TRs: {tr_annotations_df['lee_speaking'].sum()}")
    print(f"Girl speaking TRs: {tr_annotations_df['girl_speaking'].sum()}")
    print(f"Lee-girl interaction TRs: {tr_annotations_df['lee_girl_together'].sum()}")
    print("\nPOS counts (TR-level):")
    print(f"TRs with verbs: {tr_annotations_df['has_verb'].sum()}")
    print(f"Total verbs: {tr_annotations_df['n_verb'].sum()}")
    print(f"TRs with nouns: {tr_annotations_df['has_noun'].sum()}")
    print(f"Total nouns: {tr_annotations_df['n_noun'].sum()}")
    print(f"TRs with adjectives: {tr_annotations_df['has_adj'].sum()}")
    print(f"Total adjectives: {tr_annotations_df['n_adj'].sum()}")
    print(f"TRs with adverbs: {tr_annotations_df['has_adv'].sum()}")
    print(f"Total adverbs: {tr_annotations_df['n_adv'].sum()}")

if __name__ == "__main__":
    main()