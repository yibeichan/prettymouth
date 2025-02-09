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

def main():
    load_dotenv()
    scratch_dir = os.getenv('SCRATCH_DIR')
    data_dir = os.path.join(scratch_dir, 'data', 'stimuli')
    # Load data
    segments_df, words_df = load_data(os.path.join(data_dir, 'segments_speaker.csv'), os.path.join(data_dir, 'word_onset_tagged_zz.csv'))
    
    # Create annotations
    annotations_df = create_annotations(segments_df, words_df)
    
    # Save annotations
    annotations_df.to_csv(os.path.join(data_dir, '10_story_annotations.csv'), index=False)
    
    # Print some statistics
    print("\nAnnotation Statistics:")
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

if __name__ == "__main__":
    main()